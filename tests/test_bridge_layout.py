from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tinytron.bridge import (
    BridgeContext,
    LayoutIndex,
    LayoutPlanner,
    ParallelSpec,
    Placement,
    RankCoord,
    RoutedMaterializer,
    ShardSpec,
    build_tinytron_canonical_layout,
    build_tinytron_inference_layout,
    build_tinytron_training_layout,
)
from tinytron.bridge.layout import intersect_slices, local_slices
from tinytron.model.config import ModelConfig


def tiny_moe_config() -> ModelConfig:
    return ModelConfig(
        block_size=16,
        vocab_size=32,
        num_layer=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=8,
        intermediate_size=16,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
    )


class BridgeLayoutPlannerTest(unittest.TestCase):
    def test_training_layout_shards_experts_by_sep_rank(self) -> None:
        cfg = tiny_moe_config()
        layout = build_tinytron_training_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=2, system="training"),
        )

        shards = layout.shards_for("blocks.0.mlp.experts_gate_weights")

        self.assertEqual(len(shards), 2)
        self.assertEqual([shard.global_slices[0] for shard in shards], [(0, 2), (2, 4)])
        self.assertEqual([shard.local_shape for shard in shards], [(2, 4, 8), (2, 4, 8)])
        self.assertEqual({shard.axis_tags for shard in shards}, {("expert",)})

    def test_training_to_inference_plan_handles_sep_change(self) -> None:
        cfg = tiny_moe_config()
        src = build_tinytron_training_layout(
            cfg,
            ParallelSpec(dp_size=2, sep_size=4, system="training"),
        )
        dst = build_tinytron_inference_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=2, system="inference"),
            shard_qkv=True,
        )

        plan = LayoutPlanner().plan(src, dst)
        expert_moves = [move for move in plan if move.param_name == "blocks.0.mlp.experts_gate_weights"]
        q_moves = [move for move in plan if move.param_name == "blocks.0.attn.q_proj.weight"]

        self.assertEqual([move.global_slices[0] for move in expert_moves], [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertEqual([move.src.rank for move in expert_moves], [0, 1, 2, 3])
        self.assertEqual([move.dst.rank for move in expert_moves], [0, 0, 1, 1])
        self.assertEqual([move.global_slices[0] for move in q_moves], [(0, 4), (4, 8)])
        self.assertEqual([move.dst.rank for move in q_moves], [0, 1])

    def test_canonical_to_inference_plan_slices_qkv(self) -> None:
        cfg = tiny_moe_config()
        src = build_tinytron_canonical_layout(cfg)
        dst = build_tinytron_inference_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=2, system="inference"),
            shard_qkv=True,
        )

        plan = LayoutPlanner().plan(src, dst)
        q_moves = [move for move in plan if move.param_name == "blocks.0.attn.q_proj.weight"]
        k_moves = [move for move in plan if move.param_name == "blocks.0.attn.k_proj.weight"]
        v_moves = [move for move in plan if move.param_name == "blocks.0.attn.v_proj.weight"]

        self.assertEqual([move.global_slices[0] for move in q_moves], [(0, 4), (4, 8)])
        self.assertEqual([move.global_slices[0] for move in k_moves], [(0, 2), (2, 4)])
        self.assertEqual([move.global_slices[0] for move in v_moves], [(0, 2), (2, 4)])

    def test_planner_rejects_uncovered_target_shard(self) -> None:
        src_place = Placement(system="src", rank=0, coord=RankCoord.from_axes(replica=0))
        dst_place = Placement(system="dst", rank=0, coord=RankCoord.from_axes(replica=0))
        src = LayoutIndex.from_shards(
            "src",
            [
                ShardSpec(
                    param_name="weight",
                    global_shape=(4, 2),
                    global_slices=((0, 2), (0, 2)),
                    placement=src_place,
                )
            ],
        )
        dst = LayoutIndex.from_shards(
            "dst",
            [
                ShardSpec(
                    param_name="weight",
                    global_shape=(4, 2),
                    global_slices=((0, 4), (0, 2)),
                    placement=dst_place,
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "does not cover target shard"):
            LayoutPlanner().plan(src, dst)

    def test_slice_helpers_return_expected_overlap_and_local_offsets(self) -> None:
        overlap = intersect_slices(((0, 4), (0, 8)), ((2, 6), (3, 5)))

        self.assertEqual(overlap, ((2, 4), (3, 5)))
        self.assertEqual(local_slices(((0, 4), (0, 8)), overlap), ((2, 4), (3, 5)))
        self.assertEqual(local_slices(((2, 6), (3, 5)), overlap), ((0, 2), (0, 2)))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class BridgeMaterializerTest(unittest.TestCase):
    def test_state_dict_materializer_copies_overlapping_slices(self) -> None:
        import torch

        from tinytron.bridge import StateDictShardFileStore, StateDictTensorStore, localize_layout

        src_place = Placement(system="src", rank=0, coord=RankCoord.from_axes(replica=0))
        dst_place = Placement(system="dst", rank=0, coord=RankCoord.from_axes(replica=0))
        src = LayoutIndex.from_shards(
            "src",
            [
                ShardSpec(
                    param_name="weight",
                    global_shape=(4, 2),
                    global_slices=((0, 4), (0, 2)),
                    placement=src_place,
                )
            ],
        )
        dst = LayoutIndex.from_shards(
            "dst",
            [
                ShardSpec(
                    param_name="weight",
                    global_shape=(4, 2),
                    global_slices=((1, 3), (0, 2)),
                    placement=dst_place,
                )
            ],
        )
        src_store = StateDictTensorStore(
            {src_place: {"weight": torch.arange(8, dtype=torch.float32).view(4, 2)}}
        )
        dst_store = StateDictTensorStore()

        plan = LayoutPlanner().plan(src, dst)
        RoutedMaterializer().materialize(plan, BridgeContext(src_store, dst_store))

        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        self.assertTrue(torch.equal(dst_store.state_dict_for(dst_place)["weight"], expected))

    def test_file_shard_store_materializes_sep_reshard_without_rank0_gather(self) -> None:
        import torch

        from tinytron.bridge import StateDictShardFileStore, StateDictTensorStore, localize_layout

        cfg = tiny_moe_config()
        src_layout = build_tinytron_training_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=4, system="training"),
        )
        dst_layout = build_tinytron_inference_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=2, system="inference"),
            shard_qkv=True,
        )
        dst_placement = [placement for placement in dst_layout.placements() if placement.rank == 0][0]
        dst_local_layout = localize_layout(dst_layout, dst_placement)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for placement in src_layout.placements():
                torch.save(
                    self._state_dict_for_placement(src_layout, placement),
                    root / f"rank{placement.rank:05d}.pt",
                )

            src_store = StateDictShardFileStore(lambda placement: root / f"rank{placement.rank:05d}.pt")
            dst_store = StateDictTensorStore()
            plan = LayoutPlanner().plan(src_layout, dst_local_layout)
            RoutedMaterializer().materialize(plan, BridgeContext(src_store, dst_store))

            out = dst_store.state_dict_for(dst_placement)
            expert = out["blocks.0.mlp.experts_gate_weights"]
            q_proj = out["blocks.0.attn.q_proj.weight"]

            self.assertEqual(tuple(expert.shape), (2, 4, 8))
            self.assertEqual(tuple(q_proj.shape), (4, 8))
            self.assertTrue(torch.equal(expert[0], self._filled((1, 4, 8), 0)[0]))
            self.assertTrue(torch.equal(expert[1], self._filled((1, 4, 8), 1000)[0]))
            self.assertTrue(torch.equal(q_proj, self._filled((8, 8), 0)[:4]))

    def test_training_checkpoint_load_reshards_when_sep_changes(self) -> None:
        import torch

        from tinytron.training import checkpoint as training_checkpoint

        cfg = tiny_moe_config()
        src_layout = build_tinytron_training_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=4, system="training"),
        )
        target_parallel = ParallelSpec(dp_size=1, sep_size=2, system="training")
        target_layout = build_tinytron_training_layout(cfg, target_parallel)
        target_placement = [placement for placement in target_layout.placements() if placement.rank == 0][0]
        meta = {
            "model_sharded": True,
            "model_layout": {
                "layout_kind": "training",
                "parallel": {
                    "dp_size": 1,
                    "sep_size": 4,
                    "system": "training",
                    "group_id": None,
                },
                "shard_qkv": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "00001")
            for placement in src_layout.placements():
                torch.save(
                    self._state_dict_for_placement(src_layout, placement),
                    training_checkpoint.checkpoint_model_shard_path(prefix, int(placement.rank)),
                )

            with mock.patch.object(
                training_checkpoint,
                "current_tinytron_parallel_spec",
                return_value=target_parallel,
            ), mock.patch.object(
                training_checkpoint,
                "current_tinytron_placement",
                return_value=target_placement,
            ):
                state_dict = training_checkpoint.load_model_state_dict_for_training(
                    checkpoint_prefix=prefix,
                    model_config=cfg,
                    rank=0,
                    meta=meta,
                )

            expert = state_dict["blocks.0.mlp.experts_gate_weights"]
            self.assertEqual(tuple(expert.shape), (2, 4, 8))
            self.assertTrue(torch.equal(expert[0], self._filled((1, 4, 8), 0)[0]))
            self.assertTrue(torch.equal(expert[1], self._filled((1, 4, 8), 1000)[0]))

    def test_inference_checkpoint_load_reshards_sharded_training_checkpoint(self) -> None:
        import torch

        from tinytron.inference import checkpoint as inference_checkpoint

        cfg = tiny_moe_config()
        cfg = replace(cfg, inference_shard_qkv=True)
        src_layout = build_tinytron_training_layout(
            cfg,
            ParallelSpec(dp_size=1, sep_size=4, system="training"),
        )
        target_parallel = ParallelSpec(dp_size=1, sep_size=2, system="inference")
        target_layout = build_tinytron_inference_layout(cfg, target_parallel, shard_qkv=True)
        target_placement = [placement for placement in target_layout.placements() if placement.rank == 0][0]
        meta = {
            "config": {"model": cfg.__dict__},
            "model_sharded": True,
            "model_layout": {
                "layout_kind": "training",
                "parallel": {
                    "dp_size": 1,
                    "sep_size": 4,
                    "system": "training",
                    "group_id": None,
                },
                "shard_qkv": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            prefix = str(Path(tmp) / "00001")
            torch.save(meta, inference_checkpoint.checkpoint_meta_path(prefix))
            for placement in src_layout.placements():
                torch.save(
                    self._state_dict_for_placement(src_layout, placement),
                    inference_checkpoint.checkpoint_model_shard_path(prefix, int(placement.rank)),
                )

            with mock.patch.object(
                inference_checkpoint,
                "current_tinytron_parallel_spec",
                return_value=target_parallel,
            ), mock.patch.object(
                inference_checkpoint,
                "current_tinytron_placement",
                return_value=target_placement,
            ):
                state_dict = inference_checkpoint.load_model_state_dict_for_inference(
                    checkpoint_path=f"{prefix}_model.pt",
                    model_config=cfg,
                )

            expert = state_dict["blocks.0.mlp.experts_gate_weights"]
            q_proj = state_dict["blocks.0.attn.q_proj.weight"]
            self.assertEqual(tuple(expert.shape), (2, 4, 8))
            self.assertEqual(tuple(q_proj.shape), (4, 8))
            self.assertTrue(torch.equal(expert[0], self._filled((1, 4, 8), 0)[0]))
            self.assertTrue(torch.equal(expert[1], self._filled((1, 4, 8), 1000)[0]))
            self.assertTrue(torch.equal(q_proj, self._filled((8, 8), 0)[:4]))

    def _state_dict_for_placement(self, layout: LayoutIndex, placement: Placement) -> dict:
        rank = int(placement.rank or 0)
        return {
            shard.param_name: self._filled(shard.local_shape, rank * 1000)
            for shard in layout.local_shards(placement)
        }

    def _filled(self, shape: tuple[int, ...], offset: int):
        import torch

        numel = 1
        for dim in shape:
            numel *= dim
        return torch.arange(offset, offset + numel, dtype=torch.float32).reshape(shape)


if __name__ == "__main__":
    unittest.main()
