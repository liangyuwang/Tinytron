"""Checkpoint save/load, TP resharding, and PP dimension tests.

Single-process tests cover the layout rule table and roundtrips; the
multi-process tests (gloo) cover the pp dimension: same-layout roundtrip
and cross-pp load (pp=2 -> pp=1) with exact weight match.
"""

import functools
import os
import tempfile
import unittest

import torch

from aimegatron.core.config import ModelConfig
from aimegatron.model.gpt import GPT
from aimegatron.model.pipeline import PipelineStage
from aimegatron.train import checkpoint as ckpt
from aimegatron.train.layout import get_layout, reshard_tensor
from common import run_distributed


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16, hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=16, num_layer=2, block_size=8, tied_lm_head=True,
    )


def _pp_model_config() -> ModelConfig:
    # untied: pp>1 keeps wte and lm_head on different stages.
    return ModelConfig(
        vocab_size=16, hidden_size=8, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=16, num_layer=2, block_size=8, tied_lm_head=False,
    )


def _perturb(module) -> None:
    """Deterministic per-key perturbation so a load actually moves values."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            p.add_(torch.linspace(0.0, 1.0, p.numel()).reshape(p.shape) * 0.01)


def _worker_pp2_roundtrip(rank, out_dir):
    """Save a pp=2 stage, load it into a fresh stage of the same layout."""
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    _perturb(stage)
    before = {k: v.detach().clone() for k, v in stage.state_dict().items()}
    path = ckpt.save_checkpoint(out_dir, 3, stage, extra_meta={"num_layer": cfg.num_layer})

    fresh = PipelineStage(cfg)
    meta = ckpt.load_checkpoint(path, fresh)
    assert meta["step"] == 3 and meta["pp_size"] == 2
    for name, value in fresh.state_dict().items():
        assert torch.equal(value, before[name]), f"pp roundtrip mismatch at {name}"


def _worker_pp2_save(rank, out_dir):
    """Save a perturbed pp=2 checkpoint + the merged reference weights."""
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    _perturb(stage)
    torch.save({k: v.detach().cpu() for k, v in stage.state_dict().items()},
               os.path.join(out_dir, f"stage_{rank}_ref.pt"))
    ckpt.save_checkpoint(os.path.join(out_dir, "ckpt"), 7, stage,
                         extra_meta={"num_layer": cfg.num_layer})


def _worker_pp1_cross_load(rank, out_dir):
    """pp=2 checkpoint -> pp=1 GPT: every weight must match the merged stages."""
    cfg = _pp_model_config()
    merged = {}
    for stage_rank in (0, 1):
        merged.update(torch.load(os.path.join(out_dir, f"stage_{stage_rank}_ref.pt"),
                                 map_location="cpu", weights_only=True))
    gpt = GPT(cfg)
    meta = ckpt.load_checkpoint(os.path.join(out_dir, "ckpt"), gpt)
    assert meta["step"] == 7
    state = gpt.state_dict()
    assert set(state.keys()) == set(merged.keys()), \
        f"key mismatch: {set(state) ^ set(merged)}"
    for name, value in state.items():
        assert torch.equal(value, merged[name]), f"cross-pp load mismatch at {name}"


class TestCheckpoint(unittest.TestCase):

    def test_same_layout_roundtrip(self):
        model = GPT(_tiny_model_config())
        with tempfile.TemporaryDirectory() as log_dir:
            path = ckpt.save_checkpoint(log_dir, 10, model)
            fresh = GPT(_tiny_model_config())
            meta = ckpt.load_checkpoint(path, fresh)
        self.assertEqual(meta["step"], 10)
        self.assertEqual(meta["tp_size"], 1)
        for (name_a, a), (name_b, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(a, b), f"roundtrip mismatch at {name_a}")

    def test_reshard_tp1_to_tp2_and_back(self):
        model = GPT(_tiny_model_config())
        state = model.state_dict()

        for name, tensor in state.items():
            layout = get_layout(name)
            # tp=1 -> tp=2: each target shard equals the manual chunk.
            for target_rank in (0, 1):
                got = reshard_tensor([tensor], name, source_tp_size=1,
                                     target_tp_size=2, target_tp_rank=target_rank)
                if layout.shard_dim is None:
                    expected = tensor
                else:
                    expected = tensor.chunk(2, dim=layout.shard_dim)[target_rank]
                self.assertTrue(torch.equal(got, expected), f"reshard 1->2 mismatch at {name}")

            if layout.shard_dim is not None:
                # tp=2 -> tp=1: concatenating the shards restores the tensor.
                shards = list(tensor.chunk(2, dim=layout.shard_dim))
                got_full = reshard_tensor(shards, name, source_tp_size=2,
                                          target_tp_size=1, target_tp_rank=0)
                self.assertTrue(torch.equal(got_full, tensor), f"reshard 2->1 mismatch at {name}")

    def test_find_latest_checkpoint(self):
        model = GPT(_tiny_model_config())
        with tempfile.TemporaryDirectory() as log_dir:
            self.assertIsNone(ckpt.find_latest_checkpoint(log_dir))
            ckpt.save_checkpoint(log_dir, 5, model)
            ckpt.save_checkpoint(log_dir, 15, model)
            latest = ckpt.find_latest_checkpoint(log_dir)
            self.assertTrue(latest.endswith("step_0000015"))

    def test_pp2_same_layout_roundtrip(self):
        with tempfile.TemporaryDirectory() as out_dir:
            run_distributed(2, functools.partial(_worker_pp2_roundtrip, out_dir=out_dir),
                            pp_size=2)

    def test_cross_pp_load_pp2_to_pp1(self):
        with tempfile.TemporaryDirectory() as out_dir:
            run_distributed(2, functools.partial(_worker_pp2_save, out_dir=out_dir),
                            pp_size=2)
            run_distributed(1, functools.partial(_worker_pp1_cross_load, out_dir=out_dir),
                            pp_size=1)

    def test_cross_ep_load_raises(self):
        model = GPT(_tiny_model_config())
        with tempfile.TemporaryDirectory() as log_dir:
            path = ckpt.save_checkpoint(log_dir, 1, model)
            meta = torch.load(os.path.join(path, ckpt.META_FILE), weights_only=True)
            meta["ep_size"] = 2
            torch.save(meta, os.path.join(path, ckpt.META_FILE))
            with self.assertRaises(NotImplementedError):
                ckpt.load_checkpoint(path, GPT(_tiny_model_config()))


if __name__ == "__main__":
    unittest.main()
