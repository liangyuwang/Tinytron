from __future__ import annotations

import glob
import os
from dataclasses import asdict

import torch

from tinytron.bridge import (
    BridgeContext,
    LayoutPlanner,
    ParallelSpec,
    RoutedMaterializer,
    StateDictShardFileStore,
    StateDictTensorStore,
    bridge_metadata,
    build_tinytron_canonical_layout,
    build_tinytron_training_layout,
    canonical_state_dict_to_local_training,
    current_tinytron_parallel_spec,
    current_tinytron_placement,
    localize_layout,
)
from tinytron.model.config import ModelConfig


def checkpoint_prefix_from_model_path(model_path: str) -> str:
    if model_path.endswith("_model.pt"):
        return model_path.removesuffix("_model.pt")
    if "_model_rank" in model_path and model_path.endswith(".pt"):
        return model_path.rsplit("_model_rank", 1)[0]
    if model_path.endswith("_model_canonical.pt"):
        return model_path.removesuffix("_model_canonical.pt")
    raise ValueError(f"Cannot infer checkpoint prefix from model path: {model_path}")


def checkpoint_meta_path(checkpoint_prefix: str) -> str:
    return f"{checkpoint_prefix}_meta.pt"


def checkpoint_model_path(checkpoint_prefix: str) -> str:
    return f"{checkpoint_prefix}_model.pt"


def checkpoint_model_shard_path(checkpoint_prefix: str, rank: int) -> str:
    return f"{checkpoint_prefix}_model_rank{rank:05d}.pt"


def find_latest_checkpoint_prefix(checkpoint_dir: str) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*_model.pt")))
    if not ckpts:
        return None
    return checkpoint_prefix_from_model_path(ckpts[-1])


def model_layout_metadata() -> dict:
    return bridge_metadata(
        layout_kind="training",
        parallel=current_tinytron_parallel_spec(system="training"),
        shard_qkv=False,
    )


def training_layout_matches_current(meta: dict) -> bool:
    layout = meta.get("model_layout")
    if layout is None:
        return True
    if layout.get("layout_kind") != "training":
        return False
    return layout.get("parallel") == asdict(current_tinytron_parallel_spec(system="training"))


def save_local_model_shard(model, checkpoint_prefix: str, rank: int) -> None:
    torch.save(model.state_dict(), checkpoint_model_shard_path(checkpoint_prefix, rank))


def save_rank0_legacy_model(model, checkpoint_prefix: str) -> None:
    torch.save(model.state_dict(), checkpoint_model_path(checkpoint_prefix))


def load_model_state_dict_for_training(
    checkpoint_prefix: str,
    model_config: ModelConfig,
    rank: int,
    meta: dict,
) -> dict[str, torch.Tensor]:
    rank_model_path = checkpoint_model_shard_path(checkpoint_prefix, rank)
    if os.path.exists(rank_model_path) and training_layout_matches_current(meta):
        return torch.load(rank_model_path, map_location="cpu", weights_only=True)

    if _has_sharded_model_checkpoint(checkpoint_prefix, meta):
        return _reshard_model_state_dict_from_files(checkpoint_prefix, model_config, meta)

    model_path = checkpoint_model_path(checkpoint_prefix)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if model_path.endswith("_model_canonical.pt"):
        return canonical_state_dict_to_local_training(state_dict, model_config)
    return state_dict


def _has_sharded_model_checkpoint(checkpoint_prefix: str, meta: dict) -> bool:
    return bool(meta.get("model_sharded")) and os.path.exists(checkpoint_model_shard_path(checkpoint_prefix, 0))


def _reshard_model_state_dict_from_files(
    checkpoint_prefix: str,
    model_config: ModelConfig,
    meta: dict,
) -> dict[str, torch.Tensor]:
    src_layout = _source_layout_from_meta(meta, model_config)
    dst_layout = build_tinytron_training_layout(
        model_config=model_config,
        parallel=current_tinytron_parallel_spec(system="training"),
    )
    dst_placement = current_tinytron_placement(dst_layout)
    src_store = StateDictShardFileStore(
        lambda placement: checkpoint_model_shard_path(checkpoint_prefix, int(placement.rank or 0))
    )
    dst_store = StateDictTensorStore()
    plan = LayoutPlanner().plan(src_layout, localize_layout(dst_layout, dst_placement))
    RoutedMaterializer().materialize(plan, BridgeContext(src_store=src_store, dst_store=dst_store))
    return dst_store.state_dict_for(dst_placement)


def _source_layout_from_meta(meta: dict, model_config: ModelConfig):
    layout = meta.get("model_layout") or {}
    kind = layout.get("layout_kind", "training")
    parallel = ParallelSpec(**layout.get("parallel", asdict(current_tinytron_parallel_spec(system="training"))))
    if kind == "training":
        return build_tinytron_training_layout(model_config=model_config, parallel=parallel)
    if kind == "canonical":
        return build_tinytron_canonical_layout(model_config=model_config)
    raise ValueError(f"Unsupported checkpoint model layout kind: {kind}")
