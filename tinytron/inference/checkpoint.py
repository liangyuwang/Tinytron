from __future__ import annotations

import os

import torch

from tinytron.bridge import (
    BridgeContext,
    LayoutPlanner,
    ParallelSpec,
    RoutedMaterializer,
    StateDictShardFileStore,
    StateDictTensorStore,
    build_tinytron_canonical_layout,
    build_tinytron_inference_layout,
    build_tinytron_training_layout,
    current_tinytron_parallel_spec,
    current_tinytron_placement,
    localize_layout,
    state_dict_to_local_inference,
)
from tinytron.model.config import ModelConfig


def checkpoint_prefix_from_model_path(model_path: str) -> str | None:
    if model_path.endswith("_model.pt"):
        return model_path.removesuffix("_model.pt")
    if "_model_rank" in model_path and model_path.endswith(".pt"):
        return model_path.rsplit("_model_rank", 1)[0]
    if model_path.endswith("_model_canonical.pt"):
        return model_path.removesuffix("_model_canonical.pt")
    return None


def checkpoint_meta_path(checkpoint_prefix: str) -> str:
    return f"{checkpoint_prefix}_meta.pt"


def checkpoint_model_shard_path(checkpoint_prefix: str, rank: int) -> str:
    return f"{checkpoint_prefix}_model_rank{rank:05d}.pt"


def load_model_state_dict_for_inference(
    checkpoint_path: str,
    model_config: ModelConfig,
) -> dict[str, torch.Tensor]:
    checkpoint_prefix = checkpoint_prefix_from_model_path(checkpoint_path)
    if checkpoint_prefix is not None:
        meta_path = checkpoint_meta_path(checkpoint_prefix)
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            if _has_sharded_model_checkpoint(checkpoint_prefix, meta):
                return _reshard_model_state_dict_from_files(checkpoint_prefix, model_config, meta)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return state_dict_to_local_inference(state_dict, model_config)


def _has_sharded_model_checkpoint(checkpoint_prefix: str, meta: dict) -> bool:
    return bool(meta.get("model_sharded")) and os.path.exists(checkpoint_model_shard_path(checkpoint_prefix, 0))


def _reshard_model_state_dict_from_files(
    checkpoint_prefix: str,
    model_config: ModelConfig,
    meta: dict,
) -> dict[str, torch.Tensor]:
    src_layout = _source_layout_from_meta(meta, model_config)
    dst_layout = build_tinytron_inference_layout(
        model_config=model_config,
        parallel=current_tinytron_parallel_spec(system="inference"),
        shard_qkv=model_config.inference_shard_qkv,
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
    kind = layout.get("layout_kind", "canonical")
    parallel = ParallelSpec(**layout.get("parallel", {}))
    if kind == "training":
        return build_tinytron_training_layout(model_config=model_config, parallel=parallel)
    if kind == "canonical":
        return build_tinytron_canonical_layout(model_config=model_config)
    raise ValueError(f"Unsupported checkpoint model layout kind: {kind}")
