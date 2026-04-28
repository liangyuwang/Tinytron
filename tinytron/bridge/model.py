from __future__ import annotations

from dataclasses import asdict

import torch
import torch.distributed as dist

from tinytron.model.config import ModelConfig
from tinytron.distributed import parallel_state

from .layout import LayoutIndex, Placement
from .materializers import BridgeContext, RoutedMaterializer
from .planner import LayoutPlanner
from .rules import (
    ParallelSpec,
    build_tinytron_canonical_layout,
    build_tinytron_inference_layout,
    build_tinytron_training_layout,
)
from .stores import StateDictTensorStore


EXPERT_PARAM_SUFFIXES = (
    ".mlp.experts_gate_weights",
    ".mlp.experts_up_weights",
    ".mlp.experts_down_weights",
)


def current_tinytron_parallel_spec(system: str) -> ParallelSpec:
    if dist.is_available() and dist.is_initialized():
        try:
            return ParallelSpec(
                dp_size=int(parallel_state.get_dp_world_size()),
                sep_size=int(parallel_state.get_sep_world_size()),
                system=system,
            )
        except AssertionError:
            pass
    return ParallelSpec(dp_size=1, sep_size=1, system=system)


def current_tinytron_placement(layout: LayoutIndex) -> Placement:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    for placement in layout.placements():
        if placement.rank == rank:
            return placement
    if rank == 0 and len(layout.placements()) == 1:
        return layout.placements()[0]
    raise ValueError(f"layout {layout.name!r} has no placement for global rank {rank}")


def localize_layout(layout: LayoutIndex, placement: Placement) -> LayoutIndex:
    return LayoutIndex.from_shards(
        name=f"{layout.name}:local:{placement.rank}",
        shards=layout.local_shards(placement),
    )


def bridge_metadata(
    *,
    layout_kind: str,
    parallel: ParallelSpec,
    shard_qkv: bool = False,
    format_version: int = 1,
) -> dict:
    return {
        "bridge_format_version": format_version,
        "layout_kind": layout_kind,
        "parallel": asdict(parallel),
        "shard_qkv": bool(shard_qkv),
    }


def canonical_state_dict_to_local_inference(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> dict[str, torch.Tensor]:
    if not _needs_inference_bridge(state_dict, model_config):
        return state_dict

    src_layout = build_tinytron_canonical_layout(model_config)
    dst_layout = build_tinytron_inference_layout(
        model_config=model_config,
        parallel=current_tinytron_parallel_spec(system="inference"),
        shard_qkv=model_config.inference_shard_qkv,
    )
    dst_placement = current_tinytron_placement(dst_layout)
    return _materialize_local_state_dict(
        src_state_dict=state_dict,
        src_layout=src_layout,
        dst_layout=localize_layout(dst_layout, dst_placement),
        dst_placement=dst_placement,
    )


def state_dict_to_local_inference(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> dict[str, torch.Tensor]:
    if _looks_like_local_training_state_dict(state_dict, model_config):
        return local_training_state_dict_to_local_inference(state_dict, model_config)
    return canonical_state_dict_to_local_inference(state_dict, model_config)


def local_training_state_dict_to_local_inference(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> dict[str, torch.Tensor]:
    if not _needs_inference_bridge(state_dict, model_config):
        return state_dict

    parallel = current_tinytron_parallel_spec(system="training")
    src_layout = build_tinytron_training_layout(model_config=model_config, parallel=parallel)
    src_placement = current_tinytron_placement(src_layout)
    dst_layout = build_tinytron_inference_layout(
        model_config=model_config,
        parallel=current_tinytron_parallel_spec(system="inference"),
        shard_qkv=model_config.inference_shard_qkv,
    )
    dst_placement = current_tinytron_placement(dst_layout)
    return _materialize_local_state_dict(
        src_state_dict=state_dict,
        src_layout=localize_layout(src_layout, src_placement),
        dst_layout=localize_layout(dst_layout, dst_placement),
        dst_placement=dst_placement,
    )


def canonical_state_dict_to_local_training(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> dict[str, torch.Tensor]:
    if not _needs_training_bridge(state_dict, model_config):
        return state_dict

    src_layout = build_tinytron_canonical_layout(model_config)
    dst_layout = build_tinytron_training_layout(
        model_config=model_config,
        parallel=current_tinytron_parallel_spec(system="training"),
    )
    dst_placement = current_tinytron_placement(dst_layout)
    return _materialize_local_state_dict(
        src_state_dict=state_dict,
        src_layout=src_layout,
        dst_layout=localize_layout(dst_layout, dst_placement),
        dst_placement=dst_placement,
    )


def _materialize_local_state_dict(
    *,
    src_state_dict: dict[str, torch.Tensor],
    src_layout: LayoutIndex,
    dst_layout: LayoutIndex,
    dst_placement: Placement,
) -> dict[str, torch.Tensor]:
    src_placement = src_layout.placements()[0]
    src_store = StateDictTensorStore({src_placement: src_state_dict})
    dst_store = StateDictTensorStore()
    plan = LayoutPlanner().plan(src_layout, dst_layout)
    RoutedMaterializer().materialize(plan, BridgeContext(src_store=src_store, dst_store=dst_store))
    return dst_store.state_dict_for(dst_placement)


def _needs_inference_bridge(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> bool:
    try:
        sep_size = parallel_state.get_sep_world_size()
    except AssertionError:
        sep_size = 1

    if model_config.inference_shard_qkv and sep_size and sep_size > 1:
        q_key = "blocks.0.attn.q_proj.weight"
        weight = state_dict.get(q_key)
        if weight is not None and weight.size(0) == model_config.hidden_size:
            return True

    if model_config.use_moe and sep_size and sep_size > 1:
        expert_key = "blocks.0.mlp.experts_gate_weights"
        weight = state_dict.get(expert_key)
        if weight is not None and weight.size(0) == model_config.num_experts:
            return True

    return False


def _needs_training_bridge(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> bool:
    if not model_config.use_moe:
        return False

    try:
        sep_size = parallel_state.get_sep_world_size()
    except AssertionError:
        sep_size = 1
    if not sep_size or sep_size <= 1:
        return False

    expert_key = "blocks.0.mlp.experts_gate_weights"
    weight = state_dict.get(expert_key)
    return weight is not None and weight.size(0) == model_config.num_experts


def _looks_like_local_training_state_dict(
    state_dict: dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> bool:
    if not model_config.use_moe:
        return False
    try:
        sep_size = parallel_state.get_sep_world_size()
    except AssertionError:
        sep_size = 1
    if not sep_size or sep_size <= 1:
        return False
    expert_key = "blocks.0.mlp.experts_gate_weights"
    weight = state_dict.get(expert_key)
    return weight is not None and weight.size(0) == model_config.num_experts // sep_size


def _is_expert_param(name: str) -> bool:
    return name.endswith(EXPERT_PARAM_SUFFIXES)
