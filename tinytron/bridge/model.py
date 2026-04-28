from __future__ import annotations

import os
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


def checkpoint_model_paths(model_path: str) -> tuple[str, str | None]:
    if model_path.endswith("_model_canonical.pt"):
        prefix = model_path.removesuffix("_model_canonical.pt")
        return model_path, f"{prefix}_model.pt"
    if model_path.endswith("_model.pt"):
        prefix = model_path.removesuffix("_model.pt")
        canonical_path = f"{prefix}_model_canonical.pt"
        return canonical_path if os.path.exists(canonical_path) else model_path, model_path
    return model_path, None


def checkpoint_meta_path(model_path: str) -> str:
    if model_path.endswith("_model_canonical.pt"):
        return f"{model_path.removesuffix('_model_canonical.pt')}_meta.pt"
    if model_path.endswith("_model.pt"):
        return f"{model_path.removesuffix('_model.pt')}_meta.pt"
    return model_path.replace("_model.pt", "_meta.pt")


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


@torch.no_grad()
def export_tinytron_training_state_dict_to_canonical(
    model,
    model_config: ModelConfig,
) -> dict[str, torch.Tensor] | None:
    """Export the current training layout to a canonical full state dict.

    Returns the canonical state dict only on global rank 0. Other ranks return
    None after participating in any required collectives.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}

    try:
        dp_rank = parallel_state.get_dp_rank()
        sep_rank = parallel_state.get_sep_rank()
        sep_world_size = parallel_state.get_sep_world_size()
        sep_group = parallel_state.get_sep_group()
    except AssertionError:
        return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()} if dist.get_rank() == 0 else None

    if dp_rank != 0:
        return None

    canonical_state_dict: dict[str, torch.Tensor] | None = {} if sep_rank == 0 else None
    for name, tensor in model.state_dict().items():
        local_tensor = tensor.detach().contiguous()
        if model_config.use_moe and _is_expert_param(name) and sep_world_size > 1:
            gathered = [torch.empty_like(local_tensor) for _ in range(sep_world_size)]
            dist.all_gather(gathered, local_tensor, group=sep_group)
            if sep_rank == 0:
                assert canonical_state_dict is not None
                canonical_state_dict[name] = torch.cat(gathered, dim=0).cpu()
        elif sep_rank == 0:
            assert canonical_state_dict is not None
            canonical_state_dict[name] = local_tensor.cpu()

    if dist.get_rank() == 0:
        return canonical_state_dict
    return None


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


def _is_expert_param(name: str) -> bool:
    return name.endswith(EXPERT_PARAM_SUFFIXES)
