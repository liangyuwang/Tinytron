from __future__ import annotations

from dataclasses import dataclass

from tinytron.model.config import ModelConfig

from .layout import LayoutIndex, Placement, RankCoord, ShardSpec, SliceSpec, full_slice


@dataclass(frozen=True)
class ParallelSpec:
    """Tinytron logical parallel shape without binding to live process groups."""

    dp_size: int = 1
    sep_size: int = 1
    system: str = "training"
    group_id: str | None = None

    def placements(self) -> tuple[Placement, ...]:
        placements: list[Placement] = []
        for dp_rank in range(self.dp_size):
            for sep_rank in range(self.sep_size):
                global_rank = dp_rank * self.sep_size + sep_rank
                placements.append(
                    Placement(
                        system=self.system,
                        rank=global_rank,
                        coord=RankCoord.from_axes(dp=dp_rank, sep=sep_rank, ep=sep_rank),
                        group_id=self.group_id,
                    )
                )
        return tuple(placements)


def build_tinytron_canonical_layout(
    model_config: ModelConfig,
    system: str = "canonical",
) -> LayoutIndex:
    placement = Placement(system=system, rank=0, coord=RankCoord.from_axes(replica=0))
    shards = [
        ShardSpec.replicated(name, shape, placement)
        for name, shape in _global_param_shapes(model_config)
    ]
    return LayoutIndex.from_shards("tinytron-canonical", shards)


def build_tinytron_training_layout(
    model_config: ModelConfig,
    parallel: ParallelSpec,
) -> LayoutIndex:
    placements = parallel.placements()
    shards: list[ShardSpec] = []
    for name, shape in _global_param_shapes(model_config):
        if _is_expert_param(name):
            for placement in placements:
                sep_rank = _placement_axis(placement, "sep")
                shards.append(
                    _expert_shard(
                        name=name,
                        shape=shape,
                        placement=placement,
                        sep_rank=sep_rank,
                        sep_size=parallel.sep_size,
                    )
                )
        else:
            for placement in placements:
                shards.append(
                    ShardSpec.replicated(
                        name,
                        shape,
                        placement,
                        replica_group=f"{parallel.system}:{name}",
                    )
                )
    return LayoutIndex.from_shards("tinytron-training", shards)


def build_tinytron_inference_layout(
    model_config: ModelConfig,
    parallel: ParallelSpec,
    shard_qkv: bool = False,
) -> LayoutIndex:
    placements = parallel.placements()
    shards: list[ShardSpec] = []
    for name, shape in _global_param_shapes(model_config):
        if shard_qkv and _is_qkv_param(name):
            for placement in placements:
                sep_rank = _placement_axis(placement, "sep")
                shards.append(
                    _dim0_shard(
                        name=name,
                        shape=shape,
                        placement=placement,
                        rank=sep_rank,
                        world_size=parallel.sep_size,
                        axis_tag="qkv_head",
                    )
                )
        elif _is_expert_param(name):
            for placement in placements:
                sep_rank = _placement_axis(placement, "sep")
                shards.append(
                    _expert_shard(
                        name=name,
                        shape=shape,
                        placement=placement,
                        sep_rank=sep_rank,
                        sep_size=parallel.sep_size,
                    )
                )
        else:
            for placement in placements:
                shards.append(
                    ShardSpec.replicated(
                        name,
                        shape,
                        placement,
                        replica_group=f"{parallel.system}:{name}",
                    )
                )
    return LayoutIndex.from_shards("tinytron-inference", shards)


def _global_param_shapes(model_config: ModelConfig) -> tuple[tuple[str, tuple[int, ...]], ...]:
    H = model_config.hidden_size
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    q_out = model_config.num_attention_heads * head_dim
    kv_out = model_config.num_key_value_heads * head_dim
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("wte.weight", (model_config.vocab_size, H)),
    ]

    for layer_idx in range(model_config.num_layer):
        prefix = f"blocks.{layer_idx}"
        shapes.extend(
            [
                (f"{prefix}.ln_1.weight", (H,)),
                (f"{prefix}.attn.q_proj.weight", (q_out, H)),
                (f"{prefix}.attn.k_proj.weight", (kv_out, H)),
                (f"{prefix}.attn.v_proj.weight", (kv_out, H)),
                (f"{prefix}.attn.c_proj.weight", (H, H)),
                (f"{prefix}.ln_2.weight", (H,)),
            ]
        )
        if model_config.use_moe:
            I = model_config.moe_intermediate_size
            E = model_config.num_experts
            shapes.extend(
                [
                    (f"{prefix}.mlp.router.weight", (E, H)),
                    (f"{prefix}.mlp.experts_gate_weights", (E, I, H)),
                    (f"{prefix}.mlp.experts_up_weights", (E, I, H)),
                    (f"{prefix}.mlp.experts_down_weights", (E, H, I)),
                ]
            )
        else:
            I = model_config.intermediate_size
            shapes.extend(
                [
                    (f"{prefix}.mlp.gate_proj.weight", (I, H)),
                    (f"{prefix}.mlp.up_proj.weight", (I, H)),
                    (f"{prefix}.mlp.down_proj.weight", (H, I)),
                ]
            )

    shapes.extend(
        [
            ("lnf.weight", (H,)),
            ("lm_head.weight", (model_config.vocab_size, H)),
        ]
    )
    return tuple(shapes)


def _is_qkv_param(name: str) -> bool:
    return name.endswith(
        (
            ".attn.q_proj.weight",
            ".attn.k_proj.weight",
            ".attn.v_proj.weight",
        )
    )


def _is_expert_param(name: str) -> bool:
    return name.endswith(
        (
            ".mlp.experts_gate_weights",
            ".mlp.experts_up_weights",
            ".mlp.experts_down_weights",
        )
    )


def _placement_axis(placement: Placement, axis: str) -> int:
    if placement.coord is None:
        raise ValueError(f"placement {placement} has no rank coordinate")
    value = placement.coord.get(axis)
    if value is None:
        raise ValueError(f"placement {placement} has no {axis!r} coordinate")
    return value


def _dim0_shard(
    name: str,
    shape: tuple[int, ...],
    placement: Placement,
    rank: int,
    world_size: int,
    axis_tag: str,
) -> ShardSpec:
    if shape[0] % world_size != 0:
        raise ValueError(f"{name} dim0={shape[0]} must be divisible by world_size={world_size}")
    chunk = shape[0] // world_size
    slices: list[SliceSpec] = list(full_slice(shape))
    slices[0] = (rank * chunk, (rank + 1) * chunk)
    return ShardSpec(
        param_name=name,
        global_shape=shape,
        global_slices=tuple(slices),
        placement=placement,
        axis_tags=(axis_tag,),
    )


def _expert_shard(
    name: str,
    shape: tuple[int, ...],
    placement: Placement,
    sep_rank: int,
    sep_size: int,
) -> ShardSpec:
    return _dim0_shard(
        name=name,
        shape=shape,
        placement=placement,
        rank=sep_rank,
        world_size=sep_size,
        axis_tag="expert",
    )
