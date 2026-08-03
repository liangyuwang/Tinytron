"""aimegatron.train.layout

The checkpoint layout rule table: for every parameter name, which dimension
is sharded across TP (or None when replicated). This single table drives
both checkpoint resharding and is the reference for what each TP shard
file contains.

Edit contract: when a new TP-sharded module is added, append its weight
pattern here. Everything else (checkpoint.py) picks the rule up for free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    shard_dim: int | None  # None -> replicated across TP


# (substring of parameter name, shard dimension). First match wins.
TP_SHARD_RULES: list[tuple[str, int]] = [
    ("q_proj.weight", 0),       # [heads*head_dim, hidden] sharded on output dim
    ("k_proj.weight", 0),
    ("v_proj.weight", 0),
    ("c_proj.weight", 1),       # [hidden, hidden] sharded on input dim
    ("gate_proj.weight", 0),
    ("up_proj.weight", 0),
    ("down_proj.weight", 1),
    ("wte.weight", 0),          # [vocab, hidden] sharded on vocab dim
    ("lm_head.weight", 0),
]


def get_layout(param_name: str) -> Layout:
    for pattern, shard_dim in TP_SHARD_RULES:
        if pattern in param_name:
            return Layout(shard_dim=shard_dim)
    return Layout(shard_dim=None)


def stage_of_block(block_idx: int, pp_size: int, num_layer: int) -> int:
    """Which pipeline stage owns a block (even split)."""
    per_stage = num_layer // pp_size
    return block_idx // per_stage


def source_stage_of_key(key: str, src_pp_size: int, num_layer: int) -> int:
    """Which source pipeline stage file contains a state-dict key."""
    if key.startswith("wte."):
        return 0
    if key.startswith("lm_head.") or key.startswith("lnf."):
        return src_pp_size - 1
    if key.startswith("blocks."):
        return stage_of_block(int(key.split(".")[1]), src_pp_size, num_layer)
    raise ValueError(f"cannot map checkpoint key to a pipeline stage: {key}")


def reshard_tensor(
    shards: list,
    param_name: str,
    source_tp_size: int,
    target_tp_size: int,
    target_tp_rank: int,
):
    """Rebuild the target_tp_size/target_tp_rank shard of a parameter from the
    source_tp_size shards. `shards` must be ordered by source TP rank."""
    import torch

    assert len(shards) == source_tp_size, \
        f"expected {source_tp_size} source shards for {param_name}, got {len(shards)}"
    layout = get_layout(param_name)

    if layout.shard_dim is None or source_tp_size == 1:
        full = shards[0]
    else:
        full = torch.cat(shards, dim=layout.shard_dim)

    if layout.shard_dim is None or target_tp_size == 1:
        return full.clone()

    assert full.size(layout.shard_dim) % target_tp_size == 0, \
        f"{param_name}: dim {layout.shard_dim} size {full.size(layout.shard_dim)} " \
        f"not divisible by target tp_size {target_tp_size}"
    return full.chunk(target_tp_size, dim=layout.shard_dim)[target_tp_rank].clone()
