"""aimegatron.core.mesh

Builds the process-group mesh purely from a ParallelConfig.

Topology: ranks form a pp_size x dp_size x tp_size grid with global rank
    ((pp_rank * dp_size) + dp_rank) * tp_size + tp_rank
TP groups are contiguous rank slices (Megatron convention), DP groups live
within a pipeline stage, and PP groups connect the same (dp, tp) position
across stages. Sequence parallelism rides on the TP group and needs no
extra groups.

Expert parallelism lives inside the DP dimension (Megatron-Core style):
ep_size divides dp_size, experts are sharded across contiguous EP slices of
the DP group, and ranks holding identical experts form an expert-DP group
of size dp_size / ep_size.

Edit contract: this is the only module that constructs process groups.
Everything else asks the accessors below. When parallelism is not
initialized (single-process use, e.g. tests or tooling), all accessors fall
back to size=1 / rank=0 semantics.
"""

import torch.distributed as dist

from aimegatron.core.config import ParallelConfig

_INITIALIZED = False
_SEQUENCE_PARALLEL = False

_TP_GROUP = None
_DP_GROUP = None
_PP_GROUP = None
_EP_GROUP = None
_EXPERT_DP_GROUP = None

_TP_SIZE = 1
_DP_SIZE = 1
_PP_SIZE = 1
_EP_SIZE = 1
_EXPERT_DP_SIZE = 1

_TP_RANK = 0
_DP_RANK = 0
_PP_RANK = 0
_EP_RANK = 0
_EXPERT_DP_RANK = 0


def initialize_parallel(config: ParallelConfig) -> None:
    global _INITIALIZED, _SEQUENCE_PARALLEL
    global _TP_GROUP, _DP_GROUP, _PP_GROUP, _EP_GROUP, _EXPERT_DP_GROUP
    global _TP_SIZE, _DP_SIZE, _PP_SIZE, _EP_SIZE, _EXPERT_DP_SIZE
    global _TP_RANK, _DP_RANK, _PP_RANK, _EP_RANK, _EXPERT_DP_RANK

    assert dist.is_available() and dist.is_initialized(), \
        "torch.distributed must be initialized before initialize_parallel"

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tp_size, pp_size, ep_size = config.tp_size, config.pp_size, config.ep_size
    assert world_size % (tp_size * pp_size) == 0, \
        f"world_size ({world_size}) must be divisible by tp_size * pp_size ({tp_size * pp_size})"
    dp_size = world_size // (tp_size * pp_size)
    assert dp_size % ep_size == 0, \
        f"dp_size ({dp_size}) must be divisible by ep_size ({ep_size})"

    def decompose(r: int):
        return r // (dp_size * tp_size), (r // tp_size) % dp_size, r % tp_size

    def group_for(select):
        """Create every group in `select` collectively; keep the one containing rank."""
        mine, mine_ranks = None, None
        for ranks in select():
            group = dist.new_group(ranks)
            if rank in ranks:
                mine, mine_ranks = group, ranks
        return mine, mine_ranks

    # TP groups: contiguous rank slices within a stage.
    _TP_GROUP, tp_ranks = group_for(
        lambda: ([(s * dp_size + d) * tp_size + t for t in range(tp_size)]
                 for s in range(pp_size) for d in range(dp_size)))
    _TP_RANK = tp_ranks.index(rank)

    # DP groups: same tp offset within a stage.
    _DP_GROUP, dp_ranks = group_for(
        lambda: ([(s * dp_size + d) * tp_size + t for d in range(dp_size)]
                 for s in range(pp_size) for t in range(tp_size)))
    _DP_RANK = dp_ranks.index(rank)

    # PP groups: same (dp, tp) position across stages.
    _PP_GROUP, pp_ranks = group_for(
        lambda: ([((s * dp_size) + d) * tp_size + t for s in range(pp_size)]
                 for d in range(dp_size) for t in range(tp_size)))
    _PP_RANK = pp_ranks.index(rank)

    pp_rank, dp_rank, _ = decompose(rank)
    if ep_size > 1:
        # EP groups: contiguous ep_size slices inside each DP group.
        _EP_GROUP, ep_ranks = group_for(
            lambda: ([(s * dp_size + c * ep_size + e) * tp_size + t
                      for e in range(ep_size)]
                     for s in range(pp_size) for t in range(tp_size)
                     for c in range(dp_size // ep_size)))
        _EP_RANK = ep_ranks.index(rank)

        # Expert-DP groups: ranks holding identical experts (same ep_rank).
        _EXPERT_DP_GROUP, edp_ranks = group_for(
            lambda: ([(s * dp_size + c * ep_size + e) * tp_size + t
                      for c in range(dp_size // ep_size)]
                     for s in range(pp_size) for t in range(tp_size)
                     for e in range(ep_size)))
        _EXPERT_DP_RANK = edp_ranks.index(rank)

    _TP_SIZE, _DP_SIZE, _PP_SIZE, _EP_SIZE = tp_size, dp_size, pp_size, ep_size
    _EXPERT_DP_SIZE = dp_size // ep_size
    _SEQUENCE_PARALLEL = bool(config.sequence_parallel)
    _INITIALIZED = True


def destroy_parallel() -> None:
    """Reset mesh state (test helper; does not destroy torch.distributed)."""
    global _INITIALIZED, _SEQUENCE_PARALLEL
    global _TP_GROUP, _DP_GROUP, _PP_GROUP, _EP_GROUP, _EXPERT_DP_GROUP
    global _TP_SIZE, _DP_SIZE, _PP_SIZE, _EP_SIZE, _EXPERT_DP_SIZE
    global _TP_RANK, _DP_RANK, _PP_RANK, _EP_RANK, _EXPERT_DP_RANK
    _INITIALIZED = False
    _SEQUENCE_PARALLEL = False
    _TP_GROUP = _DP_GROUP = _PP_GROUP = _EP_GROUP = _EXPERT_DP_GROUP = None
    _TP_SIZE = _DP_SIZE = _PP_SIZE = _EP_SIZE = _EXPERT_DP_SIZE = 1
    _TP_RANK = _DP_RANK = _PP_RANK = _EP_RANK = _EXPERT_DP_RANK = 0


def is_initialized() -> bool:
    return _INITIALIZED


def sequence_parallel() -> bool:
    return _SEQUENCE_PARALLEL


def get_tp_group():
    return _TP_GROUP


def get_dp_group():
    return _DP_GROUP


def get_pp_group():
    return _PP_GROUP


def get_ep_group():
    return _EP_GROUP


def get_expert_dp_group():
    return _EXPERT_DP_GROUP


def get_tp_world_size() -> int:
    return _TP_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_dp_world_size() -> int:
    return _DP_SIZE


def get_dp_rank() -> int:
    return _DP_RANK


def get_pp_world_size() -> int:
    return _PP_SIZE


def get_pp_rank() -> int:
    return _PP_RANK


def get_ep_world_size() -> int:
    return _EP_SIZE


def get_ep_rank() -> int:
    return _EP_RANK


def get_expert_dp_world_size() -> int:
    return _EXPERT_DP_SIZE


def get_expert_dp_rank() -> int:
    return _EXPERT_DP_RANK


def print_topology(master_only: bool = True) -> None:
    if not _INITIALIZED:
        return
    if master_only and dist.get_rank() != 0:
        return
    ep_note = f" ep={_EP_SIZE}" if _EP_SIZE > 1 else ""
    print(
        f"[aimegatron.mesh] world={dist.get_world_size()} "
        f"dp={_DP_SIZE} tp={_TP_SIZE} pp={_PP_SIZE}{ep_note} "
        f"sequence_parallel={_SEQUENCE_PARALLEL}",
        flush=True,
    )
