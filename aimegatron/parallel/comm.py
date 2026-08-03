"""aimegatron.parallel.comm

Autograd-aware collectives. Every operation is a (forward, backward) pair
mirroring Megatron's communication regions:

- copy_to_parallel_region:            fwd identity            / bwd all-reduce
- reduce_from_parallel_region:        fwd all-reduce          / bwd identity
- gather_from_parallel_region:        fwd all-gather on dim   / bwd local slice
- scatter_to_sequence_parallel:       fwd local slice (seq)   / bwd all-gather
- gather_from_sequence_parallel:      fwd all-gather (seq)    / bwd reduce-scatter
- reduce_scatter_to_sequence_parallel: fwd reduce-scatter (seq) / bwd all-gather

Edit contract: this is the only place raw collectives meet autograd.
Layers in aimegatron/parallel/layers.py compose these; model code never
imports this module. Sequence dimension is fixed at SEQ_DIM = 1 ([B, T, H]).
"""

import torch
import torch.distributed as dist

SEQ_DIM = 1


def _world_size(group) -> int:
    if group is None:
        return 1
    return dist.get_world_size(group)


def _rank(group) -> int:
    if group is None:
        return 0
    return dist.get_rank(group)


def _all_gather_along(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    size = _world_size(group)
    gathered = [torch.empty_like(x) for _ in range(size)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def _reduce_scatter_along(x: torch.Tensor, dim: int, group) -> torch.Tensor:
    size = _world_size(group)
    chunks = [c.contiguous() for c in x.chunk(size, dim=dim)]
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, group=group)
    return output


class _CopyToParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.contiguous()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherFromParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, dim):
        ctx.group = group
        ctx.dim = dim
        ctx.rank = _rank(group)
        ctx.size = _world_size(group)
        return _all_gather_along(x, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        local = grad_output.chunk(ctx.size, dim=ctx.dim)[ctx.rank].contiguous()
        return local, None, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.size = _world_size(group)
        ctx.rank = _rank(group)
        return x.chunk(ctx.size, dim=SEQ_DIM)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather_along(grad_output, SEQ_DIM, ctx.group), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_gather_along(x, SEQ_DIM, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along(grad_output, SEQ_DIM, ctx.group), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _reduce_scatter_along(x, SEQ_DIM, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _all_gather_along(grad_output, SEQ_DIM, ctx.group), None


def copy_to_parallel_region(x: torch.Tensor, group) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _CopyToParallelRegion.apply(x, group)


def reduce_from_parallel_region(x: torch.Tensor, group) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _ReduceFromParallelRegion.apply(x, group)


def gather_from_parallel_region(x: torch.Tensor, group, dim: int = -1) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _GatherFromParallelRegion.apply(x, group, dim)


def scatter_to_sequence_parallel_region(x: torch.Tensor, group) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _ScatterToSequenceParallelRegion.apply(x, group)


def gather_from_sequence_parallel_region(x: torch.Tensor, group) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _GatherFromSequenceParallelRegion.apply(x, group)


def reduce_scatter_to_sequence_parallel_region(x: torch.Tensor, group) -> torch.Tensor:
    if _world_size(group) <= 1:
        return x
    return _ReduceScatterToSequenceParallelRegion.apply(x, group)
