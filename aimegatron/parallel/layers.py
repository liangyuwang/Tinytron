"""aimegatron.parallel.layers

Tensor-parallel layer primitives. Model code builds layers from these three
classes and never touches collectives directly.

Sharding conventions (Megatron-style):
- ColumnParallelLinear: weight sharded on the output dim -> [out/tp, in].
  Output is a contiguous slice of the full output features.
- RowParallelLinear: weight sharded on the input dim -> [out, in/tp].
  Input must already be the matching slice; output is reduced across TP.
- VocabParallelEmbedding: embedding table sharded on the vocab dim.

With sequence_parallel enabled (mesh-level flag), activations outside TP
regions are sharded along the sequence dimension and the collectives switch
from all-reduce to reduce-scatter / all-gather, saving activation memory.

Gradient bookkeeping: parameters replicated across TP (norms, row bias) are
marked via mark_tp_replicated so gradient finalization and checkpoint layout
can treat them correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimegatron.core import mesh
from aimegatron.parallel import comm

TP_REPLICATED_ATTR = "tp_replicated"


def mark_tp_replicated(param: nn.Parameter) -> nn.Parameter:
    setattr(param, TP_REPLICATED_ATTR, True)
    return param


def is_tp_replicated(param: nn.Parameter) -> bool:
    return bool(getattr(param, TP_REPLICATED_ATTR, False))


def init_from_full_tensor_(weight: torch.Tensor, full_shape, shard_dim: int, init_fn) -> None:
    """Draw the full logical tensor from the current RNG stream and copy this
    rank's slice into `weight`. When every TP rank enters with an identically
    seeded stream, the shards become exact slices of the single-device
    weights, so parallel and single-device runs start from the same model."""
    full = torch.empty(full_shape, dtype=weight.dtype, device=weight.device)
    init_fn(full)
    tp_size, tp_rank = mesh.get_tp_world_size(), mesh.get_tp_rank()
    shard = full if tp_size <= 1 else full.chunk(tp_size, dim=shard_dim)[tp_rank]
    with torch.no_grad():
        weight.copy_(shard)


def _normal(mean: float, std: float):
    return lambda t: nn.init.normal_(t, mean=mean, std=std)


class ColumnParallelLinear(nn.Module):
    """Linear with the output dimension sharded across the TP group."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 gather_output: bool = False, device=None, dtype=None):
        super().__init__()
        tp_size = mesh.get_tp_world_size()
        assert out_features % tp_size == 0, \
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        self.out_features_per_partition = out_features // tp_size
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features, device=device, dtype=dtype))
        self.bias = (nn.Parameter(torch.zeros(self.out_features_per_partition, device=device, dtype=dtype))
                     if bias else None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def init_normal_(self, mean: float, std: float) -> None:
        init_from_full_tensor_(self.weight, (self.out_features, self.in_features), 0,
                               _normal(mean, std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = mesh.get_tp_group()
        if mesh.sequence_parallel():
            x = comm.gather_from_sequence_parallel_region(x, group)
        else:
            x = comm.copy_to_parallel_region(x, group)
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = comm.gather_from_parallel_region(out, group, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Linear with the input dimension sharded across the TP group."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        tp_size = mesh.get_tp_world_size()
        assert in_features % tp_size == 0, \
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        self.in_features_per_partition = in_features // tp_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition, device=device, dtype=dtype))
        self.bias = (nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
                     if bias else None)
        if self.bias is not None:
            # Bias is added after the TP reduction, so it is replicated across TP.
            mark_tp_replicated(self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def init_normal_(self, mean: float, std: float) -> None:
        init_from_full_tensor_(self.weight, (self.out_features, self.in_features), 1,
                               _normal(mean, std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = mesh.get_tp_group()
        out = F.linear(x, self.weight)
        if mesh.sequence_parallel():
            out = comm.reduce_scatter_to_sequence_parallel_region(out, group)
        else:
            out = comm.reduce_from_parallel_region(out, group)
        if self.bias is not None:
            out = out + self.bias
        return out


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded across the TP group on the vocab dimension."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        tp_size = mesh.get_tp_world_size()
        tp_rank = mesh.get_tp_rank()
        assert num_embeddings % tp_size == 0, \
            f"num_embeddings ({num_embeddings}) must be divisible by tp_size ({tp_size})"
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vocab_start_index = tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def init_normal_(self, mean: float, std: float) -> None:
        init_from_full_tensor_(self.weight, (self.num_embeddings, self.embedding_dim), 0,
                               _normal(mean, std))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= self.vocab_start_index) & (input_ids < self.vocab_end_index)
        local_ids = (input_ids - self.vocab_start_index).clamp(0, self.num_embeddings_per_partition - 1)
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        group = mesh.get_tp_group()
        out = comm.reduce_from_parallel_region(out, group)
        if mesh.sequence_parallel():
            out = comm.scatter_to_sequence_parallel_region(out, group)
        return out
