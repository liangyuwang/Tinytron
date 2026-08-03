"""aimegatron.model.norm

Normalization layers. Norms are elementwise over the hidden dimension, so
they run identically on a sequence-parallel shard. Their weights are
replicated across TP; under sequence parallelism each rank only sees a
sequence shard, so gradients are partial and are all-reduced across TP
during gradient finalization (see aimegatron.model.gpt.finalize_model_grads).

Edit contract: any new norm must call mark_tp_replicated on its learnable
parameters and be registered in aimegatron.core.registry.NORMS.
"""

import torch
import torch.nn as nn

from aimegatron.core import mesh
from aimegatron.parallel.layers import mark_tp_replicated


def _mark_if_sequence_parallel(*params: nn.Parameter) -> None:
    if mesh.sequence_parallel():
        for p in params:
            mark_tp_replicated(p)


def create_layernorm(hidden_size: int) -> nn.Module:
    norm = nn.LayerNorm(hidden_size)
    _mark_if_sequence_parallel(norm.weight, norm.bias)
    return norm


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        _mark_if_sequence_parallel(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm.to(dtype)


def create_rmsnorm(hidden_size: int) -> nn.Module:
    return RMSNorm(hidden_size)
