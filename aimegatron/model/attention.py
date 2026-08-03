"""aimegatron.model.attention

Grouped-query attention written as if for a single GPU: QKV projections are
ColumnParallelLinear (heads are sharded automatically), the output
projection is RowParallelLinear. Under sequence parallelism the column
layer gathers the sequence internally, so attention always sees the full
sequence for its local head shard.

Edit contract: no torch.distributed calls belong in this module; sharding
behavior lives in aimegatron/parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimegatron.core import mesh
from aimegatron.core.config import ModelConfig
from aimegatron.model.rope import apply_rope
from aimegatron.parallel.layers import ColumnParallelLinear, RowParallelLinear


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expand KV heads to match Q heads. x: [B, H_kv, T, D]."""
    if repeats == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None].expand(B, H, repeats, T, D).reshape(B, H * repeats, T, D)


class Attention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        tp_size = mesh.get_tp_world_size()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_q_heads_local = config.num_attention_heads // tp_size
        self.num_kv_heads_local = config.num_key_value_heads // tp_size
        self.kv_repeats = self.num_q_heads_local // self.num_kv_heads_local
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta

        self.q_proj = ColumnParallelLinear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = ColumnParallelLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.c_proj = RowParallelLinear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        q = self.q_proj(x).view(B, -1, self.num_q_heads_local, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, -1, self.num_kv_heads_local, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, -1, self.num_kv_heads_local, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, position_ids, theta=self.rope_theta)
        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)

        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)

        y = y.transpose(1, 2).reshape(B, y.size(2), self.num_q_heads_local * self.head_dim)
        return self.c_proj(y)
