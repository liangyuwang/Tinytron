"""aimegatron.model.mlp

SwiGLU MLP: gate/up projections are ColumnParallelLinear, the down
projection is RowParallelLinear. The elementwise SiLU-gate product happens
inside the TP region on the sharded intermediate dimension, so no extra
communication is needed.

Edit contract: no torch.distributed calls belong in this module.
"""

import torch.nn as nn
import torch.nn.functional as F

from aimegatron.core.config import ModelConfig
from aimegatron.parallel.layers import ColumnParallelLinear, RowParallelLinear


class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
