"""aimegatron.model.pipeline

This rank's slice of the GPT stack. The full model is cut on layer
boundaries: stage s owns blocks [s*L/pp, (s+1)*L/pp); stage 0 additionally
owns the embedding, the last stage owns the final norm + lm head + loss.

Activation layout across stage boundaries is whatever the stage computes:
with sequence_parallel on, hidden states stay sequence-sharded across the
whole pipeline because P2P connects identical (dp, tp) positions.

Edit contract: weight init MUST go through gpt.init_gpt_weights so every
layout starts from exact slices of the single-device weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aimegatron.core import mesh
from aimegatron.core.config import ModelConfig
from aimegatron.core.registry import NORMS
from aimegatron.model.gpt import Block, init_gpt_weights, is_moe_layer
from aimegatron.model.loss import CrossEntropyLoss
from aimegatron.model.moe import MoE
from aimegatron.parallel.layers import ColumnParallelLinear, VocabParallelEmbedding


def stage_layer_range(num_layer: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    per_stage = num_layer // pp_size
    return pp_rank * per_stage, (pp_rank + 1) * per_stage


class PipelineStage(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        pp_size, pp_rank = mesh.get_pp_world_size(), mesh.get_pp_rank()
        self.is_first = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1
        start, end = stage_layer_range(config.num_layer, pp_size, pp_rank)
        self.block_start, self.block_end = start, end

        self.wte = (VocabParallelEmbedding(config.vocab_size, config.hidden_size)
                    if self.is_first else None)
        # Global block indices in the state dict, so checkpoint keys are
        # identical across pp_size values.
        self.blocks = nn.ModuleDict(
            {str(i): Block(config, moe=is_moe_layer(config, i)) for i in range(start, end)})
        self.lnf = NORMS.get(config.norm_type)(config.hidden_size) if self.is_last else None
        self.lm_head = (ColumnParallelLinear(config.hidden_size, config.vocab_size,
                                             gather_output=True)
                        if self.is_last else None)
        self.loss_fn = CrossEntropyLoss() if self.is_last else None
        self._position_ids: torch.Tensor | None = None

        init_gpt_weights(
            config,
            wte=self.wte,
            lm_head=self.lm_head if not config.tied_lm_head else None,
            blocks_with_idx=[(int(i), b) for i, b in self.blocks.items()],
            tied_lm_head=config.tied_lm_head,
        )

    def _get_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._position_ids is None or self._position_ids.size(0) < seq_len \
                or self._position_ids.device != device:
            self._position_ids = torch.arange(seq_len, device=device)
        return self._position_ids[:seq_len]

    def _total_moe_aux_loss(self):
        total = None
        for m in self.modules():
            if isinstance(m, MoE):
                total = m.last_aux_loss if total is None else total + m.last_aux_loss
        return total

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        """First stage: x = input_ids -> (hidden, aux). Middle stage:
        hidden -> (hidden, aux). Last stage: (hidden, labels) ->
        (ce_loss, logging_loss, aux)."""
        if self.is_first:
            input_ids = x
            hidden = self.wte(input_ids)
            position_ids = self._get_position_ids(input_ids.size(1), input_ids.device)
        else:
            hidden = x
            # Sequence dim may be SP-sharded; position ids are always full.
            position_ids = self._get_position_ids(
                hidden.size(1) * mesh.get_tp_world_size() if mesh.sequence_parallel()
                else hidden.size(1), hidden.device)

        for block in self.blocks.values():
            hidden = block(hidden, position_ids)

        aux = self._total_moe_aux_loss()

        if not self.is_last:
            return hidden, aux

        hidden = self.lnf(hidden)
        logits = self.lm_head(hidden)
        loss, logging_loss = self.loss_fn(logits, labels)
        return loss, logging_loss, aux
