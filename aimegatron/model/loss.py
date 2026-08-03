"""aimegatron.model.loss

Cross-entropy that is agnostic to the parallel layout. The model gathers
logits before the loss, so every rank computes the loss on full logits and
targets; the value is identical across TP and only needs a DP average for
logging.

Edit contract: keep loss math layout-agnostic; normalization is by valid
token count so sequence-parallel sharding upstream cannot bias the value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aimegatron.core import mesh


class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logits: [B, T, V] (full), targets: [B, T]. Returns (loss, logging_loss)."""
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
        )
        logging_loss = loss.detach()
        dp_group = mesh.get_dp_group()
        dp_size = mesh.get_dp_world_size()
        if dp_size > 1:
            logging_loss = logging_loss.clone()
            torch.distributed.all_reduce(logging_loss, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            logging_loss = logging_loss / dp_size
        return loss, logging_loss
