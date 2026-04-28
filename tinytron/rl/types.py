from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RLLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


@dataclass
class RolloutBatch:
    """
    response_mask, old_log_probs, reference_log_probs, and advantages are aligned
    to responses, not to the full prompt+response sequence.
    """

    prompts: torch.Tensor
    responses: torch.Tensor
    sequences: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor | None = None
    reference_log_probs: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "RolloutBatch":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if torch.is_tensor(value) else value
        return RolloutBatch(**values)
