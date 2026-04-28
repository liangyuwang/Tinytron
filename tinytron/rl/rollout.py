from __future__ import annotations

import torch

from .logprobs import build_response_mask
from .types import RolloutBatch


def make_rollout_batch(
    prompts: torch.Tensor,
    sequences: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> RolloutBatch:
    if prompts.dim() != 2 or sequences.dim() != 2:
        raise ValueError("prompts and sequences must be [B, T]")
    if prompts.size(0) != sequences.size(0):
        raise ValueError("prompts and sequences must share batch size")
    prompt_len = prompts.size(1)
    if sequences.size(1) < prompt_len:
        raise ValueError("sequences must include the prompt prefix")
    responses = sequences[:, prompt_len:]
    full_response_mask = build_response_mask(sequences, prompt_len=prompt_len, eos_token_id=eos_token_id)
    response_mask = full_response_mask[:, prompt_len - 1 :]
    if old_log_probs is not None and old_log_probs.shape != response_mask.shape:
        raise ValueError("old_log_probs must be aligned to response tokens")
    return RolloutBatch(
        prompts=prompts,
        responses=responses,
        sequences=sequences,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
    )


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalize rewards within each prompt group, as used by GRPO-style trainers.
    """
    if rewards.dim() != 1:
        raise ValueError("rewards must be [B]")
    if group_size <= 0 or rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
    return ((grouped - mean) / std).reshape_as(rewards)
