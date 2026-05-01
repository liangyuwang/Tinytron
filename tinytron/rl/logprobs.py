from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Return log p(token_ids) for logits shaped [..., vocab].
    """
    if token_ids.shape == logits.shape:
        raise ValueError("token_ids should not include the vocab dimension")
    if token_ids.shape != logits.shape[:-1]:
        raise ValueError(f"token_ids shape {tuple(token_ids.shape)} must match logits shape {tuple(logits.shape[:-1])}")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def causal_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Convert full-sequence LM logits into per-token logprobs for input_ids[:, 1:].
    """
    if logits.dim() != 3 or input_ids.dim() != 2:
        raise ValueError("logits must be [B, T, V] and input_ids must be [B, T]")
    if logits.size(0) != input_ids.size(0) or logits.size(1) != input_ids.size(1):
        raise ValueError("logits and input_ids must share batch and sequence dimensions")
    return gather_log_probs(logits[:, :-1, :], input_ids[:, 1:])


def response_log_probs(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return per-response-token logprobs from full-sequence logits.

    response_mask is aligned to sequences[:, 1:], matching the output shape.
    """
    log_probs = causal_log_probs(logits, sequences)
    if response_mask.shape != log_probs.shape:
        raise ValueError(f"response_mask shape {tuple(response_mask.shape)} must match log_probs {tuple(log_probs.shape)}")
    return log_probs * response_mask.to(log_probs.dtype)


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    if group is not None:
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = mask.to(values.dtype)
    numerator = (values * mask).sum()
    denominator = mask.sum()
    if group is not None:
        stats = torch.stack([numerator, denominator])
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        numerator, denominator = stats.unbind()
    return numerator / denominator.clamp_min(eps)


def sequence_log_probs(
    token_log_probs: torch.Tensor,
    mask: torch.Tensor,
    average: bool = False,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Reduce token logprobs to one logprob per sequence.
    """
    mask = mask.to(token_log_probs.dtype)
    logp = (token_log_probs * mask).sum(dim=-1)
    count = mask.sum(dim=-1)
    if group is not None:
        dist.all_reduce(logp, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
    if average:
        logp = logp / count.clamp_min(1.0)
    return logp


def build_response_mask(
    sequences: torch.Tensor,
    prompt_len: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Build a mask aligned to causal_log_probs(sequences): shape [B, T - 1].
    """
    if sequences.dim() != 2:
        raise ValueError("sequences must be [B, T]")
    if not 0 < prompt_len <= sequences.size(1):
        raise ValueError(f"prompt_len must be in [1, {sequences.size(1)}], got {prompt_len}")

    batch_size, total_len = sequences.shape
    mask = torch.zeros(batch_size, total_len - 1, dtype=torch.float32, device=sequences.device)
    mask[:, prompt_len - 1 :] = 1.0

    if eos_token_id is None:
        return mask

    response_tokens = sequences[:, prompt_len:]
    eos_hits = response_tokens.eq(eos_token_id)
    has_eos = eos_hits.any(dim=-1)
    first_eos = eos_hits.float().argmax(dim=-1)
    positions = torch.arange(response_tokens.size(1), device=sequences.device).unsqueeze(0)
    keep_response = positions <= first_eos.unsqueeze(-1)
    keep_response = torch.where(has_eos.unsqueeze(-1), keep_response, torch.ones_like(keep_response))
    mask[:, prompt_len - 1 :] = keep_response.to(mask.dtype)
    return mask
