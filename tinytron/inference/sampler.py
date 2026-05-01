from __future__ import annotations

import torch


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Apply temperature/top-k/top-p filtering to logits of shape [B, V]."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k=k, dim=-1)
        min_keep = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[..., 0] = False
        keep_sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=keep_sorted_logits)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample the next token ids from logits of shape [B, V]."""
    logits = filter_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_next_token_with_log_prob(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids and return their logprobs under the sampling distribution."""
    logits = filter_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    log_probs = torch.log_softmax(logits, dim=-1)
    next_token = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
    next_log_prob = log_probs.gather(dim=-1, index=next_token.unsqueeze(-1)).squeeze(-1)
    return next_token, next_log_prob
