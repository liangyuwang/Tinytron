from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .logprobs import masked_mean
from .types import RLLossOutput


def _align_advantages(advantages: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    if advantages.shape == log_probs.shape:
        return advantages
    if advantages.shape == log_probs.shape[:-1]:
        return advantages.unsqueeze(-1)
    raise ValueError(f"advantages shape {tuple(advantages.shape)} must match log_probs {tuple(log_probs.shape)} or batch shape")


def approx_kl(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    mask: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """
    Non-negative sample estimate of KL(policy || reference).
    """
    log_ratio = reference_log_probs - policy_log_probs
    values = torch.exp(log_ratio) - log_ratio - 1.0
    return masked_mean(values, mask, group=group)


def policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> RLLossOutput:
    advantages = _align_advantages(advantages, log_probs)
    loss = -masked_mean(log_probs * advantages, mask, group=group)
    return RLLossOutput(loss=loss, metrics={"pg_loss": loss.detach()})


def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
    group: dist.ProcessGroup | None = None,
) -> RLLossOutput:
    advantages = _align_advantages(advantages, log_probs)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    objective = torch.minimum(unclipped, clipped)
    loss = -masked_mean(objective, mask, group=group)

    with torch.no_grad():
        clip_frac = masked_mean((unclipped != objective).to(log_probs.dtype), mask, group=group)
        mean_ratio = masked_mean(ratio, mask, group=group)
    return RLLossOutput(
        loss=loss,
        metrics={
            "ppo_loss": loss.detach(),
            "clip_frac": clip_frac,
            "mean_ratio": mean_ratio,
        },
    )


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    reference_log_probs: torch.Tensor | None = None,
    clip_range: float = 0.2,
    kl_coef: float = 0.0,
    group: dist.ProcessGroup | None = None,
) -> RLLossOutput:
    out = ppo_policy_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        mask=mask,
        clip_range=clip_range,
        group=group,
    )
    loss = out.loss
    metrics = dict(out.metrics)
    if reference_log_probs is not None and kl_coef > 0.0:
        kl = approx_kl(log_probs, reference_log_probs, mask, group=group)
        loss = loss + kl_coef * kl
        metrics["kl"] = kl.detach()
    return RLLossOutput(loss=loss, metrics=metrics)


def dpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> RLLossOutput:
    policy_margin = chosen_log_probs - rejected_log_probs
    reference_margin = reference_chosen_log_probs - reference_rejected_log_probs
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_reward = beta * (chosen_log_probs - reference_chosen_log_probs)
        rejected_reward = beta * (rejected_log_probs - reference_rejected_log_probs)
        reward_margin = chosen_reward - rejected_reward
    return RLLossOutput(
        loss=loss,
        metrics={
            "dpo_loss": loss.detach(),
            "reward_accuracy": reward_margin.gt(0).float().mean(),
            "reward_margin": reward_margin.mean(),
            "chosen_reward": chosen_reward.mean(),
            "rejected_reward": rejected_reward.mean(),
        },
    )
