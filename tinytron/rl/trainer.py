from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from tinytron.distributed import allreduce_non_expert_grads_across_sp
from tinytron.inference import InferenceEngine
from tinytron.model.gpt import EXPERT_LOCAL_PARAM_SUFFIXES
from tinytron.training import Trainer
from tinytron.training.config import Config

from .logprobs import build_response_mask, gather_log_probs
from .losses import grpo_loss
from .rollout import group_advantages, make_rollout_batch
from .sync import ActorRolloutBridge, inference_model_config


@dataclass(frozen=True)
class RLConfig:
    max_new_tokens: int = 8
    group_size: int = 2
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None
    reward_target_token_id: int = 0
    clip_range: float = 0.2
    kl_coef: float = 0.0
    rollout_shard_qkv: bool = True


class GRPOTrainer(Trainer):
    """Minimal stage-1 RL trainer: synchronous rollout and update on the same ranks."""

    def __init__(self, config: Config, rl_config: RLConfig):
        self.rl_config = rl_config
        super().__init__(config)
        self._init_rollout_model(config, rl_config)

    def _init_rollout_model(self, config: Config, rl_config: RLConfig) -> None:
        rollout_model_config = inference_model_config(config.model, shard_qkv=rl_config.rollout_shard_qkv)
        self.rollout_engine = InferenceEngine(
            model_config=rollout_model_config,
            checkpoint_path=None,
            device=f"cuda:{self.local_rank}",
            dtype=torch.bfloat16 if config.train.precision == "bf16" else torch.float32,
            use_paged_kv_cache=True,
        )
        self.rollout_bridge = ActorRolloutBridge(
            actor_model=self.raw_model,
            rollout_model=self.rollout_engine.model,
            rollout_model_config=rollout_model_config,
        )

    def _one_training_step(self, config: Config, step: int):
        self.model.train()
        self.optimizer.zero_grad()
        self.rollout_bridge.sync()

        loss_accum = torch.zeros((), dtype=torch.float32, device=f"cuda:{self.local_rank}")
        reward_accum = torch.zeros_like(loss_accum)
        kl_accum = torch.zeros_like(loss_accum)

        for micro_step in range(self.training_info["grad_accum_steps"]):
            batch = self._next_train_batch()
            self.model.require_backward_grad_sync = (micro_step == self.training_info["grad_accum_steps"] - 1)
            loss, metrics = self._one_rl_micro_step(config, batch)
            (loss / self.training_info["grad_accum_steps"]).backward()
            loss_accum += metrics["loss"] / self.training_info["grad_accum_steps"]
            reward_accum += metrics["reward"] / self.training_info["grad_accum_steps"]
            kl_accum += metrics.get("kl", torch.zeros_like(kl_accum)) / self.training_info["grad_accum_steps"]

        allreduce_non_expert_grads_across_sp(
            model=self.raw_model,
            sp_group=self.sp_group,
            sp_world_size=self.sp_world_size,
            expert_local_param_suffixes=EXPERT_LOCAL_PARAM_SUFFIXES,
        )
        norm = self.raw_model.clip_grad_norm(config.train.grad_clip_value)
        lr = self._lr_scheduler(
            step,
            self.training_info["max_steps"],
            config.optim.warmup_steps,
            config.optim.max_lr,
            config.optim.min_lr,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        dist.all_reduce(reward_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        dist.all_reduce(kl_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["reward"] = reward_accum
        self.one_step_results["kl"] = kl_accum
        self.one_step_results["grad_norm"] = norm

    def _one_rl_micro_step(self, config: Config, data_batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prompts = data_batch["input_ids"].to(f"cuda:{self.local_rank}")
        prompts = prompts.repeat_interleave(self.rl_config.group_size, dim=0)

        self.rollout_engine.model.eval()
        sequences, rollout_info = self.rollout_engine.generate(
            prompts,
            max_new_tokens=self.rl_config.max_new_tokens,
            temperature=self.rl_config.temperature,
            top_k=self.rl_config.top_k,
            top_p=self.rl_config.top_p,
            eos_token_id=self.rl_config.eos_token_id,
            return_logprobs=True,
        )
        rollout = make_rollout_batch(
            prompts=prompts,
            sequences=sequences,
            old_log_probs=rollout_info["token_log_probs"].detach(),
            eos_token_id=self.rl_config.eos_token_id,
        )
        rewards = self._rule_rewards(rollout.responses, rollout.response_mask)
        advantages = group_advantages(rewards, self.rl_config.group_size).to(rollout.response_mask.dtype)

        log_probs, old_log_probs, response_mask = self._actor_local_log_probs(
            rollout.sequences,
            rollout.old_log_probs,
            prompt_len=rollout.prompts.size(1),
        )
        out = grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            mask=response_mask,
            clip_range=self.rl_config.clip_range,
            kl_coef=self.rl_config.kl_coef,
            group=self.sp_group,
        )
        reward = rewards.mean().detach()
        metrics = {"loss": out.loss.detach(), "reward": reward}
        metrics.update(out.metrics)
        return out.loss, metrics

    def _actor_local_log_probs(
        self,
        sequences: torch.Tensor,
        response_old_log_probs: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = sequences[:, :-1]
        target_ids = sequences[:, 1:]
        full_mask = build_response_mask(
            sequences,
            prompt_len=prompt_len,
            eos_token_id=self.rl_config.eos_token_id,
        )
        full_old_log_probs = torch.zeros_like(full_mask)
        full_old_log_probs[:, prompt_len - 1 :] = response_old_log_probs.to(full_old_log_probs.dtype)
        if input_ids.size(1) % self.sp_world_size != 0:
            raise ValueError(
                f"prompt_len + max_new_tokens - 1 must be divisible by sp_world_size. "
                f"Got {input_ids.size(1)} and sp_world_size={self.sp_world_size}."
            )

        seq_chunk_size = input_ids.size(1) // self.sp_world_size
        seq_start_idx = self.sp_rank * seq_chunk_size
        seq_end_idx = (self.sp_rank + 1) * seq_chunk_size
        x = input_ids[:, seq_start_idx:seq_end_idx].contiguous()
        y = target_ids[:, seq_start_idx:seq_end_idx].contiguous()
        local_mask = full_mask[:, seq_start_idx:seq_end_idx].contiguous()
        local_old_log_probs = full_old_log_probs[:, seq_start_idx:seq_end_idx].contiguous()

        with self._autocast_context(self.config.train.precision):
            logits = self.model(x)
        return gather_log_probs(logits, y) * local_mask, local_old_log_probs, local_mask

    def _rule_rewards(self, responses: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        target = int(self.rl_config.reward_target_token_id)
        scale = max(int(self.config.model.vocab_size) - 1, 1)
        token_scores = 1.0 - (responses.float() - target).abs().clamp(max=scale) / scale
        reward_sum = (token_scores * response_mask).sum(dim=-1)
        reward_count = response_mask.sum(dim=-1).clamp_min(1.0)
        return reward_sum / reward_count
