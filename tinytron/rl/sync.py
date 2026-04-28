from __future__ import annotations

from dataclasses import replace

import torch
import torch.distributed as dist

from tinytron.bridge import distributed_training_state_dict_to_local_inference
from tinytron.model.config import ModelConfig


def inference_model_config(model_config: ModelConfig, shard_qkv: bool = True) -> ModelConfig:
    return replace(model_config, inference_shard_qkv=shard_qkv)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def training_state_dict_to_inference(
    training_model: torch.nn.Module,
    model_config: ModelConfig,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    raw_model = unwrap_model(training_model)
    state_dict = raw_model.state_dict()
    return distributed_training_state_dict_to_local_inference(
        state_dict,
        model_config=model_config,
        process_group=process_group,
    )


class ActorRolloutBridge:
    """Synchronize a training actor into a rollout/inference model."""

    def __init__(
        self,
        actor_model: torch.nn.Module,
        rollout_model: torch.nn.Module,
        rollout_model_config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
    ):
        self.actor_model = actor_model
        self.rollout_model = rollout_model
        self.rollout_model_config = rollout_model_config
        self.process_group = process_group

    @torch.no_grad()
    def sync(self, strict: bool = True) -> dict[str, torch.Tensor]:
        state_dict = training_state_dict_to_inference(
            self.actor_model,
            model_config=self.rollout_model_config,
            process_group=self.process_group,
        )
        unwrap_model(self.rollout_model).load_state_dict(state_dict, strict=strict)
        return state_dict
