"""aimegatron.train.optimizer

ZeRO stage-1 distributed optimizer. Optimizer states (fp32 master weights +
momentum buffers) are sharded across the param's data-parallel scope; model
parameters and gradients stay as usual. Gradients must already be fully
reduced across that scope (see aimegatron.model.gpt.finalize_model_grads)
before step().

Two scopes exist: dense params replicate across the full DP group, expert
params replicate only across the expert-DP group (dp_size / ep_size ranks
hold identical experts). build_optimizer creates one DistributedOptimizer
per scope and aggregates them in a MultiOptimizer.

Per-step flow:
1. pack finalized grads into a flat fp32 buffer,
2. hand this rank's slice to the wrapped base optimizer,
3. all-gather the updated slices and write them back into model params.

Edit contract: this module only depends on mesh accessors; it does not know
about TP. TP sharding is invisible here because each rank only ever sees its
own local parameter shards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist

from aimegatron.core import mesh


class DistributedOptimizer:
    """ZeRO-1 wrapper around any torch optimizer factory."""

    def __init__(self, optimizer_factory, params, process_group=None,
                 group_size: int | None = None, group_rank: int | None = None):
        self.params = [p for p in params if p.requires_grad]
        assert len(self.params) > 0, "DistributedOptimizer received no trainable params"
        self.dp_group = process_group if process_group is not None else mesh.get_dp_group()
        self.dp_size = group_size if group_size is not None else mesh.get_dp_world_size()
        self.dp_rank = group_rank if group_rank is not None else mesh.get_dp_rank()

        # fp32 master copy of all local params, flattened.
        self._offsets: list[tuple[nn.Parameter, int, int]] = []
        flat_parts = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            self._offsets.append((p, offset, numel))
            flat_parts.append(p.detach().float().flatten())
            offset += numel
        self.flat = torch.cat(flat_parts)

        # Pad so the flat buffer splits into equal DP shards.
        pad = (self.dp_size - self.flat.numel() % self.dp_size) % self.dp_size
        if pad > 0:
            self.flat = torch.cat([self.flat, torch.zeros(pad, dtype=self.flat.dtype)])
        self.shard_size = self.flat.numel() // self.dp_size
        self.shard = nn.Parameter(
            self.flat[self.dp_rank * self.shard_size:(self.dp_rank + 1) * self.shard_size].clone())
        self.unpadded_numel = offset

        self.base_optimizer = optimizer_factory([self.shard])
        self._flat_grad = torch.zeros_like(self.flat)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
        self.base_optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        # 1. Pack grads (already DP-reduced) into the flat fp32 buffer.
        self._flat_grad.zero_()
        for p, start, numel in self._offsets:
            assert p.grad is not None, "missing gradient after finalize_model_grads"
            self._flat_grad[start:start + numel].copy_(p.grad.detach().float().flatten())
        shard_start = self.dp_rank * self.shard_size
        self.shard.grad = self._flat_grad[shard_start:shard_start + self.shard_size]

        # 2. Local optimizer step on this rank's shard.
        self.base_optimizer.step()

        # 3. All-gather updated shards and write back into model params.
        if self.dp_size > 1:
            gathered = [torch.empty_like(self.shard.data) for _ in range(self.dp_size)]
            dist.all_gather(gathered, self.shard.data, group=self.dp_group)
            flat_updated = torch.cat(gathered)[:self.unpadded_numel]
        else:
            flat_updated = self.shard.data[:self.unpadded_numel]
        for p, start, numel in self._offsets:
            p.data.copy_(flat_updated[start:start + numel].view_as(p).to(p.dtype))

    def state_dict(self) -> dict:
        return {"base_optimizer": self.base_optimizer.state_dict(), "shard": self.shard.data}

    def load_state_dict(self, state: dict) -> None:
        self.shard.data.copy_(state["shard"])
        self.base_optimizer.load_state_dict(state["base_optimizer"])


class MultiOptimizer:
    """Fan-out wrapper: applies several optimizers to disjoint param sets
    (dense vs expert) while presenting a single optimizer interface."""

    def __init__(self, optimizers: list):
        assert len(optimizers) > 0
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for opt, sub in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(sub)


def build_optimizer(train_config, params):
    """Create the optimizer selected by train_config (registry + ZeRO-1 flag).
    With ZeRO-1, dense params shard across the DP group and expert params
    across the expert-DP group."""
    from aimegatron.core.registry import OPTIMIZERS
    from aimegatron.model.moe import is_expert_parallel

    factory_cls = OPTIMIZERS.get(train_config.optimizer)

    def factory(param_list):
        return factory_cls(param_list, lr=train_config.max_lr, weight_decay=train_config.weight_decay)

    params = list(params)
    if train_config.use_distributed_optimizer:
        dense = [p for p in params if not is_expert_parallel(p)]
        expert = [p for p in params if is_expert_parallel(p)]
        optimizers = [DistributedOptimizer(factory, dense)]
        if expert:
            optimizers.append(DistributedOptimizer(
                factory, expert,
                process_group=mesh.get_expert_dp_group(),
                group_size=mesh.get_expert_dp_world_size(),
                group_rank=mesh.get_expert_dp_rank(),
            ))
        return optimizers[0] if len(optimizers) == 1 else MultiOptimizer(optimizers)
    return factory(params)
