from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinytron.training import build_config, build_parser
from tinytron.distributed import allreduce_non_expert_grads_across_sp
from tinytron.model.gpt import EXPERT_LOCAL_PARAM_SUFFIXES
from scripts.example import pretrain as example_pretrain


@dataclass
class RouterExperimentConfig:
    router_warmup_steps: int = 100
    router_bootstrap_steps: int = 100
    warmup_routing_strategy: str = "measurement_topk"
    router_probe_experts: int = 2
    router_probe_tokens: int = 128
    router_ranking_loss_weight: float = 0.01
    bootstrap_freeze_experts: bool = False
    disable_probe_ranking: bool = False


def parse_args():
    parser = build_parser()
    streaming_defaults = example_pretrain.StreamingDatasetConfig()

    parser.add_argument(
        "--streaming_data_dir",
        type=str,
        default=None,
        help="Path to processed Streaming-Dataloader data directory. "
             "If not set, falls back to --dataset_path.",
    )
    parser.add_argument(
        "--streaming_strict",
        action=argparse.BooleanOptionalAction,
        default=streaming_defaults.strict,
        help="Raise error if total samples < dp_world_size * num_workers.",
    )
    parser.add_argument(
        "--streaming_global_skip_batches",
        type=int,
        default=streaming_defaults.global_skip_batches,
        help="Number of globally-consumed samples to skip at dataset start.",
    )

    g = parser.add_argument_group("moe router experiment")
    defaults = RouterExperimentConfig()
    g.add_argument("--router_warmup_steps", type=int, default=defaults.router_warmup_steps)
    g.add_argument("--router_bootstrap_steps", type=int, default=defaults.router_bootstrap_steps)
    g.add_argument(
        "--warmup_routing_strategy",
        type=str,
        default=defaults.warmup_routing_strategy,
        choices=["measurement_topk", "random", "round_robin"],
        help=(
            "Warmup routing policy. measurement_topk uses a two-pass full-observation "
            "measurement to choose forced top-k experts and train router ranking; "
            "random/round_robin keep the legacy non-semantic sparse warmup."
        ),
    )
    g.add_argument("--router_probe_experts", type=int, default=defaults.router_probe_experts)
    g.add_argument("--router_probe_tokens", type=int, default=defaults.router_probe_tokens)
    g.add_argument("--router_ranking_loss_weight", type=float, default=defaults.router_ranking_loss_weight)
    g.add_argument("--bootstrap_freeze_experts", action="store_true")
    g.add_argument("--disable_probe_ranking", action="store_true")
    return parser.parse_args()


class MoERouterExperimentTrainer(example_pretrain.OurTrainer):
    def __init__(self, config, router_exp: RouterExperimentConfig):
        self.router_exp = router_exp
        super().__init__(config)
        if not config.model.use_moe:
            raise ValueError("The MoE router experiment requires --use_moe.")

    def _iter_moe_layers(self):
        for block in self.raw_model.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and hasattr(mlp, "routing_strategy"):
                yield mlp

    def _set_router_trainable(self, enabled: bool):
        for moe in self._iter_moe_layers():
            for param in moe.router.parameters():
                param.requires_grad_(enabled)

    def _set_experts_trainable(self, enabled: bool):
        for moe in self._iter_moe_layers():
            moe.experts_gate_weights.requires_grad_(enabled)
            moe.experts_up_weights.requires_grad_(enabled)
            moe.experts_down_weights.requires_grad_(enabled)

    def _configure_router_phase(self, step: int) -> str:
        warmup_steps = max(0, int(self.router_exp.router_warmup_steps))
        bootstrap_steps = max(0, int(self.router_exp.router_bootstrap_steps))
        in_warmup = step < warmup_steps
        in_bootstrap = warmup_steps <= step < warmup_steps + bootstrap_steps
        phase = "warmup" if in_warmup else ("bootstrap" if in_bootstrap else "joint")

        if in_warmup:
            routing_strategy = (
                "full_observation"
                if self.router_exp.warmup_routing_strategy == "measurement_topk"
                else self.router_exp.warmup_routing_strategy
            )
            router_trainable = self.router_exp.warmup_routing_strategy == "measurement_topk"
            experts_trainable = True
            ranking_loss_weight = (
                0.0
                if self.router_exp.disable_probe_ranking
                else self.router_exp.router_ranking_loss_weight
            )
        else:
            routing_strategy = "learned"
            router_trainable = True
            experts_trainable = not (in_bootstrap and self.router_exp.bootstrap_freeze_experts)
            ranking_loss_weight = 0.0 if self.router_exp.disable_probe_ranking else self.router_exp.router_ranking_loss_weight

        self._set_router_trainable(router_trainable)
        self._set_experts_trainable(experts_trainable)
        for moe in self._iter_moe_layers():
            moe.routing_strategy = routing_strategy
            moe.router_probe_experts = int(self.router_exp.router_probe_experts)
            moe.router_probe_tokens = int(self.router_exp.router_probe_tokens)
            moe.router_ranking_loss_weight = float(ranking_loss_weight)
        return phase

    def _set_moe_routing_strategy(self, routing_strategy: str):
        for moe in self._iter_moe_layers():
            moe.routing_strategy = routing_strategy

    def _clear_moe_warmup_state(self):
        for moe in self._iter_moe_layers():
            if hasattr(moe, "clear_warmup_state"):
                moe.clear_warmup_state()

    @torch.no_grad()
    def _allreduce_router_grads_across_dp(self):
        if self.dp_world_size <= 1:
            return
        for moe in self._iter_moe_layers():
            for param in moe.router.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.grad.div_(self.dp_world_size)

    def _mean_metric(self, values: list[torch.Tensor | float]) -> torch.Tensor | None:
        if not values:
            return None
        tensors = [
            v if torch.is_tensor(v) else torch.tensor(float(v), device=f"cuda:{self.local_rank}")
            for v in values
        ]
        value = torch.stack([t.detach().to(torch.float32) for t in tensors]).mean()
        dist.all_reduce(value, op=dist.ReduceOp.AVG, group=self.dp_group)
        return value

    def _collect_router_metrics(self):
        metric_names = ("router_entropy", "load_entropy", "load_max", "load_min")
        for name in metric_names:
            value = self._mean_metric([
                moe.last_routing_stats[name]
                for moe in self._iter_moe_layers()
                if name in moe.last_routing_stats
            ])
            if value is not None:
                self.one_step_results[f"moe/{name}"] = value

        probe_loss = self._mean_metric([
            moe.last_router_probe_loss
            for moe in self._iter_moe_layers()
            if moe.last_router_probe_loss is not None
        ])
        if probe_loss is not None:
            self.one_step_results["moe/probe_rank_loss"] = probe_loss

    def _one_measurement_topk_warmup_micro_step(self, config, micro_step: int, data_batch: dict):
        x, y = self._prepare_local_batch(data_batch)
        valid_token_count = (y != -100).sum().detach()
        tokens_per_dp_rank = valid_token_count.to(dtype=torch.float32)
        dist.all_reduce(tokens_per_dp_rank, op=dist.ReduceOp.SUM, group=self.sp_group)
        tokens_per_ddp_group = tokens_per_dp_rank.clone()
        dist.all_reduce(tokens_per_ddp_group, op=dist.ReduceOp.SUM, group=self.dp_group)
        loss_weight = tokens_per_dp_rank * self.dp_world_size / tokens_per_ddp_group.clamp(min=1.0)

        self._clear_moe_warmup_state()
        self._set_moe_routing_strategy("full_observation")
        with self.profiler_record_fn("warmup_measurement_forward"):
            with self._autocast_context(config.train.precision):
                _, ref_loss, _ = self.model(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))

        moe_layers = list(self._iter_moe_layers())
        reference_outputs = [moe.last_reference_output for moe in moe_layers]
        reference_grads = torch.autograd.grad(
            ref_loss,
            reference_outputs,
            retain_graph=False,
            allow_unused=True,
        )

        # reference_grads is intentionally unused. autograd.grad is used to
        # trigger each MoE reference-output hook, where q/top-k/rank-loss are
        # produced and full-expert measurement activations are released.
        del reference_grads
        rank_losses = [
            moe.last_router_rank_loss
            for moe in moe_layers
            if moe.last_router_rank_loss is not None
        ]

        rank_loss_accum = None
        if rank_losses:
            rank_loss_accum = torch.stack(rank_losses).mean()
            rank_loss = (
                rank_loss_accum
                * loss_weight.to(dtype=rank_loss_accum.dtype)
                / self.training_info["grad_accum_steps"]
            )
            with self.profiler_record_fn("warmup_router_backward"):
                rank_loss.backward()

        self._set_moe_routing_strategy("forced")
        with self.profiler_record_fn("warmup_topk_forward"):
            with self._autocast_context(config.train.precision):
                _, loss, logging_loss = self.model(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))
        loss = loss * loss_weight.to(dtype=loss.dtype) / self.training_info["grad_accum_steps"]
        with self.profiler_record_fn("warmup_topk_backward"):
            loss.backward()
        logging_loss = logging_loss * loss_weight.to(dtype=logging_loss.dtype) / self.training_info["grad_accum_steps"]
        self._clear_moe_warmup_state()
        return logging_loss, valid_token_count, rank_loss_accum

    def _one_measurement_topk_warmup_step(self, config, step: int):
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        router_rank_loss_accum = []
        valid_tokens_accum = torch.tensor(0, device=f"cuda:{self.local_rank}", dtype=torch.long)
        for micro_step in range(self.training_info["grad_accum_steps"]):
            batch = self._next_train_batch()
            self.model.require_backward_grad_sync = (micro_step == self.training_info["grad_accum_steps"] - 1)
            logging_loss, valid_token_count, rank_loss = self._one_measurement_topk_warmup_micro_step(
                config,
                micro_step,
                batch,
            )
            loss_accum += logging_loss
            valid_tokens_accum += valid_token_count
            if rank_loss is not None:
                router_rank_loss_accum.append(rank_loss.detach())

        self._allreduce_router_grads_across_dp()
        allreduce_non_expert_grads_across_sp(
            model=self.raw_model,
            sp_group=self.sp_group,
            sp_world_size=self.sp_world_size,
            expert_local_param_suffixes=EXPERT_LOCAL_PARAM_SUFFIXES,
        )
        norm = self.raw_model.clip_grad_norm(config.train.grad_clip_value)
        lr = self._lr_scheduler(step, self.training_info["max_steps"], config.optim.warmup_steps, config.optim.max_lr, config.optim.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        dist.all_reduce(valid_tokens_accum, op=dist.ReduceOp.SUM, group=self.dp_sp_group)
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["grad_norm"] = norm
        self.one_step_results["valid_tokens"] = valid_tokens_accum
        if router_rank_loss_accum:
            rank_metric = torch.stack(router_rank_loss_accum).mean()
            dist.all_reduce(rank_metric, op=dist.ReduceOp.AVG, group=self.dp_group)
            self.one_step_results["moe/probe_rank_loss"] = rank_metric

    def _one_training_step(self, config, step: int):
        phase = self._configure_router_phase(step)
        if phase == "warmup" and self.router_exp.warmup_routing_strategy == "measurement_topk":
            self._one_measurement_topk_warmup_step(config, step)
        else:
            super()._one_training_step(config, step)
        self.one_step_results["moe/router_phase"] = phase
        self._collect_router_metrics()
        if self.master_process and (step % config.logging.log_every == 0):
            extras = []
            for name in ("moe/router_entropy", "moe/load_entropy", "moe/load_max", "moe/probe_rank_loss"):
                value = self.one_step_results.get(name)
                if value is not None:
                    extras.append(f"{name.split('/')[-1]}={value.item():.4f}")
            if extras:
                print(f"[moe_router] step {step} phase={phase} " + " ".join(extras), flush=True)


def _needs_find_unused_parameters(router_exp: RouterExperimentConfig) -> bool:
    return (
        router_exp.router_warmup_steps > 0
        or (
            router_exp.router_bootstrap_steps > 0
            and router_exp.bootstrap_freeze_experts
        )
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    assert not cfg.train.do_val, "This experiment currently follows scripts/example and only wires train split."

    example_pretrain.dataset_cfg = example_pretrain.StreamingDatasetConfig(
        data_dir=args.streaming_data_dir or cfg.data.dataset_path,
        strict=bool(args.streaming_strict),
        global_skip_batches=int(args.streaming_global_skip_batches),
    )

    router_exp = RouterExperimentConfig(
        router_warmup_steps=args.router_warmup_steps,
        router_bootstrap_steps=args.router_bootstrap_steps,
        warmup_routing_strategy=args.warmup_routing_strategy,
        router_probe_experts=args.router_probe_experts,
        router_probe_tokens=args.router_probe_tokens,
        router_ranking_loss_weight=args.router_ranking_loss_weight,
        bootstrap_freeze_experts=bool(args.bootstrap_freeze_experts),
        disable_probe_ranking=bool(args.disable_probe_ranking),
    )
    if _needs_find_unused_parameters(router_exp) and not cfg.parallel.ddp_find_unused_parameters:
        cfg = replace(
            cfg,
            parallel=replace(cfg.parallel, ddp_find_unused_parameters=True),
        )
    trainer = MoERouterExperimentTrainer(cfg, router_exp)
    trainer.train()


if __name__ == "__main__":
    main()
