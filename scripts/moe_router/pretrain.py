from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinytron.training import build_config, build_parser
from scripts.example import pretrain as example_pretrain


@dataclass
class RouterExperimentConfig:
    router_warmup_steps: int = 100
    router_bootstrap_steps: int = 100
    warmup_routing_strategy: str = "round_robin"
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
        choices=["random", "round_robin"],
        help="Non-semantic expert assignment used before learned router ranking starts.",
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
            routing_strategy = self.router_exp.warmup_routing_strategy
            router_trainable = False
            experts_trainable = True
            ranking_loss_weight = 0.0
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

    def _one_training_step(self, config, step: int):
        phase = self._configure_router_phase(step)
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
    trainer = MoERouterExperimentTrainer(cfg, router_exp)
    trainer.train()


if __name__ == "__main__":
    main()
