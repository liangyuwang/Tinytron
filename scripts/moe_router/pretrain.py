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

from scripts.example import pretrain as example_pretrain
from tinytron.distributed import allreduce_non_expert_grads_across_sp
from tinytron.model.gpt import EXPERT_LOCAL_PARAM_SUFFIXES
from tinytron.training import build_config, build_parser


@dataclass
class OracleWarmupConfig:
    expert_warmup_steps: int = 0
    warmup_routing_strategy: str = "measurement_topk"
    measurement_topk_updates_model: bool = True


def parse_args():
    parser = build_parser()
    streaming_defaults = example_pretrain.StreamingDatasetConfig()

    parser.add_argument("--streaming_data_dir", type=str, default=None)
    parser.add_argument("--streaming_shuffle", action="store_true")
    parser.add_argument("--streaming_strict", action="store_true")
    parser.add_argument("--streaming_global_skip_batches", type=int, default=0)

    g = parser.add_argument_group("full-observation oracle warmup")
    defaults = OracleWarmupConfig()
    g.add_argument(
        "--expert_warmup_steps",
        type=int,
        default=defaults.expert_warmup_steps,
        help="Number of two-pass full-observation oracle-assisted training steps.",
    )
    g.add_argument(
        "--router_training_steps",
        type=int,
        default=0,
        help="Accepted for experiment-file compatibility; must remain 0 here.",
    )
    g.add_argument(
        "--warmup_routing_strategy",
        type=str,
        default=defaults.warmup_routing_strategy,
        choices=["measurement_topk"],
    )
    g.add_argument(
        "--measurement_topk_updates_model",
        action=argparse.BooleanOptionalAction,
        default=defaults.measurement_topk_updates_model,
        help="Use measurement-selected top-k experts for the actual model update.",
    )
    g.add_argument(
        "--router_ranking_loss_weight",
        type=float,
        default=0.0,
        help="Accepted for experiment-file compatibility; must remain 0 here.",
    )

    args = parser.parse_args()
    if args.router_training_steps != 0:
        raise ValueError("Full-observation oracle-assisted training expects router_training_steps=0.")
    if args.router_ranking_loss_weight != 0:
        raise ValueError("Full-observation oracle-assisted training expects router_ranking_loss_weight=0.")
    if not args.measurement_topk_updates_model:
        raise ValueError("Full-observation oracle-assisted training requires measurement_topk_updates_model.")
    if args.streaming_data_dir is None:
        args.streaming_data_dir = streaming_defaults.data_dir
    return args


class FullObservationOracleTrainer(example_pretrain.OurTrainer):
    def __init__(self, config, oracle_cfg: OracleWarmupConfig):
        self.oracle_cfg = oracle_cfg
        super().__init__(config)
        if not config.model.use_moe:
            raise ValueError("Full-observation oracle training requires --use_moe.")

    def _iter_moe_layers(self):
        for block in self.raw_model.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and hasattr(mlp, "routing_strategy"):
                yield mlp

    def _prepare_local_batch(self, data_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data_batch["input_ids"], data_batch["labels"]
        _, T = x.shape
        assert T % self.sp_world_size == 0, "sequence length must be divisible by sp_world_size"
        seq_chunk_size = T // self.sp_world_size
        seq_start_idx = self.sp_rank * seq_chunk_size
        seq_end_idx = (self.sp_rank + 1) * seq_chunk_size
        x = x[:, seq_start_idx:seq_end_idx].to(f"cuda:{self.local_rank}")
        y = y[:, seq_start_idx:seq_end_idx].to(f"cuda:{self.local_rank}")
        return x, y

    def _set_router_trainable(self, enabled: bool) -> None:
        for moe in self._iter_moe_layers():
            for param in moe.router.parameters():
                param.requires_grad_(enabled)

    def _set_moe_routing_strategy(self, routing_strategy: str) -> None:
        for moe in self._iter_moe_layers():
            moe.routing_strategy = routing_strategy

    def _clear_moe_warmup_state(self) -> None:
        for moe in self._iter_moe_layers():
            moe.clear_warmup_state()

    def _configure_oracle_phase(self, step: int) -> str:
        in_warmup = step < max(0, int(self.oracle_cfg.expert_warmup_steps))
        self._set_router_trainable(not in_warmup)
        self._set_moe_routing_strategy("full_observation" if in_warmup else "learned")
        return "expert_warmup" if in_warmup else "joint"

    def _measurement_topk_micro_step(
        self,
        config,
        data_batch: dict,
    ) -> list[torch.Tensor]:
        x, y = self._prepare_local_batch(data_batch)
        self._clear_moe_warmup_state()
        self._set_moe_routing_strategy("full_observation")

        with self.profiler_record_fn("oracle_measurement_forward"):
            with self._autocast_context(config.train.precision):
                _, ref_loss, _ = self.model(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))

        moe_layers = list(self._iter_moe_layers())
        reference_outputs = [moe.last_reference_output for moe in moe_layers]
        torch.autograd.grad(
            ref_loss,
            reference_outputs,
            retain_graph=False,
            allow_unused=True,
        )
        forced_routing_cache = []
        for moe in moe_layers:
            selected = moe.last_warmup_selected_experts
            if selected is None:
                raise RuntimeError("Oracle measurement did not produce forced routing.")
            forced_routing_cache.append(selected.detach().to(device="cpu", dtype=torch.int16))
        self._clear_moe_warmup_state()
        return forced_routing_cache

    def _restore_forced_routing_cache(self, forced_routing_cache: list[torch.Tensor]) -> None:
        for moe, selected_experts in zip(self._iter_moe_layers(), forced_routing_cache, strict=True):
            moe.set_forced_routing(selected_experts)

    def _actual_topk_lm_micro_step(
        self,
        config,
        data_batch: dict,
        forced_routing_cache: list[torch.Tensor],
    ) -> torch.Tensor:
        x, y = self._prepare_local_batch(data_batch)
        self._clear_moe_warmup_state()
        self._restore_forced_routing_cache(forced_routing_cache)
        self._set_moe_routing_strategy("learned")
        with self.profiler_record_fn("oracle_topk_forward"):
            with self._autocast_context(config.train.precision):
                _, loss, logging_loss = self.model(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1))
        loss = loss / self.training_info["grad_accum_steps"]
        with self.profiler_record_fn("oracle_topk_backward"):
            loss.backward()
        self._clear_moe_warmup_state()
        return logging_loss / self.training_info["grad_accum_steps"]

    def _one_oracle_warmup_step(self, config, step: int) -> None:
        self.model.train()
        batches = [self._next_train_batch() for _ in range(self.training_info["grad_accum_steps"])]
        self.optimizer.zero_grad()
        forced_routing_caches = [
            self._measurement_topk_micro_step(config, batch)
            for batch in batches
        ]

        loss_accum = 0.0
        for micro_step, batch in enumerate(batches):
            self.model.require_backward_grad_sync = micro_step == self.training_info["grad_accum_steps"] - 1
            loss_accum += self._actual_topk_lm_micro_step(
                config,
                batch,
                forced_routing_caches[micro_step],
            )

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
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["grad_norm"] = norm

    def _one_training_step(self, config, step: int):
        phase = self._configure_oracle_phase(step)
        if phase == "expert_warmup":
            self._one_oracle_warmup_step(config, step)
        else:
            super()._one_training_step(config, step)
        self.one_step_results["moe/router_phase"] = phase
        if self.master_process and step % config.logging.log_every == 0:
            print(f"[moe_router] step {step} phase={phase}", flush=True)


def main():
    args = parse_args()
    cfg = build_config(args)
    assert not cfg.train.do_val, "This experiment follows scripts/example and only wires train split."

    example_pretrain.dataset_cfg = example_pretrain.StreamingDatasetConfig(
        data_dir=args.streaming_data_dir or cfg.data.dataset_path,
        shuffle=bool(args.streaming_shuffle),
        strict=bool(args.streaming_strict),
        global_skip_batches=int(args.streaming_global_skip_batches),
    )

    oracle_cfg = OracleWarmupConfig(
        expert_warmup_steps=args.expert_warmup_steps,
        warmup_routing_strategy=args.warmup_routing_strategy,
        measurement_topk_updates_model=bool(args.measurement_topk_updates_model),
    )
    if oracle_cfg.expert_warmup_steps > 0 and not cfg.parallel.ddp_find_unused_parameters:
        cfg = replace(
            cfg,
            parallel=replace(cfg.parallel, ddp_find_unused_parameters=True),
        )
    trainer = FullObservationOracleTrainer(cfg, oracle_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
