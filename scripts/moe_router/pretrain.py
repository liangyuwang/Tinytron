from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.example import pretrain as example_pretrain
from tinytron.distributed import allreduce_non_expert_grads_across_sp
from tinytron.model.gpt import EXPERT_LOCAL_PARAM_SUFFIXES
from tinytron.training import build_config, build_parser
from tinytron.utils import compute_mfu


@dataclass
class OracleWarmupConfig:
    expert_warmup_steps: int = 0
    warmup_routing_strategy: str = "measurement_topk"
    measurement_topk_updates_model: bool = True
    router_training_steps: int = 0


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
        help="Number of router-only training steps to run after oracle-assisted expert training.",
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
    if args.router_training_steps < 0:
        raise ValueError("router_training_steps must be >= 0.")
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
        self._initial_streaming_global_skip_batches = getattr(
            self.train_dataset,
            "global_skip_batches",
            None,
        )

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

    def _is_router_param(self, name: str) -> bool:
        return ".router." in name or name.startswith("router.")

    def _set_router_only_trainable(self) -> None:
        for name, param in self.raw_model.named_parameters():
            param.requires_grad_(self._is_router_param(name))
        self._set_moe_routing_strategy("learned")
        self._clear_moe_warmup_state()

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

    def _reset_train_data_for_router_phase(self) -> None:
        self.train_loader_iter_idx = 0
        if self.train_sampler is not None:
            self.train_sampler_epoch = 0
            self.train_sampler.set_epoch(self.train_sampler_epoch)
        if self._initial_streaming_global_skip_batches is not None and hasattr(
            self.train_dataset,
            "global_skip_batches",
        ):
            self.train_dataset.global_skip_batches = self._initial_streaming_global_skip_batches
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(0)
        self.train_loader_iter = enumerate(self.train_loader)

    def train_router(self) -> None:
        router_steps = int(self.oracle_cfg.router_training_steps)
        if router_steps <= 0:
            return

        if self.master_process:
            print(f"[moe_router] starting router-only phase for {router_steps} steps", flush=True)

        self._set_router_only_trainable()
        self._reset_train_data_for_router_phase()

        for step in tqdm(
            range(router_steps),
            total=router_steps,
            desc="Train Router",
            disable=not self.master_process,
        ):
            self.one_step_results = {}
            t0 = time.time()
            last_step = step == router_steps - 1
            with self.profiler_record_fn("router_training_step"):
                super()._one_training_step(self.config, step)
            self.one_step_results["moe/router_phase"] = "router_only"
            torch.cuda.synchronize()
            if self.profiler:
                self.profiler.step()
            if (
                self.config.ckpt.do_save
                and not self.config.train.debug
                and step > 0
                and (step % self.config.ckpt.save_every_steps == 0 or last_step)
            ):
                self.save(self.training_info["max_steps"] + step)

            t1 = time.time()
            dt = t1 - t0
            tokens_processed = (
                self.config.train.batch_size
                * self.config.train.seq_len
                * self.training_info["grad_accum_steps"]
                * self.dp_world_size
            )
            tokens_per_sec = tokens_processed / dt
            mfu, _, _ = compute_mfu(
                self.raw_model,
                self.config.train.batch_size,
                self.config.train.seq_len,
                dt,
                self.training_info["grad_accum_steps"],
                dtype="bf16",
            )
            if self.master_process:
                tqdm.write(
                    f"router step {step:5d}/{router_steps} | "
                    f"loss: {self.one_step_results['loss'].item():.6f} | "
                    f"lr {self.one_step_results['lr']:.4e} | "
                    f"grad norm: {self.one_step_results['grad_norm']:.4f} | "
                    f"dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | "
                    f"MFU: {mfu*100:.2f}%"
                )
                with open(self.log_file, "a") as f:
                    f.write(f"router {step} train {self.one_step_results['loss'].item():.6f}\n")
            self.results[f"router/{step}"] = self.one_step_results

        if self.master_process:
            print("[moe_router] router-only phase finished", flush=True)

    def _after_train_loop(self):
        self.train_router()


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
        router_training_steps=int(args.router_training_steps),
    )
    if (
        (oracle_cfg.expert_warmup_steps > 0 or oracle_cfg.router_training_steps > 0)
        and not cfg.parallel.ddp_find_unused_parameters
    ):
        cfg = replace(
            cfg,
            parallel=replace(cfg.parallel, ddp_find_unused_parameters=True),
        )
    trainer = FullObservationOracleTrainer(cfg, oracle_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
