"""aimegatron.train.trainer

The training loop. Responsibilities, in order: initialize torch.distributed
and the mesh, build model/optimizer/data, then run a step loop of
forward -> backward -> finalize grads -> clip -> ZeRO-1 step.

Edit contract:
- Data pipeline swaps belong in _init_dataset (subclass and override).
- Anything about sharding belongs in aimegatron.parallel / train.layout,
  not here; the trainer only calls finalize_model_grads + clip_grad_norm.
"""

import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from aimegatron.core import mesh
from aimegatron.core.config import Config
from aimegatron.model.gpt import GPT, clip_grad_norm, finalize_model_grads
from aimegatron.model.pipeline import PipelineStage
from aimegatron.parallel.pipeline import OneFOneBSchedule
from aimegatron.train import checkpoint as ckpt
from aimegatron.train.optimizer import build_optimizer
from aimegatron.utils import count_parameters, get_lr, set_seed


class MockDataset(Dataset):
    """Deterministic random-token dataset: index-only generation, so any
    layout/parallelism change replays identical data."""

    def __init__(self, length: int, seq_len: int, vocab_size: int = 50304, seed: int = 0):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + int(idx))
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long, generator=g)
        return {"input_ids": data[:self.seq_len], "labels": data[1:self.seq_len + 1]}


class Trainer:

    def __init__(self, config: Config):
        self.config = config

        if not dist.is_initialized():
            # Allow plain `python scripts/pretrain.py` single-process launches.
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            dist.init_process_group(backend=config.parallel.backend, init_method=config.parallel.init_method)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))

        mesh.initialize_parallel(config.parallel)
        config.validate(self.world_size)
        mesh.print_topology()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
        set_seed(config.train.seed + self.rank, deterministic=config.train.deterministic)

        self.dtype = torch.bfloat16 if config.train.dtype == "bf16" else torch.float32
        if config.parallel.pp_size > 1:
            # Each rank holds only its stage; the schedule drives micro-batches.
            self.model = PipelineStage(config.model).to(device=self.device, dtype=self.dtype)
        else:
            self.model = GPT(config.model).to(device=self.device, dtype=self.dtype)
        self.optimizer = build_optimizer(config.train, self.model.parameters())

        self._init_dataset(config)
        # Shard the data stream across DP ranks; TP ranks within one DP group
        # must consume identical batches, so the sampler keys on the DP rank.
        dp_size = mesh.get_dp_world_size()
        sampler = None
        if dp_size > 1:
            sampler = DistributedSampler(
                self.train_dataset, num_replicas=dp_size,
                rank=mesh.get_dp_rank(), shuffle=False,
            )
        self.dataloader = DataLoader(
            self.train_dataset, batch_size=config.train.batch_size, shuffle=False,
            sampler=sampler,
            num_workers=0, drop_last=True, pin_memory=(self.device.type == "cuda"),
        )

        tokens_per_micro_step = config.train.batch_size * config.train.seq_len * dp_size
        self.grad_accum_steps = config.train.total_batch_size // tokens_per_micro_step

        self.schedule = None
        if config.parallel.pp_size > 1:
            self.schedule = OneFOneBSchedule(
                self.model, config.train.batch_size, config.train.seq_len,
                config.model.hidden_size, self.dtype, self.grad_accum_steps,
                aux_loss_coeff=config.model.moe_aux_loss_coeff,
            )

        self.master_process = self.rank == 0
        if self.master_process:
            print(f"[aimegatron] params={count_parameters(self.model):,} (per-rank local), "
                  f"grad_accum_steps={self.grad_accum_steps}", flush=True)
            os.makedirs(config.train.log_dir, exist_ok=True)

        self.start_step = 0
        self._maybe_resume()

    # -- extension point ---------------------------------------------------

    def _init_dataset(self, config: Config):
        """Override in a subclass to plug in a real data pipeline. The dataset
        must yield dicts with 'input_ids' and 'labels' of length seq_len."""
        assert config.data.use_mock_data, \
            "default trainer only provides mock data; subclass and override _init_dataset"
        self.train_dataset = MockDataset(
            config.data.mock_data_num_samples, config.train.seq_len,
            vocab_size=config.model.vocab_size, seed=config.train.seed,
        )

    # -- checkpoint resume --------------------------------------------------

    def _maybe_resume(self):
        path = self.config.train.resume_path or ckpt.find_latest_checkpoint(self.config.train.log_dir)
        if not path:
            return
        meta = ckpt.load_checkpoint(path, self.model, self.optimizer)
        # meta["step"] records completed steps, which is exactly the next step index.
        self.start_step = meta["step"]
        if self.master_process:
            print(f"[aimegatron] resumed from {path} at step {self.start_step}", flush=True)

    # -- training -----------------------------------------------------------

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train(self):
        config = self.config
        data_iter = iter(self.dataloader)

        def next_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                return next(data_iter)

        # Fast-forward the data stream to the resume point.
        for _ in range(self.start_step * self.grad_accum_steps):
            next_batch()

        flops_fn = getattr(self.model, "get_flops_per_fwd_bwd", None)
        flops_per_token = flops_fn(1, 1) if flops_fn is not None else 0.0
        log_file = os.path.join(config.train.log_dir, "log.txt") if self.master_process else None

        self.model.train()
        for step in range(self.start_step, config.train.max_steps):
            step_start = time.time()
            self._set_lr(get_lr(step, config.train.warmup_steps, config.train.max_steps,
                                config.train.max_lr, config.train.min_lr))
            self.optimizer.zero_grad()

            if self.schedule is not None:
                # 1F1B: the schedule owns the micro-batch loop and P2P.
                local_loss = self.schedule.run_step(next_batch)
                loss_sum = self.schedule.broadcast_loss(local_loss)
            else:
                loss_sum = 0.0
                for _ in range(self.grad_accum_steps):
                    batch = next_batch()
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)
                    _, loss, logging_loss = self.model(input_ids, labels)
                    (loss / self.grad_accum_steps).backward()
                    loss_sum += logging_loss.item() / self.grad_accum_steps

            finalize_model_grads(self.model)
            grad_norm = clip_grad_norm(self.model, config.train.grad_clip_value)
            self.optimizer.step()

            if self.master_process and (step % config.train.log_every_steps == 0
                                        or step == config.train.max_steps - 1):
                dt = time.time() - step_start
                tokens_per_sec = config.train.total_batch_size / max(dt, 1e-9)
                msg = f"{step} train {loss_sum:.4f} grad_norm {grad_norm.item():.3f} tok/s {tokens_per_sec:.0f}"
                if config.train.peak_flops > 0:
                    mfu = tokens_per_sec * flops_per_token / (config.train.peak_flops * 1e12)
                    msg += f" mfu {mfu * 100:.1f}%"
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(f"{step} train {loss_sum:.4f}\n")

            if config.train.do_save and (step + 1) % config.train.save_every_steps == 0:
                ckpt.save_checkpoint(config.train.log_dir, step + 1, self.model, self.optimizer,
                                     extra_meta={"num_layer": config.model.num_layer})

        if config.train.do_save:
            ckpt.save_checkpoint(config.train.log_dir, config.train.max_steps, self.model,
                                 self.optimizer, extra_meta={"num_layer": config.model.num_layer})
        dist.barrier()
