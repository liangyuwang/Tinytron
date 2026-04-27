from __future__ import annotations

import os
import math
import glob
import random
from tqdm.auto import tqdm
from contextlib import contextmanager, nullcontext
from itertools import cycle, islice
from dataclasses import dataclass, field, asdict
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.checkpoint import state_dict_saver, state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader
from torch.nn.parallel import DistributedDataParallel as DDP

from tinytron.training.config import Config
from tinytron.model import GPT
from tinytron.model.gpt import EXPERT_LOCAL_PARAM_SUFFIXES
import tinytron.optim
from tinytron.distributed import (
    DistributedOptimizer,
    parallel_state,
    allreduce_non_expert_grads_across_sp,
)
from tinytron.utils import (
    get_training_info,
    set_seed,
    get_model_params,
    compute_mfu,
)


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self._init_distributed(config)
        assert config.train.total_batch_size % (config.train.batch_size * config.train.seq_len * self.dp_world_size) == 0, \
            f"make sure total_batch_size {config.train.total_batch_size} is divisible by batch_size {config.train.batch_size} * seq_len {config.train.seq_len} * dp_world_size {self.dp_world_size}"
        if self.master_process:
            print(f"Training config: {config.as_dict()}")
        self._init_dataset(config)
        self.training_info = get_training_info(
            self.num_train_samples, config.train.seq_len, config.train.total_batch_size, config.train.batch_size, self.dp_world_size, config.train.max_steps, config.train.max_epochs)
        if config.optim.warmup_steps >= self.training_info["max_steps"]:
            raise ValueError(f"warmup_steps must be < max_steps. Got warmup_steps={config.optim.warmup_steps}, max_steps={self.training_info['max_steps']}.")
        if self.master_process:
            print(f"The training process will train {self.training_info['epochs']} epochs, {self.training_info['max_steps']} steps.")
            print(f"=> total tokens per step: {config.train.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {self.training_info['grad_accum_steps']}")
        self._init_model(config)
        self._init_optimizer(config)
        self._init_logging(config)

    def _init_distributed(self, config: Config):
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend=config.parallel.backend, init_method=config.parallel.init_method)
        device = f'cuda:{self.local_rank}'
        torch.cuda.set_device(device)
        set_seed(config.seed.seed + self.rank, deterministic=config.seed.deterministic)
        self.master_process = self.rank == 0 # this process will do logging, checkpointing etc.
        
        # initialize data / sequence parallel groups
        parallel_state.initialize_model_parallel(config=config.parallel)
        self.dp_group = parallel_state.get_dp_group()
        self.dp_rank = parallel_state.get_dp_rank()
        self.dp_world_size = parallel_state.get_dp_world_size()
        self.sp_group = parallel_state.get_sp_group()
        self.sp_rank = parallel_state.get_sp_rank()
        self.sp_world_size = parallel_state.get_sp_world_size()
        self.dp_sp_group = parallel_state.get_dp_sp_group()
        self.dp_sp_rank = parallel_state.get_dp_sp_rank()
        self.dp_sp_world_size = parallel_state.get_dp_sp_world_size()
        parallel_state.print_model_parallel_topology(self.master_process, tag="Init Distributed Topology")

    def _init_dataset(self, config: Config):
        if config.data.use_mock_data:
            from torch.utils.data import Dataset
            class MockDataset(Dataset):
                def __init__(self, length: int, seq_len: int, vocab_size: int=50304, deterministic: bool=False, base_seed: int=0, seed_offset: int=0):
                    self.length = length
                    self.seq_len = seq_len
                    self.vocab_size = vocab_size
                    self.deterministic = deterministic
                    self.base_seed = int(base_seed)
                    self.seed_offset = int(seed_offset)
                def __len__(self):
                    return self.length
                def __getitem__(self, idx):
                    if self.deterministic:
                        # Make each sample depend only on its global index, so different
                        # distributed strategies produce identical mock tokens for the same idx.
                        g = torch.Generator(device="cpu")
                        g.manual_seed(self.base_seed + self.seed_offset + int(idx))
                        data = torch.randint(0, self.vocab_size, (self.seq_len+1,), dtype=torch.long, generator=g)
                    else:
                        data = torch.randint(0, self.vocab_size, (self.seq_len+1,), dtype=torch.long)
                    x = data[:self.seq_len]
                    y = data[1:self.seq_len+1]
                    return {"input_ids": x, "labels": y}
            self.train_dataset = MockDataset(config.data.mock_data_num_samples, config.train.seq_len, deterministic=config.seed.deterministic, base_seed=config.seed.seed)
            if config.train.do_val:
                self.val_dataset = MockDataset(config.data.mock_data_num_samples // 10, config.train.seq_len, deterministic=config.seed.deterministic, base_seed=config.seed.seed, seed_offset=1_000_000_000)
        else:
            class CustomDataset: ...
            self.train_dataset = CustomDataset(dataset_path=config.data.dataset_path, split="train")
            if config.train.do_val:
                self.val_dataset = CustomDataset(dataset_path=config.data.dataset_path, split="validation")
        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config.train.batch_size, sampler=self.train_sampler, num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
        if config.train.do_val:
            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.dp_world_size, rank=self.dp_rank, shuffle=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=config.train.batch_size, sampler=self.val_sampler, num_workers=config.data.num_workers, pin_memory=config.data.pin_memory)
        else:
            self.val_dataset = self.val_loader = None
        self.num_train_samples = len(self.train_dataset)
        self.train_sampler_epoch = 0
        self.train_loader_iter_idx = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.train_sampler_epoch)

    def _init_model(self, config: Config):
        torch.set_float32_matmul_precision('high')
        self.model_config = config.model
        # self.model_config.seed = config.seed.seed
        model = GPT(self.model_config)
        params_config = get_model_params(self.model_config)
        if self.master_process:
            print(f"Params config: {params_config}")
        if config.train.use_compile and hasattr(torch, 'compile'):
            model = torch.compile(model)
        #TODO: Here ZeRO-1 actually only need 'reduce' not 'all-reduce' used in DDP, we can develop a custom wrapper for ZeRO-1
        self.model = DDP(model, process_group=self.dp_group, find_unused_parameters=config.parallel.ddp_find_unused_parameters, gradient_as_bucket_view=config.parallel.ddp_gradient_as_bucket_view)
        self.raw_model = self.model.module

    def _init_optimizer(self, config: Config):
        if config.optim.optimizer == "adam":
            self.optimizer = tinytron.optim.AdamW(
                self.raw_model.parameters(), 
                lr=config.optim.max_lr,
                weight_decay=config.optim.weight_decay,
                betas=(config.optim.adam_beta1, config.optim.adam_beta2),
                eps=config.optim.adam_eps,
            )
        elif config.optim.optimizer == "muon":
            muon_params = []
            adam_params = []
            for name, param in self.raw_model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim == 2 and "wte" not in name and "lm_head" not in name:
                    muon_params.append(param)
                else:
                    adam_params.append(param)
            param_groups = [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "use_adam": False,
                    "lr": config.optim.max_lr,
                    "momentum": config.optim.muon_momentum,
                    "weight_decay": config.optim.weight_decay,
                },
                {
                    "params": adam_params,
                    "use_muon": False,
                    "use_adam": True,
                    "lr": config.optim.max_lr,
                    "betas": (config.optim.adam_beta1, config.optim.adam_beta2),
                    "eps": config.optim.adam_eps,
                    "weight_decay": config.optim.weight_decay,
                }
            ]
            self.optimizer = tinytron.optim.Muon(param_groups)
        if config.parallel.use_distributed_optimizer:
            self.optimizer = DistributedOptimizer(
                optimizer=self.optimizer,
                process_group=self.dp_group,
            )
            self.raw_optimizer = self.optimizer.optimizer
        else:
            self.raw_optimizer = self.optimizer

    def _init_profiler(self, config: Config):
        @contextmanager
        def dummy_record_function(name: str):
            yield
        def trace_handler(prof):
            if self.master_process:
                prof.export_chrome_trace(f"{self.log_dir}/rank{self.rank}_trace.json")
        if config.train.use_profiler:
            assert config.train.steps_to_profile[0] >= 1, "steps_to_profile[0] should be >= 1"
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=config.train.steps_to_profile[0]-1,
                    warmup=1,
                    active=config.train.steps_to_profile[1]+1-config.train.steps_to_profile[0],
                    repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=True,
                with_flops=True,
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            )
        else:
            self.profiler = None
        self.profiler_record_fn = torch.profiler.record_function if config.train.use_profiler else dummy_record_function

    def _init_logging(self, config: Config):
        # create the log directory we will write checkpoints to and log to
        self.log_dir = os.path.join(
            config.logging.log_dir,
            f"modelsize_{sum(p.numel() for p in self.raw_model.parameters())}_"
            f"lr{config.optim.max_lr}_"
            f"BS{config.train.batch_size}_"
            f"SL{config.train.seq_len}_"
            f"DP{self.dp_world_size}_"
            f"SEP{parallel_state.get_sep_world_size()}_"
        )
        if self.master_process:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, f"log.txt")
            with open(self.log_file, "w") as f: # open for writing to clear the file
                pass
        dist.barrier()

    def _lr_scheduler(self, it: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)

    def _autocast_context(self, precision: str):
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            raise NotImplementedError("FP16 precision is not supported yet")
        if precision == "fp32":
            return nullcontext()
        raise ValueError(f"Unsupported precision: {precision}. Supported precisions are: bf16, fp32.")

    def _prepare_local_batch(self, data_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data_batch["input_ids"], data_batch["labels"]
        _, T = x.shape
        assert T % self.sp_world_size == 0, "sequence length must be divisible by sp_world_size"
        seq_chunk_size = T // self.sp_world_size
        seq_start_idx = self.sp_rank * seq_chunk_size
        seq_end_idx = (self.sp_rank + 1) * seq_chunk_size
        x = x[:, seq_start_idx:seq_end_idx]
        y = y[:, seq_start_idx:seq_end_idx]
        x = x.to(f'cuda:{self.local_rank}')
        y = y.to(f'cuda:{self.local_rank}')
        return x, y
        
    def _one_training_micro_step(self, config: Config, micro_step: int, data_batch: dict):
        x, y = self._prepare_local_batch(data_batch)
        with self.profiler_record_fn("forward"):
            with self._autocast_context(config.train.precision):
                _, loss, logging_loss = self.model(x.reshape(x.shape[0],-1), y.reshape(y.shape[0],-1))
        loss = loss / self.training_info["grad_accum_steps"]
        with self.profiler_record_fn("backward"):
            loss.backward()
        return logging_loss / self.training_info["grad_accum_steps"]

    def _one_training_step(self, config: Config, step: int):
        self.model.train()
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.training_info["grad_accum_steps"]):
            batch = self._next_train_batch()
            self.model.require_backward_grad_sync = (micro_step == self.training_info["grad_accum_steps"] - 1)
            loss_accum += self._one_training_micro_step(config, micro_step, batch)
        # TODO: Refactor optimizer/grad communication by parameter group (dense/router vs expert-local).
        allreduce_non_expert_grads_across_sp(
            model=self.raw_model,
            sp_group=self.sp_group,
            sp_world_size=self.sp_world_size,
            expert_local_param_suffixes=EXPERT_LOCAL_PARAM_SUFFIXES,
        )
        norm = self.raw_model.clip_grad_norm(config.train.grad_clip_value)
        lr = self._lr_scheduler(step, self.training_info["max_steps"], config.optim.warmup_steps, config.optim.max_lr, config.optim.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        self.one_step_results["lr"] = lr
        self.one_step_results["loss"] = loss_accum
        self.one_step_results["grad_norm"] = norm
    
    def _next_train_batch(self):
        try:
            iter_idx, batch = next(self.train_loader_iter)
        except StopIteration:
            self.train_sampler_epoch += 1
            self.train_loader_iter_idx = 0
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.train_sampler_epoch)
            self.train_loader_iter = enumerate(self.train_loader)
            iter_idx, batch = next(self.train_loader_iter)
        self.train_loader_iter_idx = iter_idx + 1
        return batch

    def _resume_from_checkpoint(self):
        ckpt_dir = self.config.ckpt.resume_path or self.log_dir
        pattern = os.path.join(ckpt_dir, "*_model.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            self.start_step = 0
            return
        ckpt_prefix = ckpts[-1].replace("_model.pt", "")
        meta_path = f"{ckpt_prefix}_meta.pt"
        meta = torch.load(meta_path, map_location="cpu")
        # 1) model
        state_dict = torch.load(f"{ckpt_prefix}_model.pt", map_location="cpu", weights_only=True)
        self.raw_model.load_state_dict(state_dict)
        # 2) optimizer
        opt_key = f"optimizer/rank{self.rank}"
        opt_state_placeholder = {opt_key: self.raw_optimizer.state_dict()}
        state_dict_loader.load(
            state_dict=opt_state_placeholder,
            storage_reader=FileSystemReader(f"{ckpt_prefix}_opt"),
        )
        self.raw_optimizer.load_state_dict(opt_state_placeholder[opt_key])
        # 3) dataset state
        sampler_state = meta.get('sampler_state', {})
        epoch = int(sampler_state.get('epoch', 0))
        iter_idx = int(sampler_state.get('iter_idx', 0))
        self.train_sampler_epoch = epoch
        self.train_loader_iter_idx = iter_idx
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.train_loader_iter = enumerate(islice(self.train_loader, iter_idx, None), start=iter_idx)
        # 4) next step 
        step = meta.get('step', None)
        self.start_step = (step + 1) if (step is not None) else 0
        if self.master_process:
            print(f"=> Resumed from {ckpt_dir} | next_step={self.start_step}, "
                f"sampler_epoch={epoch}, dataloader_iter_idx={iter_idx}")
        # 5) RNG: finally load RNG state
        rng_path = f"{ckpt_prefix}_rng/rank{self.rank}.pt"
        if os.path.exists(rng_path):
            rng = torch.load(rng_path, map_location='cpu')
            torch.set_rng_state(rng['torch'].to(torch.uint8).cpu())
            torch.cuda.set_rng_state(rng['cuda'].to(torch.uint8).cpu(), self.local_rank)
            np.random.set_state(rng['numpy'])
            if 'python' in rng:
                random.setstate(rng['python'])
        dist.barrier()
        torch.cuda.synchronize()
    
    def train(self):
        self.results = {}
        self.train_loader_iter = enumerate(self.train_loader)
        self._resume_from_checkpoint()
        # training loop
        self._init_profiler(self.config)
        if self.profiler:
            self.profiler.start()
        for step in tqdm(range(self.start_step, self.training_info["max_steps"]), 
                        initial=self.start_step, total=self.training_info["max_steps"], 
                        desc="Train", disable=not self.master_process):
            self.one_step_results = {}
            t0 = time.time()
            last_step = (step == self.training_info["max_steps"] - 1)
            # 1) train
            with self.profiler_record_fn("training_step"):
                self._one_training_step(self.config, step)
            torch.cuda.synchronize()
            if self.profiler:
                self.profiler.step()
            # 2) eval
            if not self.config.train.debug and self.config.train.do_val and (step % self.config.train.val_every_steps == 0 or last_step):
                self.eval()
                if self.master_process:
                    tqdm.write(f"validation loss: {self.one_step_results['val_loss'].item():.4f}")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} val {self.one_step_results['val_loss'].item():.4f}\n")
            # 3) save
            if self.config.ckpt.do_save and not self.config.train.debug and step > 0 and (step % self.config.ckpt.save_every_steps == 0 or last_step):
                self.save(step)
            # 4) print
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_processed = self.config.train.batch_size * self.config.train.seq_len * self.training_info["grad_accum_steps"] * self.dp_world_size
            tokens_per_sec = tokens_processed / dt
            mfu, actual, peak = compute_mfu(
                self.raw_model, self.config.train.batch_size, self.config.train.seq_len, dt, self.training_info["grad_accum_steps"], dtype="bf16")
            if self.master_process:
                tqdm.write(f"step {step:5d}/{self.training_info['max_steps']} | loss: {self.one_step_results['loss'].item():.6f} | lr {self.one_step_results['lr']:.4e} | grad norm: {self.one_step_results['grad_norm']:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | MFU: {mfu*100:.2f}%")
                with open(self.log_file, "a") as f:
                    f.write(f"{step} train {self.one_step_results['loss'].item():.6f}\n")
            self.results[step] = self.one_step_results
        dist.destroy_process_group()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = len(self.val_loader)
            for batch in tqdm(self.val_loader, desc="Val", disable=not self.master_process):
                x, y = self._prepare_local_batch(batch)
                with self._autocast_context(self.config.train.precision):
                    logits, _, logging_loss = self.model(x.reshape(x.shape[0],-1), y.reshape(y.shape[0],-1))
                loss = logging_loss / val_loss_steps
                val_loss_accum += loss
        torch.cuda.synchronize()
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG, group=self.dp_group)
        self.one_step_results["val_loss"] = val_loss_accum
    
    def save(self, step: int | None = None):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(self.log_dir, f"{step:05d}")
        rng_dir = f"{checkpoint_path}_rng"
        os.makedirs(rng_dir, exist_ok=True)
        state_dict_saver.save(
            state_dict={f"optimizer/rank{self.rank}": self.raw_optimizer.state_dict()},
            storage_writer=FileSystemWriter(f"{checkpoint_path}_opt"),
        )
        rng_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(self.local_rank),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        torch.save(rng_state, os.path.join(rng_dir, f"rank{self.rank}.pt"))
        dist.barrier()
        if self.master_process:
            torch.save(self.raw_model.state_dict(), f"{checkpoint_path}_model.pt")
            checkpoint = {
                'config': self.config.as_dict(),
                'step': step,
                'this_step_results': self.one_step_results,
                'opt_part_assignment': self.optimizer.part_assignment if hasattr(self.optimizer, 'part_assignment') else None,
                'sampler_state': {
                    'epoch': self.train_sampler_epoch,
                    'iter_idx': self.train_loader_iter_idx,
                },
                'rng_state': rng_state,
            }
            torch.save(checkpoint, f"{checkpoint_path}_meta.pt")
        dist.barrier()
