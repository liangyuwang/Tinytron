"""Per-key weight drift diagnosis for pp=2 vs single device."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

import functools

import torch

from aimegatron.core.config import Config, DataConfig, ModelConfig, ParallelConfig, TrainConfig
from aimegatron.train.trainer import Trainer
from common import run_distributed


def align_config(log_dir, pp_size=1):
    return Config(
        model=ModelConfig(
            vocab_size=64, hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
            intermediate_size=32, num_layer=2, block_size=8, tied_lm_head=pp_size == 1,
        ),
        parallel=ParallelConfig(tp_size=1, pp_size=pp_size, ep_size=1, backend="gloo"),
        train=TrainConfig(
            total_batch_size=64, batch_size=2, seq_len=8,
            max_steps=2, max_lr=3e-2, min_lr=1e-3, warmup_steps=2,
            weight_decay=0.0, dtype="fp32",
            log_every_steps=1, log_dir=log_dir, use_distributed_optimizer=True,
        ),
        data=DataConfig(use_mock_data=True, mock_data_num_samples=64),
    )


def worker_ref(rank, out_dir):
    trainer = Trainer(align_config(os.path.join(out_dir, "log")))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, "ref.pt"))


def worker_pp(rank, out_dir):
    trainer = Trainer(align_config(os.path.join(out_dir, "log"), pp_size=2))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, f"pp_{rank}.pt"))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as out_dir:
        run_distributed(1, functools.partial(worker_ref, out_dir=out_dir))
        run_distributed(2, functools.partial(worker_pp, out_dir=out_dir), pp_size=2)
        ref = torch.load(os.path.join(out_dir, "ref.pt"), weights_only=True)
        merged = {}
        for r in range(2):
            merged.update(torch.load(os.path.join(out_dir, f"pp_{r}.pt"), weights_only=True))
        print(f"keys ref={len(ref)} merged={len(merged)} missing={set(ref) - set(merged)}")
        for k in sorted(ref):
            d = (ref[k].float() - merged[k].float()).abs().max().item()
            flag = "  <<< DRIFT" if d > 1e-6 else ""
            print(f"{k:50s} {d:.3e}{flag}")
