"""End-to-end training smoke tests on CPU/gloo.

Covered layouts:
- tp=2 + sequence parallel + ZeRO-1 (loss decreases)
- tp=2 without sequence parallel (loss decreases)
- tp=1, dp=2 + ZeRO-1 across DP (loss decreases, data sharded per DP rank)
- pp=2 with the 1F1B schedule (loss decreases)
- dp=2, ep=2 MoE with ZeRO-1 dense+expert optimizers (loss decreases)
- cross-TP resume: train at tp=2 with checkpointing, resume at tp=1 via
  layout-driven resharding, eval loss must match
"""

import functools
import os
import tempfile
import unittest

import torch

from aimegatron.core.config import Config, DataConfig, ModelConfig, ParallelConfig, TrainConfig
from aimegatron.train.trainer import Trainer
from common import run_distributed


def _smoke_config(log_dir: str, tp_size: int = 2, sequence_parallel: bool = True,
                  max_steps: int = 30, do_save: bool = False) -> Config:
    return Config(
        model=ModelConfig(
            vocab_size=64, hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
            intermediate_size=32, num_layer=2, block_size=8, tied_lm_head=True,
        ),
        parallel=ParallelConfig(tp_size=tp_size, sequence_parallel=sequence_parallel, backend="gloo"),
        train=TrainConfig(
            total_batch_size=64, batch_size=2, seq_len=8,
            max_steps=max_steps, max_lr=3e-2, min_lr=1e-3, warmup_steps=2,
            weight_decay=0.0, dtype="fp32",
            log_every_steps=10, log_dir=log_dir, use_distributed_optimizer=True,
            do_save=do_save,
        ),
        data=DataConfig(use_mock_data=True, mock_data_num_samples=64),
    )


def _eval_loss(trainer: Trainer, num_batches: int = 4) -> float:
    trainer.model.eval()
    total = 0.0
    loader = iter(trainer.dataloader)
    with torch.no_grad():
        for _ in range(num_batches):
            batch = next(loader)
            _, loss, _ = trainer.model(batch["input_ids"], batch["labels"])
            total += loss.item()
    trainer.model.train()
    return total / num_batches


def _worker(rank, log_dir):
    trainer = Trainer(_smoke_config(log_dir))
    initial_loss = _eval_loss(trainer)
    trainer.train()
    final_loss = _eval_loss(trainer)
    assert torch.isfinite(torch.tensor(final_loss)), f"loss not finite: {final_loss}"
    assert final_loss < initial_loss, \
        f"loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"


def _worker_tp2_no_sp(rank, log_dir):
    trainer = Trainer(_smoke_config(log_dir, sequence_parallel=False))
    initial_loss = _eval_loss(trainer)
    trainer.train()
    final_loss = _eval_loss(trainer)
    assert final_loss < initial_loss, \
        f"tp-no-sp loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"


def _worker_dp2(rank, log_dir):
    # world=2, tp=1 -> dp=2; ZeRO-1 shards optimizer state across the DP group
    # and the data stream must be sharded per DP rank.
    trainer = Trainer(_smoke_config(log_dir, tp_size=1, sequence_parallel=False, max_steps=20))
    assert len(trainer.dataloader) < len(trainer.train_dataset) // 2 + 1, \
        "DP rank should only see its own data shard"
    initial_loss = _eval_loss(trainer)
    trainer.train()
    final_loss = _eval_loss(trainer)
    assert final_loss < initial_loss, \
        f"dp2 loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"


def _worker_train_tp2_and_dump_loss(rank, log_dir):
    trainer = Trainer(_smoke_config(log_dir, max_steps=4, do_save=True))
    trainer.train()
    # Eval must run on ALL TP ranks: the forward contains TP collectives, so a
    # rank-0-only eval would deadlock. Only the file write is rank-gated.
    loss = _eval_loss(trainer)
    if rank == 0:
        with open(os.path.join(log_dir, "tp2_loss.txt"), "w") as f:
            f.write(f"{loss:.8f}")


def _worker_pp2(rank, log_dir):
    # pp=2 forbids weight tying; the loss trajectory comes from the trainer
    # log since eval would need a full-model forward across stages.
    cfg = _smoke_config(log_dir, tp_size=1, sequence_parallel=False, max_steps=20)
    cfg.model.tied_lm_head = False
    cfg.parallel.pp_size = 2
    trainer = Trainer(cfg)
    trainer.train()
    if rank == 0:
        losses = []
        with open(os.path.join(log_dir, "log.txt")) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "train":
                    losses.append(float(parts[2]))
        assert len(losses) >= 2, f"expected logged losses, got {losses}"
        assert losses[-1] < losses[0], \
            f"pp2 loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"


def _worker_moe_ep2(rank, log_dir):
    # world=2, tp=1 -> dp=2; ep=2 shards the 4 experts across the DP group.
    cfg = _smoke_config(log_dir, tp_size=1, sequence_parallel=False, max_steps=20)
    cfg.parallel.ep_size = 2
    cfg.model.num_experts = 4
    cfg.model.num_experts_per_tok = 2
    cfg.model.moe_every = 1
    trainer = Trainer(cfg)
    initial_loss = _eval_loss(trainer)   # collective: runs on both EP ranks
    trainer.train()
    final_loss = _eval_loss(trainer)
    assert torch.isfinite(torch.tensor(final_loss)), f"loss not finite: {final_loss}"
    assert final_loss < initial_loss, \
        f"moe ep2 loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"


def _worker_resume_tp1(rank, log_dir):
    # Fresh tp=1 run resumes the tp=2 checkpoint via layout-driven resharding.
    trainer = Trainer(_smoke_config(log_dir, tp_size=1, sequence_parallel=False, max_steps=4))
    assert trainer.start_step == 4, f"expected resume at step 4, got {trainer.start_step}"
    with open(os.path.join(log_dir, "tp2_loss.txt")) as f:
        expected = float(f.read().strip())
    got = _eval_loss(trainer)
    assert abs(got - expected) < 1e-4, \
        f"resharded model eval loss mismatch: tp2={expected:.6f} tp1={got:.6f}"


class TestTrainSmoke(unittest.TestCase):

    def test_tp2_sp_zero1_training(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker, log_dir=log_dir),
                            tp_size=2, sequence_parallel=True)

    def test_tp2_no_sp_training(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker_tp2_no_sp, log_dir=log_dir),
                            tp_size=2, sequence_parallel=False)

    def test_dp2_zero1_training(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker_dp2, log_dir=log_dir), tp_size=1)

    def test_pp2_1f1b_training(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker_pp2, log_dir=log_dir), pp_size=2)

    def test_moe_ep2_training(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker_moe_ep2, log_dir=log_dir),
                            ep_size=2)

    def test_resume_across_tp_resharding(self):
        with tempfile.TemporaryDirectory() as log_dir:
            run_distributed(2, functools.partial(_worker_train_tp2_and_dump_loss, log_dir=log_dir),
                            tp_size=2, sequence_parallel=True)
            run_distributed(1, functools.partial(_worker_resume_tp1, log_dir=log_dir), tp_size=1)


if __name__ == "__main__":
    unittest.main()
