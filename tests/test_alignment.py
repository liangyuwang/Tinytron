"""Parallel-vs-single-device alignment tests on CPU/gloo.

Every parallel layout must reproduce single-device training: same config, same
data stream, same seed. The only allowed deviation is floating-point rounding
from chunked GEMMs and different reduction orders.

Compared after NUM_STEPS optimizer steps:
- final weights (TP shards are merged via train/layout rules before comparing)
- final eval loss (TP layouts only; with dp>1 each rank evaluates its own data
  shard, so the eval loss is not directly comparable by construction)
"""

import functools
import os
import tempfile
import unittest

import torch

from aimegatron.core.config import Config, DataConfig, ModelConfig, ParallelConfig, TrainConfig
from aimegatron.train.layout import reshard_tensor
from aimegatron.train.trainer import Trainer
from common import run_distributed
from test_train_smoke import _eval_loss

NUM_STEPS = 10
# Measured drift on CPU/fp32 after 10 AdamW steps is ~2e-6 (weights) and
# ~1e-7 (loss); 1e-5 leaves headroom for kernel/library rounding variance.
WEIGHT_ATOL = 1e-5
LOSS_ATOL = 1e-5

REF_WEIGHTS = "ref_weights.pt"
REF_LOSS = "ref_loss.txt"


def _align_config(log_dir: str, tp_size: int = 1, sequence_parallel: bool = False,
                  pp_size: int = 1, ep_size: int = 1, moe: bool = False) -> Config:
    model_kwargs = dict(
        vocab_size=64, hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=32, num_layer=2, block_size=8,
        # pp>1 forbids weight tying (wte and lm_head live on different stages).
        tied_lm_head=pp_size == 1,
    )
    if moe:
        # aux coefficient is zero on purpose: the balance loss uses local
        # token statistics, which legitimately differ between the sharded
        # layouts and the single-device run; zeroing it isolates the
        # routing/dispatch math this test verifies.
        model_kwargs.update(num_experts=4, num_experts_per_tok=2, moe_every=1,
                            moe_aux_loss_coeff=0.0)
    return Config(
        model=ModelConfig(**model_kwargs),
        parallel=ParallelConfig(tp_size=tp_size, pp_size=pp_size, ep_size=ep_size,
                                sequence_parallel=sequence_parallel, backend="gloo"),
        train=TrainConfig(
            total_batch_size=64, batch_size=2, seq_len=8,
            max_steps=NUM_STEPS, max_lr=3e-2, min_lr=1e-3, warmup_steps=2,
            weight_decay=0.0, dtype="fp32",
            log_every_steps=NUM_STEPS + 1, log_dir=log_dir, use_distributed_optimizer=True,
        ),
        data=DataConfig(use_mock_data=True, mock_data_num_samples=64),
    )


def _worker_reference(rank, out_dir, moe=False):
    """Single-device reference run: tp=1, dp=1."""
    trainer = Trainer(_align_config(os.path.join(out_dir, "log"), moe=moe))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, REF_WEIGHTS))
    loss = _eval_loss(trainer)
    with open(os.path.join(out_dir, REF_LOSS), "w") as f:
        f.write(f"{loss:.8f}")


def _worker_tp(rank, out_dir, sequence_parallel):
    """tp=2 run; every rank saves its own shard, rank 0 dumps the eval loss."""
    trainer = Trainer(_align_config(os.path.join(out_dir, "log"), tp_size=2,
                                    sequence_parallel=sequence_parallel))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, f"tp_{rank}.pt"))
    loss = _eval_loss(trainer)  # collective forward: must run on all TP ranks
    if rank == 0:
        with open(os.path.join(out_dir, "loss.txt"), "w") as f:
            f.write(f"{loss:.8f}")


def _worker_dp2(rank, out_dir, sequence_parallel=False):
    """tp=1, dp=2 run; after gradient sync every rank holds a full replica.
    (sequence_parallel is accepted only to match _run_pair's worker kwarg.)"""
    trainer = Trainer(_align_config(os.path.join(out_dir, "log"), tp_size=1, sequence_parallel=False))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, f"dp_{rank}.pt"))


def _worker_pp2(rank, out_dir, sequence_parallel=False):
    """pp=2 run; each stage saves its own state (global block keys)."""
    trainer = Trainer(_align_config(os.path.join(out_dir, "log"), pp_size=2))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, f"pp_{rank}.pt"))


def _worker_moe_ep2(rank, out_dir, sequence_parallel=False):
    """tp=1, dp=2, ep=2 MoE run; each rank saves its state (local experts)."""
    trainer = Trainer(_align_config(os.path.join(out_dir, "log"), ep_size=2, moe=True))
    trainer.train()
    torch.save({k: v.detach().cpu() for k, v in trainer.model.state_dict().items()},
               os.path.join(out_dir, f"ep_{rank}.pt"))


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _max_diff(a: dict, b: dict) -> float:
    assert set(a.keys()) == set(b.keys()), f"key mismatch: {set(a) ^ set(b)}"
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def _merged_tp_weights(shard0: dict, shard1: dict) -> dict:
    """Merge the two tp=2 shard dicts back to tp=1 layout."""
    return {
        name: reshard_tensor([shard0[name], shard1[name]], name, 2, 1, 0)
        for name in shard0
    }


def _merged_ep_weights(w0: dict, w1: dict, experts_per_partition: int) -> dict:
    """Merge two ep shards by expert-index ownership: rank r's local expert i
    is global expert r * experts_per_partition + i."""
    merged = dict(w0)
    for name, value in w1.items():
        parts = name.split(".")
        if "experts" in parts:
            idx = parts.index("experts")
            parts[idx + 1] = str(int(parts[idx + 1]) + experts_per_partition)
        merged[".".join(parts)] = value
    return merged


class TestParallelAlignment(unittest.TestCase):

    def _run_pair(self, world_size, worker, tp_size=1, pp_size=1, ep_size=1,
                  sequence_parallel=False, moe=False):
        """Reference run + parallel run in one tmp dir; every artifact is loaded
        before the directory is torn down."""
        with tempfile.TemporaryDirectory() as out_dir:
            run_distributed(1, functools.partial(_worker_reference, out_dir=out_dir, moe=moe))
            run_distributed(world_size,
                            functools.partial(worker, out_dir=out_dir, sequence_parallel=sequence_parallel),
                            tp_size=tp_size, pp_size=pp_size, ep_size=ep_size,
                            sequence_parallel=sequence_parallel)
            payload = {}
            for name in os.listdir(out_dir):
                path = os.path.join(out_dir, name)
                if not os.path.isfile(path):
                    continue
                if name.endswith(".pt"):
                    payload[name] = _load(path)
                else:
                    with open(path) as f:
                        payload[name] = f.read().strip()
            return payload

    def test_tp2_no_sp_aligns_with_single_device(self):
        payload = self._run_pair(2, _worker_tp, tp_size=2, sequence_parallel=False)
        merged = _merged_tp_weights(payload["tp_0.pt"], payload["tp_1.pt"])
        diff = _max_diff(payload[REF_WEIGHTS], merged)
        self.assertLess(diff, WEIGHT_ATOL, f"tp2-no-sp weight drift vs single device: {diff:.3e}")
        ref_loss, loss = float(payload[REF_LOSS]), float(payload["loss.txt"])
        self.assertLess(abs(loss - ref_loss), LOSS_ATOL,
                        f"tp2-no-sp eval loss {loss:.6f} vs single {ref_loss:.6f}")

    def test_tp2_sp_aligns_with_single_device(self):
        payload = self._run_pair(2, _worker_tp, tp_size=2, sequence_parallel=True)
        merged = _merged_tp_weights(payload["tp_0.pt"], payload["tp_1.pt"])
        diff = _max_diff(payload[REF_WEIGHTS], merged)
        self.assertLess(diff, WEIGHT_ATOL, f"tp2-sp weight drift vs single device: {diff:.3e}")
        ref_loss, loss = float(payload[REF_LOSS]), float(payload["loss.txt"])
        self.assertLess(abs(loss - ref_loss), LOSS_ATOL,
                        f"tp2-sp eval loss {loss:.6f} vs single {ref_loss:.6f}")

    def test_dp2_aligns_with_single_device(self):
        payload = self._run_pair(2, _worker_dp2, tp_size=1)
        w0, w1 = payload["dp_0.pt"], payload["dp_1.pt"]
        # DP replicas must agree with each other and with single device.
        replica_diff = _max_diff(w0, w1)
        self.assertLess(replica_diff, WEIGHT_ATOL,
                        f"dp replicas diverged: {replica_diff:.3e}")
        diff = _max_diff(payload[REF_WEIGHTS], w0)
        self.assertLess(diff, WEIGHT_ATOL, f"dp2 weight drift vs single device: {diff:.3e}")

    def test_pp2_aligns_with_single_device(self):
        payload = self._run_pair(2, _worker_pp2, pp_size=2)
        # Stages use global block keys, so the union IS the full model.
        merged = {**payload["pp_0.pt"], **payload["pp_1.pt"]}
        diff = _max_diff(payload[REF_WEIGHTS], merged)
        self.assertLess(diff, WEIGHT_ATOL, f"pp2 weight drift vs single device: {diff:.3e}")

    def test_moe_ep2_aligns_with_single_device(self):
        payload = self._run_pair(2, _worker_moe_ep2, ep_size=2, moe=True)
        merged = _merged_ep_weights(payload["ep_0.pt"], payload["ep_1.pt"],
                                    experts_per_partition=2)
        diff = _max_diff(payload[REF_WEIGHTS], merged)
        self.assertLess(diff, WEIGHT_ATOL, f"moe ep2 weight drift vs single device: {diff:.3e}")


if __name__ == "__main__":
    unittest.main()
