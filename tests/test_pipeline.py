"""Pipeline-parallel tests on CPU/gloo.

- Stage contents: each pp rank owns exactly its layer slice (+wte first,
  +lnf/lm_head last).
- Forward/loss equivalence: pp=2 one-microbatch loss equals the
  single-device GPT forward.
- Gradient equivalence: one full 1F1B step (2 micro-batches) produces the
  same per-parameter gradients as the single-device gradient-accumulation
  loop, and the same logging loss.
"""

import unittest

import torch

from aimegatron.core.config import ModelConfig
from aimegatron.model.gpt import GPT
from aimegatron.model.pipeline import PipelineStage, stage_layer_range
from aimegatron.parallel.pipeline import OneFOneBSchedule
from aimegatron.train.trainer import MockDataset
from common import run_distributed

BATCH_SIZE = 2
SEQ_LEN = 8
NUM_MICRO = 2


def _pp_model_config() -> ModelConfig:
    # tied_lm_head=False: pp>1 forbids weight tying (wte and lm_head live on
    # different stages).
    return ModelConfig(
        vocab_size=64, hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=32, num_layer=2, block_size=8, tied_lm_head=False,
    )


def _batches(num_micro: int):
    ds = MockDataset(num_micro * BATCH_SIZE, SEQ_LEN, vocab_size=64, seed=0)
    return [
        {"input_ids": torch.stack([ds[i]["input_ids"] for i in range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)]),
         "labels": torch.stack([ds[i]["labels"] for i in range(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)])}
        for j in range(num_micro)
    ]


def _worker_stage_contents(rank):
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    start, end = stage_layer_range(cfg.num_layer, 2, rank)
    assert sorted(int(k) for k in stage.blocks) == list(range(start, end)), \
        f"stage {rank} owns wrong blocks: {list(stage.blocks)}"
    assert stage.is_first == (rank == 0)
    assert stage.is_last == (rank == 1)
    assert (stage.wte is not None) == (rank == 0), "only the first stage owns wte"
    assert (stage.lnf is not None) == (rank == 1), "only the last stage owns lnf"
    assert (stage.lm_head is not None) == (rank == 1), "only the last stage owns lm_head"


def _worker_weights_match_gpt(rank):
    """Every stage parameter is the exact slice of the single-device GPT."""
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    gpt = GPT(cfg)
    gpt_state = gpt.state_dict()
    for name, value in stage.state_dict().items():
        assert name in gpt_state, f"stage key {name} missing from GPT state dict"
        assert torch.equal(value, gpt_state[name]), \
            f"stage {rank} init mismatch at {name}"


def _worker_forward_loss(rank):
    """pp=2 one-microbatch loss equals the single-device forward loss."""
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    batches = _batches(1)

    schedule = OneFOneBSchedule(stage, BATCH_SIZE, SEQ_LEN, cfg.hidden_size,
                                torch.float32, num_microbatches=1)
    it = iter(batches)
    loss = schedule.run_step(lambda: next(it))
    loss = schedule.broadcast_loss(loss)

    # Reference: same batch through the full single-device GPT.
    gpt = GPT(cfg)
    with torch.no_grad():
        _, ref_loss, ref_logging = gpt(batches[0]["input_ids"], batches[0]["labels"])
    assert abs(loss - ref_logging.item()) < 1e-6, \
        f"pp2 loss {loss:.6f} != single device {ref_logging.item():.6f}"


def _worker_grads_match(rank):
    """One full 1F1B step's grads equal the single-device accumulation grads."""
    cfg = _pp_model_config()
    stage = PipelineStage(cfg)
    batches = _batches(NUM_MICRO)

    schedule = OneFOneBSchedule(stage, BATCH_SIZE, SEQ_LEN, cfg.hidden_size,
                                torch.float32, num_microbatches=NUM_MICRO)
    it = iter(batches)
    logging_loss = schedule.run_step(lambda: next(it))

    gpt = GPT(cfg)
    gpt.zero_grad()
    ref_logging = 0.0
    for batch in batches:
        _, loss, logging = gpt(batch["input_ids"], batch["labels"])
        (loss / NUM_MICRO).backward()
        ref_logging += logging.item() / NUM_MICRO

    gpt_grads = dict(gpt.named_parameters())
    for name, p in stage.named_parameters():
        assert p.grad is not None, f"stage {rank} param {name} has no grad"
        ref = gpt_grads[name].grad
        diff = (p.grad - ref).abs().max().item()
        assert diff < 1e-6, f"grad mismatch at {name}: max diff {diff:.3e}"

    if stage.is_last:
        assert abs(logging_loss - ref_logging) < 1e-6, \
            f"logging loss {logging_loss:.6f} != single device {ref_logging:.6f}"


class TestPipeline(unittest.TestCase):

    def test_stage_contents(self):
        run_distributed(2, _worker_stage_contents, pp_size=2)

    def test_stage_weights_match_single_device(self):
        run_distributed(2, _worker_weights_match_gpt, pp_size=2)

    def test_pp2_forward_loss_equals_single_device(self):
        run_distributed(2, _worker_forward_loss, pp_size=2)

    def test_pp2_1f1b_grads_equal_single_device(self):
        run_distributed(2, _worker_grads_match, pp_size=2)


if __name__ == "__main__":
    unittest.main()
