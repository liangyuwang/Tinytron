"""MoE + expert-parallel tests on CPU/gloo.

ep=2 dispatch/combine must reproduce the single-device expert loop: same
routing, same outputs, same gradients. Both EP ranks are fed IDENTICAL
tokens here (module-level test, not the trainer's DP-sharded stream), so:
- outputs / input grads / router grads match the reference 1:1,
- expert weight grads are exactly ep_size times the reference, because each
  rank dispatches its own copy of the tokens to the expert owner (with
  distinct DP data in real training this sum is the correct full-batch
  gradient; see finalize_model_grads).
"""

import functools
import os
import tempfile
import unittest

import torch
import torch.distributed as dist

from aimegatron.core.config import ModelConfig
from aimegatron.model.moe import MoE
from common import run_distributed

REF = "moe_ref.pt"
ATOL = 1e-6


def _moe_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64, hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=32, num_layer=1, block_size=8,
        num_experts=4, num_experts_per_tok=2, moe_every=1,
    )


def _input():
    g = torch.Generator().manual_seed(3)
    return torch.randn(6, 8, 16, generator=g)   # [B, T, H], identical on all ranks


def _build_moe() -> MoE:
    """MoE with the canonical layout-invariant init from a shared seed, so
    every ep layout holds exact slices of the same global weights."""
    moe = MoE(_moe_config())
    torch.manual_seed(123)
    moe.init_normal_(0.0, 0.02)
    return moe


def _worker_reference(rank, out_dir):
    """Single-device MoE (ep=1): save weights, output, and all grads."""
    moe = _build_moe()
    x = _input().requires_grad_(True)
    y = moe(x)
    aux = moe.last_aux_loss.detach().clone()
    y.sum().backward()
    torch.save({
        "state": {k: v.detach().clone() for k, v in moe.state_dict().items()},
        "y": y.detach(),
        "x_grad": x.grad.detach(),
        "grads": {k: v.grad.detach().clone() for k, v in moe.named_parameters()},
        "aux": aux,
    }, os.path.join(out_dir, REF))


def _worker_ep2(rank, out_dir):
    moe = _build_moe()
    ref = torch.load(os.path.join(out_dir, REF), map_location="cpu", weights_only=True)

    # Layout-invariant init: local experts are exact slices of the reference.
    for i in range(moe.experts_per_partition):
        g = i + moe.expert_start
        for name, p in moe.experts[i].named_parameters():
            assert torch.equal(p.detach(), ref["state"][f"experts.{g}.{name}"]), \
                f"ep rank {rank} expert {g} init mismatch at {name}"
    assert torch.equal(moe.router.weight.detach(), ref["state"]["router.weight"]), \
        "router init mismatch"

    x = _input().requires_grad_(True)
    y = moe(x)
    assert torch.allclose(y, ref["y"], atol=ATOL), \
        f"ep2 output mismatch: {(y - ref['y']).abs().max().item():.3e}"
    assert torch.allclose(moe.last_aux_loss, ref["aux"], atol=ATOL), \
        f"aux loss mismatch: {moe.last_aux_loss.item()} vs {ref['aux'].item()}"

    # Routing determinism: both EP ranks must compute the identical output.
    y0 = y.detach().clone()
    dist.broadcast(y0, src=0)
    assert torch.allclose(y, y0, atol=ATOL), "EP ranks disagree on the MoE output"

    y.sum().backward()
    assert torch.allclose(x.grad, ref["x_grad"], atol=ATOL), \
        f"input grad mismatch: {(x.grad - ref['x_grad']).abs().max().item():.3e}"
    assert torch.allclose(moe.router.weight.grad, ref["grads"]["router.weight"], atol=ATOL), \
        "router grad mismatch"

    ep_size = 2
    for i in range(moe.experts_per_partition):
        g = i + moe.expert_start
        for name, p in moe.experts[i].named_parameters():
            got, want = p.grad, ref["grads"][f"experts.{g}.{name}"]
            # Duplicate tokens across both EP ranks -> exactly ep_size times
            # the single-device gradient (see module docstring).
            assert torch.allclose(got, ep_size * want, atol=ATOL), \
                f"expert {g} grad mismatch at {name}: " \
                f"{(got - ep_size * want).abs().max().item():.3e}"


class TestMoE(unittest.TestCase):

    def test_ep2_matches_single_device(self):
        with tempfile.TemporaryDirectory() as out_dir:
            run_distributed(1, functools.partial(_worker_reference, out_dir=out_dir))
            run_distributed(2, functools.partial(_worker_ep2, out_dir=out_dir), ep_size=2)


if __name__ == "__main__":
    unittest.main()
