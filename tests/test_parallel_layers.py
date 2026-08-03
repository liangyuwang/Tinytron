"""Numerical equivalence of TP-sharded layers vs single-device baselines.

Each test spawns 2 gloo processes (tp_size=2) and asserts that the sharded
forward/backward exactly reproduces the baseline linear / embedding math,
both with and without sequence parallelism.
"""

import unittest

import torch
import torch.nn as nn

from aimegatron.core import mesh
from aimegatron.parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from common import run_distributed

B, T, IN, OUT = 2, 8, 16, 32
V, H = 16, 8


def _baseline_linear():
    g = torch.Generator().manual_seed(123)
    w = torch.randn(OUT, IN, generator=g)
    x = torch.randn(B, T, IN, generator=g)
    return w, x


def _baseline_grads(w, x, gout):
    x_base = x.clone().requires_grad_(True)
    y = x_base @ w.t()
    y.backward(gout)
    w_grad = gout.reshape(-1, OUT).t() @ x.reshape(-1, IN)
    return y.detach(), x_base.grad.detach(), w_grad


def _worker_column(rank):
    w, x = _baseline_linear()
    gout = torch.ones(B, T, OUT)
    y, x_grad, w_grad = _baseline_grads(w, x, gout)

    col = ColumnParallelLinear(IN, OUT)
    with torch.no_grad():
        col.weight.copy_(w.chunk(2, dim=0)[rank])
    assert mesh.sequence_parallel() is False

    xin = x.clone().requires_grad_(True)
    out = col(xin)
    assert torch.allclose(out.detach(), y.chunk(2, dim=-1)[rank], atol=1e-5), "column fwd mismatch"
    out.backward(gout.chunk(2, dim=-1)[rank])
    assert torch.allclose(xin.grad, x_grad, atol=1e-5), "column input-grad mismatch"
    assert torch.allclose(col.weight.grad, w_grad.chunk(2, dim=0)[rank], atol=1e-5), "column weight-grad mismatch"


def _worker_column_sp(rank):
    w, x = _baseline_linear()
    gout = torch.ones(B, T, OUT)
    y, x_grad, w_grad = _baseline_grads(w, x, gout)

    col = ColumnParallelLinear(IN, OUT)
    with torch.no_grad():
        col.weight.copy_(w.chunk(2, dim=0)[rank])
    assert mesh.sequence_parallel() is True

    xin = x.chunk(2, dim=1)[rank].clone().contiguous().requires_grad_(True)
    out = col(xin)  # gathers sequence internally -> [B, T, OUT/2]
    assert out.shape == (B, T, OUT // 2)
    assert torch.allclose(out.detach(), y.chunk(2, dim=-1)[rank], atol=1e-5), "column-SP fwd mismatch"
    out.backward(gout.chunk(2, dim=-1)[rank])
    expected_x_grad_shard = x_grad.chunk(2, dim=1)[rank]
    assert torch.allclose(xin.grad, expected_x_grad_shard, atol=1e-5), "column-SP input-grad mismatch"
    assert torch.allclose(col.weight.grad, w_grad.chunk(2, dim=0)[rank], atol=1e-5), "column-SP weight-grad mismatch"


def _worker_row(rank):
    w, x = _baseline_linear()
    gout = torch.ones(B, T, OUT)
    y, x_grad, w_grad = _baseline_grads(w, x, gout)

    row = RowParallelLinear(IN, OUT)
    with torch.no_grad():
        row.weight.copy_(w.chunk(2, dim=1)[rank])
    assert mesh.sequence_parallel() is False

    xin = x.chunk(2, dim=-1)[rank].clone().contiguous().requires_grad_(True)
    out = row(xin)  # all-reduced -> [B, T, OUT]
    assert torch.allclose(out.detach(), y, atol=1e-5), "row fwd mismatch"
    out.backward(gout)
    assert torch.allclose(xin.grad, x_grad.chunk(2, dim=-1)[rank], atol=1e-5), "row input-grad mismatch"
    assert torch.allclose(row.weight.grad, w_grad.chunk(2, dim=1)[rank], atol=1e-5), "row weight-grad mismatch"


def _worker_row_sp(rank):
    w, x = _baseline_linear()
    gout = torch.ones(B, T, OUT)
    y, x_grad, w_grad = _baseline_grads(w, x, gout)

    row = RowParallelLinear(IN, OUT)
    with torch.no_grad():
        row.weight.copy_(w.chunk(2, dim=1)[rank])
    assert mesh.sequence_parallel() is True

    xin = x.chunk(2, dim=-1)[rank].clone().contiguous().requires_grad_(True)
    out = row(xin)  # reduce-scattered -> [B, T/2, OUT]
    assert out.shape == (B, T // 2, OUT)
    assert torch.allclose(out.detach(), y.chunk(2, dim=1)[rank], atol=1e-5), "row-SP fwd mismatch"
    out.backward(gout.chunk(2, dim=1)[rank])
    assert torch.allclose(xin.grad, x_grad.chunk(2, dim=-1)[rank], atol=1e-5), "row-SP input-grad mismatch"
    assert torch.allclose(row.weight.grad, w_grad.chunk(2, dim=1)[rank], atol=1e-5), "row-SP weight-grad mismatch"


def _worker_embedding(rank):
    g = torch.Generator().manual_seed(7)
    w = torch.randn(V, H, generator=g)
    ids = torch.randint(0, V, (B, T), generator=g)

    baseline = nn.Embedding(V, H)
    with torch.no_grad():
        baseline.weight.copy_(w)
    expected = baseline(ids)

    vpe = VocabParallelEmbedding(V, H)
    with torch.no_grad():
        vpe.weight.copy_(w.chunk(2, dim=0)[rank])
    out = vpe(ids)
    assert torch.allclose(out.detach(), expected, atol=1e-6), "vocab-parallel embedding mismatch"


class TestParallelLayers(unittest.TestCase):

    def test_column_parallel(self):
        run_distributed(2, _worker_column, tp_size=2)

    def test_column_parallel_sp(self):
        run_distributed(2, _worker_column_sp, tp_size=2, sequence_parallel=True)

    def test_row_parallel(self):
        run_distributed(2, _worker_row, tp_size=2)

    def test_row_parallel_sp(self):
        run_distributed(2, _worker_row_sp, tp_size=2, sequence_parallel=True)

    def test_vocab_parallel_embedding(self):
        run_distributed(2, _worker_embedding, tp_size=2)


if __name__ == "__main__":
    unittest.main()
