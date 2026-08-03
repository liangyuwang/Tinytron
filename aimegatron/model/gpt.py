"""aimegatron.model.gpt

GPT assembly from the module set, plus the two distributed-aware gradient
routines (finalize + clip). The forward path reads like a single-GPU model:
parallelism only enters through the layer primitives and the embedding/lm
head boundary.

Activation layout:
- sequence_parallel off: hidden states are replicated across TP; TP regions
  reduce with all-reduce.
- sequence_parallel on: hidden states are sharded along the sequence dim
  outside TP regions; column layers gather, row layers reduce-scatter.

Edit contract: finalize_model_grads and clip_grad_norm rely on the
tp_replicated parameter marker from aimegatron.parallel.layers. Any module
that adds TP-replicated parameters must mark them at construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist

from aimegatron.core import mesh
from aimegatron.core.config import ModelConfig
from aimegatron.core.registry import NORMS
from aimegatron.model.attention import Attention
from aimegatron.model.loss import CrossEntropyLoss
from aimegatron.model.mlp import MLP
from aimegatron.model.moe import MoE, is_expert_parallel, is_sp_grad_complete
from aimegatron.parallel.layers import (ColumnParallelLinear, RowParallelLinear,
                                         VocabParallelEmbedding, is_tp_replicated)


def is_moe_layer(config: ModelConfig, layer_idx: int) -> bool:
    return config.moe_every > 0 and layer_idx % config.moe_every == 0


def init_gpt_weights(config: ModelConfig, wte, lm_head, blocks_with_idx,
                     tied_lm_head: bool) -> None:
    """Deterministic, layout-invariant init shared by the full GPT and by
    pipeline stages. Every rank draws the FULL logical tensor of every
    weight from the same seeded stream and keeps its slice, so shards are
    exact slices of the single-device weights regardless of tp/pp/ep size.

    Components a rank does not own are still drawn (and discarded) so the
    RNG stream position stays identical across layouts. `blocks_with_idx`
    pairs GLOBAL layer indices with their Block modules.
    """
    ref = wte.weight if wte is not None else next(iter(blocks_with_idx))[1].ln_1.weight
    with torch.random.fork_rng(devices=[ref.device] if ref.device.type == "cuda" else []):
        torch.manual_seed(0)
        if wte is not None:
            wte.init_normal_(0.0, config.init_std)
        else:
            _draw_and_discard((config.vocab_size, config.hidden_size), config.init_std,
                              ref.device, ref.dtype)
        if not tied_lm_head:
            if lm_head is not None:
                lm_head.init_normal_(0.0, config.init_std)
            else:
                _draw_and_discard((config.vocab_size, config.hidden_size), config.init_std,
                                  ref.device, ref.dtype)
        for layer_idx, block in blocks_with_idx:
            torch.manual_seed((layer_idx + 1) % (2 ** 31))
            for name, m in block.named_modules():
                if isinstance(m, MoE):
                    # Router + every global expert in canonical order, so the
                    # RNG stream is identical across ep/tp layouts.
                    m.init_normal_(0.0, config.init_std)
                elif isinstance(m, (ColumnParallelLinear, RowParallelLinear)) \
                        and ".experts." not in name:
                    m.init_normal_(0.0, config.init_std)


def _draw_and_discard(shape, std, device, dtype) -> None:
    tmp = torch.empty(shape, device=device, dtype=dtype)
    nn.init.normal_(tmp, mean=0.0, std=std)


class Block(nn.Module):

    def __init__(self, config: ModelConfig, moe: bool = False):
        super().__init__()
        self.ln_1 = NORMS.get(config.norm_type)(config.hidden_size)
        self.attn = Attention(config)
        self.ln_2 = NORMS.get(config.norm_type)(config.hidden_size)
        self.mlp = MoE(config) if moe else MLP(config)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), position_ids)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [Block(config, moe=is_moe_layer(config, i)) for i in range(config.num_layer)])
        self.lnf = NORMS.get(config.norm_type)(config.hidden_size)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, gather_output=True)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.loss_fn = CrossEntropyLoss()
        self._position_ids: torch.Tensor | None = None
        self._init_weights(config.init_std)

    def _init_weights(self, init_std: float) -> None:
        init_gpt_weights(
            self.config, wte=self.wte, lm_head=None if self.config.tied_lm_head else self.lm_head,
            blocks_with_idx=list(enumerate(self.blocks)), tied_lm_head=self.config.tied_lm_head)

    def _get_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._position_ids is None or self._position_ids.size(0) < seq_len \
                or self._position_ids.device != device:
            self._position_ids = torch.arange(seq_len, device=device)
        return self._position_ids[:seq_len]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block_size is only {self.config.block_size}"
        # Under SP the embedding output is already sequence-sharded; attention
        # restores the full sequence internally, so position ids are always full.
        x = self.wte(idx)
        position_ids = self._get_position_ids(T, idx.device)
        for block in self.blocks:
            x = block(x, position_ids)
        x = self.lnf(x)
        logits = self.lm_head(x)  # [B, T, vocab] gathered
        if targets is not None:
            loss, logging_loss = self.loss_fn(logits, targets)
            aux = self.total_moe_aux_loss()
            if aux is not None:
                loss = loss + self.config.moe_aux_loss_coeff * aux
            return logits, loss, logging_loss
        return logits

    def total_moe_aux_loss(self):
        """Sum of the last forward's MoE load-balance losses, or None for
        dense models. Only valid right after a forward pass."""
        total = None
        for m in self.modules():
            if isinstance(m, MoE):
                total = m.last_aux_loss if total is None else total + m.last_aux_loss
        return total

    def get_flops_per_fwd_bwd(self, batch_size: int, seq_len: int) -> float:
        """Approximate forward+backward FLOPs for the whole model."""
        c = self.config
        d, l, v = c.hidden_size, c.num_layer, c.vocab_size
        # Attention projections + output projection: 4 * B*T*d^2 per layer.
        attn_proj = 8 * batch_size * seq_len * d * d
        # QK^T and attention-over-V: 4 * B*T^2*d per layer.
        attn_core = 4 * batch_size * seq_len * seq_len * d
        # SwiGLU: three d<->ff projections.
        mlp = 6 * batch_size * seq_len * d * c.intermediate_size
        # Embedding + lm head lookup/projection.
        emb = 6 * batch_size * seq_len * d * v
        return l * (attn_proj + attn_core + mlp) + emb


def finalize_model_grads(model: nn.Module) -> None:
    """Make every gradient complete before clipping / optimizer step.

    1. Under sequence parallelism, TP-replicated params (norms, row biases)
       only saw a sequence shard -> all-reduce their grads across TP. Params
       marked sp_grad_complete (the MoE router) gathered their own input and
       are skipped.
    2. All-reduce gradients across the data-parallel scope: expert params
       only across the expert-DP group (identical experts), everything else
       across the full DP group.
    """
    tp_group, tp_size = mesh.get_tp_group(), mesh.get_tp_world_size()
    if mesh.sequence_parallel() and tp_size > 1:
        for p in model.parameters():
            if p.grad is not None and is_tp_replicated(p) and not is_sp_grad_complete(p):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=tp_group)

    dp_group, dp_size = mesh.get_dp_group(), mesh.get_dp_world_size()
    edp_group, edp_size = mesh.get_expert_dp_group(), mesh.get_expert_dp_world_size()
    if dp_size > 1:
        for p in model.parameters():
            if p.grad is None:
                continue
            if is_expert_parallel(p):
                # An expert's grad already sums contributions from every EP
                # peer's tokens (all-to-all dispatch), each scaled by the
                # peer's local 1/batch-size. Summing the edp replicas then
                # covers all dp_size data shards, so normalize by dp_size to
                # land exactly on the single-device mean-batch gradient.
                if edp_size > 1:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=edp_group)
                p.grad.div_(dp_size)
            else:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                p.grad.div_(dp_size)


def clip_grad_norm(model: nn.Module, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """Global grad norm correct under TP, EP, and DP.

    Three buckets: tp_replicated params are identical across TP after
    finalization (counted once); TP-sharded params sum their squared norms
    across the TP group; expert params are sharded across both TP and EP,
    so their squared norms are summed across both groups. After
    finalize_model_grads, gradients are identical across the data-parallel
    scope of each param, so no DP communication is needed.
    """
    device = next(model.parameters()).device
    replicated_sq = torch.zeros((), dtype=torch.float32, device=device)
    tp_sharded_sq = torch.zeros((), dtype=torch.float32, device=device)
    expert_sq = torch.zeros((), dtype=torch.float32, device=device)
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_sq = torch.linalg.vector_norm(p.grad.detach(), norm_type).to(torch.float32) ** norm_type
        if is_expert_parallel(p):
            expert_sq += grad_sq
        elif is_tp_replicated(p):
            replicated_sq += grad_sq
        else:
            tp_sharded_sq += grad_sq

    tp_group, tp_size = mesh.get_tp_group(), mesh.get_tp_world_size()
    if tp_size > 1:
        dist.all_reduce(tp_sharded_sq, op=dist.ReduceOp.SUM, group=tp_group)
        if expert_sq.item() > 0 or mesh.get_ep_world_size() > 1:
            dist.all_reduce(expert_sq, op=dist.ReduceOp.SUM, group=tp_group)
    ep_size = mesh.get_ep_world_size()
    if ep_size > 1:
        dist.all_reduce(expert_sq, op=dist.ReduceOp.SUM, group=mesh.get_ep_group())
    total_norm = (replicated_sq + tp_sharded_sq + expert_sq) ** (1.0 / norm_type)

    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef)
    return total_norm
