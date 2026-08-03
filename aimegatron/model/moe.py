"""aimegatron.model.moe

Mixture-of-Experts MLP with expert parallelism inside the DP dimension.

Routing is dropless and capacity-free: every token goes to its top-k
experts. Tokens are permuted by destination expert and exchanged with the
EP group via a gather of the expert-sorted chunks, local per-expert
slicing, and an all-reduce scatter back, all inside one autograd Function
(gloo's alltoallv cannot express dropless routing, and autograd-external
collectives would be scheduled in nondeterministic order across ranks).
Routing is a pure function of the (TP-replicated) hidden states, so EP
runs are deterministic replays of the single-device math.

Sequence-parallel contract (same as MLP): the module gathers the sequence
on entry and reduce-scatters the output, so expert/router gradients see
the full sequence and need no SP finalization.

Edit contract: expert parameters carry the expert_parallel marker (see
mark_expert_parallel); gradient finalization and clip_grad_norm key off it.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from aimegatron.core import mesh
from aimegatron.model.mlp import MLP
from aimegatron.parallel import comm

EXPERT_PARALLEL_ATTR = "expert_parallel"
# Params that are replicated across TP but live inside a self-gathering
# module (the router), so their gradients are already complete under SP and
# must NOT be all-reduced across TP during SP finalization.
SP_GRAD_COMPLETE_ATTR = "sp_grad_complete"


def mark_expert_parallel(param: nn.Parameter) -> nn.Parameter:
    setattr(param, EXPERT_PARALLEL_ATTR, True)
    return param


def is_expert_parallel(param: nn.Parameter) -> bool:
    return bool(getattr(param, EXPERT_PARALLEL_ATTR, False))


def mark_sp_grad_complete(param: nn.Parameter) -> nn.Parameter:
    setattr(param, SP_GRAD_COMPLETE_ATTR, True)
    return param


def is_sp_grad_complete(param: nn.Parameter) -> bool:
    return bool(getattr(param, SP_GRAD_COMPLETE_ATTR, False))


class _EPExchange(torch.autograd.Function):
    """Dropless, capacity-free token exchange across the EP group, wrapped in
    ONE autograd Function so every collective is issued in a fixed order.

    Gloo's alltoallv requires every rank's total send == total recv, which
    dropless routing cannot guarantee, so the exchange is a gather of the
    expert-sorted token chunks, local per-expert slicing, and an all-reduce
    scatter back. Keeping every collective inside one Function matters:
    collectives issued from separate autograd nodes run on the engine's
    worker threads and can be ordered differently on different ranks, which
    deadlocks gloo. Here backward issues exactly two all-reduces on every
    rank, always in the same order.

    Each expert run sees the same peer-ordered token list the single-device
    expert loop would see, so the math matches exactly.
    """

    @staticmethod
    def forward(ctx, group, ep_rank, experts, sorted_tokens, counts):
        ep_size = dist.get_world_size(group=group)
        epp = len(experts)
        hidden = sorted_tokens.size(1)
        n_local = sorted_tokens.size(0)

        count_list = [torch.empty_like(counts) for _ in range(ep_size)]
        dist.all_gather(count_list, counts.contiguous(), group=group)
        t = torch.tensor([n_local], dtype=torch.long)
        size_list = [torch.empty_like(t) for _ in range(ep_size)]
        dist.all_gather(size_list, t, group=group)
        sizes = [int(s) for s in size_list]
        chunk_offsets = [0]
        for s in sizes:
            chunk_offsets.append(chunk_offsets[-1] + s)

        max_n = max(sizes)
        padded = sorted_tokens if n_local == max_n else torch.cat(
            [sorted_tokens, sorted_tokens.new_zeros(max_n - n_local, hidden)])
        gathered = [torch.empty_like(padded) for _ in range(ep_size)]
        dist.all_gather(gathered, padded.contiguous(), group=group)
        chunks = [g[:s] for g, s in zip(gathered, sizes)]
        cumsums = [torch.cat([c.new_zeros(1), c.cumsum(0)]) for c in count_list]

        # Iterate GLOBAL experts in canonical order so each expert's token
        # rows (in peer order) match the single-device expert loop exactly;
        # each rank computes only the experts it owns.
        outs = []
        for g in range(ep_size * epp):
            if not (ep_rank * epp <= g < (ep_rank + 1) * epp):
                continue
            rows = []
            for q in range(ep_size):
                s, t2 = int(cumsums[q][g].item()), int(cumsums[q][g + 1].item())
                if t2 > s:
                    rows.append(chunks[q][s:t2])
            inp = torch.cat(rows, dim=0) if rows else sorted_tokens.new_empty(0, hidden)
            outs.append(experts[g - ep_rank * epp](inp))

        # Lay the expert outputs back into the peer-major token order: each
        # global row is written by exactly one rank (its expert owner), the
        # others contribute zeros, so the all-reduce SUM is a pure scatter.
        full = sorted_tokens.new_zeros(chunk_offsets[-1], hidden)
        idx = 0
        for g in range(ep_size * epp):
            if not (ep_rank * epp <= g < (ep_rank + 1) * epp):
                continue
            out_e = outs[idx]
            idx += 1
            c = 0
            for q in range(ep_size):
                s, t2 = int(cumsums[q][g].item()), int(cumsums[q][g + 1].item())
                if t2 > s:
                    full[chunk_offsets[q] + s:chunk_offsets[q] + t2] = out_e[c:c + t2 - s]
                    c += t2 - s
        dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
        result = full[chunk_offsets[ep_rank]:chunk_offsets[ep_rank] + n_local].clone()
        # Peer-major token values, needed to replay the expert runs in backward.
        token_full = torch.cat(chunks, dim=0).contiguous()

        ctx.group = group
        ctx.ep_rank = ep_rank
        ctx.experts = experts
        ctx.epp = epp
        ctx.sizes = sizes
        ctx.chunk_offsets = chunk_offsets
        ctx.cumsums = cumsums
        ctx.save_for_backward(full, token_full)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        group, ep_rank = ctx.group, ctx.ep_rank
        experts, epp = ctx.experts, ctx.epp
        sizes, chunk_offsets, cumsums = ctx.sizes, ctx.chunk_offsets, ctx.cumsums
        ep_size = len(sizes)
        full, token_full = ctx.saved_tensors
        hidden = full.size(1)

        # Transpose of the all-reduce scatter: place this rank's grad rows
        # into a zero tensor and sum-reduce; every rank ends with the grads
        # of ALL output rows.
        full_grad = full.new_zeros(full.shape)
        start = chunk_offsets[ep_rank]
        full_grad[start:start + grad_output.size(0)] = grad_output
        dist.all_reduce(full_grad, op=dist.ReduceOp.SUM, group=group)

        # Replay the local expert runs on the gathered TOKEN values with the
        # gathered grad rows to get expert param grads and the grads of the
        # gathered input rows.
        chunks = [token_full[chunk_offsets[q]:chunk_offsets[q] + s]
                  for q, s in enumerate(sizes)]
        chunk_grads = [c.new_zeros(c.shape) for c in chunks]
        with torch.enable_grad():
            for g in range(ep_size * epp):
                if not (ep_rank * epp <= g < (ep_rank + 1) * epp):
                    continue
                expert = experts[g - ep_rank * epp]
                rows, spans = [], []
                for q in range(ep_size):
                    s, t2 = int(cumsums[q][g].item()), int(cumsums[q][g + 1].item())
                    if t2 > s:
                        rows.append(chunks[q][s:t2])
                        spans.append((q, s, t2))
                if not spans:
                    # Zero-token expert: still materialize zero param grads,
                    # matching the empty-input run on single device.
                    (sum(p.sum() for p in expert.parameters()) * 0).backward()
                    continue
                inp = torch.cat(rows, dim=0).detach().requires_grad_(True)
                grad_rows = torch.cat(
                    [full_grad[chunk_offsets[q] + s:chunk_offsets[q] + t2]
                     for q, s, t2 in spans], dim=0)
                expert(inp).backward(grad_rows)
                c = 0
                for q, s, t2 in spans:
                    chunk_grads[q][s:t2] += inp.grad[c:c + t2 - s]
                    c += t2 - s

        # Transpose of the gather: each chunk's grad is the sum of every
        # rank's contribution, so stack all chunk grads into one padded
        # tensor, all-reduce once, then keep this rank's own chunk.
        max_n = max(sizes)
        stacked = torch.cat([c if c.size(0) == max_n
                             else torch.cat([c, c.new_zeros(max_n - c.size(0), hidden)])
                             for c in chunk_grads], dim=0)
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=group)
        sorted_tokens_grad = stacked.view(ep_size, max_n, hidden)[ep_rank, :sizes[ep_rank]].clone()

        return None, None, None, sorted_tokens_grad, None


class MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        ep_size, ep_rank = mesh.get_ep_world_size(), mesh.get_ep_rank()
        assert self.num_experts % ep_size == 0
        self.experts_per_partition = self.num_experts // ep_size
        self.expert_start = ep_rank * self.experts_per_partition

        # Router is replicated across TP and EP; it sits inside this
        # self-gathering module, so its gradient is complete under SP.
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        mark_sp_grad_complete(self.router.weight)

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.experts_per_partition)])
        for expert in self.experts:
            for p in expert.parameters():
                mark_expert_parallel(p)

        # Set on every forward; the enclosing stage/model folds it into the
        # loss (or the pipeline backward pass).
        self.last_aux_loss = torch.zeros(())

    def init_normal_(self, mean: float, std: float) -> None:
        """Layout-invariant init: draw the router, then EVERY global expert
        in canonical order from the shared seeded stream. Owned experts keep
        their TP slice via the parallel layers' init; non-owned experts are
        drawn and discarded so the RNG stream position stays identical
        across ep/tp sizes (shards are exact slices of the single-device
        weights)."""
        nn.init.normal_(self.router.weight, mean=mean, std=std)
        shapes = ((self.intermediate_size, self.hidden_size),    # gate_proj
                  (self.intermediate_size, self.hidden_size),    # up_proj
                  (self.hidden_size, self.intermediate_size))    # down_proj
        attrs = ("gate_proj", "up_proj", "down_proj")
        for e in range(self.num_experts):
            owned = self.expert_start <= e < self.expert_start + self.experts_per_partition
            expert = self.experts[e - self.expert_start] if owned else None
            for full_shape, attr in zip(shapes, attrs):
                if owned:
                    getattr(expert, attr).init_normal_(mean, std)
                else:
                    tmp = torch.empty(full_shape, dtype=self.router.weight.dtype,
                                      device=self.router.weight.device)
                    nn.init.normal_(tmp, mean=mean, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group = mesh.get_tp_group()
        if mesh.sequence_parallel():
            x = comm.gather_from_sequence_parallel_region(x, group)
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])                    # [N, H]

        probs = F.softmax(self.router(tokens), dim=-1)       # [N, E]
        topk_w, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        n = tokens.size(0)
        token_ids = torch.arange(n, device=x.device).repeat_interleave(self.top_k)
        dest_expert = topk_idx.reshape(-1)                   # [N*k]
        order = torch.argsort(dest_expert, stable=True)      # group by expert
        sorted_tokens = tokens[token_ids[order]]
        counts = torch.bincount(dest_expert, minlength=self.num_experts)

        aux = self._load_balance_aux(probs, dest_expert)
        self.last_aux_loss = aux

        ep_group = mesh.get_ep_group()
        ep_size = mesh.get_ep_world_size()
        if ep_size > 1:
            combined = _EPExchange.apply(ep_group, mesh.get_ep_rank(), self.experts,
                                         sorted_tokens, counts)
        else:
            outs = []
            offset = 0
            for expert, num in zip(self.experts, counts.tolist()):
                # Empty slices keep the autograd edge, so every expert param
                # still receives a (zero) gradient.
                outs.append(expert(sorted_tokens[offset:offset + num]))
                offset += num
            combined = torch.cat(outs, dim=0)

        # Undo the permutation with the routing weights.
        out_tokens = tokens.new_zeros(n, shape[-1])
        out_tokens.index_add_(0, token_ids[order],
                              combined * topk_w.reshape(-1)[order].unsqueeze(-1))
        out = out_tokens.reshape(shape)
        if mesh.sequence_parallel():
            out = comm.reduce_scatter_to_sequence_parallel_region(out, group)
        return out

    def _load_balance_aux(self, probs, dest_expert) -> torch.Tensor:
        """Switch-style balance loss: E * sum_i f_i * P_i, with f_i the routed
        token fraction and P_i the mean router probability for expert i."""
        f = torch.bincount(dest_expert, minlength=self.num_experts).to(probs.dtype)
        f = f / dest_expert.numel()
        p = probs.mean(dim=0)
        return self.num_experts * (f * p).sum()
