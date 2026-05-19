import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tinytron.model.config import ModelConfig
from tinytron.distributed import (
    parallel_state,
    ep_all_to_all,
    ep_all_to_all_no_grad,
)

class MLP(nn.Module):
    """Dense MLP or Single expert in MoE"""
    def __init__(self, config: ModelConfig, layer_idx: int, use_moe: bool = False):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if use_moe else config.intermediate_size
        self.use_moe = use_moe

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=self.device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=self.device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=self.device)
        self.act_fn = nn.SiLU()
        self._init_weights(config.seed, layer_idx)

    def _init_weights(self, base_seed: int, layer_idx: int):
        with torch.random.fork_rng(devices=[self.gate_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.gate_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.up_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.up_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.down_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.down_proj.weight, mean=0.0, std=self.config.init_std)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, top_k: int | None = None):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = top_k if top_k is not None else config.num_experts_per_tok
        assert self.top_k <= self.num_experts, f"top_k must be less than or equal to num_experts, got {self.top_k} and {self.num_experts}"
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.grouped_gemm_supported = (
            torch.cuda.is_available()
            and torch.version.cuda is not None
            and hasattr(F, "grouped_mm")
            and torch.cuda.get_device_capability(self.device)[0] >= 8.0
        )
        rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        if not self.grouped_gemm_supported and self.layer_idx == 0 and rank0:
            print(f"⚠️ [Performance Warning] torch.nn.functional.grouped_mm is NOT supported or hardware requirements (SM >= 8.0) are not met."
                  f"MoE will fallback to the padded batched matmul loop (Slow Path).")

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=self.device)

        if dist.is_available() and dist.is_initialized():
            try:
                self.ep_size = parallel_state.get_ep_world_size()
            except AssertionError:
                self.ep_size = 1
        else:
            self.ep_size = 1
        assert self.num_experts % self.ep_size == 0
        self.num_local_experts = self.num_experts // self.ep_size

        self.experts_gate_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_up_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_down_weights = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size, device=self.device))
        self.experts_act_fn = nn.SiLU()
        self._init_expert_weights(config.seed, layer_idx)

    def _trace(self, message: str, *, sync: bool = False) -> None:
        if os.environ.get("TINYTRON_MOE_TRACE", "0") != "1":
            return
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        now = time.strftime("%H:%M:%S")
        print(f"[moe-trace {now} rank={rank} layer={self.layer_idx}] {message}", flush=True)
    
    def _init_expert_weights(self, base_seed: int, layer_idx: int):
        if dist.is_available() and dist.is_initialized():
            try:
                ep_rank = parallel_state.get_ep_rank()
            except AssertionError:
                ep_rank = 0
        else:
            ep_rank = 0
        
        with torch.random.fork_rng(devices=[self.experts_gate_weights.device]):
            for local_idx in range(self.num_local_experts):
                global_expert_idx = ep_rank * self.num_local_experts + local_idx
                
                expert_seed = base_seed + layer_idx + global_expert_idx
                torch.manual_seed(expert_seed)
                
                nn.init.normal_(self.experts_gate_weights[local_idx], mean=0.0, std=self.config.init_std)
                nn.init.normal_(self.experts_up_weights[local_idx], mean=0.0, std=self.config.init_std)
                nn.init.normal_(self.experts_down_weights[local_idx], mean=0.0, std=self.config.init_std)

        with torch.random.fork_rng(devices=[self.router.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            nn.init.normal_(self.router.weight, mean=0.0, std=self.config.init_std)

    def _zero_expert_weight_dependency(self) -> torch.Tensor:
        return (
            self.experts_gate_weights.reshape(-1)[0]
            + self.experts_up_weights.reshape(-1)[0]
            + self.experts_down_weights.reshape(-1)[0]
        ) * 0.0

    def _apply_local_experts(
        self,
        received_x: torch.Tensor,
        received_experts: torch.Tensor,
    ) -> torch.Tensor:
        if received_experts.numel() == 0:
            # Preserve the autograd edge to the preceding EP all-to-all. If this
            # returns a fresh empty tensor, ranks with no local tokens skip that
            # all-to-all in backward while other ranks still enter it.
            target_dtype = (
                torch.get_autocast_gpu_dtype()
                if torch.is_autocast_enabled("cuda")
                else received_x.dtype
            )
            return received_x.reshape(0, self.hidden_size).to(dtype=target_dtype)

        local_expert_indices = received_experts % self.num_local_experts
        local_sort_idx = torch.argsort(local_expert_indices)
        local_x = received_x[local_sort_idx].contiguous()
        local_expert_indices = local_expert_indices[local_sort_idx]
        counts = torch.bincount(local_expert_indices, minlength=self.num_local_experts)
        offs = torch.cumsum(counts, dim=0).to(torch.int32)
        self._trace(
            f"apply_local start tokens={local_x.size(0)} counts={counts.detach().cpu().tolist()}",
            sync=True,
        )
        if self.grouped_gemm_supported:
            gate_out = F.grouped_mm(local_x, self.experts_gate_weights, offs=offs)
            up_out = F.grouped_mm(local_x, self.experts_up_weights, offs=offs)
            act_out = self.experts_act_fn(gate_out) * up_out
            down_out = F.grouped_mm(act_out, self.experts_down_weights, offs=offs)
        else:   # slow
            max_tokens = counts.max().item()
            if max_tokens == 0:
                padded_x = local_x.view(self.num_local_experts, 0, self.hidden_size)
                gate_out_padded = torch.bmm(padded_x, self.experts_gate_weights.transpose(1, 2))
                up_out_padded = torch.bmm(padded_x, self.experts_up_weights.transpose(1, 2))
                act_out_padded = self.experts_act_fn(gate_out_padded) * up_out_padded
                down_out_padded = torch.bmm(act_out_padded, self.experts_down_weights.transpose(1, 2))
                down_out = down_out_padded.view(0, self.hidden_size)
            else:
                starts = torch.zeros_like(offs)
                starts[1:] = offs[:-1]
                relative_idx = torch.arange(len(local_x), device=local_x.device) - starts[local_expert_indices]
                padded_x = torch.zeros(
                    self.num_local_experts, max_tokens, self.hidden_size,
                    dtype=local_x.dtype, device=local_x.device
                )
                padded_x = padded_x.index_put((local_expert_indices, relative_idx), local_x)
                self._trace(f"slow_path padded_bmm start max_tokens={max_tokens}", sync=True)

                gate_out_padded = torch.bmm(padded_x, self.experts_gate_weights.transpose(1, 2))
                up_out_padded = torch.bmm(padded_x, self.experts_up_weights.transpose(1, 2))
                act_out_padded = self.experts_act_fn(gate_out_padded) * up_out_padded
                down_out_padded = torch.bmm(act_out_padded, self.experts_down_weights.transpose(1, 2))
                down_out = down_out_padded[local_expert_indices, relative_idx]
                self._trace("slow_path padded_bmm done", sync=True)

        rev_local_sort_idx = torch.argsort(local_sort_idx)
        self._trace("apply_local done", sync=True)
        return down_out[rev_local_sort_idx].contiguous()

    def _forward_sep_local_reduce_inference(
        self,
        x: torch.Tensor,
        flat_x: torch.Tensor,
        selected_experts: torch.Tensor,
        weights: torch.Tensor,
        ep_group,
        ep_rank: int,
    ) -> torch.Tensor:
        """
        During SEP-only inference, every rank in the EP/SEP group sees the same tokens.
        Each rank computes only its local-expert contribution, then all-reduces the
        per-token partial output across the group.
        """
        B, T, D = x.size()
        target_ep_ranks = selected_experts // self.num_local_experts
        local_mask = target_ep_ranks == ep_rank

        partial_flat = flat_x.new_zeros((flat_x.size(0), D))
        if local_mask.any():
            local_indices = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
            local_x = flat_x[local_indices].contiguous()
            local_experts = selected_experts[local_indices].contiguous()
            local_out = self._apply_local_experts(local_x, local_experts)
            partial_flat = partial_flat.index_put((local_indices,), local_out)

        partial_flat = partial_flat * weights.unsqueeze(-1)
        final_x = partial_flat.view(B * T, self.top_k, D).sum(dim=1).reshape(B, T, D)
        dist.all_reduce(final_x, op=dist.ReduceOp.SUM, group=ep_group)
        return final_x
    
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.size()
        if dist.is_available() and dist.is_initialized():
            try:
                ep_group = parallel_state.get_ep_group()
                ep_world_size = parallel_state.get_ep_world_size()
                ep_rank = parallel_state.get_ep_rank()
            except AssertionError:
                ep_group = None
                ep_world_size = 1
                ep_rank = 0
        else:
            ep_group = None
            ep_world_size = 1
            ep_rank = 0
        # router_logits: (batch * N, n_experts)
        gate_logits = self.router(x) # [B, T, total_experts]
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1).view(-1)      # [B * T * top_k]
        selected_experts = selected_experts.view(-1)       # [B * T * top_k]
        flat_x = x.view(-1, D).repeat_interleave(self.top_k, dim=0)

        if use_cache and ep_world_size > 1:
            final_x = self._forward_sep_local_reduce_inference(
                x=x,
                flat_x=flat_x,
                selected_experts=selected_experts,
                weights=weights,
                ep_group=ep_group,
                ep_rank=ep_rank,
            )
            return final_x, gate_logits

        if ep_world_size > 1:
            target_ep_ranks = selected_experts // self.num_local_experts
            global_sort_idx = torch.argsort(target_ep_ranks)
            sorted_x = flat_x[global_sort_idx].contiguous()
            sorted_experts = selected_experts[global_sort_idx].contiguous()
            send_splits_tensor = torch.bincount(target_ep_ranks, minlength=ep_world_size)
            recv_splits_tensor = torch.empty_like(send_splits_tensor)
            self._trace(f"splits all_to_all start send={send_splits_tensor.detach().cpu().tolist()}", sync=True)
            dist.all_to_all_single(recv_splits_tensor, send_splits_tensor, group=ep_group)
            send_splits, recv_splits = torch.stack([send_splits_tensor, recv_splits_tensor]).cpu().tolist()
            self._trace(f"splits all_to_all done recv={recv_splits}", sync=True)
            self._trace(f"x all_to_all start send={send_splits} recv={recv_splits}", sync=True)
            received_x = ep_all_to_all(sorted_x, send_splits, recv_splits, ep_group)
            self._trace(f"x all_to_all done received={received_x.size(0)}", sync=True)
            self._trace("experts all_to_all start", sync=True)
            received_experts = ep_all_to_all_no_grad(
                sorted_experts,
                send_splits,
                recv_splits,
                ep_group,
            )
            self._trace("experts all_to_all done", sync=True)
        else:
            received_x = flat_x
            received_experts = selected_experts

        out_x = self._apply_local_experts(received_x, received_experts)
        if ep_world_size > 1:
            self._trace(f"out all_to_all start send={recv_splits} recv={send_splits}", sync=True)
            combined_x = ep_all_to_all(out_x, recv_splits, send_splits, ep_group)
            self._trace("out all_to_all done", sync=True)
            rev_global_sort_idx = torch.argsort(global_sort_idx)
            unpermuted_x = combined_x[rev_global_sort_idx]
        else:
            unpermuted_x = out_x
        unpermuted_x = unpermuted_x * weights.unsqueeze(-1)
        final_x = unpermuted_x.view(B * T, self.top_k, D).sum(dim=1)
        if self.training:
            final_x = final_x + self._zero_expert_weight_dependency()

        return final_x.reshape(B, T, D), gate_logits
