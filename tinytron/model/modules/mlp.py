import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tinytron.training.config import ModelConfig
from tinytron.distributed import (
    parallel_state,
    ep_all_to_all,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        if not self.grouped_gemm_supported and self.layer_idx == 0 and dist.get_rank() == 0:
            print(f"⚠️ [Performance Warning] torch.nn.functional.grouped_mm is NOT supported or hardware requirements (SM >= 8.0) are not met."
                  f"MoE will fallback to the padded batched matmul loop (Slow Path).")

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, device=self.device)

        self.ep_size = parallel_state.get_ep_world_size()
        assert self.num_experts % self.ep_size == 0
        self.num_local_experts = self.num_experts // self.ep_size

        self.experts_gate_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_up_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_down_weights = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size, device=self.device))
        self.experts_act_fn = nn.SiLU()
        self._router_debug_stats_buffer: list[dict[str, float]] = []
        self._route_trace_tensors_buffer: list[torch.Tensor] = []
        self._init_expert_weights(config.seed, layer_idx)
    
    def _init_expert_weights(self, base_seed: int, layer_idx: int):
        ep_rank = parallel_state.get_ep_rank()
        
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.size()
        ep_group = parallel_state.get_ep_group()
        ep_world_size = parallel_state.get_ep_world_size()
        # router_logits: (batch * N, n_experts)
        gate_logits = self.router(x) # [B, T, total_experts]
        topk_out = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(topk_out.values, dim=-1).view(-1)      # [B * T * top_k]
        selected_experts = topk_out.indices.view(-1)               # [B * T * top_k]
        if self.config.moe_router_debug:
            self._record_router_debug_stats(gate_logits, topk_out.indices)
        if self.config.moe_route_trace:
            self._route_trace_tensors_buffer.append(topk_out.indices.detach().to(torch.int32).contiguous())
        flat_x = x.view(-1, D).repeat_interleave(self.top_k, dim=0)

        target_ep_ranks = selected_experts // self.num_local_experts
        global_sort_idx = torch.argsort(target_ep_ranks)
        sorted_x = flat_x[global_sort_idx].contiguous()
        sorted_experts = selected_experts[global_sort_idx].contiguous()
        send_splits_tensor = torch.bincount(target_ep_ranks, minlength=ep_world_size)
        recv_splits_tensor = torch.empty_like(send_splits_tensor)
        dist.all_to_all_single(recv_splits_tensor, send_splits_tensor, group=ep_group)
        send_splits, recv_splits = torch.stack([send_splits_tensor, recv_splits_tensor]).cpu().tolist()
        received_x = ep_all_to_all(sorted_x, send_splits, recv_splits, ep_group)
        received_experts = torch.empty(sum(recv_splits), dtype=sorted_experts.dtype, device=x.device)
        dist.all_to_all_single(
            received_experts, sorted_experts,
            output_split_sizes=recv_splits, input_split_sizes=send_splits, group=ep_group
        )

        local_expert_indices = received_experts % self.num_local_experts
        local_sort_idx = torch.argsort(local_expert_indices)
        local_x = received_x[local_sort_idx].contiguous()
        local_expert_indices = local_expert_indices[local_sort_idx]
        counts = torch.bincount(local_expert_indices, minlength=self.num_local_experts)
        offs = torch.cumsum(counts, dim=0).to(torch.int32)
        if self.grouped_gemm_supported:
            gate_out = F.grouped_mm(local_x, self.experts_gate_weights, offs=offs)
            up_out = F.grouped_mm(local_x, self.experts_up_weights, offs=offs)
            act_out = self.experts_act_fn(gate_out) * up_out
            down_out = F.grouped_mm(act_out, self.experts_down_weights, offs=offs)
        else:   # slow
            max_tokens = counts.max().item()
            if max_tokens == 0:
                # down_out = torch.empty_like(local_x)  # will break autograd graph, change to the following code
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
                # padded_x[local_expert_indices, relative_idx] = local_x    # # will break autograd graph, change to the following code
                padded_x = padded_x.index_put((local_expert_indices, relative_idx), local_x)
                
                gate_out_padded = torch.bmm(padded_x, self.experts_gate_weights.transpose(1, 2))
                up_out_padded = torch.bmm(padded_x, self.experts_up_weights.transpose(1, 2))
                act_out_padded = self.experts_act_fn(gate_out_padded) * up_out_padded
                down_out_padded = torch.bmm(act_out_padded, self.experts_down_weights.transpose(1, 2))
                down_out = down_out_padded[local_expert_indices, relative_idx]

        rev_local_sort_idx = torch.argsort(local_sort_idx)
        out_x = down_out[rev_local_sort_idx].contiguous()
        combined_x = ep_all_to_all(out_x, recv_splits, send_splits, ep_group)
        rev_global_sort_idx = torch.argsort(global_sort_idx)

        unpermuted_x = combined_x[rev_global_sort_idx]
        unpermuted_x = unpermuted_x * weights.unsqueeze(-1)
        final_x = unpermuted_x.view(B * T, self.top_k, D).sum(dim=1)

        return final_x.reshape(B, T, D), gate_logits

    def pop_router_debug_stats(self) -> list[dict[str, float]]:
        stats = self._router_debug_stats_buffer
        self._router_debug_stats_buffer = []
        return stats

    def pop_route_trace_tensors(self) -> list[torch.Tensor]:
        traces = self._route_trace_tensors_buffer
        self._route_trace_tensors_buffer = []
        return traces

    def _record_router_debug_stats(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor):
        with torch.no_grad():
            next_rank = min(self.top_k + 1, self.num_experts)
            if next_rank > self.top_k:
                topk_plus = torch.topk(gate_logits, next_rank, dim=-1).values
                boundary_margin = topk_plus[..., self.top_k - 1] - topk_plus[..., self.top_k]
            else:
                boundary_margin = torch.zeros_like(gate_logits[..., 0])

            margin = boundary_margin.reshape(-1).float()
            counts = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts).float()
            probs = counts / counts.sum().clamp(min=1.0)
            valid_probs = probs[probs > 0]
            entropy = -(valid_probs * valid_probs.log()).sum()
            if self.num_experts > 1:
                entropy = entropy / torch.log(torch.tensor(float(self.num_experts), device=entropy.device))

            stats = {
                "router_margin_mean": margin.mean().item(),
                "router_margin_min": margin.min().item(),
                "router_margin_p01": torch.quantile(margin, 0.01).item(),
                "router_margin_p001": torch.quantile(margin, 0.001).item(),
                "router_margin_le_1e-3_ratio": (margin <= 1e-3).float().mean().item(),
                "router_margin_le_1e-4_ratio": (margin <= 1e-4).float().mean().item(),
                "router_margin_le_1e-5_ratio": (margin <= 1e-5).float().mean().item(),
                "router_topk_entropy_norm": entropy.item(),
            }
            self._router_debug_stats_buffer.append(stats)
