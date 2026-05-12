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


class MoERouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size, device=device))
        self.forced_selected_experts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def set_forced_routing(self, selected_experts: torch.Tensor) -> None:
        self.forced_selected_experts = selected_experts.detach()

    def clear_forced_routing(self) -> None:
        self.forced_selected_experts = None

    def has_forced_routing(self) -> bool:
        return self.forced_selected_experts is not None

    def route_forced(self, x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        selected_experts = self.forced_selected_experts
        if selected_experts is None:
            raise RuntimeError("Forced routing was requested without selected experts.")
        B, T, _ = x.size()
        if tuple(selected_experts.shape) != (B, T, top_k):
            raise ValueError(
                "Forced routing shape mismatch: "
                f"expected {(B, T, top_k)}, got {tuple(selected_experts.shape)}"
            )
        selected_experts = selected_experts.to(device=x.device, dtype=torch.long)
        weights = x.new_full((B, T, top_k), 1.0 / top_k)
        return weights, selected_experts


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

        self.router = MoERouter(self.hidden_size, self.num_experts, device=self.device)
        self.routing_strategy = "learned"
        self.last_full_expert_outputs = None
        self.last_reference_output = None
        self.last_measurement_input = None
        self.last_measurement_q = None
        self.last_warmup_selected_experts = None

        self.ep_size = parallel_state.get_ep_world_size()
        assert self.num_experts % self.ep_size == 0
        self.num_local_experts = self.num_experts // self.ep_size

        self.experts_gate_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_up_weights = nn.Parameter(torch.empty(self.num_local_experts, self.intermediate_size, self.hidden_size, device=self.device))
        self.experts_down_weights = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_size, self.intermediate_size, device=self.device))
        self.experts_act_fn = nn.SiLU()
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

    def clear_warmup_state(self) -> None:
        self.router.clear_forced_routing()
        self.last_full_expert_outputs = None
        self.last_reference_output = None
        self.last_measurement_input = None
        self.last_measurement_q = None
        self.last_warmup_selected_experts = None

    def set_forced_routing(self, selected_experts: torch.Tensor) -> None:
        self.router.set_forced_routing(selected_experts)

    def _apply_experts(
        self,
        flat_x: torch.Tensor,
        selected_experts: torch.Tensor,
        ep_group,
        ep_world_size: int,
    ) -> torch.Tensor:
        selected_experts = selected_experts.reshape(-1)

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
        return unpermuted_x

    def _apply_all_experts(self, x: torch.Tensor, ep_group, ep_world_size: int) -> torch.Tensor:
        B, T, D = x.size()
        num_tokens = B * T
        flat_x = x.reshape(num_tokens, D).repeat_interleave(self.num_experts, dim=0)
        all_experts = (
            torch.arange(self.num_experts, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(num_tokens, self.num_experts)
            .reshape(-1)
        )
        out = self._apply_experts(flat_x, all_experts, ep_group, ep_world_size)
        return out.view(B, T, self.num_experts, D)

    def _prepare_warmup_routing_from_reference_grad(
        self,
        reference_grad: torch.Tensor | None,
    ) -> None:
        if (
            self.last_full_expert_outputs is None
            or self.last_measurement_input is None
            or reference_grad is None
        ):
            return
        expert_outputs = self.last_full_expert_outputs.to(torch.float32)
        grad = reference_grad.detach().to(torch.float32)
        q = -(expert_outputs * grad.unsqueeze(2)).sum(dim=-1)
        selected_experts = torch.topk(q, self.top_k, dim=-1).indices
        self.last_measurement_q = q.detach()
        self.last_warmup_selected_experts = selected_experts.detach()
        self.set_forced_routing(selected_experts)

    def _capture_reference_grad(self, reference_grad: torch.Tensor) -> None:
        self._prepare_warmup_routing_from_reference_grad(reference_grad)
        self.last_full_expert_outputs = None
        self.last_reference_output = None
        self.last_measurement_input = None
        return None

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.router.has_forced_routing():
            weights, selected_experts = self.router.route_forced(x, self.top_k)
            return weights, selected_experts, None

        if self.routing_strategy != "learned":
            raise ValueError(
                f"Unsupported routing_strategy={self.routing_strategy!r}. "
                "Expected one of: learned, full_observation."
            )

        gate_logits = self.router(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, selected_experts, gate_logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.size()
        ep_group = parallel_state.get_ep_group()
        ep_world_size = parallel_state.get_ep_world_size()

        if self.routing_strategy == "full_observation":
            all_outputs = self._apply_all_experts(x, ep_group, ep_world_size)
            reference_output = all_outputs.mean(dim=2)
            self.last_full_expert_outputs = all_outputs.detach()
            self.last_reference_output = reference_output
            self.last_measurement_input = x.detach()
            self.last_measurement_q = None
            self.last_warmup_selected_experts = None
            if reference_output.requires_grad:
                reference_output.register_hook(self._capture_reference_grad)
            return reference_output, None

        weights, selected_experts, gate_logits = self._route(x)
        weights = weights.view(-1)                         # [B * T * top_k]
        selected_experts = selected_experts.view(-1)       # [B * T * top_k]
        flat_x = x.view(-1, D).repeat_interleave(self.top_k, dim=0)
        unpermuted_x = self._apply_experts(flat_x, selected_experts, ep_group, ep_world_size)
        unpermuted_x = unpermuted_x * weights.unsqueeze(-1)
        final_x = unpermuted_x.view(B * T, self.top_k, D).sum(dim=1)

        return final_x.reshape(B, T, D), gate_logits
