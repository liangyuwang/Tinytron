import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tinytron.model.config import ModelConfig
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
        # Runtime knobs used by scripts/moe_router. They default to the standard
        # learned top-k router so existing training scripts are unchanged.
        self.routing_strategy = "learned"
        self.router_probe_experts = 0
        self.router_probe_tokens = 0
        self.router_ranking_loss_weight = 0.0
        self.last_router_probe_loss = None
        self.last_routing_stats = {}

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

    def _apply_local_experts(
        self,
        received_x: torch.Tensor,
        received_experts: torch.Tensor,
    ) -> torch.Tensor:
        if received_experts.numel() == 0:
            return received_x.new_empty((0, self.hidden_size))

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
                
                gate_out_padded = torch.bmm(padded_x, self.experts_gate_weights.transpose(1, 2))
                up_out_padded = torch.bmm(padded_x, self.experts_up_weights.transpose(1, 2))
                act_out_padded = self.experts_act_fn(gate_out_padded) * up_out_padded
                down_out_padded = torch.bmm(act_out_padded, self.experts_down_weights.transpose(1, 2))
                down_out = down_out_padded[local_expert_indices, relative_idx]

        rev_local_sort_idx = torch.argsort(local_sort_idx)
        return down_out[rev_local_sort_idx].contiguous()

    def _apply_experts(
        self,
        flat_x: torch.Tensor,
        selected_experts: torch.Tensor,
        ep_group,
        ep_world_size: int,
    ) -> torch.Tensor:
        if ep_world_size > 1:
            target_ep_ranks = selected_experts // self.num_local_experts
            global_sort_idx = torch.argsort(target_ep_ranks)
            sorted_x = flat_x[global_sort_idx].contiguous()
            sorted_experts = selected_experts[global_sort_idx].contiguous()
            send_splits_tensor = torch.bincount(target_ep_ranks, minlength=ep_world_size)
            recv_splits_tensor = torch.empty_like(send_splits_tensor)
            dist.all_to_all_single(recv_splits_tensor, send_splits_tensor, group=ep_group)
            send_splits, recv_splits = torch.stack([send_splits_tensor, recv_splits_tensor]).cpu().tolist()
            received_x = ep_all_to_all(sorted_x, send_splits, recv_splits, ep_group)
            received_experts = torch.empty(sum(recv_splits), dtype=sorted_experts.dtype, device=flat_x.device)
            dist.all_to_all_single(
                received_experts,
                sorted_experts,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=ep_group,
            )
            out_x = self._apply_local_experts(received_x, received_experts)
            combined_x = ep_all_to_all(out_x, recv_splits, send_splits, ep_group)
            rev_global_sort_idx = torch.argsort(global_sort_idx)
            return combined_x[rev_global_sort_idx]

        return self._apply_local_experts(flat_x, selected_experts)

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
    
    def _route(
        self,
        x: torch.Tensor,
        gate_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.size()
        num_tokens = B * T

        if self.routing_strategy == "learned":
            assert gate_logits is not None
            weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
            weights = F.softmax(weights, dim=-1)
            return weights, selected_experts, gate_logits

        if self.routing_strategy == "random":
            random_scores = torch.rand(
                num_tokens,
                self.num_experts,
                device=x.device,
                dtype=torch.float32,
            )
            selected_experts = torch.topk(random_scores, self.top_k, dim=-1).indices.view(B, T, self.top_k)
            weights = x.new_full((B, T, self.top_k), 1.0 / self.top_k)
            return weights, selected_experts, None

        if self.routing_strategy == "round_robin":
            offsets = torch.arange(self.top_k, device=x.device, dtype=torch.long)
            token_ids = torch.arange(num_tokens, device=x.device, dtype=torch.long).unsqueeze(-1)
            selected_experts = (
                token_ids * self.top_k + offsets + self.layer_idx * self.top_k
            ) % self.num_experts
            selected_experts = selected_experts.view(B, T, self.top_k)
            weights = x.new_full((B, T, self.top_k), 1.0 / self.top_k)
            return weights, selected_experts, None

        raise ValueError(
            f"Unsupported routing_strategy={self.routing_strategy!r}. "
            "Expected one of: learned, random, round_robin."
        )

    def _record_routing_stats(
        self,
        gate_logits: torch.Tensor | None,
        selected_experts: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            flat_selected = selected_experts.reshape(-1)
            load = torch.bincount(flat_selected, minlength=self.num_experts).to(torch.float32)
            load = load / load.sum().clamp_min(1.0)
            stats = {
                "load_max": load.max().detach(),
                "load_min": load.min().detach(),
                "load_entropy": (-(load.clamp_min(1e-9) * load.clamp_min(1e-9).log()).sum()).detach(),
            }
            if gate_logits is not None:
                probs = F.softmax(gate_logits.reshape(-1, self.num_experts), dim=-1)
                entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
                stats["router_entropy"] = entropy.detach()
            self.last_routing_stats = stats

    def _compute_probe_ranking_loss(
        self,
        x: torch.Tensor,
        gate_logits: torch.Tensor | None,
        selected_experts: torch.Tensor,
        final_x: torch.Tensor,
        ep_group,
        ep_world_size: int,
    ) -> torch.Tensor | None:
        if (
            gate_logits is None
            or self.router_ranking_loss_weight <= 0.0
            or self.router_probe_experts <= 0
            or self.router_probe_tokens <= 0
            or not self.training
        ):
            return None

        B, T, D = x.size()
        num_tokens = B * T
        sample_count = min(int(self.router_probe_tokens), num_tokens)
        if sample_count <= 0:
            return None

        flat_x = x.reshape(num_tokens, D)
        flat_target = final_x.detach().reshape(num_tokens, D)
        flat_selected = selected_experts.reshape(num_tokens, self.top_k)
        flat_logits = gate_logits.reshape(num_tokens, self.num_experts)

        token_idx = torch.randperm(num_tokens, device=x.device)[:sample_count]
        selected_candidates = flat_selected[token_idx]
        probe_candidates = torch.randint(
            low=0,
            high=self.num_experts,
            size=(sample_count, int(self.router_probe_experts)),
            device=x.device,
        )
        candidates = torch.cat([selected_candidates, probe_candidates], dim=-1)
        num_candidates = candidates.size(-1)

        with torch.no_grad():
            candidate_x = (
                flat_x[token_idx]
                .unsqueeze(1)
                .expand(sample_count, num_candidates, D)
                .reshape(sample_count * num_candidates, D)
            )
            candidate_out = self._apply_experts(
                candidate_x,
                candidates.reshape(-1),
                ep_group=ep_group,
                ep_world_size=ep_world_size,
            ).view(sample_count, num_candidates, D)
            proxy_error = (candidate_out - flat_target[token_idx].unsqueeze(1)).pow(2).mean(dim=-1)
            best_idx = proxy_error.argmin(dim=-1)
            worst_idx = proxy_error.argmax(dim=-1)

        candidate_scores = flat_logits[token_idx].gather(1, candidates)
        best_scores = candidate_scores.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        worst_scores = candidate_scores.gather(1, worst_idx.unsqueeze(-1)).squeeze(-1)
        loss = -F.logsigmoid(best_scores - worst_scores).mean()
        return loss * self.router_ranking_loss_weight

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        gate_logits = self.router(x) if self.routing_strategy == "learned" else None
        weights, selected_experts, gate_logits_for_loss = self._route(x, gate_logits)
        self._record_routing_stats(gate_logits_for_loss, selected_experts)
        selected_experts_for_probe = selected_experts
        weights = weights.view(-1)                         # [B * T * top_k]
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
            self.last_router_probe_loss = None
            return final_x, gate_logits_for_loss, None

        if ep_world_size > 1:
            unpermuted_x = self._apply_experts(flat_x, selected_experts, ep_group, ep_world_size)
        else:
            unpermuted_x = self._apply_local_experts(flat_x, selected_experts)
        unpermuted_x = unpermuted_x * weights.unsqueeze(-1)
        final_x = unpermuted_x.view(B * T, self.top_k, D).sum(dim=1)

        final_x = final_x.reshape(B, T, D)
        probe_loss = self._compute_probe_ranking_loss(
            x=x,
            gate_logits=gate_logits_for_loss,
            selected_experts=selected_experts_for_probe,
            final_x=final_x,
            ep_group=ep_group,
            ep_world_size=ep_world_size,
        )
        self.last_router_probe_loss = None if probe_loss is None else probe_loss.detach()

        return final_x, gate_logits_for_loss, probe_loss
