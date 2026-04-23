import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from tinytron.training.config import ModelConfig
from .modules.attn import Attention
from .modules.mlp import MLP, MoE
from .modules.norm import LayerNorm
from .modules.loss import CrossEntropyLoss, ExpertLoadBalancingLoss
from tinytron.distributed import parallel_state

EXPERT_LOCAL_PARAM_SUFFIXES = (
    "mlp.experts_gate_weights",
    "mlp.experts_up_weights",
    "mlp.experts_down_weights",
)

class Block(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int, use_moe: bool = True, top_k: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_moe = use_moe
        self.device = torch.cuda.current_device()
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config, layer_idx)
        self.ln_2 = LayerNorm(config)
        self.mlp = MoE(config, layer_idx, top_k) if use_moe else MLP(config, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        attn_out, new_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache, position_offset=position_offset)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x), use_cache=use_cache)
        x = x + mlp_out[0] if self.use_moe else x + mlp_out
        gate_logits = mlp_out[1] if self.use_moe else None
        return x, gate_logits, new_kv

# GPT-like Model

class GPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.config = config
        self.pos = None
        self.use_moe = config.use_moe
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, device=self.device)
        self.blocks = nn.ModuleList([Block(config, layer_idx, self.use_moe) for layer_idx in range(config.num_layer)])
        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=self.device)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.loss_fn = CrossEntropyLoss()
        self.expert_loss_fn = ExpertLoadBalancingLoss(config.num_experts, config.num_experts_per_tok)
        self._init_weights(config.seed)

    def _init_weights(self, base_seed: int):
        with torch.random.fork_rng(devices=[self.wte.weight.device]):
            torch.manual_seed(base_seed)
            torch.nn.init.normal_(self.wte.weight, mean=0.0, std=self.config.init_std)
        if not self.config.tied_lm_head:
            with torch.random.fork_rng(devices=[self.lm_head.weight.device]):
                torch.manual_seed(base_seed)
                torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        total_aux_loss = 0.0
        new_past_key_values = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            x, gate_logits, new_kv = block(x, past_kv=layer_past, use_cache=use_cache, position_offset=position_offset)
            if gate_logits is not None and targets is not None:
                total_aux_loss += self.expert_loss_fn(gate_logits)
            if use_cache:
                new_past_key_values.append(new_kv)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            loss, logging_loss = self.loss_fn(logits, targets)
            return logits, loss + total_aux_loss, logging_loss
        if use_cache:
            return logits, new_past_key_values
        return logits
    
    def get_flops_per_fwd_bwd(self, batch_size, seq_len):
        """
        Approximate model FLOPs for forward and backward pass.
        """
        sp_size = parallel_state.get_sp_world_size()
        ep_size = parallel_state.get_ep_world_size()
        # Q, K, V projection (3D->D) + Out projection (D->D)
        qkv_out_flops = 8 * batch_size * seq_len * self.config.hidden_size * self.config.hidden_size  # 6BLD^2 + 2BLD^2
        # QK^T + softmaxV
        attn_matmul_flops = 4 * batch_size * seq_len * seq_len * self.config.hidden_size
        attn_flops = qkv_out_flops + attn_matmul_flops
        # FFN: SwiGLU = gate_proj(D->D_int) + up_proj(D->D_int) + elementwise + down_proj(D_int->D)
        intermediate_size = self.config.moe_intermediate_size if self.use_moe else self.config.intermediate_size
        ffn_flops = 2 * batch_size * seq_len * self.config.hidden_size * intermediate_size \
                + 2 * batch_size * seq_len * self.config.hidden_size * intermediate_size \
                + batch_size * seq_len * intermediate_size \
                + 2 * batch_size * seq_len * intermediate_size * self.config.hidden_size
        expert_gate_flops = 2 * batch_size * seq_len * self.config.hidden_size * self.config.num_experts  # gate_proj(D->E)
        if self.use_moe:
            per_layer_flops = (attn_flops + expert_gate_flops) / sp_size \
                            + (self.config.num_experts_per_tok * ffn_flops) / ep_size    # FLOPs per layer
        else:
            per_layer_flops = (attn_flops + ffn_flops) / sp_size    # total FLOPs
        total_flops = 3 * self.config.num_layer * per_layer_flops  # fwd + bwd ≈ 3 × fwd FLOPs
        return total_flops
    
    def clip_grad_norm(self, max_norm: float, norm_type: float = 2):
        normal_sq_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        expert_sq_sum = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            param_norm_sq = torch.linalg.vector_norm(p.grad, norm_type).to(torch.float32) ** norm_type
            if name.endswith(EXPERT_LOCAL_PARAM_SUFFIXES):
                expert_sq_sum += param_norm_sq
            else:
                normal_sq_sum += param_norm_sq
        dist.all_reduce(expert_sq_sum, op=dist.ReduceOp.SUM, group=parallel_state.get_ep_group())
        total_sq_sum = normal_sq_sum + expert_sq_sum
        total_norm = total_sq_sum ** (1.0 / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in self.parameters():
            p.grad.detach().mul_(clip_coef_clamped)
        return total_norm
