import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tinytron.model.config import ModelConfig
from tinytron.distributed import (
    parallel_state,
    ulysses_all_to_all,
)


def rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HF Llama/Qwen-style RoPE:
    - rotate_half uses half-split layout
    - cos/sin are built from emb = cat(freqs, freqs)
    q, k: [B, H, T, D]
    position_ids: [T] or [B, T]
    """
    B, H, T, D = q.shape
    assert D % 2 == 0, f"head_dim must be even for RoPE, got {D}"

    # inv_freq: [D/2]
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32, device=q.device) / D)
    )

    # position_ids -> [B, T]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
    elif position_ids.dim() == 2:
        if position_ids.size(0) == 1 and B > 1:
            position_ids = position_ids.expand(B, -1)
        elif position_ids.size(0) != B:
            raise ValueError(f"position_ids batch mismatch: got {position_ids.shape}, expected batch={B}")
    else:
        raise ValueError(f"position_ids must be [T] or [B,T], got shape={position_ids.shape}")

    # freqs: [B, T, D/2]
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)

    # HF style: duplicate by concatenation (half-split compatible)
    # emb: [B, T, D]
    emb = torch.cat((freqs, freqs), dim=-1)

    # cos/sin: [B, 1, T, D] for broadcasting over heads
    cos = emb.cos().unsqueeze(1).to(dtype=q.dtype)
    sin = emb.sin().unsqueeze(1).to(dtype=q.dtype)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def gqa_impl(k: torch.Tensor, v: torch.Tensor, num_key_value_heads: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    k,v: (B, H_kv, T, D)
    return: (B, H_q, T, D)
    """
    if num_key_value_heads == num_attention_heads:
        return k, v
    elif num_key_value_heads == 1:
        k = k.expand(-1, num_attention_heads, -1, -1)
        v = v.expand(-1, num_attention_heads, -1, -1)
        return k, v
    elif num_attention_heads % num_key_value_heads == 0:
        repeat_factor = num_attention_heads // num_key_value_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        return k, v
    else:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


def build_cache_causal_mask(
    query_len: int,
    key_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a causal mask for incremental decoding with a KV cache."""
    query_positions = past_len + torch.arange(query_len, device=device).unsqueeze(-1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    return key_positions <= query_positions


def _is_paged_kv_cache(past_kv) -> bool:
    return hasattr(past_kv, "append") and hasattr(past_kv, "get_kv") and hasattr(past_kv, "current_length")


class Attention(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.device = torch.cuda.current_device()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.inference_shard_qkv = bool(config.inference_shard_qkv)
        self.n_embd = config.hidden_size
        self.dropout = config.dropout
        self.pos = None
        self.attn_fn = F.scaled_dot_product_attention
        self.q_proj_heads = self.num_attention_heads
        self.kv_proj_heads = self.num_key_value_heads
        if self.inference_shard_qkv:
            try:
                sp_size = parallel_state.get_sp_world_size()
            except AssertionError:
                sp_size = 1
            if sp_size is None:
                sp_size = 1
            if sp_size > 1:
                if self.num_attention_heads % sp_size != 0:
                    raise ValueError(
                        f"num_attention_heads ({self.num_attention_heads}) must be divisible by sp_size ({sp_size}) "
                        "for inference_shard_qkv."
                    )
                if self.num_key_value_heads % sp_size != 0:
                    raise ValueError(
                        f"num_key_value_heads ({self.num_key_value_heads}) must be divisible by sp_size ({sp_size}) "
                        "for inference_shard_qkv."
                    )
                self.q_proj_heads = self.num_attention_heads // sp_size
                self.kv_proj_heads = self.num_key_value_heads // sp_size
        # key, query, value projections for all heads, but in a batch        
        self.q_proj = nn.Linear(config.hidden_size, self.q_proj_heads * self.head_dim, bias=False, device=self.device)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_proj_heads * self.head_dim, bias=False, device=self.device)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_proj_heads * self.head_dim, bias=False, device=self.device)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, device=self.device)
        self._init_weights(config.seed, layer_idx)

    def _init_weights(self, base_seed: int, layer_idx: int):
        with torch.random.fork_rng(devices=[self.q_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.k_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.v_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.c_proj.weight.device]):
            torch.manual_seed(base_seed + layer_idx)
            torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=self.config.init_std)

    def _gather_heads_across_sp(
        self,
        y_local: torch.Tensor,
        sp_group,
        sp_size: int,
    ) -> torch.Tensor:
        gathered = [torch.empty_like(y_local) for _ in range(sp_size)]
        dist.all_gather(gathered, y_local.contiguous(), group=sp_group)
        return torch.cat(gathered, dim=1)

    def _get_sp_context(self) -> tuple[object | None, int, int]:
        if dist.is_available() and dist.is_initialized():
            try:
                return (
                    parallel_state.get_sp_group(),
                    parallel_state.get_sp_world_size(),
                    parallel_state.get_sp_rank(),
                )
            except AssertionError:
                pass
        return None, 1, 0

    def _concat_nonpaged_past(
        self,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        past_kv,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if past_kv is None or _is_paged_kv_cache(past_kv):
            return k_local, v_local

        k_past, v_past = past_kv
        k_local = torch.cat([k_past, k_local], dim=2)
        v_local = torch.cat([v_past, v_local], dim=2)
        return k_local, v_local

    def _prepare_sp_cache_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_kv,
        sp_size: int,
        sp_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        H = self.num_attention_heads

        if self.inference_shard_qkv:
            H_local = self.q_proj_heads
            H_kv_local = self.kv_proj_heads
            q_local = q
            k_local, v_local = gqa_impl(k, v, H_kv_local, H_local)
            k_local, v_local = self._concat_nonpaged_past(k_local, v_local, past_kv)
            return q_local, k_local, v_local, H_local

        assert H % sp_size == 0, f"Attention heads ({H}) must be divisible by sp_size ({sp_size})"
        H_local = H // sp_size
        head_slice = slice(sp_rank * H_local, (sp_rank + 1) * H_local)
        q_local = q[:, head_slice]

        if self.num_key_value_heads % sp_size == 0:
            H_kv_local = self.num_key_value_heads // sp_size
            kv_slice = slice(sp_rank * H_kv_local, (sp_rank + 1) * H_kv_local)
            k_local = k[:, kv_slice]
            v_local = v[:, kv_slice]
            k_local, v_local = gqa_impl(k_local, v_local, H_kv_local, H_local)
        else:
            k_full, v_full = gqa_impl(k, v, self.num_key_value_heads, H)
            k_local = k_full[:, head_slice]
            v_local = v_full[:, head_slice]

        k_local, v_local = self._concat_nonpaged_past(k_local, v_local, past_kv)
        return q_local, k_local, v_local, H_local

    def _prepare_sp_training_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group,
        sp_size: int,
        B: int,
        T_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        H = self.num_attention_heads
        assert H % sp_size == 0, f"Attention heads ({H}) must be divisible by sp_size ({sp_size})"
        H_local = H // sp_size
        T_full = sp_size * T_local

        if self.num_key_value_heads % sp_size == 0:
            H_kv_local = self.num_key_value_heads // sp_size

            q = q.view(B, sp_size, H_local, T_local, self.head_dim)
            q = q.permute(1, 0, 2, 3, 4).contiguous()
            q = ulysses_all_to_all(q, sp_group)
            q = q.permute(1, 2, 0, 3, 4).contiguous()
            q_local = q.view(B, H_local, T_full, self.head_dim)

            kv = torch.stack([k, v], dim=0)
            kv = kv.view(2, B, sp_size, H_kv_local, T_local, self.head_dim)
            kv = kv.permute(2, 0, 1, 3, 4, 5).contiguous()
            kv = ulysses_all_to_all(kv, sp_group)
            kv = kv.permute(1, 2, 3, 0, 4, 5).contiguous()
            kv = kv.view(2, B, H_kv_local, T_full, self.head_dim)
            k_local, v_local = kv[0], kv[1]
            k_local, v_local = gqa_impl(k_local, v_local, H_kv_local, H_local)
            return q_local, k_local, v_local, H_local

        k, v = gqa_impl(k, v, self.num_key_value_heads, H)
        qkv = torch.stack([q, k, v], dim=0)
        qkv = qkv.view(3, B, sp_size, H_local, T_local, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()
        qkv = ulysses_all_to_all(qkv, sp_group)
        qkv = qkv.permute(1, 2, 3, 0, 4, 5).contiguous()
        qkv = qkv.view(3, B, H_local, T_full, self.head_dim)
        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]
        return q_local, k_local, v_local, H_local

    def _prepare_local_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_kv,
        use_cache: bool,
        sp_group,
        sp_size: int,
        sp_rank: int,
        B: int,
        T_local: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        sp_cache_inference = use_cache and sp_size > 1

        if sp_cache_inference:
            q_local, k_local, v_local, H_local = self._prepare_sp_cache_qkv(
                q=q,
                k=k,
                v=v,
                past_kv=past_kv,
                sp_size=sp_size,
                sp_rank=sp_rank,
            )
            return q_local, k_local, v_local, H_local, True

        if sp_size > 1:
            q_local, k_local, v_local, H_local = self._prepare_sp_training_qkv(
                q=q,
                k=k,
                v=v,
                sp_group=sp_group,
                sp_size=sp_size,
                B=B,
                T_local=T_local,
            )
            return q_local, k_local, v_local, H_local, False

        H_local = self.num_attention_heads
        q_local = q
        k_local, v_local = gqa_impl(k, v, self.num_key_value_heads, H_local)
        k_local, v_local = self._concat_nonpaged_past(k_local, v_local, past_kv)
        return q_local, k_local, v_local, H_local, False

    def _prepare_attention_inputs(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        past_kv,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        attn_mask = None
        is_causal = True

        if _is_paged_kv_cache(past_kv):
            past_len = past_kv.append(k_local, v_local)
            k_local, v_local = past_kv.get_kv()
        elif past_kv is not None:
            past_len = past_kv[0].size(2)
        else:
            past_len = 0

        if past_len > 0:
            attn_mask = build_cache_causal_mask(
                query_len=q_local.size(2),
                key_len=k_local.size(2),
                past_len=past_len,
                device=q_local.device,
            )
            is_causal = False

        return k_local, v_local, attn_mask, is_causal

    def _restore_attention_output_layout(
        self,
        y: torch.Tensor,
        B: int,
        H: int,
        H_local: int,
        T_local: int,
        sp_group,
        sp_size: int,
        sp_cache_inference: bool,
    ) -> torch.Tensor:
        if sp_cache_inference:
            y = self._gather_heads_across_sp(y, sp_group, sp_size)
        elif sp_size > 1:
            y = y.view(B, H_local, sp_size, T_local, self.head_dim)
            y = y.permute(2, 0, 1, 3, 4).contiguous()
            y = ulysses_all_to_all(y, sp_group)
            y = y.permute(1, 0, 2, 3, 4).contiguous()
        return y.view(B, H, T_local, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        B, T_local, C = x.size()
        sp_group, sp_size, sp_rank = self._get_sp_context()
        sp_cache_inference = use_cache and sp_size > 1

        q = self.q_proj(x).view(B, T_local, self.q_proj_heads, self.head_dim).transpose(-2, -3)
        k = self.k_proj(x).view(B, T_local, self.kv_proj_heads, self.head_dim).transpose(-2, -3)
        v = self.v_proj(x).view(B, T_local, self.kv_proj_heads, self.head_dim).transpose(-2, -3)

        start_pos = position_offset if sp_cache_inference else sp_rank * T_local + position_offset
        end_pos = start_pos + T_local
        if self.pos is None or self.pos.size(1) != T_local or self.pos[0, 0] != start_pos:
            self.pos = torch.arange(start_pos, end_pos, device=x.device).unsqueeze(0)
        q, k = rope_impl(q, k, self.pos)

        H = self.num_attention_heads
        q_local, k_local, v_local, H_local, sp_cache_inference = self._prepare_local_qkv(
            q=q,
            k=k,
            v=v,
            past_kv=past_kv,
            use_cache=use_cache,
            sp_group=sp_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
            B=B,
            T_local=T_local,
        )

        dropout_p = self.dropout if self.training else 0.0
        k_local, v_local, attn_mask, is_causal = self._prepare_attention_inputs(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            past_kv=past_kv,
        )
        y = self.attn_fn(
            q_local, k_local, v_local,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

        y = self._restore_attention_output_layout(
            y=y,
            B=B,
            H=H,
            H_local=H_local,
            T_local=T_local,
            sp_group=sp_group,
            sp_size=sp_size,
            sp_cache_inference=sp_cache_inference,
        )
        y = y.transpose(-2, -3).reshape(B, T_local, C)
        y = self.c_proj(y)
        if use_cache:
            new_kv = past_kv if _is_paged_kv_cache(past_kv) else (k_local, v_local)
            return y, new_kv
        return y, None
