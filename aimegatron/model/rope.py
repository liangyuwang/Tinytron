"""aimegatron.model.rope

Rotary position embeddings, HF Llama/Qwen-style (half-split rotate_half and
cat(freqs, freqs) layout). RoPE is head-local: it operates on [B, H, T, D]
and therefore works unchanged on a TP head shard.

Edit contract: keep this module free of any distributed calls.
"""

import torch


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE in place of positional embeddings.

    q, k: [B, H, T, D]; position_ids: [T] or [B, T].
    """
    B, _, T, D = q.shape
    assert D % 2 == 0, f"head_dim must be even for RoPE, got {D}"

    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32, device=q.device) / D))

    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
    elif position_ids.size(0) == 1 and B > 1:
        position_ids = position_ids.expand(B, -1)

    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)  # [B, T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)                            # [B, T, D]
    cos = emb.cos().unsqueeze(1).to(dtype=q.dtype)                     # [B, 1, T, D]
    sin = emb.sin().unsqueeze(1).to(dtype=q.dtype)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
