from __future__ import annotations

import math

import torch

from tinytron.distributed import parallel_state
from tinytron.model.config import ModelConfig


class LayerPagedKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        self.current_length = 0
        num_pages = math.ceil(max_seq_len / page_size)
        shape = (batch_size, num_heads, num_pages, page_size, head_dim)
        self._k_pages = torch.empty(shape, device=device, dtype=dtype)
        self._v_pages = torch.empty(shape, device=device, dtype=dtype)
        self._k_flat = self._k_pages.view(batch_size, num_heads, num_pages * page_size, head_dim)
        self._v_flat = self._v_pages.view(batch_size, num_heads, num_pages * page_size, head_dim)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> int:
        append_len = k.size(2)
        end = self.current_length + append_len
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: requested sequence length {end}, max supported is {self.max_seq_len}"
            )
        self._k_flat[:, :, self.current_length:end, :].copy_(k)
        self._v_flat[:, :, self.current_length:end, :].copy_(v)
        previous_length = self.current_length
        self.current_length = end
        return previous_length

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._k_flat[:, :, : self.current_length, :],
            self._v_flat[:, :, : self.current_length, :],
        )


class PagedKVCache:
    def __init__(self, layer_caches: list[LayerPagedKVCache]) -> None:
        self._layer_caches = layer_caches

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        batch_size: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "PagedKVCache":
        try:
            sep_size = parallel_state.get_sep_world_size()
        except AssertionError:
            sep_size = 1
        if sep_size is None:
            sep_size = 1
        if sep_size > 1:
            if model_config.num_attention_heads % sep_size != 0:
                raise ValueError(
                    f"num_attention_heads ({model_config.num_attention_heads}) must be divisible by sep_size ({sep_size})"
                )
            cache_heads = model_config.num_attention_heads // sep_size
        else:
            cache_heads = model_config.num_attention_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        layer_caches = [
            LayerPagedKVCache(
                batch_size=batch_size,
                num_heads=cache_heads,
                head_dim=head_dim,
                max_seq_len=model_config.block_size,
                page_size=page_size,
                device=device,
                dtype=dtype,
            )
            for _ in range(model_config.num_layer)
        ]
        return cls(layer_caches)

    def get_layer_cache(self, layer_idx: int) -> LayerPagedKVCache:
        return self._layer_caches[layer_idx]
