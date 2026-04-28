from __future__ import annotations

import torch
import torch.distributed as dist
from contextlib import nullcontext
import time

from tinytron.model import GPT
from tinytron.model.config import ModelConfig
from tinytron.distributed import parallel_state
from tinytron.bridge import (
    canonical_state_dict_to_local_inference,
    checkpoint_model_paths,
)
from .cache import PagedKVCache
from .sampler import sample_next_token


class InferenceEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_paged_kv_cache: bool = True,
        kv_cache_page_size: int = 128,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model_config = model_config
        self.use_paged_kv_cache = use_paged_kv_cache
        self.kv_cache_page_size = kv_cache_page_size
        self.model = GPT(model_config).to(self.device)
        if checkpoint_path:
            checkpoint_path, _ = checkpoint_model_paths(checkpoint_path)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = canonical_state_dict_to_local_inference(state_dict, model_config)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.dtype = dtype

    def _slice_qkv_state_dict_for_local_sep_rank(
        self,
        state_dict: dict[str, torch.Tensor],
        model_config: ModelConfig,
    ) -> dict[str, torch.Tensor]:
        try:
            sep_rank = parallel_state.get_sep_rank()
            sep_size = parallel_state.get_sep_world_size()
        except AssertionError:
            return state_dict

        if sep_size is None or sep_size <= 1:
            return state_dict

        head_dim = model_config.hidden_size // model_config.num_attention_heads
        q_total_out = model_config.num_attention_heads * head_dim
        kv_total_out = model_config.num_key_value_heads * head_dim
        q_out_per_rank = q_total_out // sep_size
        kv_out_per_rank = kv_total_out // sep_size

        sliced_state_dict = dict(state_dict)
        for layer_idx in range(model_config.num_layer):
            q_key = f"blocks.{layer_idx}.attn.q_proj.weight"
            k_key = f"blocks.{layer_idx}.attn.k_proj.weight"
            v_key = f"blocks.{layer_idx}.attn.v_proj.weight"

            for key, shard_width in ((q_key, q_out_per_rank), (k_key, kv_out_per_rank), (v_key, kv_out_per_rank)):
                weight = sliced_state_dict[key]
                if weight.size(0) == shard_width:
                    continue
                if weight.size(0) not in (q_total_out, kv_total_out):
                    raise ValueError(f"Unexpected checkpoint shape for {key}: {tuple(weight.shape)}")
                start = sep_rank * shard_width
                end = start + shard_width
                sliced_state_dict[key] = weight[start:end, :].contiguous()

        return sliced_state_dict

    def _sample_next_token_synced(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        try:
            sep_group = parallel_state.get_sep_group()
            sep_rank = parallel_state.get_sep_rank()
            sep_world_size = parallel_state.get_sep_world_size()
            sep_src = parallel_state.get_sep_global_rank(0)
        except AssertionError:
            return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        if sep_world_size <= 1:
            return sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        if sep_rank == 0:
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            next_token = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        dist.broadcast(next_token, src=sep_src, group=sep_group)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T], got shape={tuple(input_ids.shape)}")
        prompt = input_ids.to(self.device)
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + max_new_tokens
        if total_len > self.model_config.block_size:
            raise ValueError(
                f"Requested total sequence length {total_len} exceeds block_size {self.model_config.block_size}"
            )
        tokens = torch.empty(batch_size, total_len, dtype=prompt.dtype, device=self.device)
        tokens[:, :prompt_len] = prompt
        current_length = prompt_len
        past_key_values = (
            PagedKVCache.from_model_config(
                self.model_config,
                batch_size=batch_size,
                page_size=self.kv_cache_page_size,
                device=self.device,
                dtype=self.dtype,
            )
            if self.use_paged_kv_cache
            else None
        )

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            prefill_tokens = int(prompt.numel())
            prefill_t0 = time.perf_counter()
            logits, past_key_values = self.model(prompt, use_cache=True, past_key_values=past_key_values, position_offset=0)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prefill_t1 = time.perf_counter()
            decode_steps = 0
            decode_t0 = time.perf_counter()
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]
                next_token = self._sample_next_token_synced(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                tokens[:, current_length] = next_token
                current_length += 1
                decode_steps += 1

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break

                decode_input = next_token.unsqueeze(-1)
                logits, past_key_values = self.model(
                    decode_input,
                    use_cache=True,
                    past_key_values=past_key_values,
                    position_offset=current_length - 1,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            decode_t1 = time.perf_counter()

        tokens = tokens[:, :current_length]

        if not return_stats:
            return tokens

        prefill_time = max(prefill_t1 - prefill_t0, 1e-9)
        decode_time = max(decode_t1 - decode_t0, 1e-9)
        decode_tokens = int(tokens.size(0) * decode_steps)
        stats = {
            "prefill_tokens_per_sec": prefill_tokens / prefill_time,
            "decode_tokens_per_sec": decode_tokens / decode_time if decode_steps > 0 else 0.0,
            "prefill_tokens": float(prefill_tokens),
            "decode_tokens": float(decode_tokens),
        }
        return tokens, stats
