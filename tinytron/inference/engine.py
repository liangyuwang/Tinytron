from __future__ import annotations

import torch
from contextlib import nullcontext
import time

from tinytron.model import GPT
from tinytron.training.config import ModelConfig
from .sampler import sample_next_token


class InferenceEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = GPT(model_config).to(self.device)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
        self.model.eval()
        self.dtype = dtype

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
        tokens = input_ids.to(self.device)
        past_key_values = None

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.dtype)
            if self.device.type == "cuda" and self.dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            prefill_tokens = int(tokens.numel())
            prefill_t0 = time.perf_counter()
            logits, past_key_values = self.model(tokens, use_cache=True, past_key_values=None, position_offset=0)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            prefill_t1 = time.perf_counter()
            decode_steps = 0
            decode_t0 = time.perf_counter()
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1, :]
                next_token = sample_next_token(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)
                decode_steps += 1

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break

                decode_input = next_token.unsqueeze(-1)
                logits, past_key_values = self.model(
                    decode_input,
                    use_cache=True,
                    past_key_values=past_key_values,
                    position_offset=tokens.size(1) - 1,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            decode_t1 = time.perf_counter()

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
