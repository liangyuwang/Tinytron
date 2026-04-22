from __future__ import annotations

import argparse
import torch

from tinytron.training.config import ModelConfig
from tinytron.inference import InferenceEngine


def parse_prompt_token_ids(prompt: str) -> list[int]:
    values = [p.strip() for p in prompt.split(",") if p.strip()]
    if not values:
        raise ValueError("--prompt_token_ids must contain at least one token id")
    return [int(v) for v in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("tinytron-inference", allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to *_model.pt checkpoint")
    parser.add_argument("--init_from_scratch", action="store_true", help="Run inference with random initialized weights")
    parser.add_argument("--prompt_token_ids", type=str, required=True, help="Comma-separated token ids, e.g. '1,2,3'")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--eos_token_id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])

    # model shape args (keep same names as ModelConfig)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--num_layer", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=128)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--intermediate_size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--init_std", type=float, default=0.013)
    parser.add_argument("--tied_lm_head", action=argparse.BooleanOptionalAction, default=True)

    # MoE
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_experts_per_tok", type=int, default=8)
    parser.add_argument("--moe_intermediate_size", type=int, default=256)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.init_from_scratch and not args.checkpoint_path:
        parser.error("Either provide --checkpoint_path, or set --init_from_scratch for random-weight smoke testing.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_config = ModelConfig(
        seed=args.seed,
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        num_layer=args.num_layer,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        init_std=args.init_std,
        tied_lm_head=args.tied_lm_head,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
    )

    prompt = torch.tensor([parse_prompt_token_ids(args.prompt_token_ids)], dtype=torch.long)
    engine = InferenceEngine(
        model_config=model_config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=dtype,
    )
    output, stats = engine.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=args.eos_token_id,
        return_stats=True,
    )
    print(",".join(str(int(x)) for x in output[0].tolist()))
    print(f"prefill tok/s: {stats['prefill_tokens_per_sec']:.2f}")
    print(f"decode tok/s: {stats['decode_tokens_per_sec']:.2f}")


if __name__ == "__main__":
    main()
