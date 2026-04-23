from __future__ import annotations

import argparse


def parse_prompt_token_ids(prompt: str) -> list[int]:
    values = [p.strip() for p in prompt.split(",") if p.strip()]
    if not values:
        raise ValueError("--prompt_token_ids must contain at least one token id")
    return [int(v) for v in values]


def add_all_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = _add_inference_args(parser)
    parser = _add_sampling_args(parser)
    parser = _add_parallel_args(parser)
    parser = _add_model_args(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("tinytron-inference", allow_abbrev=False)
    parser = add_all_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args


def _add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("inference")
    g.add_argument("--checkpoint_path", type=str, default=None, help="Path to *_model.pt checkpoint")
    g.add_argument("--init_from_scratch", action="store_true", help="Run inference with random initialized weights")
    g.add_argument("--prompt_token_ids", type=str, required=True, help="Comma-separated token ids, e.g. '1,2,3'")
    g.add_argument("--max_new_tokens", type=int, default=64)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return parser


def _add_sampling_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("sampling")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top_k", type=int, default=None)
    g.add_argument("--top_p", type=float, default=None)
    g.add_argument("--eos_token_id", type=int, default=None)
    return parser


def _add_parallel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("distributed")
    g.add_argument("--backend", type=str, default="nccl")
    g.add_argument("--init_method", type=str, default="env://")
    g.add_argument("--sep_size", type=int, default=1, help="SEP size for distributed KV-cache inference")
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("model")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--block_size", type=int, default=4096)
    g.add_argument("--vocab_size", type=int, default=50304)
    g.add_argument("--num_layer", type=int, default=32)
    g.add_argument("--num_attention_heads", type=int, default=128)
    g.add_argument("--num_key_value_heads", type=int, default=8)
    g.add_argument("--hidden_size", type=int, default=1024)
    g.add_argument("--intermediate_size", type=int, default=4096)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--init_std", type=float, default=0.013)
    g.add_argument("--tied_lm_head", action=argparse.BooleanOptionalAction, default=True)

    g.add_argument("--use_moe", action="store_true")
    g.add_argument("--num_experts", type=int, default=128)
    g.add_argument("--num_experts_per_tok", type=int, default=8)
    g.add_argument("--moe_intermediate_size", type=int, default=256)
    return parser
