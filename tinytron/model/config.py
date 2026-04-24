from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 1337
    block_size: int = 4096
    vocab_size: int = 50304
    num_layer: int = 32
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 4096
    dropout: float = 0.0
    init_std: float = 0.013
    tied_lm_head: bool = True

    # MoE
    use_moe: bool = False
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 256

    # Inference-only optimization: shard q/k/v projections by SEP head shard.
    inference_shard_qkv: bool = False


def add_model_config_args(parser: argparse._ArgumentGroup | argparse.ArgumentParser) -> argparse._ArgumentGroup | argparse.ArgumentParser:
    defaults = ModelConfig()
    parser.add_argument("--block_size", type=int, default=defaults.block_size)
    parser.add_argument("--vocab_size", type=int, default=defaults.vocab_size)
    parser.add_argument("--num_layer", type=int, default=defaults.num_layer)
    parser.add_argument("--num_attention_heads", type=int, default=defaults.num_attention_heads)
    parser.add_argument("--num_key_value_heads", type=int, default=defaults.num_key_value_heads)
    parser.add_argument("--hidden_size", type=int, default=defaults.hidden_size)
    parser.add_argument("--intermediate_size", type=int, default=defaults.intermediate_size)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--init_std", type=float, default=defaults.init_std)
    parser.add_argument("--tied_lm_head", action=argparse.BooleanOptionalAction, default=defaults.tied_lm_head)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--num_experts", type=int, default=defaults.num_experts)
    parser.add_argument("--num_experts_per_tok", type=int, default=defaults.num_experts_per_tok)
    parser.add_argument("--moe_intermediate_size", type=int, default=defaults.moe_intermediate_size)
    return parser


def build_model_config(args: argparse.Namespace, **overrides) -> ModelConfig:
    values = dict(
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
    values.update(overrides)
    return ModelConfig(**values)
