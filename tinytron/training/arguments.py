from __future__ import annotations

import argparse


def add_all_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = _add_logging_args(parser)
    parser = _add_seed_args(parser)
    parser = _add_data_args(parser)
    parser = _add_training_args(parser)
    parser = _add_optim_args(parser)
    parser = _add_checkpoint_args(parser)
    parser = _add_parallel_args(parser)
    parser = _add_model_args(parser)
    return parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("pretrain", allow_abbrev=False)
    parser = add_all_arguments(parser)
    return parser

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args



def _add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("training")
    # tokens-based global batch size (like repo)
    g.add_argument("--total_batch_size", type=int, default=524288, help="global batch size in tokens")
    g.add_argument("--batch_size", type=int, default=8, help="micro batch size per device")
    g.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    g.add_argument("--max_steps", type=int, default=None)
    g.add_argument("--max_epochs", type=int, default=1)

    g.add_argument("--do_val", action="store_true")
    g.add_argument("--val_every_steps", type=int, default=250)
    g.add_argument("--split_rate", type=float, default=None, help="train split ratio; default depends on do_val")

    g.add_argument("--grad_clip_value", type=float, default=1.0)
    g.add_argument("--debug", action="store_true")

    # precision / compile / profiler
    g.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    g.add_argument("--disable_tf32", action="store_true", help="disable TF32 matmul/cuDNN kernels for stricter numerical consistency")
    g.add_argument("--use_compile", action="store_true")
    g.add_argument("--use_profiler", action="store_true")
    g.add_argument("--steps_to_profile", type=int, nargs="+", default=[15, 20])
    return parser


def _add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("logging")
    g.add_argument("--exp_name", type=str, default="gpt")
    g.add_argument("--log_dir", type=str, default="./log/")
    g.add_argument("--log_every", type=int, default=10)
    g.add_argument("--print_config", action="store_true")
    return parser


def _add_seed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("reproducibility")
    g.add_argument("--seed", type=int, default=1337)
    g.add_argument("--deterministic", action="store_true")
    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("data")
    g.add_argument("--dataset_path", type=str, default="../data/fineweb-edu-sample-10BT/")
    g.add_argument("--use_mock_data", action="store_true")
    g.add_argument("--mock_data_num_samples", type=int, default=1280)
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--pin_memory", action="store_true")
    return parser

def _add_optim_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("optimizer")
    g.add_argument("--optimizer", type=str, default="adam")

    g.add_argument("--max_lr", type=float, default=4e-3)
    g.add_argument("--min_lr", type=float, default=3e-5)
    g.add_argument("--warmup_steps", type=int, default=1000)
    g.add_argument("--weight_decay", type=float, default=0.1)

    # (optional) AdamW knobs
    g.add_argument("--adam_beta1", type=float, default=0.9)
    g.add_argument("--adam_beta2", type=float, default=0.95)
    g.add_argument("--adam_eps", type=float, default=1e-8)

    # Muon
    g.add_argument("--muon_momentum", type=float, default=0.95)
    return parser


def _add_checkpoint_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("checkpoint")
    g.add_argument("--do_save", action="store_true")
    g.add_argument("--save_every_steps", type=int, default=5000)
    g.add_argument("--resume_path", type=str, default=None, help="path to a checkpoint directory")
    return parser


def _add_parallel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("distributed")
    g.add_argument("--backend", type=str, default="nccl")
    g.add_argument("--init_method", type=str, default="env://")
    g.add_argument("--sep_size", type=int, default=8, help="SEP size (SP/EP joint parallel group size)")
    g.add_argument("--ddp_find_unused_parameters", action="store_true")
    g.add_argument("--ddp_gradient_as_bucket_view", action="store_true")
    g.add_argument("--use_distributed_optimizer", action="store_true", help="enable ZeRO-1 style optimizer sharding (DistributedOptimizer)")
    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("model")
    # Map to model.GPTConfig fields
    g.add_argument("--block_size", type=int, default=4096)
    g.add_argument("--vocab_size", type=int, default=50304)
    g.add_argument("--num_layer", type=int, default=32)
    g.add_argument("--num_attention_heads", type=int, default=128)
    g.add_argument("--num_key_value_heads", type=int, default=8)
    g.add_argument("--hidden_size", type=int, default=1024)
    g.add_argument("--intermediate_size", type=int, default=4096)
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--init_std", type=float, default=0.013)
    g.add_argument("--tied_lm_head", action="store_true")

    # MoE
    g.add_argument("--use_moe", action="store_true")
    g.add_argument("--num_experts", type=int, default=128)
    g.add_argument("--num_experts_per_tok", type=int, default=8)
    g.add_argument("--moe_intermediate_size", type=int, default=256)
    g.add_argument("--moe_router_debug", action="store_true", help="log MoE router top-k boundary statistics")
    g.add_argument("--moe_route_trace", action="store_true", help="dump per-token MoE top-k routes for flip-rate comparison")
    return parser
