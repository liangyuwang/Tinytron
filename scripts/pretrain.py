"""aimegatron pre-training entrypoint.

Usage (single process, mock data):
    python scripts/pretrain.py --model_size 0.03B

Usage (multi-GPU, tensor parallel + sequence parallel):
    torchrun --nproc_per_node=8 scripts/pretrain.py \
        --model_size 0.25B --tp_size 8 --sequence_parallel

See scripts/pretrain.sh for the env-var driven launcher.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aimegatron.core.config import Config, DataConfig, ModelConfig, ParallelConfig, TrainConfig  # noqa: E402
from aimegatron.train.trainer import Trainer  # noqa: E402

# Dense model presets. All dims are divisible by 8 so TP up to 8 always works
# (except noted kv-head sizes); vocab 50304 is divisible by 8.
MODEL_SIZES = {
    "0.03B": dict(num_layer=4, hidden_size=256, num_attention_heads=8,
                  num_key_value_heads=8, intermediate_size=1024),
    "0.03B-MoE": dict(num_layer=4, hidden_size=256, num_attention_heads=8,
                      num_key_value_heads=8, intermediate_size=1024,
                      num_experts=8, num_experts_per_tok=2, moe_every=1),
    "0.1B": dict(num_layer=8, hidden_size=768, num_attention_heads=16,
                 num_key_value_heads=8, intermediate_size=3072),
    "0.25B": dict(num_layer=12, hidden_size=1024, num_attention_heads=16,
                  num_key_value_heads=8, intermediate_size=4096),
    "1B": dict(num_layer=24, hidden_size=2048, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=8192),
    "7B": dict(num_layer=32, hidden_size=4096, num_attention_heads=32,
               num_key_value_heads=8, intermediate_size=14336),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="aimegatron pre-training")

    parser.add_argument("--model_size", type=str, default="0.25B", choices=sorted(MODEL_SIZES))
    # model overrides
    parser.add_argument("--num_layer", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--norm_type", type=str, default=None)
    parser.add_argument("--no_tied_lm_head", action="store_true")
    # MoE overrides
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--num_experts_per_tok", type=int, default=None)
    parser.add_argument("--moe_every", type=int, default=None)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=None)
    # parallelism
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--sequence_parallel", action="store_true")
    parser.add_argument("--backend", type=str, default="nccl")
    # training
    parser.add_argument("--total_batch_size", type=int, default=524288)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--no_distributed_optimizer", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--peak_flops", type=float, default=0.0)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=5000)
    parser.add_argument("--resume_path", type=str, default="")
    # data
    parser.add_argument("--mock_data_num_samples", type=int, default=1280)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    model_kwargs = dict(MODEL_SIZES[args.model_size])
    for key in ("num_layer", "hidden_size", "num_attention_heads", "num_key_value_heads",
                "intermediate_size", "vocab_size", "block_size", "dropout", "norm_type",
                "num_experts", "num_experts_per_tok", "moe_every", "moe_aux_loss_coeff"):
        value = getattr(args, key)
        if value is not None:
            model_kwargs[key] = value
    model = ModelConfig(tied_lm_head=not args.no_tied_lm_head, **model_kwargs)
    if model.block_size < args.seq_len:
        model.block_size = args.seq_len

    parallel = ParallelConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        ep_size=args.ep_size,
        sequence_parallel=args.sequence_parallel,
        backend=args.backend,
    )
    train = TrainConfig(
        total_batch_size=args.total_batch_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_value=args.grad_clip_value,
        optimizer=args.optimizer,
        use_distributed_optimizer=not args.no_distributed_optimizer,
        dtype=args.dtype,
        seed=args.seed,
        log_every_steps=args.log_every_steps,
        log_dir=args.log_dir,
        peak_flops=args.peak_flops,
        do_save=args.do_save,
        save_every_steps=args.save_every_steps,
        resume_path=args.resume_path,
    )
    data = DataConfig(use_mock_data=True, mock_data_num_samples=args.mock_data_num_samples)
    return Config(model=model, parallel=parallel, train=train, data=data)


def main():
    args = parse_args()
    config = build_config(args)
    Trainer(config).train()


if __name__ == "__main__":
    main()
