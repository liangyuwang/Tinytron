from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tinytron.rl import GRPOTrainer, RLConfig
from tinytron.training import build_config, build_parser


def add_rl_args(parser):
    g = parser.add_argument_group("rl")
    g.add_argument("--rl_max_new_tokens", type=int, default=8)
    g.add_argument("--rl_group_size", type=int, default=2)
    g.add_argument("--rl_temperature", type=float, default=1.0)
    g.add_argument("--rl_top_k", type=int, default=0, help="0 disables top-k filtering")
    g.add_argument("--rl_top_p", type=float, default=0.0, help="0 disables top-p filtering")
    g.add_argument("--rl_eos_token_id", type=int, default=None)
    g.add_argument("--rl_reward_target_token_id", type=int, default=0, help="mock reward favors tokens closer to this id")
    g.add_argument("--rl_clip_range", type=float, default=0.2)
    g.add_argument("--rl_kl_coef", type=float, default=0.0)
    g.add_argument("--rl_rollout_shard_qkv", action="store_true")
    return parser


def build_rl_config(args) -> RLConfig:
    return RLConfig(
        max_new_tokens=args.rl_max_new_tokens,
        group_size=args.rl_group_size,
        temperature=args.rl_temperature,
        top_k=args.rl_top_k if args.rl_top_k > 0 else None,
        top_p=args.rl_top_p if 0.0 < args.rl_top_p < 1.0 else None,
        eos_token_id=args.rl_eos_token_id,
        reward_target_token_id=args.rl_reward_target_token_id,
        clip_range=args.rl_clip_range,
        kl_coef=args.rl_kl_coef,
        rollout_shard_qkv=args.rl_rollout_shard_qkv,
    )


def main():
    parser = add_rl_args(build_parser())
    args = parser.parse_args()
    cfg = build_config(args)
    trainer = GRPOTrainer(cfg, build_rl_config(args))
    trainer.train()


if __name__ == "__main__":
    main()
