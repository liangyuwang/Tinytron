from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import argparse

@dataclass(frozen=True)
class Config:
    logging: LoggingConfig
    seed: SeedConfig
    data: DataConfig
    train: TrainingConfig
    optim: OptimConfig
    ckpt: CheckpointConfig
    parallel: ParallelConfig
    model: ModelConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "logging": asdict(self.logging),
            "seed": asdict(self.seed),
            "data": asdict(self.data),
            "train": asdict(self.train),
            "optim": asdict(self.optim),
            "ckpt": asdict(self.ckpt),
            "parallel": asdict(self.parallel),
            "model": asdict(self.model),
        }


@dataclass(frozen=True)
class TrainingConfig:
    total_batch_size: int = 524288  # tokens
    batch_size: int = 8  # micro batch per device
    seq_len: int = 4096  # sequence length

    max_steps: int | None = None
    max_epochs: int | None = 1

    do_val: bool = False
    val_every_steps: int = 250
    split_rate: float | None = None  # if None, derive from do_val

    grad_clip_value: float = 1.0
    debug: bool = False

    precision: str = "bf16"  # bf16/fp16/fp32
    disable_tf32: bool = False
    use_compile: bool = False
    use_profiler: bool = False
    steps_to_profile: list[int] = field(default_factory=lambda: [15, 20])

    def with_derived(self) -> "TrainingConfig":
        # derive split_rate default
        if self.split_rate is None:
            split = 0.99 if self.do_val else 1.0
            return TrainingConfig(**{**self.__dict__, "split_rate": split})
        return self

@dataclass(frozen=True)
class LoggingConfig:
    exp_name: str = "gpt"
    log_dir: str = "./log/"
    log_every: int = 10
    print_config: bool = False

@dataclass(frozen=True)
class SeedConfig:
    seed: int = 1337
    deterministic: bool = False

@dataclass(frozen=True)
class DataConfig:
    dataset_path: str = ""
    use_mock_data: bool = False
    mock_data_num_samples: int = 1280
    num_workers: int = 0
    pin_memory: bool = True

@dataclass(frozen=True)
class OptimConfig:
    optimizer: str = "adam"

    max_lr: float = 4e-3
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    muon_momentum: float = 0.95

@dataclass(frozen=True)
class CheckpointConfig:
    do_save: bool = False
    save_every_steps: int = 5000
    resume_path: str | None = None

@dataclass(frozen=True)
class ParallelConfig:
    backend: str = "nccl"
    init_method: str = "env://"
    sep_size: int = 1

    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    use_distributed_optimizer: bool = True

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
    moe_router_debug: bool = False
    moe_route_trace: bool = False


def build_config(args: argparse.Namespace) -> Config:
    logging_cfg = LoggingConfig(
        exp_name=args.exp_name,
        log_dir=args.log_dir,
        log_every=args.log_every,
        print_config=getattr(args, "print_config", False),
    )
    seed_cfg = SeedConfig(seed=args.seed, deterministic=args.deterministic)
    data_cfg = DataConfig(
        dataset_path=args.dataset_path,
        use_mock_data=args.use_mock_data,
        mock_data_num_samples=args.mock_data_num_samples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    train_cfg = TrainingConfig(
        total_batch_size=args.total_batch_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        do_val=args.do_val,
        val_every_steps=args.val_every_steps,
        split_rate=args.split_rate,
        grad_clip_value=args.grad_clip_value,
        debug=args.debug,
        precision=args.precision,
        disable_tf32=args.disable_tf32,
        use_compile=args.use_compile,
        use_profiler=args.use_profiler,
        steps_to_profile=args.steps_to_profile,
    ).with_derived()

    optim_cfg = OptimConfig(
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )

    ckpt_cfg = CheckpointConfig(
        do_save=args.do_save,
        save_every_steps=args.save_every_steps,
        resume_path=args.resume_path,
    )

    parallel_cfg = ParallelConfig(
        backend=args.backend,
        init_method=args.init_method,
        sep_size=args.sep_size,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        use_distributed_optimizer=args.use_distributed_optimizer,
    )

    model_cfg = ModelConfig(
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
        moe_router_debug=args.moe_router_debug,
        moe_route_trace=args.moe_route_trace,
    )

    cfg = Config(
        logging=logging_cfg,
        seed=seed_cfg,
        data=data_cfg,
        train=train_cfg,
        optim=optim_cfg,
        ckpt=ckpt_cfg,
        parallel=parallel_cfg,
        model=model_cfg,
    )
    validate_static(cfg)
    return cfg

def validate_static(cfg: Config) -> None:
    # Basic sanity checks; runtime checks (world_size divisibility etc.) in Trainer after dist init.
    if cfg.train.seq_len <= 0:
        raise ValueError("T must be positive.")
    if cfg.train.batch_size <= 0:
        raise ValueError("micro batch size B must be positive.")
    if cfg.train.total_batch_size <= 0:
        raise ValueError("total_batch_size must be positive (tokens).")
    if cfg.parallel.sep_size <= 0:
        raise ValueError("sep_size must be positive.")
    if cfg.train.precision not in ("bf16", "fp16", "fp32"):
        raise ValueError(f"Unsupported precision: {cfg.train.precision}")
    if cfg.train.max_steps is not None and cfg.optim.warmup_steps >= cfg.train.max_steps:
        raise ValueError("warmup_steps must be < max_steps when max_steps is set.")
    # Model shape constraints (minimal)
    if cfg.model.hidden_size % cfg.model.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    if cfg.model.num_attention_heads % cfg.model.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads (GQA).")
    if cfg.model.use_moe and cfg.model.num_experts_per_tok <= 0:
        raise ValueError("num_experts_per_tok must be positive when use_moe is enabled.")
