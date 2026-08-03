"""aimegatron.core.config

Single home for every experiment knob. All subsystems read from these
dataclasses; no subsystem defines its own flags.

Edit contract: adding a knob means exactly three edits --
(1) one field here, (2) one CLI argument in scripts/pretrain.py,
(3) one use site. All cross-field validation lives in Config.validate so
experiments fail fast with an actionable message.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    block_size: int = 4096          # maximum sequence length the model supports
    vocab_size: int = 50304
    num_layer: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 8    # GQA; equal to num_attention_heads for MHA
    hidden_size: int = 1024
    intermediate_size: int = 4096   # SwiGLU MLP intermediate size
    dropout: float = 0.0
    tied_lm_head: bool = True
    init_std: float = 0.02
    rope_theta: float = 10000.0
    norm_type: str = "layernorm"    # layernorm | rmsnorm (see aimegatron.model.norm)
    # MoE knobs; num_experts == 0 keeps the model dense.
    num_experts: int = 0            # experts per MoE layer
    num_experts_per_tok: int = 2    # top-k routing
    moe_every: int = 0              # every Nth layer is MoE; 0 = dense model
    moe_aux_loss_coeff: float = 0.01  # load-balance aux loss weight


@dataclass
class ParallelConfig:
    tp_size: int = 1                # tensor parallel size
    pp_size: int = 1                # pipeline parallel size; world = pp * dp * tp
    ep_size: int = 1                # expert parallel size; must divide dp_size
    sequence_parallel: bool = False # Megatron-style SP, rides on the TP group
    backend: str = "nccl"           # nccl for GPU, gloo for CPU/tests
    init_method: str = "env://"


@dataclass
class TrainConfig:
    total_batch_size: int = 524288  # global batch in tokens
    batch_size: int = 8             # micro batch per DP rank
    seq_len: int = 4096
    max_steps: int = 1000
    max_lr: float = 4e-3
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip_value: float = 1.0
    optimizer: str = "adam"         # registry key in aimegatron.core.registry.OPTIMIZERS
    use_distributed_optimizer: bool = True  # ZeRO-1 optimizer-state sharding
    dtype: str = "bf16"             # bf16 | fp32
    seed: int = 1337
    deterministic: bool = False
    log_every_steps: int = 10
    do_save: bool = False
    save_every_steps: int = 5000
    resume_path: str = ""           # explicit checkpoint dir; empty = auto-detect in log_dir
    log_dir: str = "./log"
    peak_flops: float = 0.0         # device peak TFLOPs (dense); 0 disables MFU logging


@dataclass
class DataConfig:
    use_mock_data: bool = True
    mock_data_num_samples: int = 1280


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self, world_size: int) -> None:
        m, p, t = self.model, self.parallel, self.train
        assert world_size % (p.tp_size * p.pp_size) == 0, \
            f"world_size ({world_size}) must be divisible by tp_size * pp_size"
        dp_size = world_size // (p.tp_size * p.pp_size)
        checks = [
            (not p.sequence_parallel or p.tp_size > 1,
             "sequence_parallel requires tp_size > 1"),
            (m.hidden_size % m.num_attention_heads == 0,
             f"hidden_size ({m.hidden_size}) must be divisible by num_attention_heads ({m.num_attention_heads})"),
            (m.num_attention_heads % p.tp_size == 0,
             f"num_attention_heads ({m.num_attention_heads}) must be divisible by tp_size ({p.tp_size})"),
            (m.num_key_value_heads % p.tp_size == 0,
             f"num_key_value_heads ({m.num_key_value_heads}) must be divisible by tp_size ({p.tp_size}); "
             "v1 does not replicate KV heads across TP"),
            (m.intermediate_size % p.tp_size == 0,
             f"intermediate_size ({m.intermediate_size}) must be divisible by tp_size ({p.tp_size})"),
            (m.vocab_size % p.tp_size == 0,
             f"vocab_size ({m.vocab_size}) must be divisible by tp_size ({p.tp_size})"),
            (m.norm_type in ("layernorm", "rmsnorm"),
             f"norm_type must be layernorm or rmsnorm, got {m.norm_type}"),
            (t.seq_len <= m.block_size,
             f"seq_len ({t.seq_len}) exceeds model block_size ({m.block_size})"),
            (t.dtype in ("bf16", "fp32"),
             f"train.dtype must be bf16 or fp32, got {t.dtype}"),
            # Pipeline parallelism.
            (m.num_layer % p.pp_size == 0,
             f"num_layer ({m.num_layer}) must be divisible by pp_size ({p.pp_size})"),
            (not m.tied_lm_head or p.pp_size == 1,
             "tied_lm_head requires pp_size == 1 (embedding and lm_head live on different stages)"),
            # Expert parallelism (lives inside the DP dimension).
            (dp_size % p.ep_size == 0,
             f"dp_size ({dp_size}) must be divisible by ep_size ({p.ep_size})"),
            (p.ep_size == 1 or m.num_experts > 0,
             "ep_size > 1 requires num_experts > 0"),
            (m.num_experts % p.ep_size == 0,
             f"num_experts ({m.num_experts}) must be divisible by ep_size ({p.ep_size})"),
            (m.num_experts == 0 or 1 <= m.num_experts_per_tok <= m.num_experts,
             f"num_experts_per_tok ({m.num_experts_per_tok}) must be in [1, num_experts]"),
        ]
        tokens_per_micro_step = t.batch_size * t.seq_len * dp_size
        checks.append(
            (tokens_per_micro_step > 0 and t.total_batch_size % tokens_per_micro_step == 0,
             f"total_batch_size ({t.total_batch_size}) must be divisible by "
             f"batch_size * seq_len * dp_size = {tokens_per_micro_step}")
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)
