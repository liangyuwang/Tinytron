# Tinytron

Tinytron is a compact, research-oriented pre-training and inference stack for GPT-style language models. It is built for researchers who want a codebase that can be read, modified, and instrumented quickly without fighting a large framework.

The design goal is simple: each subsystem should be small enough to hack directly. Attention, dense MLPs, MoE routing, loss computation, inference-time KV cache, optimizer variants, training state, and launch scripts are split into independent modules with explicit boundaries. That makes Tinytron a useful base for model architecture experiments, optimizer studies, KV-cache experiments, inference-path prototyping, and data-pipeline swaps.

## Research Scope

Tinytron is best viewed as a transparent GPT experimentation base rather than a general-purpose distributed-systems framework. The distributed layer is intentionally narrow: it provides DDP, sequence-expert parallel groups, expert all-to-all communication, and ZeRO-1 style optimizer sharding so that model and inference experiments can run beyond a single GPU. It is not designed to expose a large search space of parallelism policies.

Good research fits include:

- **Model architecture**: GQA/MQA/MHA variants, RoPE and long-context changes, dense MLP versus MoE, router design, expert layout, normalization, and loss variants.
- **Inference strategy**: prefill/decode behavior, sampling methods, MoE inference paths, sharded QKV inference, and lightweight decoding prototypes.
- **KV-cache design**: paged versus contiguous cache, page size, cache layout, prefix reuse, sliding-window cache, and cache quantization experiments.
- **Optimizer studies**: AdamW versus Muon, parameter-group policies, router/expert-specific learning rates, weight decay choices, and gradient clipping behavior.
- **Training and measurement**: small-scale scaling studies, architecture ablations, MFU tracking, profiler-driven bottleneck analysis, and throughput sweeps for a fixed training stack.

## Features

- **Hackable model components**:
  - Grouped Query Attention (GQA)
  - Mixture of Experts (MoE)
  - Separate attention, MLP/MoE, normalization, embedding, and loss modules
  - Shared training and inference model path
  
- **Distributed support for larger experiments**:
  - DistributedDataParallel (DDP) for multi-GPU training
  - Sequence-Expert joint parallelism via `SEP_SIZE` / `--sep_size` (SEP)
  - Expert parallel all-to-all communication
  - ZeRO-1 optimizer state partitioning for memory efficiency
  - Sharded model checkpoints with file-based model resharding across SEP layouts
  - Bridge utilities for parameter-layout conversion between training and inference
  - Native support for Muon + ZeRO-1
  - Gradient accumulation for large effective batch sizes
  
- **Training and measurement tools**:
  - Mixed precision training (BFloat16)
  - Gradient clipping
  - Cosine learning rate schedule with warmup
  - Automatic checkpoint resumption with full state recovery
  - Model FLOPs Utilization (MFU) tracking
  - PyTorch profiler integration
  - Auto-tune script for throughput search (`scripts/autotune.sh`)
  - RL primitives for rollout batches, response masks, token logprobs, actor-to-rollout bridge sync, DPO, PPO-style, and GRPO-style losses

- **Fast iteration paths**:
  - Mock data mode for rapid debugging
  - Streaming-Dataloader example for real pre-training data
  - Size-based model presets for dense and MoE experiments
  - Minimal dependencies and plain shell launchers

## Research Workflow

Tinytron is intended to be edited in place. A typical loop is:

1. Pick a dense or MoE preset with `MODEL_SIZE=<size>`.
2. Change the specific module under study, such as attention, MoE routing, the optimizer, the loss, or the data loader.
3. Run `scripts/debug/pretrain.sh` with mock data to check correctness, multi-GPU behavior, and throughput quickly.
4. Move the same change to `scripts/example/pretrain.sh` when you want to run against Streaming-Dataloader data.

Most experiment surfaces are deliberately local: model code lives under `tinytron/model`, parallel collectives under `tinytron/distributed`, training state under `tinytron/training`, optimizer variants under `tinytron/optim`, and launch defaults under `scripts`.

For AI-assisted research workflows, see `research-skills/tinytron-research/`. It contains a repo-specific skill and compact references for architecture, inference/KV-cache, optimizer, and experiment-protocol work.

## Project Structure

```
.
├── tinytron/
│   ├── model/                              # Model architecture
│   │   ├── __init__.py
│   │   ├── gpt.py                          # GPT model implementation
│   │   └── modules/                        # Modular components
│   │       ├── attn.py                     # Attention mechanisms
│   │       ├── mlp.py                      # Dense MLP and MoE layers
│   │       ├── norm.py                     # Normalization layers
│   │       ├── loss.py                     # SP-aware cross entropy loss
│   │       └── emb.py                      # Embedding layers
│   │
│   ├── inference/                          # Inference helpers
│   │   ├── arguments.py                    # Inference CLI arguments
│   │   ├── cache.py                        # KV-cache data structures
│   │   ├── checkpoint.py                   # Inference checkpoint loading policy
│   │   ├── engine.py                       # Autoregressive decode engine
│   │   └── sampler.py                      # Sampling utilities
│   │
│   ├── training/                           # Training pipeline
│   │   ├── __init__.py
│   │   ├── checkpoint.py                   # Training checkpoint save/load policy
│   │   ├── config.py                       # Config dataclasses (ModelConfig, etc.)
│   │   ├── arguments.py                    # CLI argument definitions
│   │   └── trainer.py                      # Trainer and dataset init
│   │
│   ├── rl/                                 # RL training primitives
│   │   ├── __init__.py
│   │   ├── logprobs.py                     # Token logprob and response-mask helpers
│   │   ├── losses.py                       # DPO / PPO-style / GRPO-style losses
│   │   ├── rollout.py                      # Rollout batch helpers
│   │   ├── sync.py                         # Actor training to rollout model bridge sync
│   │   └── types.py                        # Lightweight RL dataclasses
│   │
│   ├── bridge/                             # Parameter-layout bridge infrastructure
│   │   ├── layout.py                       # Layout, placement, and shard metadata
│   │   ├── planner.py                      # Source/target shard movement planning
│   │   ├── stores.py                       # State-dict and shard-file tensor stores
│   │   ├── materializers.py                # Route-based plan materialization
│   │   ├── rules.py                        # Tinytron model layout rules
│   │   └── model.py                        # Tinytron model-state layout helpers
│   │
│   ├── optim/                              # Optimizer implementations
│   │   └── muon.py                         # Muon optimizer
│   │
│   ├── distributed/                        # Distributed training components
│   │   ├── __init__.py
│   │   ├── parallel_state.py               # DP/SEP process group construction
│   │   ├── zero1/
│   │   │   └── distributed_optimizer.py    # ZeRO-1 implementation
│   │   ├── sequence_parallel/
│   │   │   └── ulysses.py                  # SP collectives and grad sync helpers
│   │   └── expert_parallel/
│   │       └── comm.py                     # EP all-to-all communication
│   │
│   └── utils/                              # Utility functions
│       ├── __init__.py
│       ├── model.py                        # Model utilities (param counting, etc.)
│       ├── training.py                     # Schedule helpers (get_training_info, etc.)
│       └── profile.py                      # Profiling and MFU computation
│
├── scripts/                                # Launch scripts
│   ├── autotune.sh                         # Auto-tune SEP_SIZE/BATCH_SIZE by tok/sec
│   ├── debug/
│   │   ├── inference.py                    # Debug inference entry (KV cache)
│   │   ├── inference.sh                    # Inference debug launch script
│   │   ├── pretrain.py                     # Debug entry (mock data, minimal deps)
│   │   └── pretrain.sh                     # Configurable debug launch script
│   ├── example/
│   │   ├── pretrain.py                     # Example entry (Streaming-Dataloader)
│   │   └── pretrain.sh                     # Configurable real-data launch script
│
├── research-skills/                        # Repo-specific research skills
│   └── tinytron-research/
│       ├── SKILL.md
│       └── references/
│
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA/NCCL support
- tqdm
- numpy

Install minimal runtime dependencies:

```bash
pip install torch tqdm numpy
```

For `scripts/example/pretrain.py`, also clone Streaming-Dataloader into `external/streaming_dataloader`:

```bash
git clone https://github.com/liangyuwang/Streaming-Dataloader.git external/streaming_dataloader
```

## Quick Start

### 1. Single Node Training

**Using training scripts (recommended):**

```bash
# Train the default 0.25B dense model with mock data (8 GPUs)
bash scripts/debug/pretrain.sh

# Train a 0.3B MoE preset with mock data
MODEL_SIZE=0.3B-A0.17B bash scripts/debug/pretrain.sh

# Override SEP (sequence-expert joint) parallel size
SEP_SIZE=2 bash scripts/debug/pretrain.sh

# Try a larger dense preset
MODEL_SIZE=7B bash scripts/debug/pretrain.sh
```

The debug, example, and inference launch scripts share the same size names:
`0.03B`, `0.1B`, `0.25B`, `1B`, `1.3B`, `7B`, `13B`, `30B`, `70B`, `0.17B-A0.1B`, `0.3B-A0.17B`, `0.7B-A0.25B`, `2.7B-A1B`, `14B-A4.5B`, `104B-A4.5B`.

**Direct command for quick testing:**

```bash
torchrun --nproc_per_node=8 scripts/debug/pretrain.py \
  --use_mock_data \
  --mock_data_num_samples 1280 \
  --total_batch_size 524288 \
  --batch_size 8 \
  --seq_len 4096 \
  --sep_size 1 \
  --max_epochs 1 \
  --debug
```

### 2. Multi-Node Training

All training scripts support multi-node training via environment variables:

```bash
# Node 0 (master, e.g. IP: 192.168.1.100)
NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 \
bash scripts/debug/pretrain.sh

# Node 1 (worker)
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
bash scripts/debug/pretrain.sh
```

When running under some distributed training platforms, you do not need to specify `--node_rank`, `--nnodes`, or `--master_addr`; `torchrun` can detect injected values from `env://`.

### 3. Data Pipeline

Use `scripts/example/pretrain.py` for the Streaming-Dataloader path. It subclasses `Trainer` and overrides `_init_dataset`, so swapping the data pipeline does not require touching the core trainer. The mock-data path lives in `scripts/debug/pretrain.py` and is useful for checking model, optimizer, and parallelism changes without a real dataset.

You can also replace `_init_dataset` in your own entry script and return any dataset whose batches provide `input_ids` and `labels`.

### 4. Auto-Tune Throughput (`tok/sec`)

The repository includes an auto-tuner at `scripts/autotune.sh` to search throughput-friendly combinations of `SEP_SIZE` and `BATCH_SIZE`.

Default search space:
- `SEP_SIZES="1 2 4 8"`
- `BATCH_SIZES="1 2 4 8 16 32"`
- `RUN_SCRIPT="scripts/debug/pretrain.sh"`

Run with defaults:

```bash
bash scripts/autotune.sh
```

Run with custom search space and target script:

```bash
MODEL_SIZE=0.3B-A0.17B \
SEP_SIZES="1 2 4" \
BATCH_SIZES="4 8 16" \
TARGET_STEPS=80 \
WARMUP_STEPS=20 \
RUN_SCRIPT="scripts/debug/pretrain.sh" \
bash scripts/autotune.sh
```

Outputs:
- Summary CSV: `autotune_results.csv`
- Temporary log (auto-cleaned): `autotune_temp.log`
- Best config printed at the end as `SEP_SIZE=<...>, BATCH_SIZE=<...>`

### 5. Inference Baseline (KV Cache)

The repository includes a minimal inference entrypoint `scripts/debug/inference.py` that reuses the same `tinytron/model` code path and supports autoregressive decoding with per-layer KV cache.

Example:

```bash
python scripts/debug/inference.py \
  --checkpoint_path ./log/your_run/00010_model.pt \
  --prompt_token_ids 1,2,3,4 \
  --max_new_tokens 32 \
  --temperature 1.0 \
  --top_k 50
```

SEP distributed inference is also supported through `torchrun`:

```bash
torchrun --nproc_per_node 2 scripts/debug/inference.py \
  --checkpoint_path ./log/your_run/00010_model.pt \
  --prompt_token_ids 1,2,3,4 \
  --max_new_tokens 32 \
  --temperature 1.0 \
  --top_k 50 \
  --sep_size 2
```

Notes:
- For `sep_size > 1`, inference keeps the prompt replicated within each SEP group and shards KV-cache/state by attention head during decode.
- For `use_moe` with `sep_size > 1`, SEP inference keeps routing local on each rank, computes only local-expert contributions, and merges the replicated MoE outputs with an EP/SEP `all_reduce`.
- Sampling is synchronized within each SEP group so all ranks advance with the same next token.
- Output is printed as comma-separated token ids.
- `scripts/debug/inference.py` also prints prefill/decode throughput (`tok/s`) for quick performance checks.
- Use `scripts/debug/inference.sh` for a one-command debug launch with model-size presets (`0.03B`, `0.1B`, `0.25B`, `1B`, `1.3B`, `7B`, `13B`, `30B`, `70B`, `0.17B-A0.1B`, `0.3B-A0.17B`, `0.7B-A0.25B`, `2.7B-A1B`, `14B-A4.5B`, `104B-A4.5B`).
- `SEP_SIZE=<n>` makes the debug script switch to `torchrun --nproc_per_node <n>`.
- MoE presets such as `MODEL_SIZE=0.17B-A0.1B` can run in single-process inference mode (no `init_process_group`) with EP fallback to 1.

Smoke test without a checkpoint (random initialized weights):

```bash
python scripts/debug/inference.py \
  --init_from_scratch \
  --prompt_token_ids 1,2,3,4 \
  --max_new_tokens 8 \
  --device cuda
```

Or use the debug script:

```bash
# Random-weight smoke test
bash scripts/debug/inference.sh

# Checkpoint run with a preset size
MODEL_SIZE=0.1B CKPT_PATH=/path/to/00500_model.pt \
bash scripts/debug/inference.sh
```

## Configuration

Configuration is built from CLI arguments via `tinytron/training/arguments.py` and assembled into a unified `Config` in `tinytron/training/config.py`. Configs are ordinary dataclasses, so adding a new experiment knob usually means adding one CLI argument, one config field, and one use site.

### Model Configuration (`ModelConfig` in `tinytron/model/config.py`)

```python
@dataclass
class ModelConfig:
    block_size: int = 4096              # Maximum sequence length
    vocab_size: int = 50304             # Vocabulary size
    num_layer: int = 32                 # Number of transformer layers
    num_attention_heads: int = 128       # Number of attention heads
    num_key_value_heads: int = 8        # Number of KV heads (GQA)
    hidden_size: int = 1024             # Hidden dimension
    intermediate_size: int = 4096       # FFN intermediate size
    dropout: float = 0.0                # Dropout rate
    tied_lm_head: bool = True           # Tie input/output embeddings

    # Mixture of Experts (optional)
    use_moe: bool = False               # Enable MoE
    num_experts: int = 128              # Total number of experts
    num_experts_per_tok: int = 8        # Active experts per token
    moe_intermediate_size: int = 256    # Expert FFN size
    moe_balance_loss_weight: float = 0.01 # Set 0 to disable MoE balance loss
```

### Training Arguments (CLI → `TrainingConfig`)

Key CLI options (see `tinytron/training/arguments.py` for full list):

| Option | Default | Description |
|--------|---------|-------------|
| `--total_batch_size` | `524288` | Global batch size in tokens |
| `--batch_size` | `8` | Micro batch size per device |
| `--seq_len` | `4096` | Sequence length |
| `--max_lr` / `--min_lr` | `4e-3` / `3e-5` | Learning rate range |
| `--weight_decay` | `0.1` | AdamW weight decay |
| `--grad_clip_value` | `1.0` | Gradient clipping |
| `--warmup_steps` | `1000` | LR warmup steps |
| `--max_epochs` | `1` | Training epochs |
| `--do_save` | `False` | Enable checkpoint saving |
| `--save_every_steps` | `5000` | Checkpoint frequency |
| `--do_val` | `False` | Enable validation during training |
| `--val_every_steps` | `250` | Validation frequency (when `--do_val` is enabled) |
| `--optimizer` | `adam` | Optimizer type (`adam` / `muon`) |
| `--use_distributed_optimizer` | `True` | Enable ZeRO-1-style optimizer sharding |
| `--pin_memory` | `True` | Enable DataLoader pinned memory |
| `--tied_lm_head` | `True` | Tie token embedding and LM head weights |
| `--moe_balance_loss_weight` | `0.01` | MoE load-balancing loss weight (`0` disables it) |
| `--use_compile` | flag | PyTorch 2.0 compilation |

### Parallelism Configuration

`sep_size` controls SEP group size (sequence-expert joint parallelism).

- CLI flag: `--sep_size` (default: `1` in `tinytron/training/arguments.py`)
- Script env var: `SEP_SIZE` (mapped to `--sep_size`, script default is `1`)
- Dense models (`--use_moe` disabled): SEP degenerates to pure SP.
- Constraints:
  - `WORLD_SIZE % sep_size == 0`
  - sequence length must be divisible by SEP size (`seq_len % sep_size == 0`)

Example:

```bash
torchrun --nproc_per_node=8 scripts/debug/pretrain.py \
  --batch_size 8 \
  --seq_len 4096 \
  --sep_size 2 \
  --max_epochs 1
```

## Training Features

### Checkpoint Saving and Resumption

Checkpoint saving is disabled by default. Enable it with:

```bash
--do_save --save_every_steps 5000
```

When enabled, the trainer can save and resume checkpoints, preserving:
- Model weights (`*_model_rankXXXXX.pt` per-rank shards; `*_model.pt` is a rank-0 legacy compatibility file)
- Optimizer states (`*_opt/` directory)
- Training metadata (`*_meta.pt`): step counter, RNG state, dataloader position

Model checkpointing is sharded by rank to avoid gathering MoE experts or other local shards onto rank 0. The metadata records the model layout used for the checkpoint. If you resume with the same layout, Tinytron restores the matching local model shard and optimizer shard. If you resume with a different SEP layout, Tinytron uses the bridge planner plus shard-file reads to rebuild each rank's local model state from the source rank files, and skips optimizer restore because optimizer-state resharding is intentionally not supported.

To resume, restart the same training command. The trainer searches checkpoints under the current `log_dir` by default, or you can specify `--resume_path` explicitly.

Inference loading uses its own checkpoint policy. It can load regular single-file checkpoints or sharded training checkpoints; for sharded checkpoints, each inference rank reads only the source shard slices required by its current inference layout.

### ZeRO-1 Optimizer

Memory-efficient optimizer state partitioning:
- Optimizer states are sharded across GPUs
- Model parameters remain replicated
- Automatic gradient synchronization and parameter broadcasting

Enable it with:

```bash
--use_distributed_optimizer
```

Native support for Muon + ZeRO-1 is also available:

```bash
--optimizer muon --use_distributed_optimizer
```

### MoE Balance Loss

MoE models add load-balancing loss by default with `--moe_balance_loss_weight 0.01`. Disable it or tune it from either the CLI or launch scripts:

```bash
# Direct CLI
--moe_balance_loss_weight 0

# Training scripts
MODEL_SIZE=0.3B-A0.17B MOE_BALANCE_LOSS_WEIGHT=0 bash scripts/debug/pretrain.sh
```

### Gradient Accumulation

Automatically computed based on:
```
grad_accum_steps = total_batch_size / (batch_size × seq_len × num_dp_ranks)
```

### Learning Rate Schedule

Implements cosine annealing with linear warmup:
1. Linear warmup: 0 → max_lr over `warmup_steps`
2. Cosine decay: max_lr → min_lr over remaining steps

### Validation

Validation is optional and disabled by default.

```bash
--do_val --val_every_steps 250
```

When enabled, validation runs every `val_every_steps` and on the last step (unless `--debug` is set).

### Model FLOPs Utilization (MFU)

Real-time tracking of hardware efficiency:
```
MFU = (Actual FLOPs) / (Peak Hardware FLOPs)
```

## Profiling

Enable PyTorch profiler for performance analysis:

```bash
python scripts/debug/pretrain.py \
  --use_profiler \
  --steps_to_profile 15 20  # profile on step 15 to 20
```

This generates a Chrome trace file at `<log_dir>/rank{rank}_trace.json` (for the exporting process) that can be viewed in `chrome://tracing`.

## Testing

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

## Example Model Configurations

### GPT-0.25B (12 layers)
```bash
--num_layer 12 \
--num_attention_heads 32 \
--num_key_value_heads 4 \
--hidden_size 1024 \
--intermediate_size 4096
```

### GPT-1B (24 layers)
```bash
--num_layer 24 \
--num_attention_heads 64 \
--num_key_value_heads 8 \
--hidden_size 2048 \
--intermediate_size 8192
```

### GPT-7B (32 layers)
```bash
--num_layer 32 \
--num_attention_heads 128 \
--num_key_value_heads 16 \
--hidden_size 4096 \
--intermediate_size 16384
```

## Hacking The Code

Tinytron is meant to be changed directly. The most common edit points are deliberately narrow:

- Model architecture: `tinytron/model/gpt.py` and `tinytron/model/modules/`
- Attention or sequence parallelism: `tinytron/model/modules/attn.py` and `tinytron/distributed/sequence_parallel/`
- MoE routing and experts: `tinytron/model/modules/mlp.py` and `tinytron/distributed/expert_parallel/`
- Loss behavior: `tinytron/model/modules/loss.py`
- Optimizer behavior: `tinytron/training/trainer.py`, `tinytron/optim/`, and `tinytron/distributed/zero1/`
- Data loading: subclass `Trainer` and override `_init_dataset`
- Launch defaults: `scripts/debug/pretrain.sh` and `scripts/example/pretrain.sh`

`tinytron/bridge` is supporting infrastructure, not a normal research edit surface. It maps parameter layouts across training, inference, and checkpoint shards so experiments can change SEP layouts without centralizing model weights. Prefer changing model, inference, optimizer, or data modules for research work; touch bridge code only when the research question is specifically about parameter layout, checkpoint resharding, or cross-system model-state transfer.

### Custom Dataset

Implement your dataset class and override `_init_dataset`: subclass `Trainer` in your entry script (e.g. `scripts/example/pretrain.py`) and set `self.train_dataset` to your dataset. Each item should provide tensors compatible with the trainer (e.g. contiguous token ids of length `seq_len+1` for causal LM).

### Custom Architecture

Modify components in `tinytron/model/modules/`:
- `attn.py`: Implement custom attention mechanisms
- `mlp.py`: Add new feedforward architectures
- `norm.py`: Experiment with normalization strategies

### Custom Optimizer

Replace AdamW in `_init_optimizer` in `tinytron/training/trainer.py` (or in a `Trainer` subclass):

```python
def _init_optimizer(self, config: Config):
    self.optimizer = YourOptimizer(
        self.raw_model.parameters(),
        lr=config.optim.max_lr,
    )
    self.optimizer = DistributedOptimizer(
        optimizer=self.optimizer,
        process_group=self.dp_group,
    )
```

## Logging

Training logs are saved to:
```
<log_dir>/modelsize_<...>_lr<...>_BS<...>_SL<...>_DP<...>_SEP<...>/log.txt
```

Log format:
```
<step> train <loss>
<step> val <val_loss>
```

Example:
```
0 train 10.8234
100 train 8.4521
250 val 8.3012
```

## Performance Tips

1. **Enable compilation**: Add `--use_compile` for PyTorch 2.0+ (20-30% speedup)
2. **Tune batch size**: Maximize `--batch_size` per GPU to improve throughput
3. **Run auto-tune first**: Use `bash scripts/autotune.sh` to quickly find strong `SEP_SIZE` + `BATCH_SIZE` settings
4. **Use Flash Attention**: Ensure Flash Attention is available for faster attention
5. **Gradient checkpointing**: Implement in `tinytron/model/gpt.py` for larger models
6. **Mixed precision**: BFloat16 is enabled by default (better than FP16 for training)

## Common Issues

### Out of Memory
- Reduce `--batch_size` (micro batch size)
- Enable gradient checkpointing
- Use larger `grad_accum_steps` by reducing `--batch_size`

### Slow Training
- Ensure your PyTorch/CUDA build supports optimized SDPA kernels
- Enable `--use_compile`
- For MoE, prefer grouped GEMM kernels as much as possible
- Check MFU percentage (should be >30% for efficient training)
- Increase `--batch_size` to better utilize GPU

### Checkpoint Issues
- Ensure all processes have write access to `log_dir`
- Check disk space for optimizer state storage

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tinytron,
  title = {Tinytron},
  author = {Liangyu Wang},
  year = {2026},
  url = {https://github.com/liangyuwang/Tinytron}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Acknowledgments

This implementation draws inspiration from:
- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) by NVIDIA
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO optimization

## Contributing

Contributions are welcome, especially changes that keep the code easy to inspect, easy to modify, and useful for research experiments.
