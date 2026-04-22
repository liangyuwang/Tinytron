# Tinytron

A minimal, hackable pre-training stack for GPT-style language models. This project provides a clean foundation for training transformer models from scratch with distributed training support.

## Features

- **Modular GPT Architecture**: Flexible transformer implementation with support for:
  - Grouped Query Attention (GQA)
  - Mixture of Experts (MoE)
  - Customizable attention, MLP, and normalization layers
  - Flash Attention optimization support
  
- **Distributed Training**:
  - ZeRO-1 optimizer state partitioning for memory efficiency (Native support for Muon)
  - DistributedDataParallel (DDP) for multi-GPU training
  - Sequence-Expert joint parallelism via `SEP_SIZE` / `--sep_size` (SEP)
  - Gradient accumulation for large effective batch sizes
  
- **Training Optimizations**:
  - Mixed precision training (BFloat16)
  - Gradient clipping
  - Cosine learning rate schedule with warmup
  - Automatic checkpoint resumption with full state recovery
  
- **Developer-Friendly**:
  - Comprehensive profiling utilities
  - Model FLOPs Utilization (MFU) tracking
  - Auto-tune script for fast throughput search (`scripts/autotune.sh`)
  - Mock data mode for rapid debugging
  - Minimal dependencies

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
│   ├── training/                           # Training pipeline
│   │   ├── __init__.py
│   │   ├── config.py                       # Config dataclasses (ModelConfig, etc.)
│   │   ├── arguments.py                    # CLI argument definitions
│   │   └── trainer.py                      # Trainer and dataset init
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
│   ├── debug_gpt_0.25b/
│   │   └── pretrain.sh                     # 0.25B debug (pretrain_debug.py)
│   ├── debug_gpt_0.3b_a0.17b/
│   │   └── pretrain.sh                     # 0.3B MoE debug (pretrain_debug.py)
│   ├── example_gpt_0.25b/
│   │   └── pretrain.sh                     # 0.25B example with custom data (pretrain_example.py)
│
├── pretrain_debug.py                       # Debug entry (mock data, minimal deps)
├── pretrain_example.py                     # Example entry (custom dataset / tokenizer)
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

For `pretrain_example.py`, also install:

```bash
pip install datasets transformers
```

## Quick Start

### 1. Single Node Training

**Using training scripts (recommended):**

```bash
# Train 0.25B dense model (8 GPUs)
bash scripts/debug_gpt_0.25b/pretrain.sh

# Train 0.3B MoE model (8 GPUs)
bash scripts/debug_gpt_0.3b_a0.17b/pretrain.sh

# Override SEP (sequence-expert joint) parallel size
SEP_SIZE=2 bash scripts/debug_gpt_0.25b/pretrain.sh
```

**Direct command for quick testing:**

```bash
torchrun --nproc_per_node=8 pretrain_debug.py \
  --exp_name debug_test \
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
bash scripts/debug_gpt_0.25b/pretrain.sh

# Node 1 (worker)
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 \
bash scripts/debug_gpt_0.25b/pretrain.sh
```

When running under some distributed training platforms, You do not need to specify --node_rank, --nnodes, or --master_addr. 'torchrun' automatically detects and uses these injected variables from 'env://' to set up distributed communication.

### 3. Custom Dataset

Use the example entry point and override `_init_dataset`: see `pretrain_example.py` for a subclass that uses a real dataset and tokenizer. The base implementation (mock data) lives in `tinytron/training/trainer.py`; override it in your entry script or subclass `Trainer` and pass your dataset there.
You can also use [Streaming-Dataloader](https://github.com/liangyuwang/Streaming-Dataloader) to build a memory-efficient streaming dataset pipeline for LLM pretraining.

### 4. Auto-Tune Throughput (`tok/sec`)

The repository includes an auto-tuner at `scripts/autotune.sh` to search throughput-friendly combinations of `SEP_SIZE` and `BATCH_SIZE`.

Default search space:
- `SEP_SIZES="1 2 4 8"`
- `BATCH_SIZES="1 2 4 8 16 32"`
- `RUN_SCRIPT="scripts/debug_gpt_0.25b/pretrain.sh"`

Run with defaults:

```bash
bash scripts/autotune.sh
```

Run with custom search space and target script:

```bash
SEP_SIZES="1 2 4" \
BATCH_SIZES="4 8 16" \
TARGET_STEPS=80 \
WARMUP_STEPS=20 \
RUN_SCRIPT="scripts/debug_gpt_0.3b_a0.17b/pretrain.sh" \
bash scripts/autotune.sh
```

Outputs:
- Summary CSV: `autotune_results.csv`
- Temporary log (auto-cleaned): `autotune_temp.log`
- Best config printed at the end as `SEP_SIZE=<...>, BATCH_SIZE=<...>`

### 5. Inference Baseline (KV Cache)

The repository includes a minimal inference entrypoint `inference.py` that reuses the same `tinytron/model` code path and supports autoregressive decoding with per-layer KV cache.

Example:

```bash
python inference.py \
  --checkpoint_path ./log/your_run/00010_model.pt \
  --prompt_token_ids 1,2,3,4 \
  --max_new_tokens 32 \
  --temperature 1.0 \
  --top_k 50
```

Notes:
- This baseline currently targets `sep_size == 1` for KV-cache inference.
- Output is printed as comma-separated token ids.
- `inference.py` also prints prefill/decode throughput (`tok/s`) for quick performance checks.
- Use `scripts/inference_debug.sh` for a one-command debug launch with model-size presets (`tiny`, `small`, `base`, `large`, `moe-sm`).
- `MODEL_SIZE=moe-sm` can run in single-process inference mode (no `init_process_group`) with EP fallback to 1.

Smoke test without a checkpoint (random initialized weights):

```bash
python inference.py \
  --init_from_scratch \
  --prompt_token_ids 1,2,3,4 \
  --max_new_tokens 8 \
  --device cuda
```

Or use the debug script:

```bash
# Random-weight smoke test
bash scripts/inference_debug.sh

# Checkpoint run with a preset size
MODEL_SIZE=small CKPT_PATH=./log/debug_gpt_0.25b/00500_model.pt \
bash scripts/inference_debug.sh
```

## Configuration

Configuration is built from CLI arguments via `tinytron/training/arguments.py` and assembled into a unified `Config` in `tinytron/training/config.py`.

### Model Configuration (`ModelConfig` in `tinytron/training/config.py`)

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
    tied_lm_head: bool = False          # Tie input/output embeddings (enable via --tied_lm_head)

    # Mixture of Experts (optional)
    use_moe: bool = False               # Enable MoE
    num_experts: int = 128              # Total number of experts
    num_experts_per_tok: int = 8        # Active experts per token
    moe_intermediate_size: int = 256    # Expert FFN size
```

### Training Arguments (CLI → `TrainingConfig`)

Key CLI options (see `tinytron/training/arguments.py` for full list):

| Option | Default | Description |
|--------|---------|-------------|
| `--exp_name` | `gpt` | Experiment name |
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
| `--use_distributed_optimizer` | `False` | Enable ZeRO-1-style optimizer sharding |
| `--pin_memory` | `False` | Enable DataLoader pinned memory |
| `--tied_lm_head` | `False` | Tie token embedding and LM head weights |
| `--use_compile` | flag | PyTorch 2.0 compilation |

### Parallelism Configuration

`sep_size` controls SEP group size (sequence-expert joint parallelism).

- CLI flag: `--sep_size` (default: `8` in `tinytron/training/arguments.py`)
- Script env var: `SEP_SIZE` (mapped to `--sep_size`, script default is `1`)
- Dense models (`--use_moe` disabled): SEP degenerates to pure SP.
- Constraints:
  - `WORLD_SIZE % sep_size == 0`
  - sequence length must be divisible by SEP size (`seq_len % sep_size == 0`)

Example:

```bash
torchrun --nproc_per_node=8 pretrain_debug.py \
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
- Model weights (`*_model.pt`)
- Optimizer states (`*_opt/` directory)
- Training metadata (`*_meta.pt`): step counter, RNG state, dataloader position

To resume, restart the same training command. The trainer searches checkpoints under the current experiment `log_dir` by default, or you can specify `--resume_path` explicitly.

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
python pretrain_debug.py \
  --use_profiler \
  --steps_to_profile 15 20  # profile on step 15 to 20
```

This generates a Chrome trace file at `<log_dir>/rank{rank}_trace.json` (for the exporting process) that can be viewed in `chrome://tracing`.

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

## Extending the Code

### Custom Dataset

Implement your dataset class and override `_init_dataset`: subclass `Trainer` in your entry script (e.g. `pretrain_example.py`) and set `self.train_dataset` to your dataset. Each item should provide tensors compatible with the trainer (e.g. contiguous token ids of length `seq_len+1` for causal LM).

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
<log_dir>/<exp_name>_modelsize_<...>_lr<...>_BS<...>_SL<...>_DP<...>_SEP<...>/log.txt
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

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This is a minimal training stack designed for educational purposes and rapid prototyping. For production-scale training, consider using frameworks like DeepSpeed, Megatron-LM, or Composer.
