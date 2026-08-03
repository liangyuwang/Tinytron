# aimegatron

An AI-native Megatron-style distributed pre-training framework. aimegatron
keeps the parallelism power of Megatron-LM while being small enough to read,
modify, and evolve in a single sitting — by humans or AI agents.

## Design principles

1. **Parallelism is declarative.** One `ParallelConfig(tp_size, sequence_parallel)`
   and one mesh builder (`aimegatron/core/mesh.py`) derive every process group.
   No model code knows about ranks or collectives.
2. **Sharding is annotation, not code.** Models are assembled from three small
   tensor-parallel primitives — `ColumnParallelLinear`, `RowParallelLinear`,
   `VocabParallelEmbedding` (`aimegatron/parallel/layers.py`). The attention
   and MLP modules read like single-GPU code.
3. **SP is a flag, not a rewrite.** Megatron-style sequence parallelism
   (reduce-scatter / all-gather around TP regions) is turned on purely by
   config; no layer has a separate SP code path.
4. **Small modules with edit contracts.** Every module states its single
   responsibility and its edit contract in the docstring. Extension points
   are registry seams (`aimegatron/core/registry.py`), not conditionals.

## V1 scope

| Capability | Status |
|---|---|
| Data parallelism (DP) | supported |
| Tensor parallelism (TP, column/row/vocab sharding) | supported |
| Sequence parallelism (Megatron-style, rides on TP group) | supported |
| ZeRO-1 distributed optimizer (fp32 master shard across DP) | supported |
| Sharded checkpoints + layout-driven TP resharding | supported |
| Dense GPT (GQA, RoPE, SwiGLU, LayerNorm/RMSNorm) | supported |
| Pipeline parallelism / MoE / context parallelism | designed-out extension points |

## Layout

```
aimegatron/
├── core/
│   ├── config.py      # all knobs + centralized validation
│   ├── registry.py    # extension seams (optimizers, norms)
│   └── mesh.py        # ParallelConfig -> TP/DP process groups
├── parallel/
│   ├── comm.py        # autograd collectives (fwd/bwd pairs)
│   └── layers.py      # Column/Row/VocabParallel primitives
├── model/
│   ├── rope.py        # head-local RoPE
│   ├── attention.py   # GQA attention (parallelism-free)
│   ├── mlp.py         # SwiGLU MLP (parallelism-free)
│   ├── norm.py        # LayerNorm / RMSNorm
│   ├── loss.py        # layout-agnostic cross-entropy
│   └── gpt.py         # GPT + finalize_model_grads + clip_grad_norm
├── train/
│   ├── layout.py      # TP shard rule table (drives resharding)
│   ├── optimizer.py   # ZeRO-1 DistributedOptimizer
│   ├── checkpoint.py  # sharded save/load + cross-TP resharding
│   └── trainer.py     # train loop (mock data by default)
└── utils/             # seed, LR schedule, param counting
scripts/
├── pretrain.py        # CLI entrypoint with model presets
└── pretrain.sh        # env-var launcher
tests/                 # CPU/gloo correctness tests
```

## Quick start

```bash
pip install torch numpy tqdm

# single GPU / process, mock data
python scripts/pretrain.py --model_size 0.03B --max_steps 100 --dtype bf16

# 8 GPUs: TP=8 with sequence parallelism
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --model_size 0.25B --tp_size 8 --sequence_parallel

# launcher with env knobs
MODEL_SIZE=0.1B NUM_GPUS=4 TP_SIZE=4 SEQUENCE_PARALLEL=1 \
MAX_STEPS=200 bash scripts/pretrain.sh
```

Model presets: `0.03B`, `0.1B`, `0.25B`, `1B`, `7B` (dense; all dims
TP-friendly up to tp=8). Any preset field can be overridden via CLI
(`--num_layer`, `--hidden_size`, ...).

Key flags:

| Flag | Meaning |
|---|---|
| `--tp_size N` | tensor parallel size (`world_size = dp_size * tp_size`) |
| `--sequence_parallel` | enable Megatron-style SP (requires `tp_size > 1`) |
| `--no_distributed_optimizer` | disable ZeRO-1 sharding |
| `--dtype bf16|fp32` | compute dtype |
| `--do_save --save_every_steps N` | checkpointing under `--log_dir` |
| `--resume_path DIR` | resume from a specific checkpoint |

## Checkpoints and resharding

Each step saves `model_tp{rank}.pt` per TP rank plus `meta.pt`. Loading with
a *different* `tp_size` resharded automatically: `train/layout.py` maps each
parameter to its shard dimension, `train/checkpoint.py` concatenates source
shards and re-slices for the target layout. Optimizer state is restored only
when the full layout (world/tp/dp) is unchanged.

```bash
# train with TP=2, then resume the same run with TP=4
torchrun --nproc_per_node=2 scripts/pretrain.py --tp_size 2 --do_save ...
torchrun --nproc_per_node=4 scripts/pretrain.py --tp_size 4 --resume_path ./log/step_XXXXXXX ...
```

## Tests

Run from the repository root (CPU-only, gloo backend):

```bash
python -m unittest discover -s tests -v
```

- `test_config.py` — config validation rules
- `test_parallel_layers.py` — TP-sharded fwd/bwd equals single-device math (with/without SP)
- `test_checkpoint.py` — save/load roundtrip and TP 1<->2 resharding
- `test_train_smoke.py` — end-to-end training on CPU: TP=2+SP, TP=2 no-SP, DP=2 ZeRO-1,
  and cross-TP resume (train at TP=2, resume at TP=1)
- `test_alignment.py` — parallel-vs-single-device alignment: TP=2 (±SP) and DP=2 runs
  must reproduce single-device training; final weights agree within ~1e-6

Alignment is an invariant, not a hope: model init is layout-agnostic (every rank
derives its TP shard as an exact slice of the single-device weights), so a run
started from any tp_size is bit-for-bit the same model.

## Extending

- **New optimizer**: register a factory under `OPTIMIZERS` in
  `aimegatron/core/registry.py`, select with `--optimizer`.
- **New norm**: implement it in `aimegatron/model/norm.py` (mark TP-replicated
  params), register under `NORMS`, select with `--norm_type`.
- **New sharded module**: use the `Column/RowParallelLinear` primitives and add
  its weight pattern to `TP_SHARD_RULES` in `train/layout.py`.
- **Real data**: subclass `Trainer` and override `_init_dataset` (batches must
  provide `input_ids` and `labels`).

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA/NCCL for GPU training; gloo/CPU works for tests)
- numpy

## License

See [LICENSE](LICENSE).
