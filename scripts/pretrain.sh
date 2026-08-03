#!/usr/bin/env bash
# aimegatron pre-training launcher.
#
# Env knobs:
#   MODEL_SIZE          0.03B | 0.03B-MoE | 0.1B | ...        (default 0.25B)
#   NUM_GPUS            processes per node                     (default 1)
#   TP_SIZE             tensor parallel size                   (default 1)
#   PP_SIZE             pipeline parallel size (1F1B)          (default 1)
#   EP_SIZE             expert parallel size (inside DP)       (default 1)
#   SEQUENCE_PARALLEL   1 to enable Megatron-style SP          (default 0)
#   BATCH_SIZE          micro batch per DP rank                (default 8)
#   TOTAL_BATCH_SIZE    global batch in tokens                 (default 524288)
#   SEQ_LEN             sequence length                        (default 4096)
#   MAX_STEPS           training steps                         (default 100)
#   BACKEND             nccl (GPU) or gloo (CPU)               (default nccl)
#   LOG_DIR             output directory                       (default ./log)
#   EXTRA_ARGS          forwarded verbatim to pretrain.py
#
# Multi-node: export NUM_NODES / NODE_RANK / MASTER_ADDR / MASTER_PORT and
# torchrun picks them up via env://.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_SIZE="${MODEL_SIZE:-0.25B}"
NUM_GPUS="${NUM_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
SEQ_LEN="${SEQ_LEN:-4096}"
MAX_STEPS="${MAX_STEPS:-100}"
BACKEND="${BACKEND:-nccl}"
LOG_DIR="${LOG_DIR:-./log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  --model_size "$MODEL_SIZE"
  --tp_size "$TP_SIZE"
  --pp_size "$PP_SIZE"
  --ep_size "$EP_SIZE"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --seq_len "$SEQ_LEN"
  --max_steps "$MAX_STEPS"
  --backend "$BACKEND"
  --log_dir "$LOG_DIR"
)
if [ "$SEQUENCE_PARALLEL" = "1" ]; then
  ARGS+=(--sequence_parallel)
fi

echo "[aimegatron] MODEL_SIZE=$MODEL_SIZE NUM_GPUS=$NUM_GPUS TP_SIZE=$TP_SIZE PP_SIZE=$PP_SIZE EP_SIZE=$EP_SIZE SP=$SEQUENCE_PARALLEL"
torchrun --nproc_per_node="$NUM_GPUS" scripts/pretrain.py "${ARGS[@]}" $EXTRA_ARGS
