#!/bin/bash
# test_sp_ep_alignment.sh

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_DP="${MASTER_PORT_DP:-29501}"
MASTER_PORT_SP_EP="${MASTER_PORT_SP_EP:-29502}"
if [[ -n "${NPROC_PER_NODE+x}" ]]; then
  NPROC_PER_NODE_USER_SET=1
else
  NPROC_PER_NODE_USER_SET=0
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
AUTO_EXCLUDE_DISPLAY_GPU="${AUTO_EXCLUDE_DISPLAY_GPU:-1}"
AUTO_SET_CUDA_VISIBLE=0
LOG_DIR="${LOG_DIR:-./log}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$AUTO_EXCLUDE_DISPLAY_GPU" == "1" ]] && [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t _GPU_ROWS < <(nvidia-smi --query-gpu=index,display_active --format=csv,noheader,nounits 2>/dev/null || true)
  if (( ${#_GPU_ROWS[@]} > 0 )); then
    NON_DISPLAY_GPU_IDS=()
    DISPLAY_GPU_IDS=()
    for ROW in "${_GPU_ROWS[@]}"; do
      IDX="$(echo "$ROW" | cut -d',' -f1 | xargs)"
      DISPLAY_ACTIVE="$(echo "$ROW" | cut -d',' -f2 | xargs)"
      if [[ "$DISPLAY_ACTIVE" == "Enabled" ]]; then
        DISPLAY_GPU_IDS+=("$IDX")
      else
        NON_DISPLAY_GPU_IDS+=("$IDX")
      fi
    done
    if (( ${#DISPLAY_GPU_IDS[@]} > 0 )) && (( ${#NON_DISPLAY_GPU_IDS[@]} >= 2 )); then
      CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${NON_DISPLAY_GPU_IDS[*]}")"
      export CUDA_VISIBLE_DEVICES
      AUTO_SET_CUDA_VISIBLE=1
      echo "[info] Detected display-attached GPU(s): ${DISPLAY_GPU_IDS[*]}"
      echo "[info] Using non-display GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
  fi
fi

AVAILABLE_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "$AVAILABLE_GPU_COUNT" =~ ^[0-9]+$ ]] && (( AVAILABLE_GPU_COUNT > 0 )) && (( NPROC_PER_NODE > AVAILABLE_GPU_COUNT )); then
  if (( NPROC_PER_NODE_USER_SET == 0 )); then
    ADJUSTED_NPROC="$AVAILABLE_GPU_COUNT"
    if (( ADJUSTED_NPROC % 2 != 0 )); then
      ADJUSTED_NPROC=$((ADJUSTED_NPROC - 1))
    fi
    if (( ADJUSTED_NPROC < 2 )); then
      echo "Not enough visible CUDA devices to run SP+EP alignment (need at least 2)."
      exit 1
    fi
    echo "[warn] NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible GPU count=$AVAILABLE_GPU_COUNT."
    echo "[warn] Auto-adjusting NPROC_PER_NODE to $ADJUSTED_NPROC."
    NPROC_PER_NODE="$ADJUSTED_NPROC"
    if (( AUTO_SET_CUDA_VISIBLE == 1 )) && (( ${#NON_DISPLAY_GPU_IDS[@]} > ADJUSTED_NPROC )); then
      SELECTED_GPU_IDS=("${NON_DISPLAY_GPU_IDS[@]: -$ADJUSTED_NPROC}")
      CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SELECTED_GPU_IDS[*]}")"
      export CUDA_VISIBLE_DEVICES
      echo "[info] Selected GPU subset for NPROC_PER_NODE=$NPROC_PER_NODE: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
  else
    echo "NPROC_PER_NODE=$NPROC_PER_NODE exceeds visible GPU count=$AVAILABLE_GPU_COUNT."
    echo "Please reduce NPROC_PER_NODE or set CUDA_VISIBLE_DEVICES explicitly."
    exit 1
  fi
fi

if (( NPROC_PER_NODE % 2 != 0 )); then
  echo "NPROC_PER_NODE must be divisible by 2 for the SP+EP run (sep_size=2)."
  exit 1
fi

mkdir -p "$LOG_DIR"

# Shared config
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-2}"
MAX_LR="${MAX_LR:-6e-4}"
MIN_LR="${MIN_LR:-6e-5}"

SEED="${SEED:-1337}"

BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-512}"
# 2 * 512 * 4 (DP ranks) = 4096. 
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-4096}"
MOCK_DATA_NUM_SAMPLES="${MOCK_DATA_NUM_SAMPLES:-1280}"
MAX_STEPS="${MAX_STEPS:-160}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"

USE_COMPILE="${USE_COMPILE:-0}"
DETER_MODE="${DETER_MODE:-1}"
PRECISION="${PRECISION:-bf16}"
USE_MOE="${USE_MOE:-1}"
MOE_ROUTER_DEBUG="${MOE_ROUTER_DEBUG:-0}"
MOE_ROUTE_TRACE="${MOE_ROUTE_TRACE:-0}"
DISABLE_TF32="${DISABLE_TF32:-0}"

if [[ "$PRECISION" == "bf16" ]]; then
  if ! python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
sys.exit(0 if ok else 1)
PY
  then
    echo "[warn] bf16 requested but current CUDA runtime/GPU does not report bf16 support."
    echo "[warn] Falling back to fp32. You can also set PRECISION=fp32 explicitly."
    PRECISION="fp32"
  fi
fi

# Common training args
COMMON_TRAINING_ARGS="\
  --seed $SEED \
  --dataset_path ... \
  --use_mock_data \
  --mock_data_num_samples $MOCK_DATA_NUM_SAMPLES \
  --log_dir $LOG_DIR \
  --total_batch_size $TOTAL_BATCH_SIZE \
  --seq_len $SEQ_LEN \
  --max_lr $MAX_LR \
  --min_lr $MIN_LR \
  --weight_decay 0.1 \
  --grad_clip_value 1.0 \
  --warmup_steps $LR_WARMUP_STEPS \
  --max_epochs $MAX_EPOCHS \
  --max_steps $MAX_STEPS \
  --debug \
  --precision $PRECISION \
"
if [ $USE_COMPILE -eq 1 ]; then
  COMMON_TRAINING_ARGS="$COMMON_TRAINING_ARGS --use_compile"
fi
if [ $DETER_MODE -eq 1 ]; then
  COMMON_TRAINING_ARGS="$COMMON_TRAINING_ARGS --deterministic"
fi
if [ "$DISABLE_TF32" -eq 1 ]; then
  COMMON_TRAINING_ARGS="$COMMON_TRAINING_ARGS --disable_tf32"
fi

MODEL_ARGS="\
  --block_size 4096 \
  --vocab_size 50304 \
  --num_layer 2 \
  --num_attention_heads 32 \
  --num_key_value_heads 4 \
  --hidden_size 768 \
  --intermediate_size 3072 \
  --tied_lm_head \
  --dropout 0.0 \
"
if [ "$USE_MOE" -eq 1 ]; then
  MODEL_ARGS="$MODEL_ARGS \
  --use_moe \
  --num_experts 8 \
  --num_experts_per_tok 2 \
  --moe_intermediate_size 768 \
"
  if [ "$MOE_ROUTER_DEBUG" -eq 1 ]; then
    MODEL_ARGS="$MODEL_ARGS --moe_router_debug"
  fi
  if [ "$MOE_ROUTE_TRACE" -eq 1 ]; then
    MODEL_ARGS="$MODEL_ARGS --moe_route_trace"
  fi
fi

echo "========================================"
echo "RUNNING STANDARD DP"
echo "========================================"

EXP_NAME_DP="test_alignment_dp_${RUN_TAG}"
BATCH_SIZE_PER_DP_RANK=$BATCH_SIZE
SEP_SIZE=1

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT_DP" \
  pretrain_debug.py \
  --exp_name "$EXP_NAME_DP" \
  --batch_size "$BATCH_SIZE_PER_DP_RANK" \
  $COMMON_TRAINING_ARGS \
  --sep_size "$SEP_SIZE" \
  --use_distributed_optimizer \
  $MODEL_ARGS

echo "========================================"
echo "RUNNING SP + EP"
echo "========================================"

EXP_NAME_SP_EP="test_alignment_sp_ep_${RUN_TAG}"
SEP_SIZE=2
BATCH_SIZE_PER_DP_RANK=$(($BATCH_SIZE * $SEP_SIZE))

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT_SP_EP" \
  pretrain_debug.py \
  --exp_name "$EXP_NAME_SP_EP" \
  --batch_size "$BATCH_SIZE_PER_DP_RANK" \
  $COMMON_TRAINING_ARGS \
  --sep_size "$SEP_SIZE" \
  --use_distributed_optimizer \
  $MODEL_ARGS

echo "========================================"
echo "Comparing Logs"
echo "========================================"

LOG_DP_DIR=$(find "$LOG_DIR" -name "${EXP_NAME_DP}*" -type d | head -n 1 || true)
LOG_SP_EP_DIR=$(find "$LOG_DIR" -name "${EXP_NAME_SP_EP}*" -type d | head -n 1 || true)

if [[ -z "$LOG_DP_DIR" || -z "$LOG_SP_EP_DIR" ]]; then
  echo "Could not find experiment directories under $LOG_DIR."
  exit 1
fi

LOG_DP="$LOG_DP_DIR/log.jsonl"
LOG_SP_EP="$LOG_SP_EP_DIR/log.jsonl"

if [[ ! -f "$LOG_DP" || ! -f "$LOG_SP_EP" ]]; then
  echo "Expected log files were not found:"
  echo "  $LOG_DP"
  echo "  $LOG_SP_EP"
  exit 1
fi

python test_scripts/compare_loss.py "$LOG_DP" "DP Loss" "$LOG_SP_EP" "SP+EP Loss" test_scripts/loss_alignment.png

if [ "$USE_MOE" -eq 1 ] && [ "$MOE_ROUTE_TRACE" -eq 1 ]; then
  TRACE_DP="$LOG_DP_DIR/moe_route_trace_rank0.jsonl"
  TRACE_SP_EP="$LOG_SP_EP_DIR/moe_route_trace_rank0.jsonl"
  if [[ -f "$TRACE_DP" && -f "$TRACE_SP_EP" ]]; then
    echo "========================================"
    echo "Comparing MoE Route Flip Rate"
    echo "========================================"
    python test_scripts/compare_route_flip.py "$TRACE_DP" "$TRACE_SP_EP"
  else
    echo "[warn] route trace file missing, skip route flip comparison."
    echo "[warn] expected: $TRACE_DP and $TRACE_SP_EP"
  fi
fi

echo "Tailing standard vs sp+ep loss:"
paste <(grep '"stage": "train"' "$LOG_DP" | sed -n '1,20p') <(grep '"stage": "train"' "$LOG_SP_EP" | sed -n '1,20p')
