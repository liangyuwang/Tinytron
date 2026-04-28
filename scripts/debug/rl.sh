#!/bin/bash
set -euo pipefail

# Minimal synchronous RL debug launcher.
# Stage 1: the same ranks run actor training and rollout inference, with
# bridge materialization syncing training layout into inference layout.

export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29501}

MODEL_SIZE=${MODEL_SIZE:-0.03B}
SEED=${SEED:-1337}
SEP_SIZE=${SEP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
PROMPT_LEN=${PROMPT_LEN:-17}
GBS=${GBS:-64}
TOTAL_BATCH_SIZE=$((GBS * PROMPT_LEN))
WARMUP_STEPS=${WARMUP_STEPS:-1}
MAX_LR=${MAX_LR:-6e-5}
MIN_LR=${MIN_LR:-6e-6}
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/debug_rl_${MODEL_SIZE}"}

RL_MAX_NEW_TOKENS=${RL_MAX_NEW_TOKENS:-8}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-2}
RL_TEMPERATURE=${RL_TEMPERATURE:-1.0}
RL_TOP_K=${RL_TOP_K:-0}
RL_TOP_P=${RL_TOP_P:-0.0}
RL_REWARD_TARGET_TOKEN_ID=${RL_REWARD_TARGET_TOKEN_ID:-0}
RL_CLIP_RANGE=${RL_CLIP_RANGE:-0.2}
RL_KL_COEF=${RL_KL_COEF:-0.0}
RL_ROLLOUT_SHARD_QKV=${RL_ROLLOUT_SHARD_QKV:-1}

NUM_LAYER=6
NUM_ATTENTION_HEADS=8
NUM_KEY_VALUE_HEADS=2
HIDDEN_SIZE=512
INTERMEDIATE_SIZE=2048

case "$MODEL_SIZE" in
  0.03B|0.03b)
    ;;
  0.1B|0.1b)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    ;;
  0.25B|0.25b)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=32
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=1024
    INTERMEDIATE_SIZE=4096
    ;;
  *)
    echo "Unknown MODEL_SIZE='$MODEL_SIZE'. Supported for RL debug: 0.03B, 0.1B, 0.25B" >&2
    exit 1
    ;;
esac

ARGS=(
  --seed "$SEED"
  --use_mock_data
  --mock_data_num_samples 1280
  --log_dir "$LOG_DIR"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --batch_size "$BATCH_SIZE"
  --seq_len "$PROMPT_LEN"
  --max_epochs 1
  --max_lr "$MAX_LR"
  --min_lr "$MIN_LR"
  --warmup_steps "$WARMUP_STEPS"
  --weight_decay 0.0
  --grad_clip_value 1.0
  --debug
  --deterministic
  --sep_size "$SEP_SIZE"
  --backend nccl
  --block_size 4096
  --vocab_size 50304
  --num_layer "$NUM_LAYER"
  --num_attention_heads "$NUM_ATTENTION_HEADS"
  --num_key_value_heads "$NUM_KEY_VALUE_HEADS"
  --hidden_size "$HIDDEN_SIZE"
  --intermediate_size "$INTERMEDIATE_SIZE"
  --tied_lm_head
  --dropout 0.0
  --rl_max_new_tokens "$RL_MAX_NEW_TOKENS"
  --rl_group_size "$RL_GROUP_SIZE"
  --rl_temperature "$RL_TEMPERATURE"
  --rl_top_k "$RL_TOP_K"
  --rl_top_p "$RL_TOP_P"
  --rl_reward_target_token_id "$RL_REWARD_TARGET_TOKEN_ID"
  --rl_clip_range "$RL_CLIP_RANGE"
  --rl_kl_coef "$RL_KL_COEF"
)

if [ "$RL_ROLLOUT_SHARD_QKV" -eq 1 ]; then
  ARGS+=(--rl_rollout_shard_qkv)
fi

torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$NUM_GPUS" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  scripts/debug/rl.py "${ARGS[@]}"
