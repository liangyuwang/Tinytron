#!/bin/bash
set -euo pipefail

# Tinytron inference debug launcher.
#
# Usage examples:
#   # 1) smoke test with random weights
#   bash scripts/debug/inference.sh
#
#   # 2) run checkpoint inference with a 0.1B preset
#   MODEL_SIZE=0.1B CKPT_PATH=/path/to/00500_model.pt \
#   bash scripts/debug/inference.sh
#
# Preset model sizes:
#   0.03B           : 6L,  8QH,   2KVH, hidden=512,  ffn=2048
#   0.1B            : 12L, 16QH,  4KVH, hidden=768,  ffn=3072
#   0.25B           : 12L, 32QH,  4KVH, hidden=1024, ffn=4096
#   1B              : 24L, 64QH,  8KVH, hidden=2048, ffn=8192
#   1.3B            : 24L, 16QH,  8KVH, hidden=2048, ffn=6144
#   7B              : 32L, 128QH, 16KVH, hidden=4096, ffn=16384
#   13B             : 40L, 40QH,  8KVH, hidden=5120, ffn=13824
#   30B             : 60L, 52QH,  8KVH, hidden=6656, ffn=17920
#   70B             : 80L, 64QH,  8KVH, hidden=8192, ffn=28672
#   0.17B-A0.1B     : 12L, 16QH, 4KVH, hidden=768, ffn=3072 + MoE
#   0.3B-A0.17B     : 12L, 32QH, 4KVH, hidden=768, ffn=3072 + MoE
#   0.7B-A0.25B     : 24L, 32QH, 4KVH, hidden=1024, ffn=4096 + MoE
#   2.7B-A1B        : 24L, 64QH, 8KVH, hidden=2048, ffn=8192 + MoE
#   14B-A4.5B       : 32L, 128QH, 16KVH, hidden=4096, ffn=16384 + MoE
#   104B-A4.5B      : 14B-A4.5B with 64 experts

MODEL_SIZE=${MODEL_SIZE:-0.25B}
CKPT_PATH=${CKPT_PATH:-}
PROMPT_TOKEN_IDS=${PROMPT_TOKEN_IDS:-1,2,3,4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-}
EOS_TOKEN_ID=${EOS_TOKEN_ID:-}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}
SEP_SIZE=${SEP_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$SEP_SIZE}
BACKEND=${BACKEND:-nccl}
INIT_METHOD=${INIT_METHOD:-env://}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# defaults (0.25B)
NUM_LAYER=12
NUM_ATTENTION_HEADS=32
NUM_KEY_VALUE_HEADS=4
HIDDEN_SIZE=1024
INTERMEDIATE_SIZE=4096
USE_MOE=0
NUM_EXPERTS=8
NUM_EXPERTS_PER_TOK=2
MOE_INTERMEDIATE_SIZE=768

case "${MODEL_SIZE}" in
  0.03B|0.03b)
    NUM_LAYER=6
    NUM_ATTENTION_HEADS=8
    NUM_KEY_VALUE_HEADS=2
    HIDDEN_SIZE=512
    INTERMEDIATE_SIZE=2048
    ;;
  0.1B|0.1b)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    ;;
  0.25B|0.25b)
    ;;
  1B|1b)
    NUM_LAYER=24
    NUM_ATTENTION_HEADS=64
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=8192
    ;;
  1.3B|1.3b)
    NUM_LAYER=24
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=6144
    ;;
  7B|7b)
    NUM_LAYER=32
    NUM_ATTENTION_HEADS=128
    NUM_KEY_VALUE_HEADS=16
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=16384
    ;;
  13B|13b)
    NUM_LAYER=40
    NUM_ATTENTION_HEADS=40
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=5120
    INTERMEDIATE_SIZE=13824
    ;;
  30B|30b)
    NUM_LAYER=60
    NUM_ATTENTION_HEADS=52
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=6656
    INTERMEDIATE_SIZE=17920
    ;;
  70B|70b)
    NUM_LAYER=80
    NUM_ATTENTION_HEADS=64
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=8192
    INTERMEDIATE_SIZE=28672
    ;;
  0.17B-A0.1B|0.17b-a0.1b|0.17b_a0.1b)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    USE_MOE=1
    ;;
  0.3B-A0.17B|0.3b-a0.17b|0.3b_a0.17b)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=32
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    USE_MOE=1
    ;;
  0.7B-A0.25B|0.7b-a0.25b|0.7b_a0.25b)
    NUM_LAYER=24
    NUM_ATTENTION_HEADS=32
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=1024
    INTERMEDIATE_SIZE=4096
    USE_MOE=1
    MOE_INTERMEDIATE_SIZE=1024
    ;;
  2.7B-A1B|2.7b-a1b|2.7b_a1b)
    NUM_LAYER=24
    NUM_ATTENTION_HEADS=64
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=8192
    USE_MOE=1
    MOE_INTERMEDIATE_SIZE=2048
    ;;
  14B-A4.5B|14b-a4.5b|14b_a4.5b)
    NUM_LAYER=32
    NUM_ATTENTION_HEADS=128
    NUM_KEY_VALUE_HEADS=16
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=16384
    USE_MOE=1
    MOE_INTERMEDIATE_SIZE=4096
    ;;
  104B-A4.5B|104b-a4.5b|104b_a4.5b)
    NUM_LAYER=32
    NUM_ATTENTION_HEADS=128
    NUM_KEY_VALUE_HEADS=16
    HIDDEN_SIZE=4096
    INTERMEDIATE_SIZE=16384
    USE_MOE=1
    NUM_EXPERTS=64
    MOE_INTERMEDIATE_SIZE=4096
    ;;
  *)
    echo "Unknown MODEL_SIZE='${MODEL_SIZE}'. Use one of: 0.03B, 0.1B, 0.25B, 1B, 1.3B, 7B, 13B, 30B, 70B, 0.17B-A0.1B, 0.3B-A0.17B, 0.7B-A0.25B, 2.7B-A1B, 14B-A4.5B, 104B-A4.5B"
    exit 1
    ;;
esac

ARGS=(
  --prompt_token_ids "${PROMPT_TOKEN_IDS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_k "${TOP_K}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --backend "${BACKEND}"
  --init_method "${INIT_METHOD}"
  --sep_size "${SEP_SIZE}"
  --num_layer "${NUM_LAYER}"
  --num_attention_heads "${NUM_ATTENTION_HEADS}"
  --num_key_value_heads "${NUM_KEY_VALUE_HEADS}"
  --hidden_size "${HIDDEN_SIZE}"
  --intermediate_size "${INTERMEDIATE_SIZE}"
  --tied_lm_head
)

if [[ "${USE_MOE}" == "1" ]]; then
  ARGS+=(
    --use_moe
    --num_experts "${NUM_EXPERTS}"
    --num_experts_per_tok "${NUM_EXPERTS_PER_TOK}"
    --moe_intermediate_size "${MOE_INTERMEDIATE_SIZE}"
  )
fi

if [[ -n "${TOP_P}" ]]; then
  ARGS+=(--top_p "${TOP_P}")
fi

if [[ -n "${EOS_TOKEN_ID}" ]]; then
  ARGS+=(--eos_token_id "${EOS_TOKEN_ID}")
fi

if [[ -n "${CKPT_PATH}" ]]; then
  ARGS+=(--checkpoint_path "${CKPT_PATH}")
else
  ARGS+=(--init_from_scratch)
fi

echo "Running inference with MODEL_SIZE=${MODEL_SIZE}, DEVICE=${DEVICE}, DTYPE=${DTYPE}, SEP_SIZE=${SEP_SIZE}"
if [[ "${SEP_SIZE}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/debug/inference.py "${ARGS[@]}"
else
  python scripts/debug/inference.py "${ARGS[@]}"
fi
