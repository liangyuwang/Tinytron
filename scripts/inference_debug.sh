#!/bin/bash
set -euo pipefail

# Tinytron inference debug launcher.
#
# Usage examples:
#   # 1) smoke test with random weights
#   bash scripts/inference_debug.sh
#
#   # 2) run checkpoint inference with a small preset
#   MODEL_SIZE=small CKPT_PATH=./log/debug_gpt_0.25b/00500_model.pt \
#   bash scripts/inference_debug.sh
#
# Preset model sizes:
#   tiny   : 6L,  8QH,  2KVH, hidden=512,  ffn=2048
#   small  : 12L, 16QH, 4KVH, hidden=768,  ffn=3072
#   base   : 12L, 32QH, 4KVH, hidden=1024, ffn=4096
#   large  : 24L, 32QH, 8KVH, hidden=2048, ffn=8192
#   moe-sm : small + MoE(8 experts, top-2, expert_ffn=768)

MODEL_SIZE=${MODEL_SIZE:-base}
CKPT_PATH=${CKPT_PATH:-}
PROMPT_TOKEN_IDS=${PROMPT_TOKEN_IDS:-1,2,3,4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-}
EOS_TOKEN_ID=${EOS_TOKEN_ID:-}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bf16}

# defaults (base)
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
  tiny)
    NUM_LAYER=6
    NUM_ATTENTION_HEADS=8
    NUM_KEY_VALUE_HEADS=2
    HIDDEN_SIZE=512
    INTERMEDIATE_SIZE=2048
    ;;
  small)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    ;;
  base)
    ;;
  large)
    NUM_LAYER=24
    NUM_ATTENTION_HEADS=32
    NUM_KEY_VALUE_HEADS=8
    HIDDEN_SIZE=2048
    INTERMEDIATE_SIZE=8192
    ;;
  moe-sm)
    NUM_LAYER=12
    NUM_ATTENTION_HEADS=16
    NUM_KEY_VALUE_HEADS=4
    HIDDEN_SIZE=768
    INTERMEDIATE_SIZE=3072
    USE_MOE=1
    ;;
  *)
    echo "Unknown MODEL_SIZE='${MODEL_SIZE}'. Use one of: tiny, small, base, large, moe-sm"
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

echo "Running inference with MODEL_SIZE=${MODEL_SIZE}, DEVICE=${DEVICE}, DTYPE=${DTYPE}"
python inference.py "${ARGS[@]}"
