#!/bin/bash

set -euo pipefail

BASE_TAG="${BASE_TAG:-realcmp_both_$(date +%Y%m%d_%H%M%S)}"
PRECISION="${PRECISION:-fp32}"
DISABLE_TF32="${DISABLE_TF32:-1}"
DETER_MODE="${DETER_MODE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
HF_MAX_SAMPLES="${HF_MAX_SAMPLES:-200}"
LOG_DIR="${LOG_DIR:-./log}"

# Optional dataset knobs forwarded to pretrain_example.py
HF_DATASET_REPO="${HF_DATASET_REPO:-HuggingFaceFW/fineweb-edu}"
HF_DATASET_NAME="${HF_DATASET_NAME:-}"
HF_SPLIT="${HF_SPLIT:-train}"
HF_TOKENIZER="${HF_TOKENIZER:-gpt2}"
HF_MIN_CHARS="${HF_MIN_CHARS:-50}"
HF_ADD_EOS_TOKEN="${HF_ADD_EOS_TOKEN:-1}"
HF_SHUFFLE_BUFFER="${HF_SHUFFLE_BUFFER:-0}"

echo "========================================"
echo "RUN 1/2: DENSE (USE_MOE=0)"
echo "========================================"
RUN_TAG="${BASE_TAG}_dense" \
PRECISION="$PRECISION" \
DISABLE_TF32="$DISABLE_TF32" \
DETER_MODE="$DETER_MODE" \
USE_MOE=0 \
MAX_EPOCHS="$MAX_EPOCHS" \
MAX_STEPS="$MAX_STEPS" \
HF_MAX_SAMPLES="$HF_MAX_SAMPLES" \
HF_DATASET_REPO="$HF_DATASET_REPO" \
HF_DATASET_NAME="$HF_DATASET_NAME" \
HF_SPLIT="$HF_SPLIT" \
HF_TOKENIZER="$HF_TOKENIZER" \
HF_MIN_CHARS="$HF_MIN_CHARS" \
HF_ADD_EOS_TOKEN="$HF_ADD_EOS_TOKEN" \
HF_SHUFFLE_BUFFER="$HF_SHUFFLE_BUFFER" \
LOG_DIR="$LOG_DIR" \
OUTPUT_FIG="test_scripts/loss_alignment_real_data_dense.png" \
bash test_scripts/test_sp_ep_alignment_real_data.sh

echo "========================================"
echo "RUN 2/2: MOE (USE_MOE=1)"
echo "========================================"
RUN_TAG="${BASE_TAG}_moe" \
PRECISION="$PRECISION" \
DISABLE_TF32="$DISABLE_TF32" \
DETER_MODE="$DETER_MODE" \
USE_MOE=1 \
MAX_EPOCHS="$MAX_EPOCHS" \
MAX_STEPS="$MAX_STEPS" \
HF_MAX_SAMPLES="$HF_MAX_SAMPLES" \
HF_DATASET_REPO="$HF_DATASET_REPO" \
HF_DATASET_NAME="$HF_DATASET_NAME" \
HF_SPLIT="$HF_SPLIT" \
HF_TOKENIZER="$HF_TOKENIZER" \
HF_MIN_CHARS="$HF_MIN_CHARS" \
HF_ADD_EOS_TOKEN="$HF_ADD_EOS_TOKEN" \
HF_SHUFFLE_BUFFER="$HF_SHUFFLE_BUFFER" \
LOG_DIR="$LOG_DIR" \
OUTPUT_FIG="test_scripts/loss_alignment_real_data_moe.png" \
bash test_scripts/test_sp_ep_alignment_real_data.sh

echo "========================================"
echo "DONE"
echo "========================================"
echo "Dense figure: test_scripts/loss_alignment_real_data_dense.png"
echo "MoE figure:   test_scripts/loss_alignment_real_data_moe.png"
