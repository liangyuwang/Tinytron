#!/bin/bash
# ================================
# Torch Distributed Training Script
# ================================
# 
# For multi-node training, set these environment variables:
#   NUM_NODES: number of nodes (default: 1)
#   NUM_GPUS: number of GPUs per node (default: 8)
#   NODE_RANK: rank of this node, 0 for master (default: 0)
#   MASTER_ADDR: IP address of the master node (default: localhost)
#   MASTER_PORT: port for communication (default: 29500)
#
# Example for 2 nodes:
#   Node 0 (master, IP: 192.168.1.100):
#     NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash scripts/debug/pretrain.sh
#   Node 1:
#     NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash scripts/debug/pretrain.sh
#

export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# Multi-node configuration (can be overridden by environment variables)
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Custom config
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-2}
MAX_LR=${MAX_LR:-6e-4}
MIN_LR=${MIN_LR:-6e-5}

SEED=${SEED:-1337}

BATCH_SIZE=${BATCH_SIZE:-8}
SEQ_LEN=${SEQ_LEN:-4096}
GBS=${GBS:-1024}
TOTAL_BATCH_SIZE=$(($GBS * $SEQ_LEN))

SEP_SIZE=${SEP_SIZE:-1}
BATCH_SIZE_PER_DP_RANK=$(($BATCH_SIZE * $SEP_SIZE))
USE_COMPILE=${USE_COMPILE:-1}

DEBUG=${DEBUG:-1}
DETER_MODE=${DETER_MODE:-1} # deter mode for precision alignment
MODEL_SIZE=${MODEL_SIZE:-0.25B}

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

EXP_NAME=${EXP_NAME:-"debug_gpt_${MODEL_SIZE}"}
TRAINING_ARGS="\
  --exp_name $EXP_NAME \
  --seed $SEED \
  --dataset_path ... \
  --use_mock_data \
  --mock_data_num_samples 12800 \
  --log_dir ./log \
  --total_batch_size $TOTAL_BATCH_SIZE \
  --batch_size $BATCH_SIZE_PER_DP_RANK \
  --seq_len $SEQ_LEN \
  --max_lr $MAX_LR \
  --min_lr $MIN_LR \
  --weight_decay 0.1 \
  --grad_clip_value 1.0 \
  --warmup_steps $LR_WARMUP_STEPS \
  --max_epochs 1 \
  --do_save \
  --save_every_steps 500 \
"
if [ $DEBUG -eq 1 ]; then
  TRAINING_ARGS="$TRAINING_ARGS --debug"
fi
if [ $USE_COMPILE -eq 1 ]; then
  TRAINING_ARGS="$TRAINING_ARGS --use_compile"
fi
if [ $DETER_MODE -eq 1 ]; then
  TRAINING_ARGS="$TRAINING_ARGS --deterministic"
fi

PARALLELISM_ARGS="\
  --sep_size $SEP_SIZE \
  --use_distributed_optimizer \
"

case $MODEL_SIZE in
    "0.03B"|"0.03b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 6 \
          --num_attention_heads 8 \
          --num_key_value_heads 2 \
          --hidden_size 512 \
          --intermediate_size 2048 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "0.1B"|"0.1b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 12 \
          --num_attention_heads 16 \
          --num_key_value_heads 4 \
          --hidden_size 768 \
          --intermediate_size 3072 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "0.25B"|"0.25b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 12 \
          --num_attention_heads 32 \
          --num_key_value_heads 4 \
          --hidden_size 1024 \
          --intermediate_size 4096 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "1B"|"1b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 24 \
          --num_attention_heads 64 \
          --num_key_value_heads 8 \
          --hidden_size 2048 \
          --intermediate_size 8192 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "1.3B"|"1.3b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 24 \
          --num_attention_heads 16 \
          --num_key_value_heads 8 \
          --hidden_size 2048 \
          --intermediate_size 6144 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "7B"|"7b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 32 \
          --num_attention_heads 128 \
          --num_key_value_heads 16 \
          --hidden_size 4096 \
          --intermediate_size 16384 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "13B"|"13b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 40 \
          --num_attention_heads 40 \
          --num_key_value_heads 8 \
          --hidden_size 5120 \
          --intermediate_size 13824 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "30B"|"30b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 60 \
          --num_attention_heads 52 \
          --num_key_value_heads 8 \
          --hidden_size 6656 \
          --intermediate_size 17920 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "70B"|"70b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 80 \
          --num_attention_heads 64 \
          --num_key_value_heads 8 \
          --hidden_size 8192 \
          --intermediate_size 28672 \
          --tied_lm_head \
          --dropout 0.0 \
        "
        ;;
    "0.17B-A0.1B"|"0.17b-a0.1b"|"0.17b_a0.1b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 12 \
          --num_attention_heads 16 \
          --num_key_value_heads 4 \
          --hidden_size 768 \
          --intermediate_size 3072 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 8 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 768 \
        "
        ;;
    "0.3B-A0.17B"|"0.3b-a0.17b"|"0.3b_a0.17b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 12 \
          --num_attention_heads 32 \
          --num_key_value_heads 4 \
          --hidden_size 768 \
          --intermediate_size 3072 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 8 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 768 \
        "
        ;;
    "0.7B-A0.25B"|"0.7b-a0.25b"|"0.7b_a0.25b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 24 \
          --num_attention_heads 32 \
          --num_key_value_heads 4 \
          --hidden_size 1024 \
          --intermediate_size 4096 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 8 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 1024 \
        "
        ;;
    "2.7B-A1B"|"2.7b-a1b"|"2.7b_a1b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 24 \
          --num_attention_heads 64 \
          --num_key_value_heads 8 \
          --hidden_size 2048 \
          --intermediate_size 8192 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 8 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 2048 \
        "
        ;;
    "14B-A4.5B"|"14b-a4.5b"|"14b_a4.5b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 32 \
          --num_attention_heads 128 \
          --num_key_value_heads 16 \
          --hidden_size 4096 \
          --intermediate_size 16384 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 8 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 4096 \
        "
        ;;
    "104B-A4.5B"|"104b-a4.5b"|"104b_a4.5b")
        MODEL_ARGS="\
          --block_size 4096 \
          --vocab_size 50304 \
          --num_layer 32 \
          --num_attention_heads 128 \
          --num_key_value_heads 16 \
          --hidden_size 4096 \
          --intermediate_size 16384 \
          --tied_lm_head \
          --dropout 0.0 \
          --use_moe \
          --num_experts 64 \
          --num_experts_per_tok 2 \
          --moe_intermediate_size 4096 \
        "
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE" >&2
        echo "Supported MODEL_SIZE values: 0.03B, 0.1B, 0.25B, 1B, 1.3B, 7B, 13B, 30B, 70B, 0.17B-A0.1B, 0.3B-A0.17B, 0.7B-A0.25B, 2.7B-A1B, 14B-A4.5B, 104B-A4.5B" >&2
        exit 1
        ;;
esac


torchrun $DISTRIBUTED_ARGS scripts/debug/pretrain.py $TRAINING_ARGS $PARALLELISM_ARGS $MODEL_ARGS
