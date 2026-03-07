#!/bin/bash

# ================================
# Torch Distributed Training Script
# ================================
#
# For multi-node training, set these environment variables:
# NUM_NODES: number of nodes (default: 1)
# NUM_GPUS: number of GPUs per node (default: 8)
# NODE_RANK: rank of this node, 0 for master (default: 0)
# MASTER_ADDR: IP address of the master node (default: localhost)
# MASTER_PORT: port for communication (default: 29500)
#
# Example for 2 nodes:
# Node 0 (master, IP: 192.168.1.100):
# NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash scripts/streaming_gpt_0.25b/pretrain.sh
#
# Node 1:
# NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash scripts/streaming_gpt_0.25b/pretrain.sh

set -euo pipefail

# ----------------
# Multi-node config
# ----------------
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# ----------------
# Custom config
# ----------------
EXP_NAME=${EXP_NAME:-streaming_gpt_0.25b}
LOG_DIR=${LOG_DIR:-./log}

SEED=${SEED:-1337}

LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500}
MAX_LR=${MAX_LR:-3.6e-3}
MIN_LR=${MIN_LR:-6e-5}

SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-8}         # per process batch size before SEP adjustment
GBS=${GBS:-1024}                   # global batch size in number of sequences
SEP_SIZE=${SEP_SIZE:-1}

# Streaming dataset path (processed by Streaming-Dataloader)
STREAMING_DATA_DIR=${STREAMING_DATA_DIR:-./data/fineweb-edu-sample-10BT}

# Data loader
NUM_WORKERS=${NUM_WORKERS:-2}
PIN_MEMORY=${PIN_MEMORY:-1}
STREAMING_SHUFFLE=${STREAMING_SHUFFLE:-1}
STREAMING_STRICT=${STREAMING_STRICT:-1}
STREAMING_GLOBAL_SKIP_BATCHES=${STREAMING_GLOBAL_SKIP_BATCHES:-0}

# Runtime switches
USE_COMPILE=${USE_COMPILE:-1}
DEBUG=${DEBUG:-0}
DETER_MODE=${DETER_MODE:-0}

# Save / resume
DO_SAVE=${DO_SAVE:-1}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-500}
RESUME=${RESUME:-0}
RESUME_PATH=${RESUME_PATH:-}

# ----------------
# Derived config
# ----------------
TOTAL_BATCH_SIZE=$(($GBS * $SEQ_LEN))
BATCH_SIZE_PER_DP_RANK=$(($BATCH_SIZE * $SEP_SIZE))

DISTRIBUTED_ARGS="\
 --nnodes=$NUM_NODES \
 --nproc_per_node=$NUM_GPUS \
 --node_rank=$NODE_RANK \
 --master_addr=$MASTER_ADDR \
 --master_port=$MASTER_PORT \
"

TRAINING_ARGS="\
 --exp_name $EXP_NAME \
 --seed $SEED \
 --log_dir $LOG_DIR \
 --total_batch_size $TOTAL_BATCH_SIZE \
 --batch_size $BATCH_SIZE_PER_DP_RANK \
 --seq_len $SEQ_LEN \
 --max_lr $MAX_LR \
 --min_lr $MIN_LR \
 --weight_decay 0.1 \
 --grad_clip_value 1.0 \
 --warmup_steps $LR_WARMUP_STEPS \
 --max_epochs 1 \
 --num_workers $NUM_WORKERS \
 --streaming_data_dir $STREAMING_DATA_DIR \
 --streaming_global_skip_batches $STREAMING_GLOBAL_SKIP_BATCHES \
"

if [ $PIN_MEMORY -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --pin_memory"
fi

if [ $STREAMING_SHUFFLE -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --streaming_shuffle"
fi

if [ $STREAMING_STRICT -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --streaming_strict"
fi

if [ $DEBUG -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --debug"
fi

if [ $USE_COMPILE -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --use_compile"
fi

if [ $DETER_MODE -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --deterministic"
fi

if [ $DO_SAVE -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --do_save --save_every_steps $SAVE_EVERY_STEPS"
fi

if [ $RESUME -eq 1 ]; then
 TRAINING_ARGS="$TRAINING_ARGS --do_resume"
 if [ -n "$RESUME_PATH" ]; then
  TRAINING_ARGS="$TRAINING_ARGS --resume_path $RESUME_PATH"
 fi
fi

PARALLELISM_ARGS="\
 --sep_size $SEP_SIZE \
 --use_distributed_optimizer \
"

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

CMD="torchrun $DISTRIBUTED_ARGS pretrain.py $TRAINING_ARGS $PARALLELISM_ARGS $MODEL_ARGS"

echo "Running command:"
echo "$CMD"

eval $CMD