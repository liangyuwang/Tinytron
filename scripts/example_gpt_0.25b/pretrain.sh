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
#     NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.100 bash scripts/example_gpt_0.25b/pretrain.sh
#   Node 1:
#     NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.100 bash scripts/example_gpt_0.25b/pretrain.sh
#

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
GBS=${GBS:-128}
TOTAL_BATCH_SIZE=$(($GBS * $SEQ_LEN))

SEP_SIZE=${SEP_SIZE:-1}
BATCH_SIZE_PER_DP_RANK=$(($BATCH_SIZE * $SEP_SIZE))
USE_COMPILE=${USE_COMPILE:-1}

DEBUG=${DEBUG:-1}
DETER_MODE=${DETER_MODE:-0} # deter mode for precision alignment

DISTRIBUTED_ARGS="\
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
"

EXP_NAME="debug_gpt_0.25b"
TRAINING_ARGS="\
  --exp_name $EXP_NAME \
  --seed 1337 \
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

torchrun $DISTRIBUTED_ARGS pretrain_example.py $TRAINING_ARGS $PARALLELISM_ARGS $MODEL_ARGS
