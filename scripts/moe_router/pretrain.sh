#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

EXP_NAME=${EXP_NAME:-full_observation_oracle_assisted}
LOG_DIR=${LOG_DIR:-scripts/moe_router/runs/full_observation_oracle_assisted}

SEED=${SEED:-1337}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-500} # ~2%
MAX_LR=${MAX_LR:-3.6e-3}
MIN_LR=${MIN_LR:-6e-5}

SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-8}
GBS=${GBS:-1024}
SEP_SIZE=${SEP_SIZE:-1}

STREAMING_DATA_DIR=${STREAMING_DATA_DIR:-./data/fineweb-edu-sample-10BT}
NUM_WORKERS=${NUM_WORKERS:-2}
PIN_MEMORY=${PIN_MEMORY:-1}
STREAMING_SHUFFLE=${STREAMING_SHUFFLE:-1}
STREAMING_STRICT=${STREAMING_STRICT:-1}
STREAMING_GLOBAL_SKIP_BATCHES=${STREAMING_GLOBAL_SKIP_BATCHES:-0}

USE_COMPILE=${USE_COMPILE:-1}
DEBUG=${DEBUG:-0}
DETER_MODE=${DETER_MODE:-0}
MODEL_SIZE=${MODEL_SIZE:-4B-A0.5B}

DO_SAVE=${DO_SAVE:-0}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-500}
EXPERT_WARMUP_STEPS=${EXPERT_WARMUP_STEPS:-23875}
ROUTER_TRAINING_STEPS=${ROUTER_TRAINING_STEPS:-0}
WARMUP_ROUTING_STRATEGY=${WARMUP_ROUTING_STRATEGY:-measurement_topk}
MEASUREMENT_TOPK_UPDATES_MODEL=${MEASUREMENT_TOPK_UPDATES_MODEL:-1}
ROUTER_RANKING_LOSS_WEIGHT=${ROUTER_RANKING_LOSS_WEIGHT:-0}

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

PARALLELISM_ARGS="\
 --sep_size $SEP_SIZE \
 --use_distributed_optimizer \
"

case $MODEL_SIZE in
    "4B-A0.5B"|"4b-a0.5b"|"4b_a0.5b")
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
          --num_experts 64 \
          --num_experts_per_tok 4 \
          --moe_intermediate_size 768 \
        "
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE" >&2
        echo "This oracle experiment currently supports MODEL_SIZE=4B-A0.5B." >&2
        exit 1
        ;;
esac

ROUTER_EXPERIMENT_ARGS="\
 --expert_warmup_steps $EXPERT_WARMUP_STEPS \
 --router_training_steps $ROUTER_TRAINING_STEPS \
 --warmup_routing_strategy $WARMUP_ROUTING_STRATEGY \
 --router_ranking_loss_weight $ROUTER_RANKING_LOSS_WEIGHT \
"
if [ $MEASUREMENT_TOPK_UPDATES_MODEL -eq 1 ]; then
 ROUTER_EXPERIMENT_ARGS="$ROUTER_EXPERIMENT_ARGS --measurement_topk_updates_model"
else
 ROUTER_EXPERIMENT_ARGS="$ROUTER_EXPERIMENT_ARGS --no-measurement_topk_updates_model"
fi

CMD="torchrun $DISTRIBUTED_ARGS scripts/moe_router/pretrain.py $TRAINING_ARGS $PARALLELISM_ARGS $MODEL_ARGS $ROUTER_EXPERIMENT_ARGS"

echo "Running command:"
echo "$CMD"

eval $CMD
