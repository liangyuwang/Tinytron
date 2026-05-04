## 1. Normal MoE Baseline

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/baseline_normal_moe \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=0 \
DISABLE_PROBE_RANKING=1 \
bash scripts/moe_router/pretrain.sh
```

Equivalent:

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/baseline_normal_moe \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=0 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```

## 2. Full-Observation Oracle-Assisted Training

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/full_observation_oracle_assisted \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=1000 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=1 \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```

## 3. Full-Observation Router-Supervision Only

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/full_observation_router_supervision \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=1000 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=0 \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```

## 4. Two-Phase Training

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/two_phase \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=1100 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=0 \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```

## 5. Three-Phase Training

```bash
MODEL_SIZE=4B-A0.5B \
LOG_DIR=scripts/moe_router/runs/three_phase \
EXPERT_WARMUP_STEPS=100 \
ROUTER_TRAINING_STEPS=1000 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=0 \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```
