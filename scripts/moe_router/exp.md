## Baseline

Normal MoE training with the learned router from the first step. No
full-observation measurement pass and no oracle-assisted forced routing are
used.

```bash
MODEL_SIZE=4B-A0.5B \
EXP_NAME=baseline_normal_moe \
LOG_DIR=scripts/moe_router/runs \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=0 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```

## Full-Observation Oracle-Assisted Training

This experiment runs an expert-warmup phase where each MoE layer first observes
all experts, chooses the oracle top-k experts from the reference-output gradient,
and then updates the model with those forced top-k experts. The router is frozen
during this warmup. Set `ROUTER_TRAINING_STEPS` to a positive value to run a
second router-only phase over the training data with all non-router parameters
frozen.

```bash
MODEL_SIZE=4B-A0.5B \
EXP_NAME=full_observation_oracle_assisted \
LOG_DIR=scripts/moe_router/runs \
EXPERT_WARMUP_STEPS=23875 \
ROUTER_TRAINING_STEPS=0 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=1 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```

## Full-Observation Oracle-Assisted Training + Router-Only Training

This experiment first runs the same full-observation oracle-assisted expert
training phase, then makes one additional pass to train only the learned router.
All non-router parameters are frozen during the router-only phase.

```bash
MODEL_SIZE=4B-A0.5B \
EXP_NAME=full_observation_oracle_assisted_router_only \
LOG_DIR=scripts/moe_router/runs \
EXPERT_WARMUP_STEPS=23875 \
ROUTER_TRAINING_STEPS=23875 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=1 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```
