## Full-Observation Oracle-Assisted Training

```bash
MODEL_SIZE=3B-A0.6B \
LOG_DIR=scripts/moe_router/runs/full_observation_oracle_assisted \
EXPERT_WARMUP_STEPS=23875 \
ROUTER_TRAINING_STEPS=0 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
MEASUREMENT_TOPK_UPDATES_MODEL=1 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```

This experiment runs an expert-warmup phase where each MoE layer first observes
all experts, chooses the oracle top-k experts from the reference-output gradient,
and then updates the model with those forced top-k experts. The router is frozen
during this warmup; no router-ranking supervision or router-training phase is
enabled here.
