```bash
MODEL_SIZE=0.17B-A0.1B \
MAX_STEPS=1000 \
LOG_DIR=scripts/moe_router/runs/baseline_normal_moe \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=0 \
DISABLE_PROBE_RANKING=1 \
bash scripts/moe_router/pretrain.sh
```

Equivalent form:

```bash
MODEL_SIZE=0.17B-A0.1B \
MAX_STEPS=1000 \
LOG_DIR=scripts/moe_router/runs/baseline_normal_moe \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=0 \
ROUTER_RANKING_LOSS_WEIGHT=0 \
bash scripts/moe_router/pretrain.sh
```

---

```bash
MODEL_SIZE=0.17B-A0.1B \
MAX_STEPS=1000 \
LOG_DIR=scripts/moe_router/runs/full_observation_ranking_oracle \
EXPERT_WARMUP_STEPS=0 \
ROUTER_TRAINING_STEPS=1000 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```

---

```bash
MODEL_SIZE=0.17B-A0.1B \
MAX_STEPS=1000 \
LOG_DIR=scripts/moe_router/runs/three_phase \
EXPERT_WARMUP_STEPS=100 \
ROUTER_TRAINING_STEPS=100 \
WARMUP_ROUTING_STRATEGY=measurement_topk \
ROUTER_RANKING_LOSS_WEIGHT=0.01 \
bash scripts/moe_router/pretrain.sh
```
