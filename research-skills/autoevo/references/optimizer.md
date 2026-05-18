# Optimizer Research

Use this reference for optimizer, parameter-group, and training-stability experiments.

## Useful Entry Points

- `tinytron/training/trainer.py::_init_optimizer`: optimizer selection and parameter grouping.
- `tinytron/optim/muon.py`: Muon implementation.
- `tinytron/optim/__init__.py`: optimizer exports.
- `tinytron/training/config.py` and `tinytron/training/arguments.py`: CLI and config wiring.
- `tinytron/model/gpt.py::clip_grad_norm`: gradient clipping, including expert-local gradient handling.
- `tinytron/distributed/zero1/distributed_optimizer.py`: ZeRO-1 state partitioning and parameter broadcast.

## Good Experiment Shapes

- AdamW versus Muon under the same architecture and token budget.
- Separate router, expert, attention, MLP, embedding, and LM-head parameter groups.
- Router-specific learning rate or weight decay in MoE models.
- Expert-specific optimizer settings for sparse activation.
- Gradient clipping variants: global norm, expert-aware norm, or module-specific clipping.
- Warmup and cosine schedule sensitivity for dense versus MoE runs.

## Guardrails

- Keep optimizer state checkpoint compatibility in mind when changing param groups.
- With ZeRO-1 enabled, every rank must agree on parameter ordering and group structure.
- Report optimizer changes together with LR, warmup, batch size, grad accumulation, and clipping.
- Avoid changing architecture and optimizer in the same experiment unless the research question requires it.

## Minimal Validation

- Build `Config` from CLI args to confirm new flags land in the right dataclass.
- Run compile checks.
- If possible, run a tiny mock-data train step with `--optimizer adam` and the new variant.
