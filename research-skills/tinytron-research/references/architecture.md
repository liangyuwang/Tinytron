# Architecture Research

Use this reference for model architecture changes. Keep experiments narrow enough that a baseline and variant differ by one main idea.

## Useful Entry Points

- `tinytron/model/gpt.py`: block composition, residual path, loss integration, MoE auxiliary loss.
- `tinytron/model/modules/attn.py`: GQA, RoPE, causal attention, sequence-parallel attention reshaping, cache-aware attention.
- `tinytron/model/modules/mlp.py`: dense SwiGLU MLP, MoE router, expert weights, expert dispatch, inference-time local reduce path.
- `tinytron/model/modules/norm.py`: normalization implementation.
- `tinytron/model/modules/loss.py`: SP-aware cross entropy and MoE load-balancing loss.
- `tinytron/model/config.py`: model-level knobs and CLI wiring.

## Good Experiment Shapes

- GQA/MQA/MHA: vary `num_attention_heads` and `num_key_value_heads`, keeping hidden size and training budget fixed.
- RoPE variants: add explicit config fields for scaling behavior, then compare short-context parity before long-context runs.
- MoE routing: compare top-k, router noise, aux loss coefficient, shared experts, or capacity rules.
- MLP variants: compare SwiGLU to alternate activation or intermediate-size ratios.
- Norm variants: compare LayerNorm/RMSNorm or pre-norm variants with identical optimizer settings.

## Guardrails

- Keep tensor shapes explicit in comments only where shape movement is hard to infer.
- Preserve training and inference compatibility unless the experiment is explicitly training-only.
- If a new config changes checkpoint compatibility, document the default and expected migration behavior.
- Do not put architecture experiments in `tinytron/bridge`; bridge code is for parameter-layout metadata, shard planning, and checkpoint/inference reshard support.
- For MoE changes, check both single-rank and `sep_size > 1` paths when practical because EP behavior is tied to SEP groups.

## Minimal Validation

- Run `python -m compileall tinytron scripts`.
- Build a tiny config with mock data for training-path shape validation.
- For attention or MoE changes, exercise inference with random weights when possible.
