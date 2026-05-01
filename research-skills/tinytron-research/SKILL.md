---
name: tinytron-research
description: Use when planning, implementing, or evaluating research experiments in the Tinytron repository, especially model architecture, inference strategy, KV-cache design, optimizer behavior, reinforcement-learning alignment, and small-scale training measurement. Treat distributed parallelism as a fixed support layer unless the user explicitly asks to redesign it.
metadata:
  short-description: Tinytron research workflows
---

# Tinytron Research

Use this skill for research work in the Tinytron repo. Tinytron is best treated as a transparent GPT experimentation base: distributed code supports multi-GPU execution, but the main research surfaces are model architecture, inference, KV cache, optimizer choices, RL alignment, and experiment measurement.

## Default Stance

- Keep distributed topology and process-group strategy fixed unless the user explicitly asks for distributed-systems work.
- Prefer local, focused changes in `tinytron/model`, `tinytron/inference`, `tinytron/optim`, or the relevant trainer hook.
- Treat `tinytron/bridge` as model-state layout infrastructure, not a primary research surface. Modify it only for explicit layout, checkpoint resharding, or model-state transfer work.
- Start with a small correctness check, usually mock data or a minimal inference run, before proposing long experiments.
- Separate research changes from cleanup refactors. If cleanup is needed, keep it mechanical and explain why it supports the experiment.
- When reporting results, distinguish measured data from expected behavior or hypotheses.

## Pick The Reference

Load only the reference needed for the current task:

- For attention, RoPE, GQA, MLP, MoE, norm, or loss changes, read `references/architecture.md`.
- For generation, prefill/decode, sampling, paged KV cache, cache layout, or cache quantization, read `references/inference-kv-cache.md`.
- For AdamW, Muon, parameter groups, router/expert learning rates, or gradient clipping, read `references/optimizer.md`.
- For RLTrainer, GRPO/RLOO/PPO/DPO design, rollout batches, rewards, actor-rollout sync, or bridge-backed live layout transfer, read `references/rl.md`.
- For designing runs, comparing baselines, choosing metrics, or writing result summaries, read `references/experiment-protocol.md`.

## Research Workflow

1. State the hypothesis in one sentence.
2. Identify the smallest code surface that can test it.
3. Name the baseline config and the changed config.
4. Implement the change behind a config flag when the behavior should be comparable.
5. Run a quick validation path before larger runs.
6. Record loss, tok/sec, MFU, memory if available, command, commit/diff, and caveats.

## Tinytron Map

- Model path: `tinytron/model/gpt.py`, `tinytron/model/modules/attn.py`, `tinytron/model/modules/mlp.py`, `tinytron/model/modules/norm.py`, `tinytron/model/modules/loss.py`.
- Inference path: `tinytron/inference/engine.py`, `tinytron/inference/cache.py`, `tinytron/inference/sampler.py`, plus attention cache handling in `tinytron/model/modules/attn.py`.
- Checkpoint policy: `tinytron/training/checkpoint.py` for training save/resume and `tinytron/inference/checkpoint.py` for inference loading.
- Parameter-layout support: `tinytron/bridge/`. Use it as infrastructure for layout metadata, shard planning, and file-based model resharding; do not treat it as the default place for architecture, inference, optimizer, or data experiments.
- RL path: `tinytron/rl/` for rollout batch types, logprob helpers, losses, actor-rollout sync, `RLTrainer`, and algorithm subclasses such as `GRPOTrainer`; `scripts/debug/rl.py` and `scripts/debug/rl.sh` are the stage-1 debug entry points.
- Optimizer path: `tinytron/optim/`, `tinytron/training/trainer.py::_init_optimizer`, and ZeRO-1 wrapper behavior in `tinytron/distributed/zero1/distributed_optimizer.py`.
- Experiment entry points: `scripts/debug/pretrain.py`, `scripts/debug/pretrain.sh`, `scripts/example/pretrain.py`, `scripts/example/pretrain.sh`, `scripts/debug/inference.py`, `scripts/debug/inference.sh`, `scripts/debug/rl.py`, `scripts/debug/rl.sh`.
