---
name: tinytron-autoevo
description: Use when designing or running AI-assisted auto-evolution research loops in Tinytron: turning hypotheses into EvolutionSpec records, isolating hypothesis branches/worktrees, implementing small code mutations, evaluating baseline-vs-candidate results, recording evidence, and planning the next experiment. Also use for Tinytron model, inference, optimizer, RL, and small-scale measurement research.
metadata:
  short-description: Tinytron auto-evolution research
---

# Tinytron Autoevo

Use this skill for auto-evolution research work in the Tinytron repo. Tinytron is best treated as a compact, transparent GPT experimentation base optimized for research iteration throughput: small code surface, easy attribution, cheap mutation, and fast evidence accumulation.

The repo code should enforce protocol, isolation, and reproducibility. The agent should provide research judgment: hypotheses, minimal experiments, diagnosis, and next-step planning.

## Default Stance

- Keep distributed topology and process-group strategy fixed unless the user explicitly asks for distributed-systems work.
- Prefer local, focused changes in `tinytron/model`, `tinytron/inference`, `tinytron/optim`, or the relevant trainer hook.
- Treat `tinytron/bridge` as model-state layout infrastructure, not a primary research surface. Modify it only for explicit layout, checkpoint resharding, or model-state transfer work.
- Treat `evolution/` as the protocol layer for structured experiment specs, translation artifacts, promotion gates, reports, and registries.
- Start with a small correctness check, usually mock data or a minimal inference run, before proposing long experiments.
- Separate research changes from cleanup refactors. If cleanup is needed, keep it mechanical and explain why it supports the experiment.
- When reporting results, distinguish measured data from expected behavior or hypotheses.
- Keep framework-development commits separate from hypothesis/experiment commits. Do not let automatic experiment branches pollute the user's active development branch.

## Pick The Reference

Load only the reference needed for the current task:

- For hypothesis lineage, git branch/worktree isolation, base/candidate/infra commits, or avoiding conflicts between repo development and research experiments, read `references/hypothesis-git.md`.
- For attention, RoPE, GQA, MLP, MoE, norm, or loss changes, read `references/architecture.md`.
- For generation, prefill/decode, sampling, paged KV cache, cache layout, or cache quantization, read `references/inference-kv-cache.md`.
- For AdamW, Muon, parameter groups, router/expert learning rates, or gradient clipping, read `references/optimizer.md`.
- For RLTrainer, GRPO/RLOO/PPO/DPO design, rollout batches, rewards, actor-rollout sync, or bridge-backed live layout transfer, read `references/rl.md`.
- For designing runs, comparing baselines, choosing metrics, or writing result summaries, read `references/experiment-protocol.md`.

## Autoevo Workflow

1. State the hypothesis in one sentence, including expected direction and primary metric.
2. Write or update an `EvolutionSpec` before implementation. The spec should capture objective, parent/candidate model, model changes, training budget, data, eval gate, evidence placeholders, and git metadata when available.
3. Pin a clean research base commit. If code will be mutated, use a dedicated experiment worktree/branch rather than the active development worktree.
4. Identify the smallest code surface that can test the hypothesis.
5. Implement the change behind a config flag when behavior should remain comparable.
6. Run a quick validation path before larger runs.
7. Evaluate candidate against baseline with the declared suites and promotion gate.
8. Record spec, artifact, command, commit/diff, metrics, decision, report, and caveats in the registry.
9. Plan the next experiment from evidence: promote, ablate, tune, fix implementation, scale budget, or archive.

## Hypothesis Model

- Prefer spec-first, git-backed research: `EvolutionSpec` owns the research intent, Git owns the implementation lineage, and the registry owns evidence.
- Treat a hypothesis family as a branch or lineage, and a concrete experiment as a spec plus candidate commit.
- Record at least `base_commit`, `candidate_commit`, `infra_commit`, `branch`, `worktree_path`, and `diff_hash` when executing an automated experiment.
- Promotion should not merge directly into `main`. Promote to a research namespace first, then let a human or higher-level agent decide whether to merge into normal development.

## Tinytron Map

- Model path: `tinytron/model/gpt.py`, `tinytron/model/modules/attn.py`, `tinytron/model/modules/mlp.py`, `tinytron/model/modules/norm.py`, `tinytron/model/modules/loss.py`.
- Inference path: `tinytron/inference/engine.py`, `tinytron/inference/cache.py`, `tinytron/inference/sampler.py`, plus attention cache handling in `tinytron/model/modules/attn.py`.
- Checkpoint policy: `tinytron/training/checkpoint.py` for training save/resume and `tinytron/inference/checkpoint.py` for inference loading.
- Parameter-layout support: `tinytron/bridge/`. Use it as infrastructure for layout metadata, shard planning, and file-based model resharding; do not treat it as the default place for architecture, inference, optimizer, or data experiments.
- RL path: `tinytron/rl/` for rollout batch types, logprob helpers, losses, actor-rollout sync, `RLTrainer`, and algorithm subclasses such as `GRPOTrainer`; `scripts/debug/rl.py` and `scripts/debug/rl.sh` are the stage-1 debug entry points.
- Optimizer path: `tinytron/optim/`, `tinytron/training/trainer.py::_init_optimizer`, and ZeRO-1 wrapper behavior in `tinytron/distributed/zero1/distributed_optimizer.py`.
- Experiment entry points: `scripts/debug/pretrain.py`, `scripts/debug/pretrain.sh`, `scripts/example/pretrain.py`, `scripts/example/pretrain.sh`, `scripts/debug/inference.py`, `scripts/debug/inference.sh`, `scripts/debug/rl.py`, `scripts/debug/rl.sh`.
