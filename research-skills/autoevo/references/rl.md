# RL Research

Use this reference for RL alignment work in Tinytron. Keep the first implementation path simple: synchronous rollout and update on the same ranks, bridge-backed actor-to-rollout layout sync, padded variable-length tensors, and explicit masks.

## Useful Entry Points

- `tinytron/rl/trainer.py`: `RLTrainer` base class and algorithm subclasses such as `GRPOTrainer`.
- `tinytron/rl/types.py`: `RolloutBatch` and `RLLossOutput`.
- `tinytron/rl/rollout.py`: rollout batch construction and group advantage normalization.
- `tinytron/rl/logprobs.py`: token logprob gathering, causal logprobs, response masks, masked reductions, and sequence logprob reduction.
- `tinytron/rl/losses.py`: DPO, PPO-style, GRPO-style, policy-gradient, and KL helper losses.
- `tinytron/rl/sync.py`: actor training model to rollout inference model sync wrapper.
- `tinytron/inference/engine.py`: rollout generation with optional sampled token logprobs.
- `tinytron/bridge/`: layout planning and materialization for training-to-inference model-state movement.
- `scripts/debug/rl.py` and `scripts/debug/rl.sh`: stage-1 synchronous RL debug launch path.

## Core Design

`RLTrainer` inherits from the pretraining `Trainer` and owns common RL system plumbing:

- initialize the rollout/inference model;
- synchronize actor weights into rollout layout through `ActorRolloutBridge`;
- run rollout generation and record old logprobs;
- build `RolloutBatch`;
- recompute actor logprobs in training layout;
- apply optimizer, gradient clipping, SP gradient sync, and logging.

Algorithm subclasses should be narrow. For GRPO, `GRPOTrainer` implements `_policy_loss()` and delegates shared rollout/recompute mechanics to `RLTrainer`. Future `RLOOTrainer`, `PPOTrainer`, or online pairwise trainers should follow the same pattern.

## Rollout Batch Convention

Rollout batches use the same variable-length convention as language-model training:

- tensors are padded to rectangular shapes;
- `labels` is aligned to `sequences[:, 1:]`;
- invalid positions are `labels == -100`;
- prompt tokens, tokens after EOS, and padding do not contribute to policy loss;
- `response_mask` is aligned to `labels` and should match `labels != -100`;
- `old_log_probs` and `reference_log_probs`, when present, are aligned to `labels`.

Prefer padded tensors plus masks over Python ragged lists. This keeps SP slicing, DDP reduction, and vectorized losses straightforward.

## Bridge And Layout Sync

Do not special-case RL resharding outside bridge. Treat live sync as one materialization backend:

```text
actor training layout -> rollout inference layout
```

`bridge` owns layout metadata, movement planning, stores, and routes. RL owns when to sync and which rollout model to update. This separation is important for later stages:

- stage 1: same ranks do rollout and update synchronously;
- stage 2: training ranks and rollout ranks are separated but synchronized;
- stage 3: asynchronous rollout/reward/reference workers.

For stage 1, rollout can still use an inference-specific layout such as QKV-sharded attention. The debug script pads prompt and recompute lengths so SEP can split local token chunks.

## Algorithms To Add

Good near-term additions:

- `RLOOTrainer`: online scalar reward with leave-one-out baseline; minimal extension of `RLTrainer`.
- `DPOTrainer`: offline preference trainer. It may inherit directly from `Trainer` or a separate `PreferenceTrainer` rather than rollout-based `RLTrainer`.
- `PPOTrainer`: start with policy-only PPO, then add value head, GAE, rollout buffers, and multi-epoch minibatch updates.
- Online pairwise DPO: generate multiple responses per prompt, rank by reward, and train on chosen/rejected pairs.

Keep reward functions separate from policy losses. Start with rule rewards for debug; introduce reward models only after rollout and logprob accounting are stable.

## Guardrails

- Keep `Trainer` pretraining behavior untouched when adding RL algorithms.
- Put shared rollout/recompute logic in `RLTrainer`; put algorithm-specific objectives in subclasses.
- Keep `RolloutBatch` fields aligned to `sequences[:, 1:]` unless there is a strong reason to introduce a separate representation.
- Never include prompt tokens in policy-gradient loss.
- Preserve `labels == -100` semantics for invalid tokens.
- For SEP runs, ensure actor recompute length is divisible by `sp_world_size`.
- If a new algorithm needs reference logprobs, value predictions, or reward-model scores, add those as optional aligned fields instead of changing the base batch shape.

## Minimal Validation

- Run `python -m compileall tinytron/rl scripts/debug/rl.py`.
- Run `bash -n scripts/debug/rl.sh`.
- Run a small random-weight debug job with `MODEL_SIZE=0.03B`.
- Test at least one `SEP_SIZE > 1` configuration when changing bridge sync, SP slicing, or rollout logprob alignment.
