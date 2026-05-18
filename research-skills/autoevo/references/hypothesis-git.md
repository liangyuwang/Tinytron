# Hypothesis Git Protocol

Use this reference when an auto-evolution task needs Git-backed hypothesis lineage or when research experiment branches may conflict with normal repo development.

## Core Idea

Use Git for implementation lineage, not as the only source of research truth.

```text
EvolutionSpec  -> research intent, metrics, gates, evidence schema
Git            -> code diff, branch lineage, candidate commits
Registry       -> artifacts, decisions, reports, run records
```

Prefer spec-first, git-backed workflows. The agent should describe what it wants to test before mutating code.

## Two Git Semantics

Keep these separate:

- Repo development Git: normal software work on Tinytron and `evolution/`.
- Research Git: hypothesis branches, experiment branches, promoted research commits, archived failed trials.

They may share the same repository object database, but should not share the active development worktree or branch namespace.

## Recommended Namespaces

Use normal branches for human/framework development:

```text
main
dev
codex/*
feature/*
fix/*
```

Use research-specific names for automatic experiment work:

```text
hyp/<topic>
exp/<spec-id>
promoted/<spec-id>
archive/<spec-id>
```

If lower-level refs are available, prefer explicit namespaces:

```text
refs/hypotheses/<topic>
refs/experiments/<spec-id>
refs/promoted/<spec-id>
refs/archive/<spec-id>
```

## Worktree Isolation

Do not mutate the user's active development worktree for an automated experiment. Create an isolated experiment worktree from a pinned base commit:

```text
base_commit -> experiment branch -> isolated worktree -> candidate commit
```

The worktree path should be recorded in the registry. A good default layout is:

```text
.evolution/worktrees/<spec-id>/
```

Before mutation, check that the target experiment worktree is clean or that every dirty change is already registered as part of the current experiment.

## Required Git Metadata

Record these fields for each experiment:

- `base_commit`: clean Tinytron code state used for baseline and candidate branch creation.
- `candidate_commit`: commit containing the hypothesis implementation.
- `infra_commit`: commit of the auto-evolution framework or runner that created/evaluated the experiment.
- `branch`: experiment branch or ref.
- `worktree_path`: isolated path used for mutation and execution.
- `diff_hash`: stable hash of the candidate diff, useful when commits are rebased or moved.
- `spec_id`: matching `EvolutionSpec.id`.

Use `base_commit` to protect scientific comparability. Use `infra_commit` to make the automation itself auditable.

## Lifecycle

1. Read prior registry evidence and current Tinytron code map.
2. Generate or refine one hypothesis.
3. Write `EvolutionSpec` with expected direction, primary metric, guardrail metrics, budget, and git metadata placeholders.
4. Create an experiment branch/worktree from a pinned `base_commit`.
5. Implement the smallest candidate diff.
6. Run unit, smoke, and declared eval commands.
7. Commit the candidate implementation.
8. Record metrics and promotion decision.
9. Promote, ablate, tune, fix, scale, or archive based on evidence.

## Promotion Policy

Promotion should not mean direct merge to `main`.

Use a staged path:

```text
exp/<spec-id> -> promoted/<spec-id> -> research/stable -> main
```

Only merge into normal development after the research result and code quality have both been reviewed. A promoted research result may still need cleanup before entering `main`.

## Avoiding Branch Explosion

Do not keep every tiny hyperparameter trial as a long-lived branch. Use one of these patterns:

- Hypothesis family branch plus registry records for individual trials.
- Short-lived `exp/<spec-id>` branches that are archived after evidence is recorded.
- Tags or commit hashes for failed experiments when branch retention is unnecessary.

Keep the registry as the durable research memory, not the branch list.

## Failure Diagnosis

When an experiment fails, classify the failure before planning the next step:

- Hypothesis likely false.
- Implementation likely wrong.
- Budget too small.
- Eval too noisy.
- Training unstable.
- Data mixture unsuitable.
- Improvement appears only on a subset.
- Infrastructure failure or runner error.

Do not tune blindly. The next experiment should address the most likely failure mode.
