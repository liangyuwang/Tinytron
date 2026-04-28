# Experiment Protocol

Use this reference when designing or summarizing Tinytron experiments.

## Experiment Card

Record these fields for each run:

- Question: the research question in one sentence.
- Hypothesis: expected direction of change.
- Baseline: model size, data path, optimizer, batch size, seq len, SEP size, seed, precision.
- Variant: only the intentional differences from baseline.
- Command: exact launch command or script environment variables.
- Code state: commit SHA or patch summary.
- Checkpoint/layout state: checkpoint source, SEP size, whether resume used same-layout optimizer restore or model-only reshard.
- Metrics: loss, validation loss if available, tok/sec, MFU, peak memory if available, prefill/decode tokens/sec for inference.
- Notes: failures, instability, OOMs, profiler findings, caveats.

## Suggested Flow

1. Run a syntax/config check.
2. Run a tiny mock-data smoke test.
3. Run a short baseline and short variant.
4. Inspect logs for obvious instability or throughput regressions.
5. Only then run longer training or inference sweeps.

## Comparison Rules

- Change one primary factor at a time.
- Keep seeds fixed for paired comparisons when possible.
- Compare at the same token budget, not just the same step count, if batch settings differ.
- Separate throughput metrics from model-quality metrics.
- Treat random-weight inference results as systems checks, not quality evidence.

## Reporting Template

```text
Question:
Hypothesis:
Change:
Baseline command:
Variant command:
Validation:
Results:
Caveats:
Next step:
```
