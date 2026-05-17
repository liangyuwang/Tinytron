from __future__ import annotations

from .promotion import PromotionDecision
from .spec import EvolutionSpec


def render_evolution_report(
    spec: EvolutionSpec,
    decision: PromotionDecision | None = None,
) -> str:
    """Render a compact Markdown report for human review or framework migration."""
    lines = [
        f"# Evolution Report: {spec.id}",
        "",
        f"Objective: {spec.objective}",
        f"Parent model: {spec.parent_model}",
        f"Candidate model: {spec.candidate_model}",
        f"Status: {spec.evidence.status}",
        f"Confidence: {spec.evidence.confidence}",
        "",
        "## Model Changes",
    ]
    if spec.model.changes:
        for change in spec.model.changes:
            lines.append(f"- {change.id}: {change.target} / {change.type}")
    else:
        lines.append("- none")

    lines.extend(["", "## Training"])
    lines.append(f"- stage: {spec.training.stage}")
    lines.append(f"- optimizer: {spec.training.optimizer.name}")
    if spec.training.schedule is not None:
        lines.append(f"- schedule: {spec.training.schedule.type}")
    if spec.training.losses:
        losses = ", ".join(loss.name for loss in spec.training.losses)
        lines.append(f"- losses: {losses}")

    lines.extend(["", "## Data"])
    if spec.data.sources:
        for source in spec.data.sources:
            lines.append(f"- {source.name}: {source.type}")
    else:
        lines.append("- none")

    lines.extend(["", "## Eval Gate"])
    for suite in spec.eval.suites:
        comparator = ">=" if suite.higher_is_better else "<="
        lines.append(f"- {suite.name}: {suite.metric}, required delta {comparator} {suite.min_delta}")

    if decision is not None:
        lines.extend(["", "## Promotion Decision"])
        lines.append(f"- promoted: {decision.promoted}")
        lines.append(f"- reason: {decision.reason}")
        for item in decision.suite_decisions:
            lines.append(
                f"- {item.suite}: passed={item.passed}, delta={item.delta:.6g}, "
                f"min_delta={item.min_delta:.6g}, reason={item.reason}"
            )

    if spec.evidence.risks:
        lines.extend(["", "## Risks"])
        for risk in spec.evidence.risks:
            lines.append(f"- {risk}")

    if spec.evidence.conclusion:
        lines.extend(["", "## Conclusion", spec.evidence.conclusion])

    return "\n".join(lines) + "\n"
