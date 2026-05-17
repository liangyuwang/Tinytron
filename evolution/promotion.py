from __future__ import annotations

from dataclasses import dataclass, field

from .spec import EvalSpec


@dataclass(frozen=True)
class EvalResult:
    suite: str
    baseline: float
    candidate: float
    metric: str | None = None
    higher_is_better: bool = True

    @property
    def delta(self) -> float:
        raw_delta = self.candidate - self.baseline
        return raw_delta if self.higher_is_better else -raw_delta


@dataclass(frozen=True)
class SuiteDecision:
    suite: str
    passed: bool
    delta: float
    min_delta: float
    required: bool
    reason: str


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    suite_decisions: list[SuiteDecision] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    reason: str = ""


def decide_promotion(eval_spec: EvalSpec, results: list[EvalResult]) -> PromotionDecision:
    """Apply EvalSpec promotion rules to measured baseline/candidate results."""
    result_by_suite = {result.suite: result for result in results}
    spec_by_suite = {suite.name: suite for suite in eval_spec.suites}
    required = eval_spec.promotion.required or [
        suite.name for suite in eval_spec.suites if suite.required
    ]

    suite_decisions: list[SuiteDecision] = []
    missing_required: list[str] = []
    for suite in eval_spec.suites:
        result = result_by_suite.get(suite.name)
        if result is None:
            if suite.name in required:
                missing_required.append(suite.name)
            suite_decisions.append(
                SuiteDecision(
                    suite=suite.name,
                    passed=False,
                    delta=0.0,
                    min_delta=suite.min_delta,
                    required=suite.name in required,
                    reason="missing_result",
                )
            )
            continue

        delta = result.delta
        passed = delta >= suite.min_delta
        suite_decisions.append(
            SuiteDecision(
                suite=suite.name,
                passed=passed,
                delta=delta,
                min_delta=suite.min_delta,
                required=suite.name in required,
                reason="passed" if passed else "below_min_delta",
            )
        )

    unknown_results = sorted(set(result_by_suite) - set(spec_by_suite))
    if unknown_results:
        suite_decisions.extend(
            SuiteDecision(
                suite=name,
                passed=False,
                delta=0.0,
                min_delta=0.0,
                required=False,
                reason="unknown_suite",
            )
            for name in unknown_results
        )

    rule = eval_spec.promotion.rule
    if rule != "all_required_pass":
        return PromotionDecision(
            promoted=False,
            suite_decisions=suite_decisions,
            missing_required=missing_required,
            reason=f"unsupported_promotion_rule:{rule}",
        )

    required_failures = [
        decision.suite
        for decision in suite_decisions
        if decision.required and not decision.passed
    ]
    promoted = not missing_required and not required_failures
    reason = "promoted" if promoted else "required_suite_failed"
    return PromotionDecision(
        promoted=promoted,
        suite_decisions=suite_decisions,
        missing_required=missing_required,
        reason=reason,
    )
