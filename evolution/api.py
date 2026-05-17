from __future__ import annotations

from pathlib import Path

from .promotion import EvalResult, PromotionDecision, decide_promotion
from .report import render_evolution_report
from .spec import EvolutionSpec, validate_evolution_spec
from .translators import TranslationArtifact
from .translators.registry import get_translator


def load_spec(path: str | Path) -> EvolutionSpec:
    return EvolutionSpec.load_json(path)


def save_spec(spec: EvolutionSpec, path: str | Path) -> None:
    spec.save_json(path)


def validate_spec(spec: EvolutionSpec) -> EvolutionSpec:
    validate_evolution_spec(spec)
    return spec


def translate_spec(spec: EvolutionSpec, framework: str = "tinytron") -> TranslationArtifact:
    return get_translator(framework).translate(spec)


def evaluate_promotion(
    spec: EvolutionSpec,
    results: list[EvalResult],
) -> PromotionDecision:
    return decide_promotion(spec.eval, results)


def render_report(
    spec: EvolutionSpec,
    decision: PromotionDecision | None = None,
) -> str:
    return render_evolution_report(spec, decision)
