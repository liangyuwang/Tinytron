from __future__ import annotations

from typing import Any

from evolution.spec import EvolutionSpec

from .base import FrameworkTranslator, TranslationArtifact


class TinytronTranslator(FrameworkTranslator):
    """Translate the framework-neutral spec into Tinytron-oriented run metadata.

    This intentionally emits config fragments and notes rather than mutating source
    files. The experiment controller can decide whether to apply code changes,
    generate a launch script, or hand the artifact to an agent worker.
    """

    framework = "tinytron"

    def translate(self, spec: EvolutionSpec) -> TranslationArtifact:
        unsupported: list[str] = []
        config: dict[str, Any] = {
            "model": {
                "base": spec.model.base,
                "changes": [change.id for change in spec.model.changes],
            },
            "training": {
                "stage": spec.training.stage,
                "optimizer": {
                    "name": spec.training.optimizer.name,
                    **spec.training.optimizer.parameters,
                },
                "losses": [
                    {
                        "name": loss.name,
                        "weight": loss.weight,
                        **loss.parameters,
                    }
                    for loss in spec.training.losses
                ],
                "budget": {
                    "max_steps": spec.training.budget.max_steps,
                    "tokens": spec.training.budget.tokens,
                    "wall_time_seconds": spec.training.budget.wall_time_seconds,
                },
                **spec.training.parameters,
            },
            "data": {
                "sources": [source.name for source in spec.data.sources],
                "mixture": [
                    {"source": item.source, "weight": item.weight}
                    for item in spec.data.mixture
                ],
            },
            "eval": {
                "baseline_model": spec.eval.baseline_model,
                "candidate_model": spec.eval.candidate_model,
                "suites": [
                    {
                        "name": suite.name,
                        "metric": suite.metric,
                        "min_delta": suite.min_delta,
                        "required": suite.required,
                    }
                    for suite in spec.eval.suites
                ],
            },
        }

        if spec.agent is not None:
            unsupported.append("agent_runtime")

        for change in spec.model.changes:
            if change.type not in _TINYTRON_CHANGE_TYPES:
                unsupported.append(f"model_change:{change.id}:{change.type}")

        commands = _commands_for_stage(spec)
        code_notes = [
            "Apply model changes by id before launching this experiment.",
            "Use EvalSpec promotion gate before promoting candidate_model.",
        ]
        return TranslationArtifact(
            framework=self.framework,
            config=config,
            commands=commands,
            code_notes=code_notes,
            unsupported=unsupported,
            metadata={
                "spec_id": spec.id,
                "schema_version": spec.schema_version,
            },
        )


_TINYTRON_CHANGE_TYPES = {
    "architecture",
    "data_recipe",
    "inference_layout",
    "loss_term",
    "optimizer",
    "rl_recipe",
    "training_recipe",
}


def _commands_for_stage(spec: EvolutionSpec) -> list[str]:
    if spec.training.stage in {"pretrain", "sft"}:
        return ["bash scripts/debug/pretrain.sh"]
    if spec.training.stage in {"rl", "grpo"}:
        return ["bash scripts/debug/rl.sh"]
    return []
