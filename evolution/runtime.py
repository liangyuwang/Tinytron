from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .api import translate_spec
from .spec import EvolutionSpec
from .translators import TranslationArtifact


@dataclass(frozen=True)
class PreparedExperiment:
    spec: EvolutionSpec
    framework: str
    workspace: Path
    artifact: TranslationArtifact
    commands: list[str] = field(default_factory=list)

    @property
    def is_runnable(self) -> bool:
        return self.artifact.is_complete and bool(self.commands)


class ExperimentRunner:
    """Thin preparation interface for external schedulers or AI agents.

    This class does not execute shell commands yet. It resolves the framework
    adapter and returns a structured plan that a controller can inspect,
    schedule, sandbox, or ask a human to approve.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)

    def prepare(self, spec: EvolutionSpec, framework: str = "tinytron") -> PreparedExperiment:
        artifact = translate_spec(spec, framework=framework)
        return PreparedExperiment(
            spec=spec,
            framework=framework,
            workspace=self.workspace,
            artifact=artifact,
            commands=artifact.commands,
        )
