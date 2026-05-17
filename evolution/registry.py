from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .promotion import PromotionDecision
from .spec import EvolutionSpec
from .translators import TranslationArtifact


class EvolutionRegistry:
    """File-backed registry for agent-supervised evolution runs.

    The registry is intentionally simple: it writes stable JSON artifacts under
    one root so an external agent can inspect or replay state without importing
    private internals.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.specs_dir = self.root / "specs"
        self.artifacts_dir = self.root / "artifacts"
        self.decisions_dir = self.root / "decisions"
        self.reports_dir = self.root / "reports"

    def initialize(self) -> None:
        for directory in (
            self.specs_dir,
            self.artifacts_dir,
            self.decisions_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def record_spec(self, spec: EvolutionSpec) -> Path:
        self.initialize()
        path = self.spec_path(spec.id)
        spec.save_json(path)
        return path

    def load_spec(self, spec_id: str) -> EvolutionSpec:
        return EvolutionSpec.load_json(self.spec_path(spec_id))

    def record_artifact(
        self,
        spec_id: str,
        artifact: TranslationArtifact,
        *,
        name: str | None = None,
    ) -> Path:
        self.initialize()
        path = self.artifact_path(spec_id, name or artifact.framework)
        _write_json(path, asdict(artifact))
        return path

    def record_decision(self, spec_id: str, decision: PromotionDecision) -> Path:
        self.initialize()
        path = self.decision_path(spec_id)
        _write_json(path, asdict(decision))
        return path

    def record_report(self, spec_id: str, report: str) -> Path:
        self.initialize()
        path = self.report_path(spec_id)
        path.write_text(report, encoding="utf-8")
        return path

    def spec_path(self, spec_id: str) -> Path:
        return self.specs_dir / f"{_safe_id(spec_id)}.json"

    def artifact_path(self, spec_id: str, name: str) -> Path:
        return self.artifacts_dir / f"{_safe_id(spec_id)}.{_safe_id(name)}.json"

    def decision_path(self, spec_id: str) -> Path:
        return self.decisions_dir / f"{_safe_id(spec_id)}.json"

    def report_path(self, spec_id: str) -> Path:
        return self.reports_dir / f"{_safe_id(spec_id)}.md"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("._") or "unnamed"
