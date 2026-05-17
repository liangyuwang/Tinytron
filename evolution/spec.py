from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class SpecValidationError(ValueError):
    """Raised when an evolution spec is structurally invalid."""


@dataclass(frozen=True)
class ChangeSpec:
    id: str
    target: str
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeSpec":
        return cls(
            id=str(data["id"]),
            target=str(data["target"]),
            type=str(data["type"]),
            parameters=dict(data.get("parameters", {})),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class ModelSpec:
    base: str
    changes: list[ChangeSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            base=str(data["base"]),
            changes=[ChangeSpec.from_dict(item) for item in data.get("changes", [])],
        )


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerSpec":
        return cls(name=str(data["name"]), parameters=dict(data.get("parameters", {})))


@dataclass(frozen=True)
class ScheduleSpec:
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleSpec":
        return cls(type=str(data["type"]), parameters=dict(data.get("parameters", {})))


@dataclass(frozen=True)
class LossSpec:
    name: str
    weight: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LossSpec":
        weight = data.get("weight")
        return cls(
            name=str(data["name"]),
            weight=None if weight is None else float(weight),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class BudgetSpec:
    max_steps: int | None = None
    tokens: int | None = None
    wall_time_seconds: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetSpec":
        if data is None:
            return cls()
        return cls(
            max_steps=_optional_int(data.get("max_steps")),
            tokens=_optional_int(data.get("tokens")),
            wall_time_seconds=_optional_int(data.get("wall_time_seconds")),
        )


@dataclass(frozen=True)
class TrainingSpec:
    stage: str
    optimizer: OptimizerSpec
    schedule: ScheduleSpec | None = None
    losses: list[LossSpec] = field(default_factory=list)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingSpec":
        schedule = data.get("schedule")
        return cls(
            stage=str(data["stage"]),
            optimizer=OptimizerSpec.from_dict(data["optimizer"]),
            schedule=None if schedule is None else ScheduleSpec.from_dict(schedule),
            losses=[LossSpec.from_dict(item) for item in data.get("losses", [])],
            budget=BudgetSpec.from_dict(data.get("budget")),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    type: str
    path: str | None = None
    generator_model: str | None = None
    verifier: str | None = None
    acceptance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataSourceSpec":
        return cls(
            name=str(data["name"]),
            type=str(data["type"]),
            path=data.get("path"),
            generator_model=data.get("generator_model"),
            verifier=data.get("verifier"),
            acceptance=dict(data.get("acceptance", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class DataMixtureSpec:
    source: str
    weight: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataMixtureSpec":
        return cls(source=str(data["source"]), weight=float(data["weight"]))


@dataclass(frozen=True)
class DataSpec:
    sources: list[DataSourceSpec] = field(default_factory=list)
    mixture: list[DataMixtureSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DataSpec":
        if data is None:
            return cls()
        return cls(
            sources=[DataSourceSpec.from_dict(item) for item in data.get("sources", [])],
            mixture=[DataMixtureSpec.from_dict(item) for item in data.get("mixture", [])],
        )


@dataclass(frozen=True)
class AgentSpec:
    runtime: str
    task_family: str
    max_turns: int | None = None
    tools: list[str] = field(default_factory=list)
    trajectory_schema: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentSpec | None":
        if data is None:
            return None
        return cls(
            runtime=str(data["runtime"]),
            task_family=str(data["task_family"]),
            max_turns=_optional_int(data.get("max_turns")),
            tools=[str(item) for item in data.get("tools", [])],
            trajectory_schema=dict(data.get("trajectory_schema", {})),
            reward=dict(data.get("reward", {})),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class EvalSuiteSpec:
    name: str
    metric: str
    min_delta: float = 0.0
    required: bool = True
    higher_is_better: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSuiteSpec":
        return cls(
            name=str(data["name"]),
            metric=str(data["metric"]),
            min_delta=float(data.get("min_delta", 0.0)),
            required=bool(data.get("required", True)),
            higher_is_better=bool(data.get("higher_is_better", True)),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class ContaminationCheckSpec:
    enabled: bool = True
    method: str = "ngram_overlap"
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContaminationCheckSpec":
        if data is None:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", True)),
            method=str(data.get("method", "ngram_overlap")),
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class PromotionSpec:
    rule: str = "all_required_pass"
    required: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromotionSpec":
        if data is None:
            return cls()
        return cls(
            rule=str(data.get("rule", "all_required_pass")),
            required=[str(item) for item in data.get("required", [])],
            parameters=dict(data.get("parameters", {})),
        )


@dataclass(frozen=True)
class EvalSpec:
    baseline_model: str
    candidate_model: str
    suites: list[EvalSuiteSpec] = field(default_factory=list)
    promotion: PromotionSpec = field(default_factory=PromotionSpec)
    contamination_checks: ContaminationCheckSpec = field(default_factory=ContaminationCheckSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSpec":
        return cls(
            baseline_model=str(data["baseline_model"]),
            candidate_model=str(data["candidate_model"]),
            suites=[EvalSuiteSpec.from_dict(item) for item in data.get("suites", [])],
            promotion=PromotionSpec.from_dict(data.get("promotion")),
            contamination_checks=ContaminationCheckSpec.from_dict(data.get("contamination_checks")),
        )


@dataclass(frozen=True)
class RunEvidenceSpec:
    run_id: str
    seed: int | None = None
    command: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEvidenceSpec":
        return cls(
            run_id=str(data["run_id"]),
            seed=_optional_int(data.get("seed")),
            command=data.get("command"),
            metrics=dict(data.get("metrics", {})),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class AblationEvidenceSpec:
    remove: str
    effect: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationEvidenceSpec":
        return cls(
            remove=str(data["remove"]),
            effect=dict(data.get("effect", {})),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class EvidenceSpec:
    status: str = "unknown"
    confidence: str = "unknown"
    runs: list[RunEvidenceSpec] = field(default_factory=list)
    ablations: list[AblationEvidenceSpec] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    conclusion: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvidenceSpec":
        if data is None:
            return cls()
        return cls(
            status=str(data.get("status", "unknown")),
            confidence=str(data.get("confidence", "unknown")),
            runs=[RunEvidenceSpec.from_dict(item) for item in data.get("runs", [])],
            ablations=[AblationEvidenceSpec.from_dict(item) for item in data.get("ablations", [])],
            risks=[str(item) for item in data.get("risks", [])],
            conclusion=data.get("conclusion"),
        )


@dataclass(frozen=True)
class EvolutionSpec:
    id: str
    objective: str
    parent_model: str
    candidate_model: str
    model: ModelSpec
    training: TrainingSpec
    eval: EvalSpec
    data: DataSpec = field(default_factory=DataSpec)
    agent: AgentSpec | None = None
    evidence: EvidenceSpec = field(default_factory=EvidenceSpec)
    schema_version: str = "evolution-spec/v0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")

    def validate(self) -> None:
        validate_evolution_spec(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionSpec":
        spec = cls(
            id=str(data["id"]),
            objective=str(data["objective"]),
            parent_model=str(data["parent_model"]),
            candidate_model=str(data["candidate_model"]),
            model=ModelSpec.from_dict(data["model"]),
            training=TrainingSpec.from_dict(data["training"]),
            eval=EvalSpec.from_dict(data["eval"]),
            data=DataSpec.from_dict(data.get("data")),
            agent=AgentSpec.from_dict(data.get("agent")),
            evidence=EvidenceSpec.from_dict(data.get("evidence")),
            schema_version=str(data.get("schema_version", "evolution-spec/v0")),
            metadata=dict(data.get("metadata", {})),
        )
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, text: str) -> "EvolutionSpec":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str | Path) -> "EvolutionSpec":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def validate_evolution_spec(spec: EvolutionSpec) -> None:
    if not spec.id:
        raise SpecValidationError("spec.id must be non-empty")
    if not spec.objective:
        raise SpecValidationError("spec.objective must be non-empty")
    if not spec.eval.suites:
        raise SpecValidationError("spec.eval.suites must include at least one suite")

    suite_names = [suite.name for suite in spec.eval.suites]
    if len(set(suite_names)) != len(suite_names):
        raise SpecValidationError("eval suite names must be unique")

    required = spec.eval.promotion.required or [
        suite.name for suite in spec.eval.suites if suite.required
    ]
    unknown_required = sorted(set(required) - set(suite_names))
    if unknown_required:
        raise SpecValidationError(f"promotion.required references unknown suites: {unknown_required}")

    source_names = {source.name for source in spec.data.sources}
    for item in spec.data.mixture:
        if item.source not in source_names:
            raise SpecValidationError(f"data.mixture references unknown source: {item.source!r}")
        if item.weight < 0:
            raise SpecValidationError(f"data.mixture weight must be non-negative: {item.source!r}")

    change_ids = [change.id for change in spec.model.changes]
    if len(set(change_ids)) != len(change_ids):
        raise SpecValidationError("model change ids must be unique")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
