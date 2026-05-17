from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evolution.spec import EvolutionSpec


class TranslationError(ValueError):
    """Raised when an EvolutionSpec cannot be translated for a framework."""


@dataclass(frozen=True)
class TranslationArtifact:
    framework: str
    config: dict[str, Any] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    code_notes: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return not self.unsupported


class FrameworkTranslator(ABC):
    """Common adapter interface for Tinytron, Megatron, HF, torchtitan, etc."""

    framework: str

    @abstractmethod
    def translate(self, spec: EvolutionSpec) -> TranslationArtifact:
        """Translate an EvolutionSpec into framework-specific execution artifacts."""

    def validate_supported(self, spec: EvolutionSpec) -> None:
        artifact = self.translate(spec)
        if artifact.unsupported:
            raise TranslationError(
                f"{self.framework} does not support: {', '.join(artifact.unsupported)}"
            )
