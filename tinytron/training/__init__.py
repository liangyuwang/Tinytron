from __future__ import annotations

from .arguments import build_parser, parse_args
from .config import build_config

__all__ = [
    "build_parser",
    "parse_args",
    "build_config",
    "Trainer",
]


def __getattr__(name: str):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
