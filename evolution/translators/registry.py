from __future__ import annotations

from .base import FrameworkTranslator, TranslationError
from .tinytron import TinytronTranslator


_TRANSLATOR_FACTORIES = {
    "tinytron": TinytronTranslator,
}


def available_translators() -> tuple[str, ...]:
    return tuple(sorted(_TRANSLATOR_FACTORIES))


def register_translator(name: str, translator_cls: type[FrameworkTranslator]) -> None:
    normalized = _normalize_name(name)
    if normalized in _TRANSLATOR_FACTORIES:
        raise TranslationError(f"translator {normalized!r} is already registered")
    _TRANSLATOR_FACTORIES[normalized] = translator_cls


def get_translator(name: str) -> FrameworkTranslator:
    normalized = _normalize_name(name)
    try:
        return _TRANSLATOR_FACTORIES[normalized]()
    except KeyError as exc:
        available = ", ".join(available_translators())
        raise TranslationError(f"unknown translator {name!r}; available: {available}") from exc


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
