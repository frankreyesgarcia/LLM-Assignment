"""SFT source registry: each dataset adapter lives in its own
sources/*.py module and self-registers with `@register`, so adding a
dataset never touches scripts/prepare_sft_data.py -- see sources/__init__.py
for the import that populates this at module load time. Deliberately the
same shape as src/eval/registry.py (Task 4's benchmark registry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sft.sources.base import SFTSource

_REGISTRY: dict[str, type["SFTSource"]] = {}


def register(name: str):
    def decorator(cls: type["SFTSource"]) -> type["SFTSource"]:
        if name in _REGISTRY:
            raise ValueError(f"SFT source {name!r} already registered (by {_REGISTRY[name].__name__})")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_source(name: str) -> type["SFTSource"]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown SFT source {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_sources() -> list[str]:
    return sorted(_REGISTRY)
