"""Benchmark registry (Task 4): each benchmark lives in its own tasks/*.py
module and self-registers with `@register`, so adding a benchmark never
touches the runner -- see tasks/__init__.py for the import that populates
this at module load time, and scripts/run_eval.py --list-tasks for what's
available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.tasks.base import Task

_REGISTRY: dict[str, type["Task"]] = {}


def register(name: str):
    def decorator(cls: type["Task"]) -> type["Task"]:
        if name in _REGISTRY:
            raise ValueError(f"task {name!r} already registered (by {_REGISTRY[name].__name__})")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_task(name: str) -> type["Task"]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)
