"""The interface every SFT dataset adapter implements.

A source is responsible for its own HF loading and for converting whatever
schema that dataset ships (already-chatty `messages`/`conversations`
columns, prompt/completion pairs, ...) into the shared SFTExample
abstraction (src/sft/types.py). Nothing downstream -- scripts/
prepare_sft_data.py -- needs to know which shape a given dataset started
in. Same split of responsibility as src/eval/tasks/base.py::Task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Iterator

from src.sft.types import SFTExample


class SFTSource(ABC):
    name: ClassVar[str]  # set by @registry.register
    languages: ClassVar[tuple[str, ...]]  # e.g. ("pt",) or ("hi",)

    @abstractmethod
    def load_examples(self, limit: int | None = None) -> Iterator[SFTExample]:
        """Yield up to `limit` examples (None = all) as SFTExamples."""
