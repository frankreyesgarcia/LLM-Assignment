"""Task ABC (Task 4): the interface every benchmark module implements.

A task is responsible for its own data loading and prompt formatting; the
runner (src/eval/runner.py) is responsible for driving fewshot sampling,
calling EvalModel, and aggregating metrics the same way for every task. So
adding a new benchmark means writing one tasks/*.py file and registering
it -- nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from src.eval.types import Doc, RequestType


class Task(ABC):
    name: ClassVar[str]  # set by @registry.register
    language: ClassVar[str]
    request_type: ClassVar[RequestType]
    # Brief task instruction, prepended as an ICL header (pretrain mode) or
    # a chat system message (posttrain mode). Keep it short -- it's spent
    # on every single example, fewshot or not.
    description: ClassVar[str] = ""
    # GENERATE tasks only.
    stop_sequences: ClassVar[list[str]] = ["\n\n"]
    max_gen_toks: ClassVar[int] = 64
    # GENERATE tasks only: True if process_result needs a `judge=` kwarg
    # (an LLM-as-judge scorer) rather than scoring against a fixed
    # reference -- see tasks/alba.py.
    needs_judge: ClassVar[bool] = False

    @abstractmethod
    def load_docs(self, limit: int | None = None) -> list[Doc]:
        """Load up to `limit` examples (None = all) as normalized Docs."""

    @abstractmethod
    def doc_to_text(self, doc: Doc) -> str:
        """Plain-text rendering of the question/context, pretrain-mode
        prompt body -- also used as the fewshot example body in both modes.
        """

    def doc_to_target(self, doc: Doc) -> str:
        """Gold text appended after doc_to_text when doc is used as a
        fewshot example. LOGLIKELIHOOD tasks get a free default (the
        correct choice); GENERATE tasks must override.
        """
        if self.request_type == RequestType.LOGLIKELIHOOD:
            return self.doc_to_choices(doc)[self.gold_index(doc)]
        raise NotImplementedError

    def doc_to_choices(self, doc: Doc) -> list[str]:
        """LOGLIKELIHOOD tasks only: candidate continuations to rank."""
        raise NotImplementedError

    def gold_index(self, doc: Doc) -> int:
        """LOGLIKELIHOOD tasks only: index of the correct doc_to_choices entry."""
        raise NotImplementedError

    def doc_to_messages(self, doc: Doc) -> list[dict]:
        """Chat messages for posttrain mode, NOT including the final
        assistant turn (the model generates that). Default: a single user
        turn built from doc_to_text -- fine for single-turn tasks; tasks
        with native multi-turn structure (pt_culture, chatrag_hi) override.
        """
        return [{"role": "user", "content": self.doc_to_text(doc)}]

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float | None]:
        """GENERATE tasks only: metric name -> value for one example. A
        None value is dropped rather than averaged in (see runner.py) --
        for tasks where scoring can fail/be skipped, e.g. alba.py without
        a judge configured.
        """
        raise NotImplementedError
