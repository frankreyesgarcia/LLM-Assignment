"""Shared types for the eval harness (Task 4).

Kept dependency-free (no torch/transformers imports) so task modules and
tests can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalMode(str, Enum):
    """Which prompt format a checkpoint expects.

    PRETRAIN: the checkpoint has never seen the chat template, so examples
    are rendered as plain ICL/zero-shot text (few-shot examples concatenated
    as text, ending right where the model should continue).

    POSTTRAIN: the checkpoint has been instruction-tuned, so examples are
    rendered as chat messages and run through the tokenizer's chat template
    (see src/tokenizer/train.py::CHAT_TEMPLATE), with the answer read back
    out of the assistant turn the model generates.
    """

    PRETRAIN = "pretrain"
    POSTTRAIN = "posttrain"


class RequestType(str, Enum):
    """How a task's examples are scored against the model.

    LOGLIKELIHOOD: rank a fixed set of candidate continuations by the
    model's log-probability of each (cloze / multiple-choice tasks). Never
    calls .generate -- works identically, and cheaply, in both EvalModes.

    GENERATE: let the model generate text freely (greedy, until a stop
    sequence or max_gen_toks), then score the generated text against a
    reference via the task's own process_result.
    """

    LOGLIKELIHOOD = "loglikelihood"
    GENERATE = "generate"


@dataclass
class Doc:
    """One example, normalized just enough for the runner to drive it.

    `raw` keeps the original dataset row so task methods (doc_to_text,
    doc_to_choices, process_result, ...) can pull whatever fields they need
    without Doc having to know every benchmark's schema.
    """

    idx: int
    raw: dict


@dataclass
class TaskResult:
    task_name: str
    language: str
    mode: EvalMode
    num_fewshot: int
    n_examples: int
    metrics: dict[str, float]
    samples: list[dict] = field(default_factory=list)
