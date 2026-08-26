"""MATH (Task 4 extension): competition math problems (Hendrycks et al.),
via nlile/hendrycks-MATH-benchmark -- kept in English on purpose. Unlike
every other benchmark in this harness, no maintained Portuguese, Spanish,
or Hindi translation of MATH exists; forcing a machine translation of
competition-math LaTeX would risk corrupting the one thing this benchmark
actually tests (precise symbolic answers), so it's included unlocalized
as a general reasoning cross-check rather than dropped.

Scored by extracting the contents of the *last* \\boxed{...} in the
generation (brace-matched, not regex, since answers routinely contain
nested braces like \\boxed{\\frac{1}{2}}) and comparing against the
reference answer after light LaTeX normalization (whitespace and the
handful of no-op spacing commands graders conventionally strip). This is
the standard approximate MATH grader used across eval harnesses -- exact
symbolic equivalence (e.g. accepting "0.5" for "\\frac{1}{2}") would need
a CAS and is out of scope here.
"""

from __future__ import annotations

import re

from src.eval import hf_data
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "nlile/hendrycks-MATH-benchmark"
_STRIP_RE = re.compile(r"\\left|\\right|\\!|\\,|\\ |\s+")


def _normalize(answer: str) -> str:
    return _STRIP_RE.sub("", answer).strip("$")


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    content = []
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        content.append(text[i])
        i += 1
    return "".join(content)


@register("math_en")
class MATHEnglish(Task):
    language = "en"
    request_type = RequestType.GENERATE
    description = 'Solve the following math problem step by step, then give the final answer as \\boxed{<answer>}.'
    stop_sequences = ["\nProblem:", "\n\n"]
    max_gen_toks = 512

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(DATASET, None, "test", limit)
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_text(self, doc: Doc) -> str:
        return f"Problem: {doc.raw['problem']}\nSolution:"

    def doc_to_target(self, doc: Doc) -> str:
        return doc.raw["solution"]

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float]:
        pred = _last_boxed(prediction)
        gold = _last_boxed(doc.raw["solution"]) or doc.raw["answer"]
        match = pred is not None and _normalize(pred) == _normalize(gold)
        return {"exact_match": 1.0 if match else 0.0}
