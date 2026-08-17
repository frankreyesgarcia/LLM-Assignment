"""ALBA (Task 4): open-ended European Portuguese linguistic/cultural
prompts, judge-scored 1-5 (github.com/AMALIA-LLM/alba-benchmark).

Prompts (evaluation/alba_prompts.csv in that repo) span categories like
Lexicology, Syntax, Morphology, Culture-Bound Semantics, Word Plays --
"give three Portuguese political neologisms", "what's the register of this
sentence" -- with no fixed reference answer, so unlike every other task
here there's nothing to loglikelihood-rank or F1-match against. Scoring
needs an LLM judge (src/eval/judge.py); with no judge configured, this
still runs end-to-end and reports generations with judge_score omitted
(see runner.py's None-filtering) rather than failing.

No fewshot support: ALBA has no reference answers to build "context +
gold" shot examples from (unlike alba-benchmark's *judge* few-shot
examples, which score answers rather than demonstrate them -- see
judge.py). Run this with --num-fewshot 0.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import requests

from src.eval.judge import Judge, NullJudge
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

PROMPTS_URL = "https://raw.githubusercontent.com/AMALIA-LLM/alba-benchmark/main/evaluation/alba_prompts.csv"
# The 800 prompts never change mid-sweep, and load_docs runs once per
# checkpoint -- fetching fresh every time means N-checkpoints-x-M-parallel-
# shards worth of requests to the same GitHub URL in a burst, which GitHub
# rate-limits (429, observed running 4 shards concurrently). Caching to
# disk after the first fetch turns that into one request total per node.
CACHE_PATH = Path.home() / ".cache" / "llm-und-eval" / "alba_prompts.csv"


def _fetch_prompts_csv() -> str:
    if CACHE_PATH.exists():
        return CACHE_PATH.read_text()
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(PROMPTS_URL, timeout=30)
            response.raise_for_status()
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(response.text)
            return response.text
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2**attempt)  # 1s, 2s, 4s, 8s, 16s
    raise RuntimeError(f"failed to fetch {PROMPTS_URL} after 5 attempts") from last_error


@register("alba")
class ALBA(Task):
    language = "pt"
    request_type = RequestType.GENERATE
    description = "Responde ao seguinte pedido em português europeu."
    stop_sequences = ["\n\n"]
    max_gen_toks = 256
    needs_judge = True

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        reader = csv.DictReader(io.StringIO(_fetch_prompts_csv()))
        rows = list(reader)
        if limit is not None:
            rows = rows[:limit]
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_text(self, doc: Doc) -> str:
        return doc.raw["prompt"]

    def process_result(self, doc: Doc, prediction: str, judge: Judge | None = None) -> dict[str, float | None]:
        judge = judge or NullJudge()
        score = judge.score(doc.raw["prompt"], prediction, doc.raw["category"])
        return {"judge_score": score, "judged": 1.0 if score is not None else 0.0}
