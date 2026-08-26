"""GSM8K (Task 4 extension): grade-school math word problems, chain-of-
thought generation, translated to Hindi and Portuguese.

Uses each language's dedicated professional/community translation rather
than a single multilingual repo, since (unlike MMLU's MMMLU) there isn't
one aligned multilingual GSM8K release:
- Hindi:      nvidia/GSM8K-Hi (row-aligned translation of the 1,319-row
  openai/gsm8k "main/test" split)
- Portuguese: Polygl0t/gsm8k-pt (1,295 rows -- not row-aligned with the
  English test split, but same task/format)

No Spanish entry: unlike Hindi/Portuguese, no comparably-maintained
Spanish GSM8K translation exists (only small, unofficial community
copies) -- spanish_bench's mgsm_direct_es (src/eval/tasks/spanish_bench.py)
already covers Spanish math word problems via lm-evaluation-harness, so
this doesn't leave Spanish without math coverage.

Scored the standard GSM8K way: both the reference and the model's
generation encode the final answer as the last number in the text (the
reference marks it after "####"); this task extracts the last number
from each and compares numerically, so the model gets credit for
reaching the right number even if its chain-of-thought differs from the
reference's.
"""

from __future__ import annotations

import re

from src.eval import hf_data
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

_NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


def _last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def _numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


class _GSM8KTask(Task):
    request_type = RequestType.GENERATE
    max_gen_toks = 256
    DATASET: str  # set by subclass

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(self.DATASET, None, "test", limit)
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_target(self, doc: Doc) -> str:
        # Full reference chain-of-thought (not just the final number) --
        # used verbatim as the fewshot demonstration, so a fewshot-primed
        # model sees "reason step by step, then give #### <answer>", the
        # same convention its own generations are graded against.
        return doc.raw["answer"]

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float]:
        gold = _last_number(doc.raw["answer"].split("####")[-1])
        pred = _last_number(prediction)
        return {"exact_match": 1.0 if _numbers_equal(pred, gold) else 0.0}


@register("gsm8k_hi")
class GSM8KHindi(_GSM8KTask):
    language = "hi"
    description = "दिए गए गणितीय शब्द समस्या को चरण दर चरण हल करें और अंत में \"#### <उत्तर>\" के रूप में अंतिम उत्तर दें।"
    stop_sequences = ["\nप्रश्न:", "\n\n"]
    DATASET = "nvidia/GSM8K-Hi"

    def doc_to_text(self, doc: Doc) -> str:
        return f"प्रश्न: {doc.raw['question']}\nउत्तर:"


@register("gsm8k_pt")
class GSM8KPortuguese(_GSM8KTask):
    language = "pt"
    description = 'Resolve o problema matemático passo a passo e dá a resposta final no formato "#### <resposta>".'
    stop_sequences = ["\nPergunta:", "\n\n"]
    DATASET = "Polygl0t/gsm8k-pt"

    def doc_to_text(self, doc: Doc) -> str:
        return f"Pergunta: {doc.raw['question']}\nResposta:"
