"""GSM8K (Task 4 extension): grade-school math word problems, chain-of-
thought generation, translated to Hindi, Portuguese, and Spanish.

Uses each language's dedicated translation rather than a single
multilingual repo, since (unlike MMLU's MMMLU) there isn't one aligned
multilingual GSM8K release:
- Hindi:      nvidia/GSM8K-Hi (row-aligned translation of the 1,319-row
  openai/gsm8k "main/test" split)
- Portuguese: Polygl0t/gsm8k-pt (1,295 rows -- not row-aligned with the
  English test split, but same task/format)
- Spanish:    frank-rg/gsm8k-es -- no comparably-maintained Spanish GSM8K
  translation existed on HF Hub at the time (checked; only small
  unofficial community copies), so this repo's translation was produced
  in-house (scripts/translate_gsm8k_es.py + scripts/slurm/14_translate_gsm8k_es.sh),
  row-aligned with openai/gsm8k "main/test" using a locally-served
  GLM-4.7-Flash. The final "#### <number>" answer is re-appended verbatim
  from the English source after translation regardless of what the
  translation model produced there, since that marker is the
  machine-checkable grading target, not prose -- a translated/hallucinated
  number would silently corrupt every eval run against it.

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


@register("gsm8k_es")
class GSM8KSpanish(_GSM8KTask):
    language = "es"
    description = 'Resuelve el problema matemático paso a paso y da la respuesta final en el formato "#### <respuesta>".'
    stop_sequences = ["\nPregunta:", "\n\n"]
    DATASET = "frank-rg/gsm8k-es"

    def doc_to_text(self, doc: Doc) -> str:
        return f"Pregunta: {doc.raw['question']}\nRespuesta:"
