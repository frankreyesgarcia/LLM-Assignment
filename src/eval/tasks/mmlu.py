"""MMLU (Task 4 extension): 4-way multiple-choice general knowledge,
translated to pt/es/hi via OpenAI's own MMMLU release
(huggingface.co/datasets/openai/MMMLU) rather than the English-only
cais/mmlu -- MMMLU is a professional translation of the exact same 14,042
cais/mmlu "all/test" questions, aligned row-for-row across languages
(configs PT_BR/ES_LA/HI_IN), so scores are directly comparable across
the three languages and against English MMLU results elsewhere.

Scored by loglikelihood over the four answer *texts* (A/B/C/D choices),
same convention as portugal_basic_qa.py -- not the letters -- since a
small pretrain-only checkpoint has no reason to have learned letter-choice
conventions.
"""

from __future__ import annotations

from src.eval import hf_data
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "openai/MMMLU"
_LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


class _MMLUTask(Task):
    request_type = RequestType.LOGLIKELIHOOD
    CONFIG: str  # set by subclass

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(DATASET, self.CONFIG, "test", limit)
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_choices(self, doc: Doc) -> list[str]:
        return [doc.raw["A"], doc.raw["B"], doc.raw["C"], doc.raw["D"]]

    def gold_index(self, doc: Doc) -> int:
        return _LABEL_TO_INDEX[doc.raw["Answer"]]


@register("mmlu_pt")
class MMLUPortuguese(_MMLUTask):
    language = "pt"
    description = "Responde à seguinte pergunta de conhecimento geral com a opção correta."
    CONFIG = "PT_BR"

    def doc_to_text(self, doc: Doc) -> str:
        return f"Pergunta: {doc.raw['Question']}\nResposta:"


@register("mmlu_es")
class MMLUSpanish(_MMLUTask):
    language = "es"
    description = "Responde a la siguiente pregunta de conocimiento general con la opción correcta."
    CONFIG = "ES_LA"

    def doc_to_text(self, doc: Doc) -> str:
        return f"Pregunta: {doc.raw['Question']}\nRespuesta:"


@register("mmlu_hi")
class MMLUHindi(_MMLUTask):
    language = "hi"
    description = "सही विकल्प चुनकर निम्नलिखित सामान्य ज्ञान प्रश्न का उत्तर दें।"
    CONFIG = "HI_IN"

    def doc_to_text(self, doc: Doc) -> str:
        return f"प्रश्न: {doc.raw['Question']}\nउत्तर:"
