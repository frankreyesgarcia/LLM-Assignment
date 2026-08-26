"""portugal-basic-qa-ptcore (Task 4): 50 basic 3-way multiple-choice
questions about Portugal (geography/culture/history/cuisine/government),
in European Portuguese.

huggingface.co/datasets/duarteocarmo/portugal-basic-qa-ptcore -- `choices`
is a 3-item list, `label` is "a"/"b"/"c". Scored by loglikelihood: the
model's log-probability is compared across the 3 choice *texts* (not the
"a"/"b"/"c" letters), since a small pilot-scale checkpoint -- especially
in pretrain mode -- has no reason to have learned letter-choice
conventions.
"""

from __future__ import annotations

from src.eval import hf_data
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "duarteocarmo/portugal-basic-qa-ptcore"
_LABEL_TO_INDEX = {"a": 0, "b": 1, "c": 2}


@register("portugal_basic_qa")
class PortugalBasicQA(Task):
    language = "pt"
    request_type = RequestType.LOGLIKELIHOOD
    description = "Responde à seguinte pergunta sobre Portugal com a opção correta."

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(DATASET, None, "val", limit)
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_text(self, doc: Doc) -> str:
        return f"Pergunta: {doc.raw['question']}\nResposta:"

    def doc_to_choices(self, doc: Doc) -> list[str]:
        return list(doc.raw["choices"])

    def gold_index(self, doc: Doc) -> int:
        return _LABEL_TO_INDEX[doc.raw["label"]]
