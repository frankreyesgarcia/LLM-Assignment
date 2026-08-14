"""CALAME-PT (Task 4): Portuguese next-word cloze benchmark, LAMBADA-style.

huggingface.co/datasets/NOVA-vision-language/calame-pt -- `sentence` is a
passage with its final word removed, `last_word` is the target. Scored by
generating a short continuation and checking whether its first word
matches (metrics.first_word_match), the dataset card's implicit metric
(no fixed distractor set to rank via loglikelihood, unlike the
multiple-choice tasks here).
"""

from __future__ import annotations

from src.eval import hf_data, metrics
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "NOVA-vision-language/calame-pt"
# "all" = both the handwritten and GPT-3.5-generated (human-reviewed) subsets.
CONFIG = "all"


@register("calame_pt")
class CalamePT(Task):
    language = "pt"
    request_type = RequestType.GENERATE
    description = "Completa a frase seguinte com a palavra mais provável."
    stop_sequences = ["\n", ".", ",", ";", "!", "?"]
    max_gen_toks = 6

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(DATASET, CONFIG, "train", limit)
        return [Doc(i, row) for i, row in enumerate(rows)]

    def doc_to_text(self, doc: Doc) -> str:
        return doc.raw["sentence"]

    def doc_to_target(self, doc: Doc) -> str:
        return doc.raw["last_word"]

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float]:
        return {"accuracy": metrics.first_word_match(prediction, [doc.raw["last_word"]])}
