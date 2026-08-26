"""PT-Culture_Data (Task 4): multi-turn Portuguese conversations about
festivals/traditions (QA_Short, QA_Long, Verification task types).

huggingface.co/datasets/duarteocarmo/PT-Culture_Data -- `conversations` is
a ShareGPT-style list of {"role": "human"/"gpt", "content"}, always ending
on a "gpt" turn. That final turn is the eval target: everything before it
becomes the model's context (posttrain mode: real chat messages;
pretrain mode: flattened "Humano:"/"Assistente:" text), and the model's
reply is scored by token-F1 against the real final turn.
"""

from __future__ import annotations

from src.eval import hf_data, metrics
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "duarteocarmo/PT-Culture_Data"
_ROLE_MAP = {"human": "user", "gpt": "assistant"}


@register("pt_culture")
class PTCulture(Task):
    language = "pt"
    request_type = RequestType.GENERATE
    description = "Responde às perguntas sobre cultura e festividades portuguesas."
    stop_sequences = ["\nHumano:", "\n\n"]
    max_gen_toks = 128

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        rows = hf_data.load_rows(DATASET, None, "train", limit)
        return [Doc(i, row) for i, row in enumerate(rows) if len(row["conversations"]) >= 2]

    def _history(self, doc: Doc) -> list[dict]:
        return doc.raw["conversations"][:-1]

    def _target_turn(self, doc: Doc) -> str:
        return doc.raw["conversations"][-1]["content"]

    def doc_to_messages(self, doc: Doc) -> list[dict]:
        return [{"role": _ROLE_MAP[turn["role"]], "content": turn["content"]} for turn in self._history(doc)]

    def doc_to_text(self, doc: Doc) -> str:
        lines = []
        for turn in self._history(doc):
            speaker = "Humano" if turn["role"] == "human" else "Assistente"
            lines.append(f"{speaker}: {turn['content']}")
        lines.append("Assistente:")
        return "\n".join(lines)

    def doc_to_target(self, doc: Doc) -> str:
        return self._target_turn(doc)

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float]:
        reference = self._target_turn(doc)
        return {
            "token_f1": metrics.token_f1(prediction, [reference]),
            "exact_match": metrics.exact_match(prediction, [reference]),
        }
