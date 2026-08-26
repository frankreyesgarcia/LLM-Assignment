"""ChatRAG-Hi (Task 4): Hindi multi-turn conversational QA-over-documents.

huggingface.co/datasets/nvidia/ChatRAG-Hi -- 8 source-dataset configs
(doc2dial, doqa_{cooking,movies,travel}, hybridial, inscit, qrecc, quac),
each row: `messages` (chat history ending on a user turn needing a reply),
`ctxs` (100 retrieved passages, ranked, {title, text}), `answers` (gold
reference strings). Retrieved context is injected as a system message
(posttrain) / a leading "Sandarbh:" block (pretrain); only the top-ranked
passage is kept, truncated -- this repo's pilot-scale checkpoints run with
small block_size (see src/model/train.py), so keeping the prompt short
matters more than exhaustive RAG context.
"""

from __future__ import annotations

from src.eval import hf_data, metrics
from src.eval.registry import register
from src.eval.tasks.base import Task
from src.eval.types import Doc, RequestType

DATASET = "nvidia/ChatRAG-Hi"
CONFIGS = ["doc2dial", "doqa_cooking", "doqa_movies", "doqa_travel", "hybridial", "inscit", "qrecc", "quac"]
TOP_N_CTXS = 1
MAX_CTX_CHARS = 500


@register("chatrag_hi")
class ChatRAGHi(Task):
    language = "hi"
    request_type = RequestType.GENERATE
    description = "आप एक सहायक हैं जो दिए गए संदर्भ के आधार पर सवालों के जवाब देता है।"
    stop_sequences = ["\nUser:", "\n\n"]
    max_gen_toks = 96

    def load_docs(self, limit: int | None = None) -> list[Doc]:
        per_config_limit = None if limit is None else max(1, -(-limit // len(CONFIGS)))
        docs = []
        for config in CONFIGS:
            rows = hf_data.load_rows(DATASET, config, "test", per_config_limit)
            for row in rows:
                docs.append(Doc(len(docs), row))
                if limit is not None and len(docs) >= limit:
                    return docs
        return docs

    def _context_block(self, doc: Doc) -> str:
        passages = doc.raw["ctxs"][:TOP_N_CTXS]
        parts = []
        for passage in passages:
            text = passage["text"][:MAX_CTX_CHARS]
            parts.append(f"{passage['title']}: {text}" if passage["title"] else text)
        return "\n".join(parts)

    def doc_to_messages(self, doc: Doc) -> list[dict]:
        system = f"{self.description}\n\nसंदर्भ:\n{self._context_block(doc)}"
        return [{"role": "system", "content": system}] + list(doc.raw["messages"])

    def doc_to_text(self, doc: Doc) -> str:
        lines = [f"संदर्भ: {self._context_block(doc)}"]
        for turn in doc.raw["messages"]:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def doc_to_target(self, doc: Doc) -> str:
        return doc.raw["answers"][0]

    def process_result(self, doc: Doc, prediction: str) -> dict[str, float]:
        return {"token_f1": metrics.token_f1(prediction, list(doc.raw["answers"]))}
