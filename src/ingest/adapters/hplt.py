"""HPLTAdapter: for HPLT/HPLT2.0_cleaned.

Unlike the other sources, HPLT's `lang` and `prob` columns are lists (one
entry per detected language across the whole doc, sorted by probability),
not a single scalar -- so it needs its own light adapter instead of
GenericTextAdapter (see configs/sources.yaml, rows es-hplt2 / hi-hplt2).
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from src.ingest.base import Document, SourceAdapter, normalize_language_code


class HPLTAdapter(SourceAdapter):
    def __init__(
        self,
        name: str,
        repo_id: str,
        config: str,
        language: str,
        split: str = "train",
        limit: int | None = None,
        streaming: bool = True,
        min_lang_prob: float = 0.5,
    ) -> None:
        self.name = name
        self.repo_id = repo_id
        self.config = config
        self.language = language
        self.split = split
        self.limit = limit
        self.streaming = streaming
        self.min_lang_prob = min_lang_prob

    def iter_documents(self) -> Iterator[Document]:
        ds = load_dataset(self.repo_id, self.config, split=self.split, streaming=self.streaming)

        count = 0
        for row in ds:
            if self.limit is not None and count >= self.limit:
                break

            text = (row.get("text") or "").strip()
            if not text:
                continue

            langs = row.get("lang") or []
            probs = row.get("prob") or []
            if not langs:
                continue
            top_lang = normalize_language_code(langs[0])
            top_prob = probs[0] if probs else 0.0
            if top_lang != self.language or top_prob < self.min_lang_prob:
                continue

            metadata = {
                "hplt_filter": row.get("filter"),
                "collection": row.get("collection"),
                "lang_probs": dict(zip(langs, probs)),
                "robotstxt": row.get("robotstxt"),
            }

            yield Document(
                text=text,
                language=self.language,
                source=self.name,
                source_id=str(row.get("id") or ""),
                url=row.get("u"),
                metadata=metadata,
            )
            count += 1
