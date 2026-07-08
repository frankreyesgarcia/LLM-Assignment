"""GenericTextAdapter: for single-language sources with a flat text column.

Covers corpus-carolina, Portuguese-PD, corpus-ptbr-v2 (TASK1-PLAN.md sec 5, Etapa 1).
"""

from __future__ import annotations

from typing import Any, Iterator

from datasets import load_dataset

from src.ingest.base import Document, SourceAdapter


class GenericTextAdapter(SourceAdapter):
    def __init__(
        self,
        name: str,
        repo_id: str,
        language: str,
        text_column: str = "text",
        id_column: str | None = None,
        url_column: str | None = None,
        split: str = "train",
        config: str | None = None,
        limit: int | None = None,
        streaming: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.name = name
        self.repo_id = repo_id
        self.language = language
        self.text_column = text_column
        self.id_column = id_column
        self.url_column = url_column
        self.split = split
        self.config = config
        self.limit = limit
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code

    def iter_documents(self) -> Iterator[Document]:
        ds = load_dataset(
            self.repo_id,
            self.config,
            split=self.split,
            streaming=self.streaming,
            trust_remote_code=self.trust_remote_code,
        )

        count = 0
        for row in ds:
            if self.limit is not None and count >= self.limit:
                break

            text = (row.get(self.text_column) or "").strip()
            if not text:
                continue

            source_id = str(row[self.id_column]) if self.id_column and row.get(self.id_column) else ""
            url = row.get(self.url_column) if self.url_column else None
            metadata: dict[str, Any] = {
                k: v for k, v in row.items() if k not in {self.text_column, self.id_column, self.url_column}
            }

            yield Document(
                text=text,
                language=self.language,
                source=self.name,
                source_id=source_id,
                url=url,
                metadata=metadata,
            )
            count += 1
