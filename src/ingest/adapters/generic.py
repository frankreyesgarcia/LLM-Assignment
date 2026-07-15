"""GenericTextAdapter: for single-language sources with a flat text column.

Covers corpus-carolina, Portuguese-PD, corpus-ptbr-v2 (TASK1-PLAN.md sec 5, Etapa 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from src.ingest.base import Document, SourceAdapter, load_local_or_remote_dataset


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
        local_files: list[Path] | None = None,
        remote_glob_patterns: list[str] | None = None,
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
        # If set (scripts/download_sources.py already pulled this source's
        # files to local disk), read those directly instead of streaming
        # over the network -- see registry.build_adapter's `raw_dir` param.
        # A single load_dataset(..., streaming=True) connection measured
        # ~8 MB/s regardless of link speed; many concurrent connections
        # (what the download step uses) measured ~65 MB/s aggregate on the
        # same repos.
        self.local_files = local_files
        # Fallback for the no-local_files (streaming) case, for repos whose
        # own repo_id/config/split loading doesn't work as-is -- see
        # load_local_or_remote_dataset.
        self.remote_glob_patterns = remote_glob_patterns

    def iter_documents(self) -> Iterator[Document]:
        ds = load_local_or_remote_dataset(
            self.repo_id,
            self.config,
            self.split,
            self.streaming,
            self.local_files,
            self.trust_remote_code,
            self.remote_glob_patterns,
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
