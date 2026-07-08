"""Unified schema and adapter interface (see TASK1-PLAN.md, section 2.2 / Etapa 1)."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

VALID_LANGUAGES = {"pt", "es", "hi"}

LANGUAGE_CODE_MAP = {
    "spa": "es",
    "spa_latn": "es",
    "es": "es",
    "por": "pt",
    "pt": "pt",
    "ptbr": "pt",
    "pt-br": "pt",
    "hin": "hi",
    "hin_deva": "hi",
    "hi": "hi",
}


def normalize_language_code(code: str) -> str | None:
    """Map a raw language/script code to pt|es|hi, or None if unrecognized."""
    if not code:
        return None
    return LANGUAGE_CODE_MAP.get(code.strip().lower())


@dataclass
class Document:
    """A single row of the unified dataset (schema defined in TASK1-PLAN.md, sec 2.2)."""

    text: str
    language: str
    source: str
    source_id: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"language must be one of {VALID_LANGUAGES}, got: {self.language!r}"
            )
        if not self.source_id:
            self.source_id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        # `metadata` is serialized to a JSON string, not kept as a nested
        # struct: each source contributes different keys/types (e.g.
        # `educational_score` is int64 for fineweb2-bagaco2 but float64 for
        # EuroWeb), and pyarrow.concat_tables can't reconcile conflicting
        # struct field types when building the cross-language `all` config
        # (Etapa 6) -- verified: it raises ArrowTypeError. A JSON string
        # column sidesteps that entirely.
        return {
            "text": self.text,
            "language": self.language,
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "metadata": json.dumps(self.metadata, ensure_ascii=False, default=str),
        }


class SourceAdapter(ABC):
    """Interface every dataset/family adapter must implement (Etapa 1)."""

    #: short name used in the `source` column and in configs/sources.yaml
    name: str

    @abstractmethod
    def iter_documents(self) -> Iterator[Document]:
        """Yield `Document`s already mapped to the unified schema.

        Implementations should:
        - Use streaming (datasets.load_dataset(..., streaming=True)) where applicable.
        - Drop rows with no text or empty text after stripping.
        - Assign `language` normalized to pt/es/hi (see normalize_language_code).
        """
        raise NotImplementedError
