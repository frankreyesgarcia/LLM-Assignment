"""Unified schema and adapter interface (see TASK1-PLAN.md, section 2.2 / Etapa 1)."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi

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


def resolve_remote_data_files(repo_id: str, patterns: list[str]) -> list[str]:
    """Glob-match `patterns` (same shape as registry.SOURCE_DOWNLOAD_SPECS,
    e.g. "pt/high/*.parquet") against a repo's file listing, returning
    `hf://datasets/...` URIs `load_dataset("parquet", data_files=...)` can
    stream directly. Used for repos whose own loading script/config can't
    be used as-is (see GenericTextAdapter's `remote_glob_patterns`).

    Lists each pattern's non-wildcard prefix directory only (via
    `list_repo_tree(path_in_repo=...)`), not the whole repo -- for
    EuroWeb-2512 (10,219 parquet files across every language/tier) or
    CulturaX, a flat `list_repo_files()` + client-side filter would pull
    down and scan a listing orders of magnitude larger than what any single
    row actually needs.
    """
    matched: set[str] = set()
    for pattern in patterns:
        prefix = pattern.rsplit("/", 1)[0] if "/" in pattern else ""
        for item in HfApi().list_repo_tree(repo_id, path_in_repo=prefix, repo_type="dataset"):
            if fnmatch.fnmatch(item.path, pattern):
                matched.add(item.path)
    return sorted(f"hf://datasets/{repo_id}/{f}" for f in matched)


def load_local_or_remote_dataset(
    repo_id: str,
    config: str | None,
    split: str,
    streaming: bool,
    local_files: list[Path] | None,
    trust_remote_code: bool = False,
    remote_glob_patterns: list[str] | None = None,
):
    """Shared by GenericTextAdapter and HPLTAdapter: read `local_files`
    (already pulled down by scripts/download_sources.py, see
    registry.build_adapter's `raw_dir` param) if given. Otherwise, for repos
    whose loading script can't be used directly (e.g. CulturaX's own script
    predates `datasets>=4` dropping trust_remote_code, same class of issue
    as CarolinaAdapter) or that need more files than a single config/split
    covers (e.g. EuroWeb-2512's multiple quality tiers), resolve an explicit
    remote file list via `remote_glob_patterns` instead. Otherwise stream via
    the repo's own config/split as before.
    """
    if local_files:
        return load_dataset("parquet", data_files=[str(p) for p in local_files], split="train", streaming=streaming)
    if remote_glob_patterns:
        uris = resolve_remote_data_files(repo_id, remote_glob_patterns)
        return load_dataset("parquet", data_files=uris, split="train", streaming=streaming)
    return load_dataset(repo_id, config, split=split, streaming=streaming, trust_remote_code=trust_remote_code)


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
