"""Etapa 5a — Exact dedup (TASK1-PLAN.md).

SHA256 hash of normalized text (lowercase + collapsed whitespace), applied
cross-source and per-language (EuroWeb shows up in the multi bucket and in
the per-language subsets — see the note in TASK1-PLAN.md sec 2.1).
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


class ExactDeduper:
    """Stateful dedup: keeps a set of seen hashes, one instance per language."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = text_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen)
