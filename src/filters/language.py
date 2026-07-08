"""Etapa 2 — Language filter (TASK1-PLAN.md).

Hard filter: `language` is already assigned by the adapter and must be in {pt, es, hi}.
Soft filter (optional): verification with `langid` over a text sample; drops the
document if confidence in the declared language is below the threshold.
"""

from __future__ import annotations

import math

from src.ingest.base import VALID_LANGUAGES, Document

SOFT_FILTER_SAMPLE_CHARS = 2000
DEFAULT_MIN_CONFIDENCE = 0.65

try:
    import langid

    _LANGID_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _LANGID_AVAILABLE = False

# langid uses 2-letter ISO 639-1 codes, which already match pt/es/hi.
_LANGID_TO_TARGET = {"pt": "pt", "es": "es", "hi": "hi"}


def hard_filter(doc: Document) -> bool:
    """True if the document should be kept (language within scope)."""
    return doc.language in VALID_LANGUAGES


def target_language_confidence(doc: Document) -> float:
    """Confidence (softmax over langid's log-probs) in the doc's declared language.

    Returns 1.0 if `langid` is not installed (soft filter is "optional but
    recommended" per the plan — it must never block execution).
    """
    if not _LANGID_AVAILABLE:
        return 1.0

    sample = doc.text[:SOFT_FILTER_SAMPLE_CHARS]
    ranked = dict(langid.rank(sample))  # {lang: log_prob}, unnormalized
    target = _LANGID_TO_TARGET.get(doc.language)
    if target not in ranked:
        return 0.0

    max_log_prob = max(ranked.values())
    total = sum(math.exp(v - max_log_prob) for v in ranked.values())
    return math.exp(ranked[target] - max_log_prob) / total


def soft_filter(doc: Document, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> bool:
    """True if the doc passes the soft check: confidence in declared language >= threshold."""
    return target_language_confidence(doc) >= min_confidence
