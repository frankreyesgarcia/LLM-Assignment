"""Etapa 4 — Quality heuristics (TASK1-PLAN.md), C4/Gopher/FineWeb-style.

Each rule is a small composable function `(text, config) -> str | None`
returning a drop_reason string, or None if the document passes. Thresholds
are configurable (see configs/filters.yaml) instead of hardcoded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

_WORD_RE = re.compile(r"\S+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?।॥]+")  # includes Devanagari danda/double danda
# "Noise" symbols associated with code/templates/spam — deliberately excludes
# normal prose punctuation (. , ' " ! ? - : ; ( ) …) which would otherwise
# flag almost every well-formed document (see pilot run on corpus-ptbr-v2,
# where a naive "any non-word char" regex dropped 97% of documents). Also
# excludes `*` and `_`: markdown emphasis (**bold**, _italic_) and
# snake_case terms (e.g. "smtp_tls_policy_maps") are legitimate in modern
# text and were the single biggest false-positive source in the pilot.
_SYMBOL_RE = re.compile(r"[{}\[\]<>|~^`$%&=+\\@#]")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class QualityConfig:
    min_words: int = 50
    max_words: int = 100_000
    min_sentences: int = 3
    max_symbol_word_ratio: float = 0.1
    max_digit_word_ratio: float = 0.25
    max_single_word_line_ratio: float = 0.3
    max_ngram_repetitions: int = 3
    ngram_size: int = 4
    banned_substrings: list[str] = field(
        default_factory=lambda: ["lorem ipsum", "{", "javascript"]
    )
    toxic_words: list[str] = field(default_factory=list)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def rule_word_count(text: str, cfg: QualityConfig) -> str | None:
    n = len(_words(text))
    if n < cfg.min_words:
        return "too_short"
    if n > cfg.max_words:
        return "too_long"
    return None


def rule_sentence_count(text: str, cfg: QualityConfig) -> str | None:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < cfg.min_sentences:
        return "too_few_sentences"
    return None


def rule_symbol_ratio(text: str, cfg: QualityConfig) -> str | None:
    words = _words(text)
    if not words:
        return "empty"
    symbols = len(_SYMBOL_RE.findall(text))
    if symbols / len(words) > cfg.max_symbol_word_ratio:
        return "too_many_symbols"
    return None


def rule_digit_ratio(text: str, cfg: QualityConfig) -> str | None:
    words = _words(text)
    if not words:
        return "empty"
    digits = len(_DIGIT_RE.findall(text))
    if digits / len(words) > cfg.max_digit_word_ratio:
        return "too_many_digits"
    return None


def rule_single_word_lines(text: str, cfg: QualityConfig) -> str | None:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return "empty"
    single_word = sum(1 for line in lines if len(_words(line)) == 1)
    if single_word / len(lines) > cfg.max_single_word_line_ratio:
        return "too_many_single_word_lines"
    return None


def rule_banned_substrings(text: str, cfg: QualityConfig) -> str | None:
    lowered = text.lower()
    for substr in cfg.banned_substrings:
        if substr.lower() in lowered:
            return f"banned_substring:{substr}"
    return None


def rule_toxic_words(text: str, cfg: QualityConfig) -> str | None:
    if not cfg.toxic_words:
        return None
    lowered = text.lower()
    for word in cfg.toxic_words:
        if word.lower() in lowered:
            return f"toxic_word:{word}"
    return None


def rule_ngram_repetition(text: str, cfg: QualityConfig) -> str | None:
    words = _words(text)
    n = cfg.ngram_size
    if len(words) < n:
        return None
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] > cfg.max_ngram_repetitions:
            return "ngram_repetition"
    return None


DEFAULT_RULES: list[Callable[[str, QualityConfig], str | None]] = [
    rule_word_count,
    rule_sentence_count,
    rule_symbol_ratio,
    rule_digit_ratio,
    rule_single_word_lines,
    rule_banned_substrings,
    rule_toxic_words,
    rule_ngram_repetition,
]


def check_quality(
    text: str,
    cfg: QualityConfig | None = None,
    rules: list[Callable[[str, QualityConfig], str | None]] | None = None,
) -> str | None:
    """Run all quality rules in order; return the first drop_reason, or None if kept."""
    cfg = cfg or QualityConfig()
    rules = rules or DEFAULT_RULES
    for rule in rules:
        reason = rule(text, cfg)
        if reason is not None:
            return reason
    return None
