"""Tokenizer quality metrics: fertility, compression ratio, per-language report.

Used both to compare our trained tokenizer's own candidate vocab sizes
(`scripts/sweep_vocab_size.py`) and to compare it against existing
tokenizers (`scripts/compare_baselines.py`) -- accepts anything with an
`.encode` method, so it works uniformly on a raw `tokenizers.Tokenizer`
and on any `transformers` tokenizer loaded via `AutoTokenizer`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_WORD_RE = re.compile(r"\S+")


def _num_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _num_tokens(tokenizer: Any, text: str) -> int:
    """Token count for `text`, without special tokens (so bos/eos/pad
    added by different tokenizers don't skew the comparison)."""
    try:
        # transformers.PreTrainedTokenizer{,Fast} style
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # raw tokenizers.Tokenizer style
        return len(tokenizer.encode(text).ids)


def fertility(tokenizer: Any, texts: Iterable[str]) -> float:
    """Mean tokens-per-word over `texts` (lower = more efficient)."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        total_tokens += _num_tokens(tokenizer, text)
        total_words += _num_words(text)
    if total_words == 0:
        return float("nan")
    return total_tokens / total_words


def compression_ratio(tokenizer: Any, texts: Iterable[str]) -> float:
    """Mean UTF-8-bytes-per-token over `texts` (higher = more compact encoding)."""
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_tokens += _num_tokens(tokenizer, text)
    if total_tokens == 0:
        return float("nan")
    return total_bytes / total_tokens


def per_language_report(
    tokenizer: Any, docs_by_lang: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Fertility + compression ratio per language, plus an `overall` row.

    A single averaged number would hide one language being starved of
    merges by a higher-resource language sharing the same vocab.
    """
    report: dict[str, dict[str, float]] = {}
    all_texts: list[str] = []
    for lang, texts in docs_by_lang.items():
        report[lang] = {
            "fertility": fertility(tokenizer, texts),
            "compression_ratio": compression_ratio(tokenizer, texts),
        }
        all_texts.extend(texts)
    report["overall"] = {
        "fertility": fertility(tokenizer, all_texts),
        "compression_ratio": compression_ratio(tokenizer, all_texts),
    }
    return report
