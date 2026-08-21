"""Scoring functions shared across GENERATE tasks (Task 4).

Kept free of any model/dataset imports -- pure text-in, float-out -- so
they're trivial to unit test and reuse across benchmarks.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    Deliberately aggressive (SQuAD-style normalization) so surface
    variation -- capitalization, accents dropped by a lossy tokenizer,
    trailing punctuation -- doesn't count against a semantically correct
    answer.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_match(prediction: str, references: list[str]) -> float:
    pred = normalize_text(prediction)
    return 1.0 if any(pred == normalize_text(ref) for ref in references) else 0.0


def first_word_match(prediction: str, references: list[str]) -> float:
    """Like exact_match, but only the first whitespace-delimited token of
    `prediction` has to match -- for cloze/next-word tasks (CALAME-PT)
    where the model may keep generating past the target word.
    """
    pred_tokens = normalize_text(prediction).split()
    pred_word = pred_tokens[0] if pred_tokens else ""
    return 1.0 if any(pred_word == normalize_text(ref) for ref in references) else 0.0


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common: dict[str, int] = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in ref_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: list[str]) -> float:
    """Token-overlap F1 (SQuAD-style) against the best-matching reference.

    Used for open-ended answers (PT-Culture, ChatRAG-Hi) where exact_match
    is too strict -- a paraphrase with the right content words should still
    score well.
    """
    pred_tokens = normalize_text(prediction).split()
    return max((_f1(pred_tokens, normalize_text(ref).split()) for ref in references), default=0.0)


def accuracy(correct_flags: list[bool]) -> float:
    return sum(correct_flags) / len(correct_flags) if correct_flags else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
