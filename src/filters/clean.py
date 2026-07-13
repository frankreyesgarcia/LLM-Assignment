"""Etapa 3 — Text cleaning (TASK1-PLAN.md).

Conservative normalization: no lowercasing or lemmatization — modern
pre-training doesn't need it (see the "Do NOT do" note in the plan).
"""

from __future__ import annotations

import re
import unicodedata

from src.filters.language import line_language

_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0, 32) if c not in (9, 10, 13)) + "\x7f]"
)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Basic mojibake fixes (UTF-8 wrongly decoded as Latin-1/CP1252).
_MOJIBAKE_REPLACEMENTS = {
    "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
    "Ã±": "ñ", "Ã§": "ç", "Ã£": "ã", "Ãµ": "õ", "Ã¢": "â",
    "â€™": "'", "â€œ": "“", "â€\x9d": "”", "â€“": "-",
}


def fix_mojibake(text: str) -> str:
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def unescape_literal_whitespace(text: str) -> str:
    """Turn literal backslash-escape sequences (`\\n`, `\\r\\n`, `\\t`) into
    real whitespace chars.

    Some sources (observed in EuroWeb) went through a JSON round-trip
    somewhere upstream that left the escape sequences as literal two-char
    text instead of decoding them, so `text.split("\\n")` elsewhere in this
    pipeline (here, and in quality.py's per-line rules) silently treats the
    whole document as a single line for those rows.
    """
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text.replace("\x00", ""))


def collapse_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def dedupe_consecutive_lines(text: str) -> str:
    """Collapse consecutive identical lines (reduces repeated web boilerplate)."""
    out_lines: list[str] = []
    prev: str | None = None
    for line in text.split("\n"):
        if line == prev and line != "":
            continue
        out_lines.append(line)
        prev = line
    return "\n".join(out_lines)


MIN_WORDS_FOR_LINE_DROP = 8

# langid regularly misclassifies plain Devanagari Hindi as Marathi/Nepali on
# short-to-medium lines (same script, overlapping vocabulary, too little
# context to disambiguate) -- that's a langid weakness, not evidence the line
# is foreign, so treat those calls as "close enough to hi" rather than drop.
_HINDI_ACCEPTED = {"hi", "mr", "ne"}


def strip_foreign_lines(text: str, language: str) -> str:
    """Drop lines langid confidently classifies as a different language.

    We only apply this to Hindi because Hindi has a different script (Devanagari) than the other languages we ingest.
    Also, MIN_WORDS_FOR_LINE_DROP is used to avoid dropping short lines that may be misclassified by langid, especially for Hindi eg. 1. टुण्ड्रा (Tundra Biome).
    """
    if language != "hi":
        return text
    kept = [
        line
        for line in text.split("\n")
        if len(line.split()) < MIN_WORDS_FOR_LINE_DROP
        or line_language(line) in _HINDI_ACCEPTED | {None}
    ]
    return "\n".join(kept)


def clean_text(text: str, language: str | None = None) -> str:
    """Full cleaning pipeline, in the order described in the plan (Etapa 3)."""
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = fix_mojibake(text)
    text = unescape_literal_whitespace(text)
    if language is not None:
        text = strip_foreign_lines(text, language)
    text = collapse_whitespace(text)
    text = dedupe_consecutive_lines(text)
    return text
