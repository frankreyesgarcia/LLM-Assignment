"""Source-of-truth for the 4 sources this pipeline supports.

Scope was deliberately pruned to just these 4 rows -- pt/es each get one
fineweb-2 config, hi gets fineweb-2 plus Sangraha's "verified" split
(human-verified sites + OCR'd PDFs + transcribed speech -- the one hi
source that isn't itself another Common-Crawl derivative, unlike
fineweb-2/HPLT/CulturaX). Filtering (language/quality/cleaning) and
dedup are handled by `scripts/run_dedup_datatrove.py` (datatrove), not
by a bespoke per-source adapter layer -- these are plain flat-parquet
sources with a `text` column, so there's nothing left for a custom
ingestion layer to add.
"""

from __future__ import annotations

VALID_LANGUAGES = {"pt", "es", "hi"}

# repo_id + glob pattern (relative to a row's local download dir, e.g.
# `data/raw/<row>/...`) -- confirmed against each repo's real file layout,
# not guessed. Used by scripts/download_sources.py (bulk pre-download) and
# scripts/run_dedup_datatrove.py (datatrove ParquetReader construction).
# Language is looked up via ROWS_BY_LANGUAGE below, not stored per-row here.
SOURCE_SPECS: dict[str, dict] = {
    "pt-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "pattern": "data/por_Latn/train/*.parquet",
        "text_key": "text",
        "id_key": "id",
    },
    "es-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "pattern": "data/spa_Latn/train/*.parquet",
        "text_key": "text",
        "id_key": "id",
    },
    "hi-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "pattern": "data/hin_Deva/train/*.parquet",
        "text_key": "text",
        "id_key": "id",
    },
    "hi-sangraha": {
        "repo_id": "ai4bharat/sangraha",
        "pattern": "verified/hin/*.parquet",
        "text_key": "text",
        "id_key": "doc_id",
    },
}

# Rows per language, in the order they're fed into the shared per-language
# MinHash pass (scripts/run_dedup_datatrove.py chains readers back-to-back
# so hi's two sources dedup against each other, not just internally).
ROWS_BY_LANGUAGE: dict[str, list[str]] = {
    "pt": ["pt-fineweb2"],
    "es": ["es-fineweb2"],
    "hi": ["hi-fineweb2", "hi-sangraha"],
}
