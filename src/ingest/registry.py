"""Adapter registry: one entry per pre-training source row (TASK1-PLAN.md sec 2.1).

Central place mapping each of the 12 assignment sources to a configured
`SourceAdapter`. Sources that can't be ingested yet (script-based loading
code, gated access) raise a clear, typed error instead of silently
returning nothing -- see BLOCKED_SOURCES below.
"""

from __future__ import annotations

from pathlib import Path

from src.ingest.adapters.carolina import TAXONOMY_FOLDERS, CarolinaAdapter
from src.ingest.adapters.generic import GenericTextAdapter
from src.ingest.adapters.hplt import HPLTAdapter
from src.ingest.base import SourceAdapter

# EuroWeb-2512 has 5 quality tiers per language (high/medium-high/medium/
# medium-low/low), classified by utter-project/EuroFilter-v1 (an educational-
# quality classifier -- see its model card; it isn't a toxicity/safety
# filter). Hindi is hard-restricted to "high" only: non-high Hindi tiers are
# asserted to contain sexual content per the assignment brief -- that claim
# isn't documented in EuroWeb-2512's own dataset card or EuroFilter-v1's
# model card (checked both), so it can't be independently verified from
# this repo; treated as a hard constraint anyway since it's clearly a
# deliberate safety rule, not a guess. No such restriction is documented for
# es/pt, so those use all 5 tiers for volume (quality is still screened by
# our own downstream language/quality/dedup filters regardless of tier).
_EUROWEB_ALL_TIERS = ["high", "medium-high", "medium", "medium-low", "low"]
_EUROWEB_HI_TIERS = ["high"]

BLOCKED_SOURCES: dict[str, str] = {}

# One entry per AVAILABLE_ROWS: the Hub repo + the file glob patterns that
# cover exactly what build_adapter() below actually reads for that row (same
# config/split/language subset, nothing more). scripts/download_sources.py
# uses this to bulk-download each source with huggingface_hub.snapshot_download
# (many concurrent connections) *before* filtering runs, instead of each
# source being pulled one file at a time over a single throttled streaming
# connection -- see that script's docstring for the measured throughput
# difference. build_adapter's `raw_dir` param below glob-matches these same
# patterns against the local download to build each adapter's local_files;
# the same patterns also drive `remote_glob_patterns` for adapters whose
# repo can't be streamed via a plain repo_id/config/split load_dataset call
# (CulturaX's own loading script predates datasets>=4 dropping
# trust_remote_code, same issue as corpus-carolina; EuroWeb needs multiple
# tiers glued together, not a single split).
#
# Patterns were confirmed against each repo's real file layout (not
# guessed): EuroWeb-2512 is "{lang}/{tier}/*.parquet", fineweb-2 is
# "data/{config}/train/*.parquet", HPLT2.0_cleaned is "{config}/train-*.parquet",
# CulturaX is "{lang}/{lang}_part_*.parquet", fineweb2-bagaco2's "all"
# config maps to "fineweb2-ptpt-prototype/*.parquet" per its README
# `configs:` frontmatter (its `classification/` folder is a separate,
# unused config). Portuguese-PD, corpus-ptbr-v2, Spanish-PD-Books and
# Spanish-PD-Newspapers are single flat parquet sets -- "*.parquet" grabs
# everything relevant.
SOURCE_DOWNLOAD_SPECS: dict[str, dict] = {
    "pt-fineweb2-bagaco2": {
        "repo_id": "duarteocarmo/fineweb2-bagaco2",
        "patterns": ["fineweb2-ptpt-prototype/*.parquet"],
    },
    "pt-corpus-carolina": {
        "repo_id": "carolina-c4ai/corpus-carolina",
        "patterns": [f"corpus/{folder}/**/*.xml.gz" for folder in TAXONOMY_FOLDERS.values()],
    },
    "pt-portuguese-pd": {
        "repo_id": "PleIAs/Portuguese-PD",
        "patterns": ["*.parquet"],
    },
    "pt-corpus-ptbr-v2": {
        "repo_id": "Madras1/corpus-ptbr-v2",
        "patterns": ["data/*.parquet"],
    },
    "pt-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "patterns": ["data/por_Latn/train/*.parquet"],
    },
    "pt-euroweb": {
        "repo_id": "utter-project/EuroWeb-2512",
        "patterns": [f"pt/{tier}/*.parquet" for tier in _EUROWEB_ALL_TIERS],
    },
    "pt-hplt2": {
        "repo_id": "HPLT/HPLT2.0_cleaned",
        "patterns": ["por_Latn/train-*.parquet"],
    },
    "pt-culturax": {
        "repo_id": "uonlp/CulturaX",
        "patterns": ["pt/pt_part_*.parquet"],
    },
    "es-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "patterns": ["data/spa_Latn/train/*.parquet"],
    },
    "es-euroweb": {
        "repo_id": "utter-project/EuroWeb-2512",
        "patterns": [f"es/{tier}/*.parquet" for tier in _EUROWEB_ALL_TIERS],
    },
    "es-hplt2": {
        "repo_id": "HPLT/HPLT2.0_cleaned",
        "patterns": ["spa_Latn/train-*.parquet"],
    },
    "es-culturax": {
        "repo_id": "uonlp/CulturaX",
        "patterns": ["es/es_part_*.parquet"],
    },
    "es-spanish-pd-books": {
        "repo_id": "PleIAs/Spanish-PD-Books",
        "patterns": ["*.parquet"],
    },
    "es-spanish-pd-newspapers": {
        "repo_id": "PleIAs/Spanish-PD-Newspapers",
        "patterns": ["*.parquet"],
    },
    "hi-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "patterns": ["data/hin_Deva/train/*.parquet"],
    },
    "hi-euroweb": {
        "repo_id": "utter-project/EuroWeb-2512",
        "patterns": [f"hi/{tier}/*.parquet" for tier in _EUROWEB_HI_TIERS],
    },
    "hi-hplt2": {
        "repo_id": "HPLT/HPLT2.0_cleaned",
        "patterns": ["hin_Deva/train-*.parquet"],
    },
    "hi-culturax": {
        "repo_id": "uonlp/CulturaX",
        "patterns": ["hi/hi_part_*.parquet"],
    },
}


def local_files_for(raw_dir: Path, row: str) -> list[Path]:
    """Glob-match SOURCE_DOWNLOAD_SPECS' patterns for `row` against a local
    download directory (raw_dir / row, as written by scripts/download_sources.py).
    """
    spec = SOURCE_DOWNLOAD_SPECS[row]
    local_dir = raw_dir / row
    files: list[Path] = []
    for pattern in spec["patterns"]:
        files.extend(sorted(local_dir.glob(pattern)))
    return files


def _build_euroweb(row: str, language: str, **kwargs) -> SourceAdapter:
    patterns = SOURCE_DOWNLOAD_SPECS[row]["patterns"]
    if language == "hi" and patterns != [f"hi/{tier}/*.parquet" for tier in _EUROWEB_HI_TIERS]:
        # Defense in depth for a content-safety rule (non-"high" Hindi
        # tiers are asserted to contain sexual content per the assignment,
        # see the _EUROWEB_ALL_TIERS/_EUROWEB_HI_TIERS comment above): catch
        # SOURCE_DOWNLOAD_SPECS["hi-euroweb"] ever being edited to include
        # other tiers, since nothing else here re-derives or re-checks this.
        raise ValueError(f"EuroWeb-2512 Hindi MUST use only the 'high' tier -- got patterns {patterns!r}")
    return GenericTextAdapter(
        name="EuroWeb-2512",
        repo_id="utter-project/EuroWeb-2512",
        language=language,
        remote_glob_patterns=patterns,
        **kwargs,
    )


def _build_culturax(row: str, language: str, **kwargs) -> SourceAdapter:
    return GenericTextAdapter(
        name="CulturaX",
        repo_id="uonlp/CulturaX",
        language=language,
        url_column="url",
        remote_glob_patterns=SOURCE_DOWNLOAD_SPECS[row]["patterns"],
        **kwargs,
    )


def build_adapter(row: str, limit: int | None = None, raw_dir: Path | None = None) -> SourceAdapter:
    """Build a configured adapter for one of the source rows.

    `row` matches the `row` key used in scripts/inspect_sources.py and
    configs/sources.yaml (e.g. "pt-corpus-ptbr-v2", "hi-euroweb").

    If `raw_dir` is given (scripts/download_sources.py already ran), the
    adapter reads that row's already-downloaded local files instead of
    streaming from the Hub -- see SOURCE_DOWNLOAD_SPECS above.
    """
    if row in BLOCKED_SOURCES:
        raise NotImplementedError(f"{row} is blocked: {BLOCKED_SOURCES[row]}")

    # pt-corpus-carolina resolves its own local files below (it needs
    # relative-path labels for per-doc taxonomy metadata, not just a flat
    # list) -- skip the glob here so it isn't computed and discarded.
    local_files = local_files_for(raw_dir, row) if raw_dir is not None and row != "pt-corpus-carolina" else None

    if row == "es-euroweb":
        return _build_euroweb(row, language="es", limit=limit, local_files=local_files)
    if row == "pt-euroweb":
        return _build_euroweb(row, language="pt", limit=limit, local_files=local_files)
    if row == "hi-euroweb":
        return _build_euroweb(row, language="hi", limit=limit, local_files=local_files)

    if row == "es-culturax":
        return _build_culturax(row, language="es", limit=limit, local_files=local_files)
    if row == "pt-culturax":
        return _build_culturax(row, language="pt", limit=limit, local_files=local_files)
    if row == "hi-culturax":
        return _build_culturax(row, language="hi", limit=limit, local_files=local_files)

    if row == "pt-fineweb2-bagaco2":
        return GenericTextAdapter(
            name="fineweb2-bagaco2",
            repo_id="duarteocarmo/fineweb2-bagaco2",
            language="pt",
            config="all",
            id_column="id",
            url_column="url",
            limit=limit,
            local_files=local_files,
        )
    if row == "pt-corpus-carolina":
        return CarolinaAdapter(
            name="corpus-carolina", limit=limit, local_dir=(raw_dir / row) if raw_dir is not None else None
        )
    if row == "pt-portuguese-pd":
        return GenericTextAdapter(
            name="Portuguese-PD",
            repo_id="PleIAs/Portuguese-PD",
            language="pt",
            id_column="identifier",
            limit=limit,
            local_files=local_files,
        )
    if row == "pt-corpus-ptbr-v2":
        return GenericTextAdapter(
            name="corpus-ptbr-v2",
            repo_id="Madras1/corpus-ptbr-v2",
            language="pt",
            limit=limit,
            local_files=local_files,
        )
    if row == "pt-fineweb2":
        return GenericTextAdapter(
            name="fineweb-2",
            repo_id="HuggingFaceFW/fineweb-2",
            language="pt",
            config="por_Latn",
            id_column="id",
            url_column="url",
            limit=limit,
            local_files=local_files,
        )
    if row == "pt-hplt2":
        return HPLTAdapter(
            name="HPLT2.0_cleaned",
            repo_id="HPLT/HPLT2.0_cleaned",
            config="por_Latn",
            language="pt",
            limit=limit,
            local_files=local_files,
        )
    if row == "es-fineweb2":
        return GenericTextAdapter(
            name="fineweb-2",
            repo_id="HuggingFaceFW/fineweb-2",
            language="es",
            config="spa_Latn",
            id_column="id",
            url_column="url",
            limit=limit,
            local_files=local_files,
        )
    if row == "es-spanish-pd-books":
        return GenericTextAdapter(
            name="Spanish-PD-Books",
            repo_id="PleIAs/Spanish-PD-Books",
            language="es",
            id_column="identifier",
            limit=limit,
            local_files=local_files,
        )
    if row == "es-spanish-pd-newspapers":
        return GenericTextAdapter(
            name="Spanish-PD-Newspapers",
            repo_id="PleIAs/Spanish-PD-Newspapers",
            language="es",
            id_column="id",
            limit=limit,
            local_files=local_files,
        )
    if row == "hi-fineweb2":
        return GenericTextAdapter(
            name="fineweb-2",
            repo_id="HuggingFaceFW/fineweb-2",
            language="hi",
            config="hin_Deva",
            id_column="id",
            url_column="url",
            limit=limit,
            local_files=local_files,
        )
    if row == "es-hplt2":
        return HPLTAdapter(
            name="HPLT2.0_cleaned",
            repo_id="HPLT/HPLT2.0_cleaned",
            config="spa_Latn",
            language="es",
            limit=limit,
            local_files=local_files,
        )
    if row == "hi-hplt2":
        return HPLTAdapter(
            name="HPLT2.0_cleaned",
            repo_id="HPLT/HPLT2.0_cleaned",
            config="hin_Deva",
            language="hi",
            limit=limit,
            local_files=local_files,
        )

    raise KeyError(f"Unknown source row: {row!r}")


# Rows actually usable today. Excludes the "multi" rows (multi-euroweb,
# multi-culturax), which are covered by their per-language counterparts --
# see TASK1-PLAN.md sec 2.1 note on EuroWeb-2512 single ingestion; the same
# applies to CulturaX now that it's accessible.
AVAILABLE_ROWS: list[str] = [
    "pt-fineweb2-bagaco2",
    "pt-corpus-carolina",
    "pt-portuguese-pd",
    "pt-corpus-ptbr-v2",
    "pt-fineweb2",
    "pt-euroweb",
    "pt-hplt2",
    "pt-culturax",
    "es-fineweb2",
    "es-euroweb",
    "es-hplt2",
    "es-culturax",
    "es-spanish-pd-books",
    "es-spanish-pd-newspapers",
    "hi-fineweb2",
    "hi-euroweb",
    "hi-hplt2",
    "hi-culturax",
]
