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

# EuroWeb-2512's "high" quality tier is used by default for all languages:
# it's mandatory for Hindi (other splits contain sexual content, per the
# assignment) and is a reasonable, smallest-volume starting point for pt/es
# too -- revisit in Fase 3 (tuning) if more volume is needed from lower tiers.
_EUROWEB_DEFAULT_SPLIT = "high"

BLOCKED_SOURCES: dict[str, str] = {
    "multi-culturax": (
        "uonlp/CulturaX is a gated dataset. Request access at "
        "https://huggingface.co/datasets/uonlp/CulturaX and set HF_TOKEN "
        "before building this adapter."
    ),
}

# One entry per AVAILABLE_ROWS: the Hub repo + the file glob patterns that
# cover exactly what build_adapter() below actually reads for that row (same
# config/split/language subset, nothing more). scripts/download_sources.py
# uses this to bulk-download each source with huggingface_hub.snapshot_download
# (many concurrent connections) *before* filtering runs, instead of each
# source being pulled one file at a time over a single throttled streaming
# connection -- see that script's docstring for the measured throughput
# difference. build_adapter's `raw_dir` param below glob-matches these same
# patterns against the local download to build each adapter's local_files.
#
# Patterns were confirmed against each repo's real file layout (not
# guessed): EuroWeb-2512 is "{lang}/{split}/*.parquet", fineweb-2 is
# "data/{config}/train/*.parquet", HPLT2.0_cleaned is "{config}/train-*.parquet",
# fineweb2-bagaco2's "all" config maps to "fineweb2-ptpt-prototype/*.parquet"
# per its README `configs:` frontmatter (its `classification/` folder is a
# separate, unused config). Portuguese-PD and corpus-ptbr-v2 are small
# enough (a few GB) to just grab in full.
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
    "es-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "patterns": ["data/spa_Latn/train/*.parquet"],
    },
    "es-euroweb": {
        "repo_id": "utter-project/EuroWeb-2512",
        "patterns": ["es/high/*.parquet"],
    },
    "es-hplt2": {
        "repo_id": "HPLT/HPLT2.0_cleaned",
        "patterns": ["spa_Latn/train-*.parquet"],
    },
    "hi-fineweb2": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "patterns": ["data/hin_Deva/train/*.parquet"],
    },
    "hi-euroweb": {
        "repo_id": "utter-project/EuroWeb-2512",
        "patterns": ["hi/high/*.parquet"],
    },
    "hi-hplt2": {
        "repo_id": "HPLT/HPLT2.0_cleaned",
        "patterns": ["hin_Deva/train-*.parquet"],
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


def _build_euroweb(language: str, split: str = _EUROWEB_DEFAULT_SPLIT, **kwargs) -> SourceAdapter:
    if language == "hi" and split != "high":
        # Hard constraint from the assignment: other Hindi splits contain
        # sexual content. Fail fast instead of silently ingesting them.
        raise ValueError(
            f"EuroWeb-2512 Hindi MUST use split='high' (got {split!r}) -- "
            "other splits contain sexual content per the assignment."
        )
    return GenericTextAdapter(
        name="EuroWeb-2512",
        repo_id="utter-project/EuroWeb-2512",
        language=language,
        config=language,
        split=split,
        **kwargs,
    )


def build_adapter(row: str, limit: int | None = None, raw_dir: Path | None = None) -> SourceAdapter:
    """Build a configured adapter for one of the 12 source rows.

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
        return _build_euroweb(language="es", limit=limit, local_files=local_files)
    if row == "hi-euroweb":
        return _build_euroweb(language="hi", limit=limit, local_files=local_files)

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


# Rows actually usable today (excludes BLOCKED_SOURCES and the two
# "multi" rows, which are covered by their per-language es/hi counterparts
# -- see TASK1-PLAN.md sec 2.1 note on EuroWeb-2512 single ingestion).
AVAILABLE_ROWS: list[str] = [
    "pt-fineweb2-bagaco2",
    "pt-corpus-carolina",
    "pt-portuguese-pd",
    "pt-corpus-ptbr-v2",
    "es-fineweb2",
    "es-euroweb",
    "es-hplt2",
    "hi-fineweb2",
    "hi-euroweb",
    "hi-hplt2",
]
