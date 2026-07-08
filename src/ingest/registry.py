"""Adapter registry: one entry per pre-training source row (TASK1-PLAN.md sec 2.1).

Central place mapping each of the 12 assignment sources to a configured
`SourceAdapter`. Sources that can't be ingested yet (script-based loading
code, gated access) raise a clear, typed error instead of silently
returning nothing -- see BLOCKED_SOURCES below.
"""

from __future__ import annotations

from src.ingest.adapters.carolina import CarolinaAdapter
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


def build_adapter(row: str, limit: int | None = None) -> SourceAdapter:
    """Build a configured adapter for one of the 12 source rows.

    `row` matches the `row` key used in scripts/inspect_sources.py and
    configs/sources.yaml (e.g. "pt-corpus-ptbr-v2", "hi-euroweb").
    """
    if row in BLOCKED_SOURCES:
        raise NotImplementedError(f"{row} is blocked: {BLOCKED_SOURCES[row]}")

    if row == "es-euroweb":
        return _build_euroweb(language="es", limit=limit)
    if row == "hi-euroweb":
        return _build_euroweb(language="hi", limit=limit)

    if row == "pt-fineweb2-bagaco2":
        return GenericTextAdapter(
            name="fineweb2-bagaco2",
            repo_id="duarteocarmo/fineweb2-bagaco2",
            language="pt",
            config="all",
            id_column="id",
            url_column="url",
            limit=limit,
        )
    if row == "pt-corpus-carolina":
        return CarolinaAdapter(name="corpus-carolina", limit=limit)
    if row == "pt-portuguese-pd":
        return GenericTextAdapter(
            name="Portuguese-PD",
            repo_id="PleIAs/Portuguese-PD",
            language="pt",
            id_column="identifier",
            limit=limit,
        )
    if row == "pt-corpus-ptbr-v2":
        return GenericTextAdapter(
            name="corpus-ptbr-v2",
            repo_id="Madras1/corpus-ptbr-v2",
            language="pt",
            limit=limit,
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
        )
    if row == "es-hplt2":
        return HPLTAdapter(
            name="HPLT2.0_cleaned", repo_id="HPLT/HPLT2.0_cleaned", config="spa_Latn", language="es", limit=limit
        )
    if row == "hi-hplt2":
        return HPLTAdapter(
            name="HPLT2.0_cleaned", repo_id="HPLT/HPLT2.0_cleaned", config="hin_Deva", language="hi", limit=limit
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
