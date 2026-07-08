#!/usr/bin/env python3
"""Fase 2 — Run Etapas 1-5 across every available source (TASK1-PLAN.md).

Same pipeline as run_pilot.py (ingest -> language filter -> clean -> quality
-> exact dedup), but across all 9 currently-ingestable sources (see
src.ingest.registry.AVAILABLE_ROWS; corpus-carolina and CulturaX are
blocked -- see registry.BLOCKED_SOURCES). Dedup is per-language (one
ExactDeduper per language), per the plan's cross-source dedup requirement.

Writes one Parquet shard per language: data/processed/{pt,es,hi}.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dedup.exact import ExactDeduper
from src.filters.clean import clean_text
from src.filters.language import hard_filter, soft_filter
from src.filters.quality import QualityConfig, check_quality
from src.ingest.base import VALID_LANGUAGES
from src.ingest.registry import AVAILABLE_ROWS, BLOCKED_SOURCES, build_adapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_quality_config(path: Path) -> QualityConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return QualityConfig(**raw["quality"])


def run(limit_per_source: int, out_dir: Path) -> None:
    quality_cfg = load_quality_config(REPO_ROOT / "configs" / "filters.yaml")
    dedupers = {lang: ExactDeduper() for lang in VALID_LANGUAGES}
    kept_rows: dict[str, list[dict]] = {lang: [] for lang in VALID_LANGUAGES}
    funnel: Counter[str] = Counter()

    for row in AVAILABLE_ROWS:
        print(f"--- {row} ---", flush=True)
        adapter = build_adapter(row, limit=limit_per_source)
        for doc in adapter.iter_documents():
            funnel[f"{row}:ingested"] += 1

            if not hard_filter(doc) or not soft_filter(doc):
                funnel[f"{row}:dropped_language"] += 1
                continue

            doc.text = clean_text(doc.text)
            if not doc.text:
                funnel[f"{row}:dropped_empty"] += 1
                continue

            drop_reason = check_quality(doc.text, quality_cfg)
            if drop_reason is not None:
                funnel[f"{row}:dropped_quality"] += 1
                continue

            if dedupers[doc.language].is_duplicate(doc.text):
                funnel[f"{row}:dropped_exact_dup"] += 1
                continue

            funnel[f"{row}:kept"] += 1
            kept_rows[doc.language].append(doc.to_dict())

    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, rows in kept_rows.items():
        if not rows:
            continue
        table = pa.Table.from_pylist(rows)
        path = out_dir / f"{lang}.parquet"
        pq.write_table(table, path)
        print(f"Wrote {len(rows):,} {lang} docs to {path}")

    print("\n" + "=" * 70)
    print("FUNNEL BY SOURCE")
    print("=" * 70)
    for row in AVAILABLE_ROWS:
        ingested = funnel[f"{row}:ingested"]
        kept = funnel[f"{row}:kept"]
        pct = 100 * kept / ingested if ingested else 0
        print(f"  {row:25s} ingested={ingested:>5,} kept={kept:>5,} ({pct:5.1f}%)")

    print("\nBLOCKED (not run):")
    for row, reason in BLOCKED_SOURCES.items():
        print(f"  {row}: {reason[:80]}...")

    print("\nTOTAL kept by language:")
    for lang, rows in kept_rows.items():
        print(f"  {lang}: {len(rows):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-source", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()
    run(args.limit_per_source, args.out_dir)

    sys.stdout.flush()
    os._exit(0)  # see run_pilot.py for why: datasets streaming + interpreter teardown crash
