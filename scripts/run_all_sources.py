#!/usr/bin/env python3
"""Fase 2 — Run Etapas 1-5 across every available source (TASK1-PLAN.md).

Same pipeline as run_pilot.py (ingest -> language filter -> clean -> quality
-> exact dedup -> near dedup), but across all 9 currently-ingestable sources
(see src.ingest.registry.AVAILABLE_ROWS; corpus-carolina and CulturaX are
blocked -- see registry.BLOCKED_SOURCES). Both dedup stages are per-language
(one ExactDeduper + one SqliteNearDeduper per language, in that order since
exact dup is cheaper and catches most of the EuroWeb multi/per-language
overlap before the pricier MinHash pass runs), per the plan's cross-source
dedup requirement. Near-dedup uses the disk-backed SqliteNearDeduper (not
the in-memory NearDeduper run_pilot.py uses) so its index doesn't have to
fit in RAM -- see src/dedup/minhash.py.

Streaming/checkpointing: kept docs are flushed to Parquet part files every
BATCH_SIZE docs instead of being held in memory for the whole run (some
sources' sizes are large enough -- see configs/sources.yaml, e.g. es-hplt2
at ~1.26TB estimated -- that nothing here can assume it fits in RAM). A
source is only marked done in checkpoint.json after it fully streams
without error AND its near-dedup writes are committed; re-running after a
crash skips already-completed sources (ExactDeduper's in-memory state is
replayed from their Parquet output; SqliteNearDeduper's state is already
durable on disk) and re-streams from scratch any source that wasn't
checkpoint-complete -- discarding its partial Parquet parts and purging any
near-dedup entries it may have already committed (SqliteNearDeduper.delete_by_source),
closing the narrow window where a crash lands between the near-dedup commit
and the checkpoint write. Streaming HF datasets don't expose a reliable
mid-stream resume offset, so source-level, not batch-level, is the
resumability granularity here.

Writes data/processed/{pt,es,hi}/<row>__part####.parquet, one
near_dedup_{lang}.sqlite3 per language, and a funnel_stats.json with
per-source/per-stage drop counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dedup.exact import ExactDeduper
from src.dedup.minhash import SqliteNearDeduper
from src.filters.clean import clean_text
from src.filters.language import hard_filter, soft_filter
from src.filters.quality import QualityConfig, check_quality
from src.ingest.base import VALID_LANGUAGES
from src.ingest.registry import AVAILABLE_ROWS, BLOCKED_SOURCES, build_adapter

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_NAME = "checkpoint.json"
DEFAULT_BATCH_SIZE = 20_000
FUNNEL_STAGES = (
    "ingested",
    "dropped_language",
    "dropped_empty",
    "dropped_quality",
    "dropped_exact_dup",
    "dropped_near_dup",
    "kept",
)
# Explicit schema for Document.to_dict() (src/ingest/base.py) -- required so
# every part file agrees on column types. Without this, pa.Table.from_pylist
# infers per-batch, and a batch where e.g. every `url` happens to be None
# (common: CarolinaAdapter never sets url) gets typed `null` instead of
# `string`, which pyarrow.dataset then fails to reconcile against other part
# files when reading the whole directory back (Etapa 6, build_final_dataset.py).
PART_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("language", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("url", pa.string()),
        ("metadata", pa.string()),
    ]
)


def load_filters_config(path: Path) -> tuple[QualityConfig, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return QualityConfig(**raw["quality"]), raw["dedup"]


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"completed_sources": {}}
    with open(path) as f:
        return json.load(f)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.replace(path)


def clear_stale_source(out_dir: Path, row: str, near_dedupers: dict) -> None:
    """Discard partial output from a previous, uncompleted attempt at `row`.

    Covers both the Parquet part files and any near-dedup SQLite entries
    tagged with this row -- the latter closes a narrow crash window where a
    prior attempt committed its near-dedup writes but died before
    checkpoint.json recorded the row as done (see SqliteNearDeduper.delete_by_source).
    """
    for lang in VALID_LANGUAGES:
        lang_dir = out_dir / lang
        if lang_dir.exists():
            for f in lang_dir.glob(f"{row}__part*.parquet*"):
                f.unlink()
        near_dedupers[lang].delete_by_source(row)


def replay_completed_source(
    out_dir: Path, row: str, parts_by_lang: dict, exact_dedupers: dict, near_dedupers: dict
) -> None:
    """Rebuild ExactDeduper's in-memory state for a source whose output is
    already on disk. near_dedupers don't strictly need this replay -- their
    SQLite files already durably hold whatever was committed -- but calling
    is_duplicate() here too is harmless (it'll just find each doc already
    indexed against itself) and keeps this function's contract uniform
    across both dedup backends.
    """
    for lang, relpaths in parts_by_lang.items():
        for relpath in relpaths:
            table = pq.read_table(out_dir / relpath, columns=["text"])
            for text in table.column("text").to_pylist():
                exact_dedupers[lang].is_duplicate(text)
                near_dedupers[lang].is_duplicate(text, source_row=row)


def process_source(
    row: str,
    limit_per_source: int | None,
    quality_cfg: QualityConfig,
    exact_dedupers: dict,
    near_dedupers: dict,
    out_dir: Path,
    funnel: Counter,
    batch_size: int,
) -> dict[str, list[str]]:
    """Stream one source end to end, flushing kept docs every `batch_size` docs.

    Returns {language: [relative part-file paths]} for what got written.
    """
    adapter = build_adapter(row, limit=limit_per_source)
    buffers: dict[str, list[dict]] = {}
    part_seq: dict[str, int] = {}
    parts_by_lang: dict[str, list[str]] = {}

    def flush(lang: str) -> None:
        rows = buffers.get(lang)
        if not rows:
            return
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        seq = part_seq.get(lang, 0)
        filename = f"{row}__part{seq:04d}.parquet"
        tmp_path = lang_dir / f".{filename}.tmp"
        final_path = lang_dir / filename
        pq.write_table(pa.Table.from_pylist(rows, schema=PART_SCHEMA), tmp_path)
        tmp_path.replace(final_path)
        parts_by_lang.setdefault(lang, []).append(f"{lang}/{filename}")
        part_seq[lang] = seq + 1
        buffers[lang] = []

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

        if exact_dedupers[doc.language].is_duplicate(doc.text):
            funnel[f"{row}:dropped_exact_dup"] += 1
            continue

        if near_dedupers[doc.language].is_duplicate(doc.text, source_row=row):
            funnel[f"{row}:dropped_near_dup"] += 1
            continue

        funnel[f"{row}:kept"] += 1
        lang_buffer = buffers.setdefault(doc.language, [])
        lang_buffer.append(doc.to_dict())
        if len(lang_buffer) >= batch_size:
            flush(doc.language)

    for lang in list(buffers):
        flush(lang)

    return parts_by_lang


def run(limit_per_source: int | None, out_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    quality_cfg, dedup_cfg = load_filters_config(REPO_ROOT / "configs" / "filters.yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    exact_dedupers = {lang: ExactDeduper() for lang in VALID_LANGUAGES}
    near_dedupers = {
        lang: SqliteNearDeduper(
            out_dir / f"near_dedup_{lang}.sqlite3",
            num_permutations=dedup_cfg["minhash_num_permutations"],
            jaccard_threshold=dedup_cfg["minhash_jaccard_threshold"],
        )
        for lang in VALID_LANGUAGES
    }
    checkpoint_path = out_dir / CHECKPOINT_NAME
    checkpoint = load_checkpoint(checkpoint_path)
    funnel: Counter[str] = Counter()

    for row in AVAILABLE_ROWS:
        completed = checkpoint["completed_sources"].get(row)
        if completed is not None and completed.get("limit_per_source") == limit_per_source:
            print(f"--- {row} (already completed, skipping re-ingest) ---", flush=True)
            funnel.update(completed["funnel"])
            replay_completed_source(out_dir, row, completed["parts"], exact_dedupers, near_dedupers)
            continue

        if completed is not None:
            print(f"--- {row} (limit changed, {completed['limit_per_source']} -> {limit_per_source}; re-running) ---", flush=True)
        else:
            print(f"--- {row} ---", flush=True)
        clear_stale_source(out_dir, row, near_dedupers)

        parts_by_lang = process_source(
            row, limit_per_source, quality_cfg, exact_dedupers, near_dedupers, out_dir, funnel, batch_size
        )
        for lang in near_dedupers:
            near_dedupers[lang].commit()
        row_funnel = {k: v for k, v in funnel.items() if k.startswith(f"{row}:")}
        checkpoint["completed_sources"][row] = {
            "limit_per_source": limit_per_source,
            "funnel": row_funnel,
            "parts": parts_by_lang,
        }
        save_checkpoint(checkpoint_path, checkpoint)

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

    kept_by_language: dict[str, int] = {}
    for lang in sorted(VALID_LANGUAGES):
        lang_dir = out_dir / lang
        total = 0
        if lang_dir.exists():
            for path in lang_dir.glob("*.parquet"):
                total += pq.ParquetFile(path).metadata.num_rows
        kept_by_language[lang] = total

    print("\nTOTAL kept by language:")
    for lang, n in kept_by_language.items():
        print(f"  {lang}: {n:,}")

    by_source = {row: {stage: funnel[f"{row}:{stage}"] for stage in FUNNEL_STAGES} for row in AVAILABLE_ROWS}
    totals = {stage: sum(counts[stage] for counts in by_source.values()) for stage in FUNNEL_STAGES}
    stats_path = out_dir / "funnel_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "by_source": by_source,
                "totals": totals,
                "kept_by_language": kept_by_language,
            },
            f,
            indent=2,
        )
    print(f"\nFunnel stats written to: {stats_path}")

    for lang in near_dedupers:
        near_dedupers[lang].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # No default here on purpose: silently falling back to "unlimited" if
    # someone forgets a flag would turn a typo into an accidental multi-TB
    # pull (see configs/sources.yaml -- es-hplt2 alone is ~1.26TB). Require
    # an explicit choice instead.
    limit_group = parser.add_mutually_exclusive_group(required=True)
    limit_group.add_argument("--limit-per-source", type=int, help="Max docs per source (pilot/smoke-test runs)")
    limit_group.add_argument(
        "--full", action="store_true", help="No cap -- stream every available source in full (real corpus run)"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Docs per Parquet part file")
    args = parser.parse_args()
    run(args.limit_per_source, args.out_dir, args.batch_size)

    sys.stdout.flush()
    os._exit(0)  # see run_pilot.py for why: datasets streaming + interpreter teardown crash
