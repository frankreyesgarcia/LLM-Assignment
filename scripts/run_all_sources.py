#!/usr/bin/env python3
"""Fase 2 — Run Etapas 1-5 across every available source (TASK1-PLAN.md).

Two-phase pipeline, split for parallelism (see scripts/download_sources.py's
docstring for the throughput numbers that motivated this):

  1. **Filter phase** (parallel, one worker process per source): ingest ->
     language filter -> clean -> quality. No shared state across sources, so
     this runs across up to --max-workers processes at once, writing
     "candidate" Parquet parts to --staging-dir/{lang}/<row>__part####.parquet.
  2. **Dedup phase** (serial): reads every source's candidate parts per
     language, in a fixed order (AVAILABLE_ROWS), through one shared
     ExactDeduper + SqliteNearDeduper *per language* -- this has to stay
     serial and in a stable order, since dedup state is explicitly
     cross-source (the plan's requirement: catch e.g. the same document
     appearing in both fineweb2 and EuroWeb). Exact dedup first (cheaper,
     catches most of the overlap) then MinHash, same as before. Writes final
     deduped parts to --out-dir/{lang}/<row>__part####.parquet -- same shape
     build_final_dataset.py (Etapa 6) already expects, so that script needs
     no changes.

Point --raw-dir at a scripts/download_sources.py output directory to read
local pre-downloaded files instead of streaming from the Hub -- recommended
for a full run: a single load_dataset(streaming=True) connection measured
~8 MB/s regardless of link speed, vs. ~65 MB/s aggregate with many
concurrent connections (see download_sources.py). Omit --raw-dir to stream
straight from the Hub as before (fine for small/limited/dev runs).

Both phases are independently resumable/checkpointed
(--staging-dir/filter_checkpoint.json, --out-dir/checkpoint.json). A
source is only marked done in a phase's checkpoint after that phase fully
finishes without error (and, for the dedup phase, after its near-dedup
writes are committed); re-running after a crash skips already-completed
sources and discards + redoes whatever was in flight (see
_clear_stale_staging / clear_stale_source) -- same crash model as before,
just split across the two phases.

Writes data/processed/{pt,es,hi}/<row>__part####.parquet, one
near_dedup_{lang}.sqlite3 per language, and a funnel_stats.json with
per-source/per-stage drop counts -- identical output shape to before.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
FILTER_CHECKPOINT_NAME = "filter_checkpoint.json"
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


def _write_part(rows: list[dict], lang_dir: Path, row: str, seq: int) -> str:
    lang_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{row}__part{seq:04d}.parquet"
    tmp_path = lang_dir / f".{filename}.tmp"
    final_path = lang_dir / filename
    pq.write_table(pa.Table.from_pylist(rows, schema=PART_SCHEMA), tmp_path)
    tmp_path.replace(final_path)
    return filename


# ---------------------------------------------------------------------------
# Phase 1: filter (parallel across sources -- no dedup, no shared state)
# ---------------------------------------------------------------------------


def filter_source(
    row: str,
    raw_dir: Path | None,
    limit_per_source: int | None,
    quality_cfg: QualityConfig,
    staging_dir: Path,
    batch_size: int,
) -> dict:
    """Ingest -> language filter -> clean -> quality for one source. Runs in
    its own worker process (see run_filter_phase) -- deliberately no dedup
    here, since that state has to stay shared and serial across sources
    (see run_dedup_phase / dedup_source below).
    """
    funnel: Counter[str] = Counter()
    adapter = build_adapter(row, limit=limit_per_source, raw_dir=raw_dir)
    buffers: dict[str, list[dict]] = {}
    part_seq: dict[str, int] = {}
    parts_by_lang: dict[str, list[str]] = {}

    def flush(lang: str) -> None:
        rows = buffers.get(lang)
        if not rows:
            return
        seq = part_seq.get(lang, 0)
        filename = _write_part(rows, staging_dir / lang, row, seq)
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

        funnel[f"{row}:kept_prefilter"] += 1
        lang_buffer = buffers.setdefault(doc.language, [])
        lang_buffer.append(doc.to_dict())
        if len(lang_buffer) >= batch_size:
            flush(doc.language)

    for lang in list(buffers):
        flush(lang)

    return {"row": row, "funnel": dict(funnel), "parts": parts_by_lang}


def _clear_stale_staging(staging_dir: Path, row: str) -> None:
    for lang in VALID_LANGUAGES:
        lang_dir = staging_dir / lang
        if lang_dir.exists():
            for f in lang_dir.glob(f"{row}__part*.parquet*"):
                f.unlink()


def run_filter_phase(
    rows: list[str],
    raw_dir: Path | None,
    staging_dir: Path,
    limit_per_source: int | None,
    quality_cfg: QualityConfig,
    batch_size: int,
    max_workers: int,
) -> Counter:
    staging_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = staging_dir / FILTER_CHECKPOINT_NAME
    checkpoint = load_checkpoint(checkpoint_path)
    funnel: Counter[str] = Counter()

    todo = []
    for row in rows:
        completed = checkpoint["completed_sources"].get(row)
        if completed is not None and completed.get("limit_per_source") == limit_per_source:
            print(f"--- {row} (filter phase already completed, skipping) ---", flush=True)
            funnel.update(completed["funnel"])
            continue
        if completed is not None:
            print(
                f"--- {row} (limit changed, {completed['limit_per_source']} -> {limit_per_source}; "
                "re-running filter) ---",
                flush=True,
            )
        _clear_stale_staging(staging_dir, row)
        todo.append(row)

    if not todo:
        return funnel

    print(f"Filter phase: {len(todo)} source(s) across up to {max_workers} worker process(es)", flush=True)
    # max_tasks_per_child=1: each source gets a fresh worker process that
    # exits right after it finishes. `datasets` streaming is known to leave
    # background threads that crash a normal Python interpreter teardown
    # (see run_pilot.py's os._exit(0) note) -- a crash during *that* worker's
    # own exit doesn't lose anything here, since its result is already sent
    # back to us the moment filter_source() returns, before the worker loops
    # around to pick up (or shut down after) its next task.
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as ex:
        futures = {
            ex.submit(filter_source, row, raw_dir, limit_per_source, quality_cfg, staging_dir, batch_size): row
            for row in todo
        }
        for fut in as_completed(futures):
            row = futures[fut]
            result = fut.result()
            funnel.update(result["funnel"])
            checkpoint["completed_sources"][row] = {
                "limit_per_source": limit_per_source,
                "funnel": result["funnel"],
                "parts": result["parts"],
            }
            save_checkpoint(checkpoint_path, checkpoint)
            print(f"--- {row} done (filter phase) ---", flush=True)

    return funnel


# ---------------------------------------------------------------------------
# Phase 2: dedup (serial, shared per-language state across all sources)
# ---------------------------------------------------------------------------


def clear_stale_source(out_dir: Path, row: str, near_dedupers: dict) -> None:
    """Discard partial output from a previous, uncompleted dedup-phase
    attempt at `row` (closes the same crash window the single-phase version
    handled -- see module docstring).
    """
    for lang in VALID_LANGUAGES:
        lang_dir = out_dir / lang
        if lang_dir.exists():
            for f in lang_dir.glob(f"{row}__part*.parquet*"):
                f.unlink()
        near_dedupers[lang].delete_by_source(row)


def replay_completed_source(out_dir: Path, row: str, parts_by_lang: dict, exact_dedupers: dict) -> None:
    """Rebuild ExactDeduper's in-memory state for a source whose deduped
    output is already on disk. near_dedupers don't need this: their SQLite
    files already durably hold whatever was committed, and querying them
    again here would just re-find each doc as a "duplicate" of itself
    (a real MinHash lookup + no-op) for no benefit.
    """
    for lang, relpaths in parts_by_lang.items():
        for relpath in relpaths:
            table = pq.read_table(out_dir / relpath, columns=["text"])
            for text in table.column("text").to_pylist():
                exact_dedupers[lang].is_duplicate(text)


def dedup_source(
    row: str,
    staging_dir: Path,
    out_dir: Path,
    exact_dedupers: dict,
    near_dedupers: dict,
    funnel: Counter,
    batch_size: int,
) -> dict[str, list[str]]:
    """Apply exact + near dedup to one source's already-filtered candidates
    (staging_dir/{lang}/<row>__part*.parquet, written by filter_source),
    writing final deduped parts to out_dir/{lang}/<row>__part####.parquet.
    """
    parts_by_lang: dict[str, list[str]] = {}
    for lang in VALID_LANGUAGES:
        staging_lang_dir = staging_dir / lang
        candidate_files = sorted(staging_lang_dir.glob(f"{row}__part*.parquet")) if staging_lang_dir.exists() else []
        if not candidate_files:
            continue

        seq = 0
        buffer: list[dict] = []
        for cf in candidate_files:
            # Dedup-check texts first and only materialize (to_pylist, which
            # copies every column incl. `metadata`) the rows that survive --
            # a dropped row would otherwise pay that conversion for nothing.
            table = pq.read_table(cf)
            keep_indices = []
            for i, text in enumerate(table.column("text").to_pylist()):
                if exact_dedupers[lang].is_duplicate(text):
                    funnel[f"{row}:dropped_exact_dup"] += 1
                    continue
                if near_dedupers[lang].is_duplicate(text, source_row=row):
                    funnel[f"{row}:dropped_near_dup"] += 1
                    continue
                funnel[f"{row}:kept"] += 1
                keep_indices.append(i)

            for record in table.take(keep_indices).to_pylist():
                buffer.append(record)
                if len(buffer) >= batch_size:
                    filename = _write_part(buffer, out_dir / lang, row, seq)
                    parts_by_lang.setdefault(lang, []).append(f"{lang}/{filename}")
                    seq += 1
                    buffer = []
        if buffer:
            filename = _write_part(buffer, out_dir / lang, row, seq)
            parts_by_lang.setdefault(lang, []).append(f"{lang}/{filename}")

    return parts_by_lang


def run_dedup_phase(
    rows: list[str],
    staging_dir: Path,
    out_dir: Path,
    limit_per_source: int | None,
    dedup_cfg: dict,
    batch_size: int,
    funnel: Counter,
) -> None:
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

    for row in rows:
        completed = checkpoint["completed_sources"].get(row)
        if completed is not None and completed.get("limit_per_source") == limit_per_source:
            print(f"--- {row} (dedup phase already completed, skipping) ---", flush=True)
            funnel.update(completed["funnel"])
            replay_completed_source(out_dir, row, completed["parts"], exact_dedupers)
            continue

        if completed is not None:
            print(
                f"--- {row} (limit changed, {completed['limit_per_source']} -> {limit_per_source}; "
                "re-running dedup) ---",
                flush=True,
            )
        else:
            print(f"--- {row} (dedup phase) ---", flush=True)
        clear_stale_source(out_dir, row, near_dedupers)

        parts_by_lang = dedup_source(row, staging_dir, out_dir, exact_dedupers, near_dedupers, funnel, batch_size)
        for lang in near_dedupers:
            near_dedupers[lang].commit()
        row_funnel = {k: v for k, v in funnel.items() if k.startswith(f"{row}:")}
        checkpoint["completed_sources"][row] = {
            "limit_per_source": limit_per_source,
            "funnel": row_funnel,
            "parts": parts_by_lang,
        }
        save_checkpoint(checkpoint_path, checkpoint)

    for lang in near_dedupers:
        near_dedupers[lang].close()


# ---------------------------------------------------------------------------


def run(
    limit_per_source: int | None,
    out_dir: Path,
    raw_dir: Path | None,
    staging_dir: Path,
    batch_size: int,
    max_workers: int,
) -> None:
    quality_cfg, dedup_cfg = load_filters_config(REPO_ROOT / "configs" / "filters.yaml")

    funnel = run_filter_phase(
        AVAILABLE_ROWS, raw_dir, staging_dir, limit_per_source, quality_cfg, batch_size, max_workers
    )
    run_dedup_phase(AVAILABLE_ROWS, staging_dir, out_dir, limit_per_source, dedup_cfg, batch_size, funnel)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # No default here on purpose: silently falling back to "unlimited" if
    # someone forgets a flag would turn a typo into an accidental multi-TB
    # pull (see configs/sources.yaml -- es-hplt2 alone is ~1.26TB). Require
    # an explicit choice instead.
    limit_group = parser.add_mutually_exclusive_group(required=True)
    limit_group.add_argument("--limit-per-source", type=int, help="Max docs per source (pilot/smoke-test runs)")
    limit_group.add_argument(
        "--full", action="store_true", help="No cap -- process every available source in full (real corpus run)"
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Pre-downloaded local data from scripts/download_sources.py; omit to stream from the Hub instead "
        "(single-connection, ~8x slower for a full run -- see download_sources.py)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Where filter-phase candidates land before dedup (default: <out-dir>/_staging)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Docs per Parquet part file")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Parallel worker processes for the filter phase (one source per worker at a time)",
    )
    args = parser.parse_args()
    staging_dir = args.staging_dir or (args.out_dir / "_staging")
    run(args.limit_per_source, args.out_dir, args.raw_dir, staging_dir, args.batch_size, args.max_workers)

    sys.stdout.flush()
    os._exit(0)  # see run_pilot.py for why: datasets streaming + interpreter teardown crash
