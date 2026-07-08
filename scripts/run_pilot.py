#!/usr/bin/env python3
"""Fase 1 — Pilot pipeline (TASK1-PLAN.md sec 7): one small source, end to end.

Runs Etapas 1-5 (ingest -> language filter -> clean -> quality -> exact dedup)
on `Madras1/corpus-ptbr-v2` (1.6 GB / 371k docs, chosen over corpus-carolina
because the latter uses a custom loading script unsupported by the streaming
adapter — see configs/sources.yaml, row pt-corpus-carolina).

Streams a bounded sample (--limit) instead of the full source, per the
plan's "Streaming first" principle and to keep the pilot fast to iterate on.
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
from src.ingest.adapters.generic import GenericTextAdapter
from src.ingest.base import Document

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_quality_config(path: Path) -> QualityConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return QualityConfig(**raw["quality"])


def run_pilot(limit: int, out_path: Path, dropped_samples_path: Path) -> None:
    quality_cfg = load_quality_config(REPO_ROOT / "configs" / "filters.yaml")
    deduper = ExactDeduper()

    adapter = GenericTextAdapter(
        name="corpus-ptbr-v2",
        repo_id="Madras1/corpus-ptbr-v2",
        language="pt",
        text_column="text",
        split="train",
        streaming=True,
        limit=limit,
    )

    funnel: Counter[str] = Counter()
    kept_rows: list[dict] = []
    dropped_samples: list[dict] = []
    MAX_DROPPED_SAMPLES = 20

    for doc in adapter.iter_documents():
        funnel["ingested"] += 1

        if not hard_filter(doc):
            funnel["dropped_lang_hard"] += 1
            continue

        if not soft_filter(doc):
            funnel["dropped_lang_soft"] += 1
            continue

        doc.text = clean_text(doc.text)
        if not doc.text:
            funnel["dropped_empty_after_clean"] += 1
            continue

        drop_reason = check_quality(doc.text, quality_cfg)
        if drop_reason is not None:
            funnel[f"dropped_quality:{drop_reason}"] += 1
            if len(dropped_samples) < MAX_DROPPED_SAMPLES:
                dropped_samples.append({"reason": drop_reason, "text": doc.text[:300]})
            continue

        if deduper.is_duplicate(doc.text):
            funnel["dropped_exact_dup"] += 1
            continue

        funnel["kept"] += 1
        kept_rows.append(doc.to_dict())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if kept_rows:
        table = pa.Table.from_pylist(kept_rows)
        pq.write_table(table, out_path)

    with open(dropped_samples_path, "w") as f:
        yaml.safe_dump(dropped_samples, f, sort_keys=False, allow_unicode=True)

    print("=" * 70)
    print("PILOT FUNNEL — Madras1/corpus-ptbr-v2")
    print("=" * 70)
    ingested = funnel["ingested"]
    for reason, count in funnel.most_common():
        pct = 100 * count / ingested if ingested else 0
        print(f"  {reason:40s} {count:>8,}  ({pct:5.1f}% of ingested)")
    print("-" * 70)
    kept = funnel["kept"]
    print(f"  Kept: {kept:,} / {ingested:,} ({100 * kept / ingested:.1f}%)" if ingested else "  Kept: 0")
    print(f"  Output written to: {out_path}")
    print(f"  Dropped samples (for manual review) written to: {dropped_samples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="Max docs to stream from the source")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pilot" / "corpus-ptbr-v2.parquet")
    parser.add_argument(
        "--dropped-samples-out", type=Path, default=REPO_ROOT / "data" / "pilot" / "dropped_samples.yaml"
    )
    args = parser.parse_args()
    run_pilot(args.limit, args.out, args.dropped_samples_out)

    # `datasets` streaming leaves background aiohttp/fsspec threads that crash
    # the interpreter's normal teardown (PyGILState_Release fatal error) on
    # this stack (datasets 5.0 + pyarrow 24). All work is done and flushed by
    # this point, so force a clean process exit instead of letting Python's
    # atexit/finalization sequence run into that known issue.
    sys.stdout.flush()
    os._exit(0)
