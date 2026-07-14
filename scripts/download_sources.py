#!/usr/bin/env python3
"""Fase 2a (new) -- bulk-download every available source's raw files to
local project storage, in parallel, before any filtering/dedup runs.

Why this exists: `datasets.load_dataset(..., streaming=True)` (what
scripts/run_all_sources.py used to read straight from the Hub) opens a
single HTTP connection per source and reads it sequentially -- measured at
~8 MB/s regardless of the cluster's actual link speed. huggingface_hub's
snapshot_download issues many concurrent file requests instead -- measured
~61-66 MB/s aggregate downloading the exact same repos (a ~7-8x
improvement; the ceiling looks like an HF-side rate limit, not our
bandwidth). At that rate the full ~14.8TB corpus downloads in ~2.6 days,
vs. an extrapolated ~690 days for the old combined stream+process design.

Downloading up front (scripts/run_all_sources.py then reads the local copy
via --raw-dir) also turns ingest from network-bound to CPU-bound, which is
what makes parallelizing the filter stage across sources worthwhile: a
single streamed connection can't be split across worker processes, but a
local parquet file can be read by any number of them at once.

Resumable for free: snapshot_download's local_dir already skips files
that are already present and up to date, so re-running this script after a
partial run/crash just picks up wherever it left off -- no custom
checkpoint needed here (unlike run_all_sources.py's per-source dedup
state).

Writes data/raw/<row>/... (mirrors each repo's real directory layout for
just the file patterns SOURCE_DOWNLOAD_SPECS needs -- see registry.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.registry import AVAILABLE_ROWS, SOURCE_DOWNLOAD_SPECS

REPO_ROOT = Path(__file__).resolve().parent.parent


def download_row(row: str, out_dir: Path, max_workers: int) -> None:
    spec = SOURCE_DOWNLOAD_SPECS[row]
    local_dir = out_dir / row
    print(f"--- {row} ({spec['repo_id']}) ---", flush=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        allow_patterns=spec["patterns"],
        local_dir=str(local_dir),
        max_workers=max_workers,
    )


def run(rows: list[str], out_dir: Path, max_workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        download_row(row, out_dir, max_workers)
    print(f"\nDone. Raw files under: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument(
        "--rows", nargs="*", default=AVAILABLE_ROWS, choices=AVAILABLE_ROWS, help="Subset of rows to download"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent connections per source (snapshot_download); measured ~61-66MB/s aggregate at 8-32 workers",
    )
    args = parser.parse_args()
    run(args.rows, args.out_dir, args.max_workers)
