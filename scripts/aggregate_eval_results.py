#!/usr/bin/env python3
"""Task 4 — aggregate a checkpoint-sweep's per-checkpoint results.json
files into one CSV, for a benchmark-score-vs-training-iteration chart.

Expects the layout scripts/run_eval.py's --out-dir produces when run once
per checkpoint into a shared parent directory, e.g.:

    runs/eval/pretrain_2.2e18_bf16_qknorm/
        iter0003246/results.json
        iter0006492/results.json
        ...

Each subdirectory's name must contain the checkpoint's iteration number
(anything matching \\d+); the exact prefix doesn't matter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ITER_RE = re.compile(r"(\d+)")


def iter_from_dirname(name: str) -> int:
    match = ITER_RE.search(name)
    if not match:
        raise ValueError(f"no iteration number found in directory name {name!r}")
    return int(match.group(1))


def load_rows(eval_dir: Path) -> list[dict]:
    rows = []
    for results_path in sorted(eval_dir.glob("*/results.json")):
        iteration = iter_from_dirname(results_path.parent.name)
        report = json.loads(results_path.read_text())
        for task_name, result in report["results"].items():
            for metric_name, value in result["metrics"].items():
                rows.append(
                    {
                        "iteration": iteration,
                        "task": task_name,
                        "metric": metric_name,
                        "value": value,
                        "n_examples": result["n_examples"],
                    }
                )
    return sorted(rows, key=lambda r: (r["task"], r["metric"], r["iteration"]))


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "task", "metric", "value", "n_examples"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="Parent directory of one results.json per checkpoint")
    parser.add_argument("--out-csv", type=Path, default=None, help="Defaults to <eval_dir>/curve.csv")
    args = parser.parse_args()

    rows = load_rows(args.eval_dir)
    if not rows:
        sys.exit(f"no */results.json found under {args.eval_dir}")

    out_csv = args.out_csv or args.eval_dir / "curve.csv"
    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows ({out_csv})")
