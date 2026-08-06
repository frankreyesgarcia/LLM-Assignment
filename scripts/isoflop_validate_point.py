#!/usr/bin/env python3
"""Task 3 -- train exactly one (FLOPs budget, width) IsoFLOP cell directly,
instead of a full scripts/isoflop_sweep.py grid.

Motivation: after fitting N_opt(C)/D_opt(C) from a sweep (e.g.
artifacts/isoflop_sweep_v2), checking whether that fit still tracks reality
at a much larger budget -- say 10x the sweep's largest budget -- would
normally mean re-running the whole grid (75+ cells) at that larger budget,
most of whose FLOPs cost comes from cells nowhere near any budget's optimum.
This script trains just the one cell needed for a cheap sanity check instead:
pick a width already characterized by the reference sweep (e.g. the width
closest to its largest budget's parabola vertex) and train it at N times
that budget, then compare the resulting loss against the reference sweep's
fitted trend by eye.

Deliberately bypasses scripts/isoflop_sweep.py::build_grid's
[MIN_ITERS_PER_CELL, MAX_ITERS_PER_CELL] window -- that window exists to
keep every cell in a *grid* sweep cheap and roughly comparable in wall-clock
cost to its neighbors, not because a single point outside it is invalid to
train on its own. A width held fixed while its budget is scaled up 10x will
often land outside that window (D = C/(6N) grows with C at fixed N), same
as it would in a real 10x-larger grid sweep.

Note this only tests "does more data help this fixed-width model" (moving
along one width's own loss-vs-tokens curve), not "does the compute-optimal
N-vs-D split still hold at 10x compute" -- the latter would require training
at the *new* budget's power-law-predicted optimal width instead, which will
usually differ from the width that was optimal at the smaller reference
budget (see src/model/scaling.py::optimal_params_and_tokens). Interpret the
comparison accordingly: expect a lower loss than the reference cell at the
same width (more compute always helps a fixed model, within reason), but
not necessarily as low as the reference sweep's power-law-extrapolated
vertex loss at the new budget, since that vertex assumes N also scales up.

Usage:
    uv run scripts/isoflop_validate_point.py \
        --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
        --out-dir "$PROJECT_STORAGE/runs/isoflop_validate_10x" \
        --flops-budget 3.16e16 --width 128 --device cuda:0

On SLURM: a one-off validation point doesn't warrant its own sbatch
wrapper script -- reuse scripts/slurm/08_isoflop_sweep.sh's #SBATCH
resources (drop --gpus to 1, single cell not a multi-GPU grid) and swap
its `uv run` line for the one above.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isoflop_sweep import append_row, run_cell  # noqa: E402
from src.tokenizer.logging_utils import tee_to_log  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "isoflop_validate_point")
    parser.add_argument("--flops-budget", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    vocab_size = json.loads((args.data_dir / "meta.json").read_text())["vocab_size"]

    print(
        f"IsoFLOP single-cell validation: flops_budget={args.flops_budget:.3e} width={args.width} "
        f"(bypassing isoflop_sweep.py's grid max_iters window -- see module docstring) -> {csv_path}"
    )
    row = run_cell(
        flops_budget=args.flops_budget,
        width=args.width,
        vocab_size=vocab_size,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
    )
    append_row(csv_path, row)
    print(
        f"n_params={row['n_params']:,} max_iters={row['max_iters']:,} "
        f"train_loss={row['final_train_loss']:.4f} val_loss={row['final_val_loss']:.4f} "
        f"({row['elapsed_s']:.1f}s) -> {csv_path}"
    )


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    with tee_to_log(args.out_dir, "isoflop_validate_point"):
        main(args)
