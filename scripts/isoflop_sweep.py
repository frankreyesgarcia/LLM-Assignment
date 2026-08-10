#!/usr/bin/env python3
"""Task 3 -- run the Chinchilla "IsoFLOP profiles" sweep: train a grid of
small GPT models, one per (FLOPs budget, model width) cell, and record
each one's final loss to a CSV. scripts/fit_isoflop_scaling_law.py then
fits a parabola per budget and a power law across budgets' minima, and
extrapolates to a real training run's much larger budget.

For a fixed budget C, model size N and token count D trade off under
FLOPs(N, D) ~= 6*N*D (src/model/scaling.py). Holding C fixed while
sweeping N (this script) and letting D = C/(6N) fall out is what makes
this an "IsoFLOP" sweep -- every cell in a budget's row costs exactly
that budget's FLOPs, regardless of which width trained it. Assumes an
effectively infinite dataset (single epoch, no repeated tokens) --
data/pretrain_full has ~510B train tokens (see its meta.json), several
orders of magnitude more than this sweep's total token usage, so that
assumption holds without needing to check it per cell.

Model width is the only knob varied -- see
src/model/scaling.py::gpt_shape_for_width for why n_layer/n_head are
derived from it instead of chosen independently.

Writes one CSV row per completed cell, flushed immediately (crash/
timeout-safe) and skips cells already present on a re-run (resumable) --
same shape of concern as a multi-hour SLURM job as scripts/
run_dedup_datatrove.py's --limit smoke-test-first pattern, just resumed
via the CSV instead of a --limit flag.

Usage:
    # Local smoke test (small grid, cheap even on CPU) before the real sweep:
    uv run scripts/isoflop_sweep.py --data-dir /path/to/pretrain_full \
        --out-dir artifacts/isoflop_sweep_smoke \
        --flop-budgets 1e11 1e12 --widths 64 128 --device cpu

    # Real sweep (see scripts/slurm/08_isoflop_sweep.sh for the SLURM wrapper):
    uv run scripts/isoflop_sweep.py --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
        --out-dir artifacts/isoflop_sweep

    # Same sweep, spread across 4 GPUs (one cell per GPU at a time) instead
    # of one GPU running all 65 cells back to back -- see --devices below.
    uv run scripts/isoflop_sweep.py --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
        --out-dir artifacts/isoflop_sweep --devices cuda:0 cuda:1 cuda:2 cuda:3
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.gpt import GPT, GPTConfig
from src.model.scaling import gpt_shape_for_width, max_iters_for_budget, tokens_for_budget
from src.model.train import TrainConfig, train_model
from src.tokenizer.logging_utils import tee_to_log

REPO_ROOT = Path(__file__).resolve().parent.parent

# 15 candidate widths (was 13, before that 5): denser throughout, and now
# reaching below WIDTH_HEAD_DIM=64 (4/8/16/32/48, n_head=1 there -- see
# src/model/scaling.py::gpt_shape_for_width) instead of stopping at 64,
# since a coarse grid -- especially its old single jump from 192 straight
# to 384 -- was flattening out the low-N side of several budgets' loss-
# vs-params parabolas. "Candidate" because, unlike the previous grid, not
# every width is actually run at every budget -- see MIN/MAX_ITERS_PER_CELL
# and build_grid() below for why a single shared width list per budget
# stops making sense once it reaches this wide.
DEFAULT_WIDTHS = [4, 8, 16, 32, 48, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768]

# A cell's max_iters = D/(batch_size*block_size) = C/(6*N*batch_size*block_size)
# falls out of (C, width) once both are fixed -- so widening DEFAULT_WIDTHS
# far enough to bracket the smallest budget's optimum (small N) also,
# unavoidably, offers those same small widths to the largest budget, where
# they mean a *huge* D and therefore max_iters -- and symmetrically, widths
# large enough to matter at the largest budget mean a tiny max_iters at the
# smallest budget. Checked directly against this grid's real numbers:
# width=768 at C=1e14 gets 21 gradient steps, width=4 at C=3.16e15 gets
# ~486,000 (vs. ~28,000 for this budget's largest *useful* width, 64 --
# itself already the slowest cell measured on real hardware, 577s). One
# shared width list per budget can't dodge both problems at once -- it
# either starves the smallest budget's largest widths of steps or forces
# the largest budget's smallest widths through a many-hundred-thousand-
# iteration cell that's nowhere near that budget's actual optimum anyway.
# So each budget instead sweeps whichever DEFAULT_WIDTHS fall in this
# window -- see build_grid().
#
# MIN_ITERS_PER_CELL: below this, a cell can't tell capacity and step-count
# effects apart -- the exact failure that sank the first pilot's 1e13/
# 3.16e13 budgets (3-297 gradient steps *even at the narrowest width*,
# see DEFAULT_FLOP_BUDGETS below): every width scored close to its near-
# random-init loss regardless of size, so the fitted "minimum" was pure
# curve-fit noise, not a real one. 100 sits comfortably above that
# regime's whole observed range.
MIN_ITERS_PER_CELL = 100
# MAX_ITERS_PER_CELL: an upper bound on any single cell's wall-clock, sized
# off the slowest cell actually measured in this grid's real run (width=64
# at C=3.16e15: 28,402 iters, 577s) -- 50,000 keeps a cap of roughly that
# same order rather than letting a far-left-of-optimum width run
# unboundedly long for a point that adds little signal near the vertex.
MAX_ITERS_PER_CELL = 50_000
# A first pilot run at 1e13/3.16e13 FLOPs produced no usable signal here --
# those budgets only bought 3-297 gradient steps even at the narrowest
# width, nowhere near enough for model *capacity* to matter yet: with that
# few updates, step count alone dominates the reachable loss regardless of
# N, so every width scored close to its near-random-init loss and the
# "minimum" was just fit noise (found N_opt(C) trending backwards,
# *smaller* optimal models for *bigger* budgets -- the opposite of any
# real scaling law). Dropped in favor of 1.78e15/3.16e15, extending
# upward from 1e15 (the smallest budget that showed a real, bracketed
# dip) instead. See README's Task 3 section for the full pilot writeup.
DEFAULT_FLOP_BUDGETS = [1e14, 3.16e14, 1e15, 1.78e15, 3.16e15]

FIELDNAMES = [
    "flops_budget",
    "width",
    "n_layer",
    "n_head",
    "n_embd",
    "n_params",
    "n_params_non_embedding",
    "tokens",
    "max_iters",
    "final_train_loss",
    "final_val_loss",
    "elapsed_s",
]


def load_completed_cells(csv_path: Path) -> dict[tuple[float, int], float]:
    """Already-trained cells -> their elapsed_s. Keys drive the resume skip;
    the values let the end-of-sweep summary report a total that spans
    earlier invocations too, not just this one's cells."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        return {(float(row["flops_budget"]), int(row["width"])): float(row["elapsed_s"]) for row in csv.DictReader(f)}


def format_duration(seconds: float) -> str:
    """Seconds -> "3h45m24s". Per-cell prints stay in raw seconds (directly
    comparable across cells that way); only the sweep totals, which run to
    hours, get this."""
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def append_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def build_grid(
    flop_budgets: list[float],
    widths: list[int],
    vocab_size: int,
    block_size: int,
    batch_size: int,
) -> list[tuple[float, int]]:
    """Per-budget (flops_budget, width) grid: each budget only sweeps the
    `widths` whose implied max_iters falls in [MIN_ITERS_PER_CELL,
    MAX_ITERS_PER_CELL] -- see the constants' docstrings above for why a
    single width list can't be both narrow enough for the smallest budget
    and wide enough for the largest one without this filter.
    """
    n_params_by_width = {}
    for width in widths:
        n_layer, n_head, n_embd = gpt_shape_for_width(width)
        model = GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd))
        n_params_by_width[width] = model.num_params(non_embedding=False)

    grid = []
    for flops_budget in flop_budgets:
        kept = 0
        for width in widths:
            max_iters = max_iters_for_budget(flops_budget, n_params_by_width[width], batch_size, block_size)
            if MIN_ITERS_PER_CELL <= max_iters <= MAX_ITERS_PER_CELL:
                grid.append((flops_budget, width))
                kept += 1
        if kept < 5:
            # fit_isoflop_parabola() fits a degree-2 polynomial -- needs at
            # least 3 points to be well-defined at all, and fewer than 5
            # leaves little margin to also bracket the vertex. Warn rather
            # than fail outright since this is a legitimate (if unusual)
            # outcome for a deliberately narrow --flop-budgets/--widths
            # smoke test.
            print(
                f"WARNING: only {kept} width(s) survive the max_iters window "
                f"[{MIN_ITERS_PER_CELL}, {MAX_ITERS_PER_CELL}] for flops_budget={flops_budget:.3e} -- "
                "too few to reliably bracket a parabola vertex; widen --widths or this budget's range."
            )
    return grid


def run_cell(
    flops_budget: float,
    width: int,
    vocab_size: int,
    data_dir: Path,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    grad_clip: float,
    device: str,
    seed: int,
) -> dict:
    n_layer, n_head, n_embd = gpt_shape_for_width(width)
    model = GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd))
    # n_params (total, embedding included) is what FLOPs accounting and the
    # power-law fit use -- Chinchilla's own convention, since embeddings
    # genuinely cost FLOPs (module docstring). n_params_non_embedding is
    # recorded alongside it purely for inspection -- e.g. to see how much
    # of a narrow-width cell's N is really "model" vs. the (width-
    # independent) vocab embedding table -- and isn't used in the fit.
    n_params = model.num_params(non_embedding=False)
    n_params_non_embedding = model.num_params(non_embedding=True)

    max_iters = max_iters_for_budget(flops_budget, n_params, batch_size, block_size)
    # A fixed warmup *fraction* (not a fixed iter count) so a 3-iter cell
    # (tiny budget, wide model) and a 9,000-iter cell (huge budget, narrow
    # model) both spend the same proportion of training ramping up --
    # a fixed absolute warmup_iters would swallow the entire run for the
    # smallest cells.
    warmup_iters = max(1, round(0.1 * max_iters))
    # Only the initial (it=0) and final (it=max_iters-1) evals are needed
    # here -- train_model() always runs those regardless of eval_interval
    # (src/model/train.py) -- so set eval_interval to max_iters to skip
    # the (here, unused) intermediate points and their eval cost.
    eval_interval = max(1, max_iters)

    cfg = TrainConfig(
        data_dir=data_dir,
        out_dir=None,
        block_size=block_size,
        batch_size=batch_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_iters=max_iters,
        lr=lr,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        device=device,
        seed=seed,
        log_every_eval=False,
    )
    start = time.time()
    result = train_model(cfg)
    elapsed = time.time() - start

    return {
        "flops_budget": flops_budget,
        "width": width,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "n_params": n_params,
        "n_params_non_embedding": n_params_non_embedding,
        "tokens": result["tokens_seen"],
        "max_iters": max_iters,
        "final_train_loss": result["final_train_loss"],
        "final_val_loss": result["final_val_loss"],
        "elapsed_s": elapsed,
    }


def _run_cell_task(kwargs: dict) -> dict:
    """Top-level (picklable) wrapper so ProcessPoolExecutor can dispatch run_cell."""
    return run_cell(**kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "isoflop_sweep")
    parser.add_argument("--flop-budgets", type=float, nargs="+", default=DEFAULT_FLOP_BUDGETS)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Run cells in parallel, one worker process pinned to each device given here "
            "(e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3 for a 4-GPU node). Cells are "
            "independent single-GPU trains -- this parallelizes *across* cells, not within "
            "one cell's training (these models are far too small, batch_size=8 and <100M "
            "params, for splitting one cell over multiple GPUs via DDP to pay off its "
            "all-reduce overhead). Defaults to a single worker on --device (old, sequential "
            "behavior). 'auto' isn't valid here with more than one device -- it would resolve "
            "to the same cuda:0 in every worker and contend for one GPU."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"

    devices = args.devices if args.devices else [args.device]
    if len(devices) > 1 and "auto" in devices:
        raise ValueError("--devices must list explicit devices (e.g. cuda:0 cuda:1 ...), not 'auto', when using more than one")

    vocab_size = json.loads((args.data_dir / "meta.json").read_text())["vocab_size"]
    completed = load_completed_cells(csv_path)

    grid = build_grid(args.flop_budgets, args.widths, vocab_size, args.block_size, args.batch_size)
    todo = [(c, w) for c, w in grid if (c, w) not in completed]
    print(
        f"IsoFLOP sweep: {len(grid)} cells ({len(args.flop_budgets)} budgets, {len(args.widths)} candidate widths each, "
        f"per-budget max_iters window [{MIN_ITERS_PER_CELL}, {MAX_ITERS_PER_CELL}]), "
        f"{len(completed)} already done, {len(todo)} to run -> {csv_path} (devices={devices})"
    )

    common_kwargs = dict(
        vocab_size=vocab_size,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    sweep_start = time.time()
    # Summed per-cell training time, not wall-clock: with --devices these
    # overlap, so this is total *device* time (the sweep's real compute
    # cost) while sweep_start measures how long it took to get through it.
    compute_s = 0.0

    if len(devices) == 1:
        for i, (flops_budget, width) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] flops_budget={flops_budget:.3e} width={width} ...")
            row = run_cell(flops_budget=flops_budget, width=width, device=devices[0], **common_kwargs)
            append_row(csv_path, row)
            compute_s += row["elapsed_s"]
            print(
                f"    n_params={row['n_params']:,} max_iters={row['max_iters']:,} "
                f"train_loss={row['final_train_loss']:.4f} val_loss={row['final_val_loss']:.4f} "
                f"({row['elapsed_s']:.1f}s, {format_duration(compute_s)} cumulative)"
            )
    else:
        # Submission order cycles through `devices`, and ProcessPoolExecutor
        # dispatches strictly FIFO to exactly `len(devices)` worker processes --
        # so any `len(devices)`-sized window of in-flight cells always contains
        # each device exactly once. No locking needed for the CSV either: only
        # this main process (in as_completed's loop below) ever writes to it.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _run_cell_task,
                    dict(flops_budget=c, width=w, device=devices[i % len(devices)], **common_kwargs),
                ): (c, w)
                for i, (c, w) in enumerate(todo)
            }
            for n_done, future in enumerate(as_completed(futures), 1):
                flops_budget, width = futures[future]
                row = future.result()
                append_row(csv_path, row)
                compute_s += row["elapsed_s"]
                print(
                    f"[{n_done}/{len(todo)}] flops_budget={flops_budget:.3e} width={width} "
                    f"n_params={row['n_params']:,} max_iters={row['max_iters']:,} "
                    f"train_loss={row['final_train_loss']:.4f} val_loss={row['final_val_loss']:.4f} "
                    f"({row['elapsed_s']:.1f}s, {format_duration(compute_s)} cumulative)"
                )

    wall_s = time.time() - sweep_start
    print(f"Done. {len(grid)} cells total in {csv_path}")
    print(
        f"  {len(todo)} cell(s) this invocation: {format_duration(compute_s)} of training time "
        f"across {len(devices)} device(s), {format_duration(wall_s)} wall-clock"
        # Perfect overlap would put compute_s at len(devices)*wall_s; the
        # shortfall is idle GPUs (stragglers at the tail, startup) plus the
        # main process's own CSV/eval bookkeeping between cells.
        + (f" ({compute_s / (wall_s * len(devices)):.0%} device utilization)" if wall_s > 0 else "")
    )
    if completed:
        # Resumed run: this invocation's total leaves out whatever earlier
        # ones already paid for, so report the whole grid's cost as well.
        grid_cells = set(grid)
        grid_total_s = sum(elapsed for cell, elapsed in load_completed_cells(csv_path).items() if cell in grid_cells)
        print(f"  {len(grid)} cell(s) including earlier invocations: {format_duration(grid_total_s)} of training time")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    with tee_to_log(args.out_dir, "isoflop_sweep"):
        main(args)
