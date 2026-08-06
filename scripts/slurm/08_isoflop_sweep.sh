#!/bin/bash
# Task 3 -- run the Chinchilla IsoFLOP profiles sweep (scripts/isoflop_sweep.py):
# small GPT training runs against the real tokenized corpus, to later fit
# a compute-optimal N(C)/D(C) scaling law
# (scripts/fit_isoflop_scaling_law.py) instead of hand-picking the real
# run's model shape.
#
# Default grid: --flop-budgets 1e14 3.16e14 1e15 1.78e15 3.16e15 x 15
# candidate widths (4 8 16 32 48 64 128 192 256 320 384 448 512 640 768, n_head
# fixed at width/64 once width>=64, else 1; n_layer grows *slower* than
# width, see src/model/scaling.py::gpt_shape_for_width) -- but *not* every
# width is actually run at every budget: scripts/isoflop_sweep.py::build_grid
# keeps only cells whose implied max_iters lands in [100, 50_000]
# (MIN/MAX_ITERS_PER_CELL), since a single shared width list can't
# simultaneously avoid starving the smallest budget's largest widths of
# gradient steps (width=768 at C=1e14 got 21 steps in the prior uniform
# grid -- the exact "too few steps for capacity to matter" failure that
# sank an earlier 1e13/3.16e13 pilot) and avoid the largest budget's
# smallest widths ballooning to hundreds of thousands of iterations
# (width=4 at C=3.16e15 -> ~486,000 iters, a many-hour outlier for a
# width nowhere near that budget's optimum anyway). This filtering drops
# the nominal 5*15=75 grid down to 59 real cells -- see
# scripts/isoflop_sweep.py::build_grid and its constants for the full
# reasoning, and the README's Task 3 section for the pilot history this
# builds on.
#
# --out-dir below is a fresh isoflop_sweep_v2, not the existing
# runs/isoflop_sweep -- that directory's results.csv is from the prior
# uniform-width grid (13 widths, every width at every budget, no
# max_iters filtering) and already has rows for cells this grid no longer
# considers valid (e.g. that same noisy width=768/C=1e14 point).
# fit_isoflop_scaling_law.py just reads whatever rows are in a CSV, with
# no knowledge of build_grid()'s window, so resuming into the old
# directory would silently mix an incompatible cell selection into the
# fit -- same concern as this project's own precedent, archiving the
# grid before this one to runs/isoflop_sweep_pilot2 rather than reusing
# its --out-dir.
#
# Cells are independent single-GPU trains (small models, batch_size=8),
# so this runs 4 of them at once across 4 GPUs (--devices below) rather
# than splitting one cell's training over multiple GPUs via DDP -- these
# runs are too small for DDP's all-reduce overhead to pay off. Estimated
# from real per-width throughput measured on runs/isoflop_sweep's 65-cell
# run (s/iter ranging ~17ms at width=16 up to ~313ms at width=768,
# extrapolated flat below width=16 since those cells are overhead- not
# compute-bound): this grid's 59 cells sum to ~3.1h of total compute,
# ~46min wall-clock across 4 GPUs under a greedy load-balanced schedule
# (the slowest single cell, width=4 at C=3.16e14, is ~14min and sets a
# floor). --time below keeps a large safety margin over that rather than
# cutting it close. -C "thin" pins this to Berzelius's 4-GPU-per-node
# partition (vs. "fat" 8-GPU nodes, see
# https://www.nsc.liu.se/support/systems/berzelius-gpu/ section 8) --
# matches --gpus=4 exactly instead of leaving 4 GPUs on a fat node idle
# for another job to (or not) share.
#
# Usage: sbatch scripts/slurm/08_isoflop_sweep.sh
#SBATCH --job-name=llm-und-isoflop-sweep
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=4
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%j-08_isoflop_sweep.out
#SBATCH --error=runs/%j-08_isoflop_sweep.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/isoflop_sweep.py \
    --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --out-dir "$PROJECT_STORAGE/runs/isoflop_sweep_v2" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    2>&1 | tee "$LOG_DIR/isoflop_sweep.log"

echo "Results CSV + logs under: $PROJECT_STORAGE/runs/isoflop_sweep_v2"
echo "Next: uv run scripts/fit_isoflop_scaling_law.py --results-csv $PROJECT_STORAGE/runs/isoflop_sweep_v2/results.csv"
