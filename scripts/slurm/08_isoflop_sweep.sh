#!/bin/bash
# Task 3 -- run the Chinchilla IsoFLOP profiles sweep (scripts/isoflop_sweep.py):
# small GPT training runs against the real tokenized corpus, to later fit
# a compute-optimal N(C)/D(C) scaling law
# (scripts/fit_isoflop_scaling_law.py) instead of hand-picking the real
# run's model shape.
#
# Cells are independent single-GPU trains (small models, batch_size=8),
# so this runs 4 of them at once across 4 GPUs (--devices below) rather
# than splitting one cell's training over multiple GPUs via DDP
# (Distributed Data Parallel) -- these runs are too small for DDP's
# all-reduce overhead to pay off. Estimated from real per-width
# throughput measured on runs/isoflop_sweep's 65-cell run (s/iter ranging
# ~17ms at width=16 up to ~313ms at width=768,
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
    --devices cuda:0 cuda:1 cuda:2 cuda:3

echo "Results CSV + logs under: $PROJECT_STORAGE/runs/isoflop_sweep_v2"
echo "Next: uv run scripts/fit_isoflop_scaling_law.py --results-csv $PROJECT_STORAGE/runs/isoflop_sweep_v2/results.csv"
