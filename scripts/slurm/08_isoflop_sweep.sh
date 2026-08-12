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
# all-reduce overhead to pay off.
#
# This grid densifies the widths around the parabola vertex (see
# scripts/isoflop_sweep.py::DEFAULT_WIDTHS), taking it from 59 to 98
# cells. --time is sized off measurement, not extrapolation: an earlier
# v3 attempt got 87 of 98 cells done in 15.4 GPU-hours before its 4h wall
# killed it, i.e. ~2.3x the 6.6h that v2's per-width s/iter predicted --
# v2 predated document masking (#15), whose FlexAttention costs real time
# per step. Median cell 210s, slowest 3695s. So ~18 GPU-hours for the
# full grid, ~4.5h wall across 4 GPUs, and 12h leaves genuine margin
# rather than the 4h that turned out to be under the actual cost.
# -C "thin" pins this to Berzelius's 4-GPU-per-node
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
#SBATCH --time=12:00:00
#SBATCH --output=runs/%j-08_isoflop_sweep.out
#SBATCH --error=runs/%j-08_isoflop_sweep.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

# --wandb: one W&B run per cell, grouped under the out-dir name. Online
# mode -- Berzelius compute nodes reach wandb.ai (the earlier pretrain runs
# logged live); add --wandb-mode offline and `wandb sync` afterwards if that
# ever stops holding.
uv run scripts/isoflop_sweep.py \
    --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --out-dir "$PROJECT_STORAGE/runs/isoflop_sweep_v3" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --wandb

echo "Results CSV + logs under: $PROJECT_STORAGE/runs/isoflop_sweep_v3"
echo "Next: uv run scripts/fit_isoflop_scaling_law.py --results-csv $PROJECT_STORAGE/runs/isoflop_sweep_v3/results.csv"
