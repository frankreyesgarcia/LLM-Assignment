#!/bin/bash
# Task 3 -- single-cell IsoFLOP validation (scripts/isoflop_validate_point.py):
# a cheap sanity check on whether artifacts/isoflop_sweep_v2's fitted
# N_opt(C)/D_opt(C) power law still tracks reality at 10x its largest swept
# budget, without re-running the full ~59-cell grid at that budget.
#
# Width/budget choice: v2's largest budget (C=3.16e15) had its parabola
# vertex at width=128 (see chat history / README), but N_opt(C) is fit to
# *grow* with C -- extrapolating v2's fit to 10x that budget (C=3.16e16)
# predicts N_opt~=11.8M params, nearest feasible width=256
# (src/model/scaling.py::optimal_params_and_tokens), not 128. So this job
# trains width=256 -- the power-law-predicted optimal width at 10x compute
# -- rather than just holding the old-budget's optimal width fixed and
# adding more tokens to it. That distinction matters: training a fixed
# width at more compute only tests "does more data help this one model"
# (loss will drop regardless); training the *newly predicted optimal*
# width is what actually exercises whether the N-vs-D tradeoff the fit
# extrapolated is trustworthy -- its loss should land close to the fit's
# predicted vertex loss (~4.37, from a power-law fit of v2's 5 per-budget
# vertex losses; not yet wired into fit_isoflop_scaling_law.py, computed
# ad hoc) if the extrapolation holds.
#
# --time: width=256 at C=3.16e15 took 602.8s for 4,873 iters (v2 results.csv).
# At C=3.16e16, max_iters_for_budget scales to 48,730 (~10x, D=C/(6N) at
# fixed N) -- linearly extrapolating wall-clock gives ~6,028s (~100min).
# --time=02:00:00 below keeps a ~20% buffer over that, same convention as
# 07_pretrain.sh's dry-run-based estimate.
#
# Usage: sbatch scripts/slurm/09_isoflop_validate_10x.sh
#SBATCH --job-name=llm-und-isoflop-validate-10x
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%j-09_isoflop_validate_10x.out
#SBATCH --error=runs/%j-09_isoflop_validate_10x.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/isoflop_validate_point.py \
    --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --out-dir "$PROJECT_STORAGE/runs/isoflop_validate_10x" \
    --flops-budget 3.16e16 --width 256 --device cuda:0 \
    2>&1 | tee "$LOG_DIR/isoflop_validate_point.log"

echo "Result CSV + logs under: $PROJECT_STORAGE/runs/isoflop_validate_10x"
