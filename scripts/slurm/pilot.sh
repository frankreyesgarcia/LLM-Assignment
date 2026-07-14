#!/bin/bash
# Fase 1 pilot (TASK1-PLAN.md sec 6/7): one small source, end to end, on a
# single node -- sanity-checks the environment/deps before committing to the
# full run in 01_run_all_sources.sh.
#
# Usage: sbatch scripts/slurm/pilot.sh
#SBATCH --job-name=llm-und-pilot
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%j-pilot.out
#SBATCH --error=runs/%j-pilot.err

# NOTE: `runs/` must already exist before submission -- SLURM opens the
# --output/--error files itself at job launch, before this script body (or
# _common.sh's mkdir) ever runs, so a per-job-id *subdirectory* here would
# fail (the parent wouldn't exist yet). Keep these flat under runs/ with
# the job id in the filename instead; `mkdir -p runs` once is enough.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/run_pilot.py \
    --limit 2000 \
    --out "$PROJECT_STORAGE/pilot/corpus-ptbr-v2.parquet" \
    --dropped-samples-out "$PROJECT_STORAGE/pilot/dropped_samples.yaml" \
    2>&1 | tee "$LOG_DIR/pilot_funnel.log"
