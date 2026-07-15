#!/bin/bash
# Fase 2 full run (TASK1-PLAN.md sec 6): filter -> clean -> quality -> exact
# dedup -> near dedup, across every available source, reading the local
# copy scripts/slurm/00_download_sources.sh already pulled down.
#
# DEVIATION FROM THE PLAN: TASK1-PLAN.md sec 6 sketches 4 separate staged
# jobs (01_ingest -> 02_filter -> 03_dedup -> 04_upload) chained with
# `sbatch --dependency=afterok`. The actual pipeline instead splits into:
# a download stage (00, network-bound, many concurrent connections -- see
# its docstring for why streaming-while-processing was both ~8x slower and
# occasionally 403/500'd from HF's Xet CDN) and this stage, which itself
# runs two phases (scripts/run_all_sources.py's module docstring): a
# parallel per-source filter phase (CPU-bound now that files are local, so
# it's worth spreading across --cpus-per-task), then a serial dedup phase
# (has to stay serial + ordered: dedup state is intentionally shared across
# sources per language, to catch cross-source duplicates). See
# scripts/slurm/README.md for the full rationale. 02 and 03 below cover
# Etapa 6 (aggregate/shuffle) and upload.
#
# Resumable by design: run_all_sources.py checkpoints both phases
# independently. If this job hits its time limit, --requeue resubmits it
# and it picks up wherever it left off -- already-completed sources are
# skipped in each phase, so requeuing is the intended way to span a
# multi-day run across several time-limited jobs. This is why --out-dir /
# --staging-dir MUST be persistent project storage, not node-local scratch
# -- see _common.sh.
#
# Chain after the download job finishes:
#   JOB0=$(sbatch --parsable scripts/slurm/00_download_sources.sh)
#   JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 scripts/slurm/01_run_all_sources.sh)
#
# Usage: sbatch scripts/slurm/01_run_all_sources.sh
#SBATCH --job-name=llm-und-ingest
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18            # filter phase's useful ceiling = number of sources (18, see registry.AVAILABLE_ROWS)
#SBATCH --mem=150G
#SBATCH --time=24:00:00              # generous but not unlimited -- rely on --requeue for multi-day runs
#SBATCH --requeue
#SBATCH --output=runs/%j-01_ingest.out
#SBATCH --error=runs/%j-01_ingest.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/run_all_sources.py \
    --full \
    --raw-dir "$PROJECT_STORAGE/data/raw" \
    --out-dir "$PROJECT_STORAGE/data/processed" \
    --staging-dir "$PROJECT_STORAGE/data/staging" \
    --batch-size 20000 \
    --max-workers "${SLURM_CPUS_PER_TASK:-8}" \
    2>&1 | tee "$LOG_DIR/run_all_sources.log"

echo "Funnel stats: $PROJECT_STORAGE/data/processed/funnel_stats.json"
