#!/bin/bash
# Fase 2 full run (TASK1-PLAN.md sec 6): ingest -> filter -> clean -> quality
# -> exact dedup -> near dedup, across every available source.
#
# DEVIATION FROM THE PLAN: TASK1-PLAN.md sec 6 sketches 4 separate staged
# jobs (01_ingest -> 02_filter -> 03_dedup -> 04_upload) chained with
# `sbatch --dependency=afterok`. The actual pipeline (scripts/run_all_sources.py)
# fuses ingest+filter+clean+quality+dedup into one streaming pass per source
# instead -- each source is only pulled over the network once, filtered and
# deduped inline, and flushed straight to Parquet part files, rather than
# materializing an intermediate "raw ingested" dump that a separate filter
# job would re-read. See scripts/slurm/README.md for the full rationale.
# This job is therefore the ingest+filter+dedup stage combined; 02 and 03
# below cover Etapa 6 (aggregate/shuffle) and upload.
#
# Resumable by design: checkpoint.json in --out-dir tracks which sources
# finished. If this job hits its time limit, --requeue resubmits it and
# run_all_sources.py picks up wherever it left off -- already-completed
# sources are skipped (no re-download), so requeuing is the intended way
# to span a multi-day run across several time-limited jobs. This is why
# --out-dir MUST be persistent project storage, not node-local scratch --
# see _common.sh.
#
# Usage: sbatch scripts/slurm/01_run_all_sources.sh
#SBATCH --job-name=llm-und-ingest
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # generous but not unlimited -- rely on --requeue for multi-day runs
#SBATCH --requeue
#SBATCH --output=runs/%j-01_ingest.out
#SBATCH --error=runs/%j-01_ingest.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Optional: some sources may need an HF token (rate limits, or if CulturaX
# ever gets unblocked -- see src/ingest/registry.py::BLOCKED_SOURCES).
# export HF_TOKEN="..."

uv run scripts/run_all_sources.py \
    --full \
    --out-dir "$PROJECT_STORAGE/data/processed" \
    --batch-size 20000 \
    2>&1 | tee "$LOG_DIR/run_all_sources.log"

echo "Funnel stats: $PROJECT_STORAGE/data/processed/funnel_stats.json"
