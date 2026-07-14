#!/bin/bash
# Fase 2a (new): bulk-download every available source's raw files to
# persistent project storage, in parallel, before any filtering/dedup runs.
#
# WHY THIS STAGE EXISTS: `datasets(streaming=True)` (what 01_run_all_sources.sh
# used to read straight from the Hub) opens one HTTP connection per source
# and reads it sequentially -- measured at ~8 MB/s regardless of link speed.
# huggingface_hub.snapshot_download (what this script uses) issues many
# concurrent file requests instead -- measured ~61-66 MB/s aggregate on the
# same repos (a ~7-8x improvement; the ceiling looks like an HF-side rate
# limit, not our bandwidth). At that rate the full ~14.8TB corpus downloads
# in ~2.6 days here, vs. an extrapolated ~690 days for the old combined
# stream+process design. See scripts/download_sources.py's docstring.
#
# Chain before the filter+dedup job:
#   JOB0=$(sbatch --parsable scripts/slurm/00_download_sources.sh)
#   JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 scripts/slurm/01_run_all_sources.sh)
#
# Resumable for free: snapshot_download's local_dir already skips files
# that are already present and up to date, so --requeue / a rerun after a
# partial download just picks up wherever it left off.
#
# Usage: sbatch scripts/slurm/00_download_sources.sh
#SBATCH --job-name=llm-und-download
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4            # this stage is network-bound, not CPU-bound
#SBATCH --mem=16G
#SBATCH --time=24:00:00              # generous but not unlimited -- rely on --requeue for a multi-day download
#SBATCH --requeue
#SBATCH --output=runs/%j-00_download.out
#SBATCH --error=runs/%j-00_download.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/download_sources.py \
    --out-dir "$PROJECT_STORAGE/data/raw" \
    --max-workers 16 \
    2>&1 | tee "$LOG_DIR/download_sources.log"

echo "Raw files under: $PROJECT_STORAGE/data/raw"
