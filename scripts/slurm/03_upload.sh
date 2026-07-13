#!/bin/bash
# Push data/final/ to the HF Hub (scripts/upload_to_hf.py). Uploads as
# PRIVATE by default -- this publishes the dataset, so submit this one
# manually and deliberately after reviewing $PROJECT_STORAGE/data/final
# (README.md doc counts, a manual sample per Etapa 7's checklist), not as
# an automatic --dependency chain off the previous jobs.
#
# Usage:
#   export HF_TOKEN=...                 # token with write access
#   export REPO_ID=frank-rg/LLM-Assignment
#   sbatch scripts/slurm/03_upload.sh
#SBATCH --job-name=llm-und-upload
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=CHANGE_ME        # cluster-specific
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00              # upload is network-bound, not compute-bound
#SBATCH --output=runs/%j-03_upload.out
#SBATCH --error=runs/%j-03_upload.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${HF_TOKEN:?export HF_TOKEN before submitting -- a token with write access to REPO_ID}"
: "${REPO_ID:?export REPO_ID before submitting, e.g. frank-rg/LLM-Assignment}"

uv run scripts/upload_to_hf.py \
    --repo-id "$REPO_ID" \
    --local-dir "$PROJECT_STORAGE/data/final" \
    2>&1 | tee "$LOG_DIR/upload.log"
    # add --public to publish openly; omitted here so private is the default
