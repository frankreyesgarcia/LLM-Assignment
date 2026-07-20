#!/bin/bash
# Etapa 6 (TASK1-PLAN.md): aggregate the per-language Parquet parts from
# scripts/run_dedup_datatrove.py into the final pt/es/hi/all config shape,
# sharded into ~500MB-1GB output files (src/aggregate.py::shuffle_into_shards
# -- streams the input, never loads a whole config into memory).
#
# Chain after the dedup job finishes:
#   JOB1=$(sbatch --parsable scripts/slurm/01_dedup_datatrove.sh)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/02_build_final.sh)
#
# --time/--mem sized for the real corpus (~1.3TB across pt/es/hi as of the
# datatrove dedup run, not the pilot-scale corpus this script was
# originally written against): shuffle_into_shards has no resume/checkpoint
# logic, so a mid-run timeout means rerunning that language's build from
# scratch -- generous margin is cheaper than a wasted multi-hour retry.
#
# Usage: sbatch scripts/slurm/02_build_final.sh
#SBATCH --job-name=llm-und-build-final
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%j-02_build_final.out
#SBATCH --error=runs/%j-02_build_final.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run python - <<PYEOF 2>&1 | tee "$LOG_DIR/build_final.log"
import sys
sys.path.insert(0, ".")
from pathlib import Path
from scripts.build_final_dataset import main

main(
    in_dir=Path("$PROJECT_STORAGE/data/processed"),
    out_dir=Path("$PROJECT_STORAGE/data/final"),
)
PYEOF
