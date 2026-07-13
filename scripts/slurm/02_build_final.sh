#!/bin/bash
# Etapa 6 (TASK1-PLAN.md): aggregate the per-language Parquet parts from
# 01_run_all_sources.sh into the final pt/es/hi/all config shape, sharded
# into ~500MB-1GB output files (src/aggregate.py::shuffle_into_shards --
# streams the input, never loads a whole config into memory).
#
# Chain after the ingest job finishes:
#   JOB1=$(sbatch --parsable scripts/slurm/01_run_all_sources.sh)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/02_build_final.sh)
#
# Usage: sbatch scripts/slurm/02_build_final.sh
#SBATCH --job-name=llm-und-build-final
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # bounded by shard size (~750MB), not corpus size
#SBATCH --time=08:00:00
#SBATCH --output=runs/%j-02_build_final.out
#SBATCH --error=runs/%j-02_build_final.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

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
