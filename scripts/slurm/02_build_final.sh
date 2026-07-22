#!/bin/bash
# Etapa 6 (TASK1-PLAN.md): aggregate the per-language Parquet parts from
# scripts/run_dedup_datatrove.py into the final pt/es/hi config shape,
# sharded into ~500MB-1GB output files (src/aggregate.py::shuffle_into_shards
# -- streams the input, never loads a whole config into memory). No combined
# `all` config -- see build_final_dataset.py's docstring for why.
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
# Calibrated against a real timed run on hi (75GB, smallest of the 3):
# 47m25s after fixing an O(batches * shards) inefficiency in
# shuffle_into_shards's pass-1 loop (was 51m12s before -- the fix helps
# more on larger shard counts than hi's 102, so treat this as a
# conservative/lower-bound rate). Extrapolating ~1.58GB/min linearly:
# pt (365GB) ~3.85h, es (849GB) ~8.95h, all (1.3TB, re-reads pt+es+hi
# combined) ~13.7h -- ~26.5h sequential total, hence --time at the
# partition max (`sinfo -p berzelius-cpu` shows 3-00:00:00) rather than a
# number that looked generous before this was actually measured. Since
# calibrated, build_final_dataset.py now also projects away the `metadata`
# column (see OUTPUT_COLUMNS there) -- strictly less data read/written per
# row than what was timed, so this remains a safe upper bound, if anything
# now pessimistic.
#
# --mem history, both real OOMs on pt's 498-shard pass 1:
#   64G:  OOM'd 5 min in -- 498 concurrently-open shard writers, each
#         buffering pt's then-9-field metadata struct (fixed by projecting
#         to just text/id, see build_final_dataset.py's OUTPUT_COLUMNS).
#   256G: OOM'd at 92% through pt's pass 1 (57m55s) even after the metadata
#         fix AND disabling each writer's dictionary encoding (ParquetWriter
#         defaults use_dictionary=True, which makes a writer accumulate a
#         growing dictionary of every unique value for its whole lifetime --
#         near-total for free text -- confirmed via a controlled local
#         before/after comparison). 498 concurrently-open writers still
#         wasn't survivable at 256G even with both fixes; getting to 92%
#         (vs. 90% pre-fix, but in 58 vs. 71 min) shows the fixes genuinely
#         helped, just not enough headroom for pt's full shard count.
#   600G: pt completed fully at this level (498 shards). es (1158 shards,
#         2.3x pt's count) OOM'd at 79% (668/849GB) -- much further than a
#         naive proportional scale-up from pt would predict, consistent
#         with the deceleration pattern seen in local before/after testing.
#         The combined `all` config was dropped entirely (see
#         build_final_dataset.py's docstring) once it became clear it was
#         both unnecessary -- pt/es/hi can be combined at load time via
#         `datasets.concatenate_datasets` -- and the single riskiest stage
#         (would have needed the most shards of any config).
#   900G: current setting -- es got most of the way at 600G, so this
#         should clear it (and hi, far fewer shards than either) with real
#         margin. build_final_dataset.py now also skips any language
#         that already has a clean completed output (see
#         _completed_row_count there), so a rerun after this only redoes
#         whichever language actually failed, not pt from scratch again.
#
# Usage: sbatch scripts/slurm/02_build_final.sh
#SBATCH --job-name=llm-und-build-final
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=900G
#SBATCH --time=3-00:00:00
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
