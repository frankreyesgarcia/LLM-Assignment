#!/bin/bash
# Fase 2 (rewrite) -- cross-source MinHash dedup for pt-fineweb2/es-fineweb2/
# hi-fineweb2/hi-sangraha via scripts/run_dedup_datatrove.py, reading the
# local copy scripts/slurm/00_download_sources.sh already pulled down.
#
# This script itself is a lightweight *driver*: with --executor slurm,
# datatrove's SlurmPipelineExecutor.run() submits its own sbatch job array
# per stage and returns immediately -- confirmed against the installed
# package's source (SlurmPipelineExecutor.run() calls launch_job(), which
# submits via sbatch and returns; it does not poll/block for completion).
# So this driver only needs to live long enough to fire off 12 sbatch calls
# (4 stages x 3 languages) -- a couple of minutes, not the whole run's
# duration. All 3 languages' 4-stage dependency chains then run
# *concurrently* on the cluster once submitted.
#
# --tasks 150 covers es-fineweb2's 146 files (the largest source) with a
# little headroom -- sharding is file-level (datatrove's DataFolder.get_shard:
# all_files[rank::world_size]), so any task beyond a source's own file count
# just gets 0 files and exits instantly (harmless, not wasted compute).
# hi-fineweb2 only has 8 files, so its 22M docs are always split across at
# most 8 concurrent tasks regardless of --tasks -- a data-layout ceiling,
# not something this flag can raise.
#
# --time 04:00:00 per stage: calibrated from a real run (hi, 40k docs,
# --tasks 1) -- stage 1 (signature) measured ~790 docs/sec/task, ~14x the
# cost of stages 2-4 combined. Extrapolating that rate across each source's
# per-task doc share (docs / its own file count) puts all 3 languages at
# ~70 minutes end to end; 4h gives ~3x margin for cluster contention/variance.
#
# berzelius-cpu capacity check (sinfo -p berzelius-cpu): 8 nodes x 128 CPUs
# = 1024 total, 832 idle at the time this was sized -- --tasks 150 (1 CPU/
# task by default) across all 3 languages concurrently (~450 CPUs worst
# case) fits comfortably. Cluster-wide MaxArraySize=1001, so --tasks 150 is
# also safely under that.
#
# Usage: sbatch scripts/slurm/01_dedup_datatrove.sh
#SBATCH --job-name=llm-und-dedup-datatrove
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2            # driver only orchestrates -- the real work runs in datatrove's own submitted array jobs
#SBATCH --mem=8G
#SBATCH --time=00:15:00              # driver submits and exits -- doesn't wait for the array jobs (see above)
#SBATCH --output=runs/%j-01_dedup_datatrove.out
#SBATCH --error=runs/%j-01_dedup_datatrove.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

# This driver's own --mem=8G above sets SLURM_MEM_PER_NODE in its
# environment; datatrove's SlurmPipelineExecutor submits each stage's array
# job via a plain `subprocess.check_output(["sbatch", ...])` call, which
# inherits this shell's environment by default -- so without unsetting these,
# the child jobs' own --mem-per-cpu directive collides with the inherited
# SLURM_MEM_PER_NODE at runtime ("SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and
# SLURM_MEM_PER_NODE are mutually exclusive", confirmed by hitting this
# directly on a real submission).
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

uv run scripts/run_dedup_datatrove.py \
    --executor slurm \
    --raw-dir "$PROJECT_STORAGE/data/raw" \
    --out-dir "$PROJECT_STORAGE/data/processed" \
    --work-dir "$PROJECT_STORAGE/data/dedup_work" \
    --tasks 150 \
    --account "$PROJECT_ACCOUNT" \
    --partition berzelius-cpu \
    --time 04:00:00 \
    2>&1 | tee "$LOG_DIR/dedup_datatrove.log"

echo "Deduped output under: $PROJECT_STORAGE/data/processed"
