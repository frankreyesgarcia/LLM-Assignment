#!/bin/bash
# Task 3 -- tokenize the full andre15silva/pretrain-pt-es-hi corpus
# (~1.3TB across pt/es/hi) into data/pretrain_full/{train,val}.bin, via
# scripts/prepare_pretrain_data_streaming.py.
#
# Unlike scripts/prepare_pretrain_data.py (loads a whole split into one
# Python list -- fine for the few-thousand-doc pilot dataset, not for this
# corpus), this streams parquet shards and does a two-pass tokenize
# (count, then write) so nothing needs the full corpus in RAM.
#
# andre15silva/pretrain-pt-es-hi is itself a copy of this repo's own
# scripts/build_final_dataset.py output (same shard layout/sizes) -- if
# $PROJECT_STORAGE/data/final already has it from an earlier run of this
# pipeline, --project-final-dir below means this job never touches HF at
# all. Otherwise it falls back to snapshot_download (resumable, see
# scripts/download_sources.py) into $PROJECT_STORAGE/data/pretrain_source.
#
# --time/--mem are rough placeholders, not calibrated against a real run
# (no cluster access yet -- see scripts/slurm/02_build_final.sh for how
# the other stages' numbers were derived from actual timed jobs). Treat
# this the same way once you can dry-run on a small --languages subset:
# time it, then size --time/--mem off that instead of trusting this
# comment.
#
# Chain before the training job:
#   JOB1=$(sbatch --parsable scripts/slurm/06_prepare_pretrain_full.sh)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/07_pretrain.sh)
#
# Usage: sbatch scripts/slurm/06_prepare_pretrain_full.sh
#SBATCH --job-name=llm-und-prepare-pretrain
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # tokenizers' Rust backend parallelizes batch_encode_plus internally
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%j-06_prepare_pretrain.out
#SBATCH --error=runs/%j-06_prepare_pretrain.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

# --project-final-dir already lives at
# /proj/assert-berzelius/users/x_andaf/llm-und/data/final -- set
# PROJECT_STORAGE accordingly so this resolves without touching HF
uv run scripts/prepare_pretrain_data_streaming.py \
    --repo-id andre15silva/pretrain-pt-es-hi \
    --project-final-dir "$PROJECT_STORAGE/data/final" \
    --local-dir "$PROJECT_STORAGE/data/pretrain_source" \
    --tokenizer-dir "$PROJECT_STORAGE/artifacts/tokenizer" \
    --out-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --val-shards-per-lang 2 \
    2>&1 | tee "$LOG_DIR/prepare_pretrain_full.log"

echo "Tokenized corpus under: $PROJECT_STORAGE/data/pretrain_full"
