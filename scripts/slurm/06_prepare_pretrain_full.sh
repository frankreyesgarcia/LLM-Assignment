#!/bin/bash
# Task 3 -- tokenize the full andre15silva/pretrain-pt-es-hi corpus
# (~1.3TB across pt/es/hi) into data/pretrain_full/{train,val}.bin, via
# scripts/prepare_pretrain_data_streaming.py.
#
# Unlike scripts/prepare_pretrain_data.py (loads a whole split into one
# Python list -- fine for the few-thousand-doc pilot dataset, not for this
# corpus), this streams parquet shards and tokenizes in a single pass
# straight to disk (see tokenize_and_write) so nothing needs the full
# corpus in RAM.
#
# andre15silva/pretrain-pt-es-hi is itself a copy of this repo's own
# scripts/build_final_dataset.py output (same shard layout/sizes) -- if
# $PROJECT_STORAGE/data/final already has it from an earlier run of this
# pipeline, --project-final-dir below means this job never touches HF at
# all. Otherwise it falls back to snapshot_download (resumable, see
# scripts/download_sources.py) into $PROJECT_STORAGE/data/pretrain_source.
#
# --cpus-per-task=128 / --time=24:00:00: a real timed test against just the
# hi language (75GB of the 1.3TB total, the smallest of the three) at the
# previous --cpus-per-task=16 and the old two-pass tokenizer measured
# ~1.9 min/shard for the count pass alone (100 train shards), extrapolating
# to ~107h for the full pt/es/hi corpus -- past this cluster's 3-day
# (72h) partition time limit, so that config could never actually finish.
# Fixed two ways: (1) tokenize_and_write is now single-pass (see its
# docstring), roughly halving the tokenization work; (2) --cpus-per-task
# raised from 16 to 128 -- `sinfo -p berzelius-cpu -N` showed several
# nodes fully idle (128/128 CPUs, 1126GB RAM free) at the time this was
# written, so 16 was using only an eighth of a node's cores for a
# CPU-bound job; requesting the full node lets tokenizers' Rust backend
# use all of it (a single process/node -- SLURM doesn't parallelize one
# job across multiple nodes, so this is the ceiling per job as written).
# Combined (~2x from single-pass, up to ~8x from cores, realistically
# less than perfectly linear), this should land well under 72h, but
# neither number has been re-verified against an actual timed run at this
# cpus-per-task/single-pass combination -- if you can, re-run the hi-only
# timing test (--languages hi) at these settings before fully trusting
# --time here. Using the full node also means it's the exclusive occupant
# until this finishes -- fine given multiple nodes were idle, but worth
# checking that's still true at submission time.
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
#SBATCH --cpus-per-task=128          # tokenizers' Rust backend parallelizes batch_encode_plus internally -- see comment above
#SBATCH --mem=96G
#SBATCH --time=24:00:00              # see comment above -- estimate, not re-verified at this cpus-per-task/single-pass combination
#SBATCH --output=runs/%j-06_prepare_pretrain.out
#SBATCH --error=runs/%j-06_prepare_pretrain.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

# --project-final-dir/--tokenizer-dir point at x_andaf's project directory
# (world-readable, this job only reads from it) rather than $PROJECT_STORAGE
# -- $PROJECT_STORAGE is your own writable project directory, which is
# where --local-dir/--out-dir below actually write output.
SOURCE_STORAGE=/proj/assert-berzelius/users/x_andaf/llm-und
uv run scripts/prepare_pretrain_data_streaming.py \
    --repo-id andre15silva/pretrain-pt-es-hi \
    --project-final-dir "$SOURCE_STORAGE/data/final" \
    --local-dir "$PROJECT_STORAGE/data/pretrain_source" \
    --tokenizer-dir "$SOURCE_STORAGE/artifacts/tokenizer" \
    --out-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --val-shards-per-lang 2

echo "Tokenized corpus under: $PROJECT_STORAGE/data/pretrain_full"
