#!/bin/bash
# Task 2 -- vocab-size sweep on the deduped local corpus
# (scripts/run_dedup_datatrove.py's output), instead of the finished
# frank-rg/LLM-Assignment dataset. Uses scripts/sweep_vocab_size.py's
# --processed-dir path (src/tokenizer/data.py::stratified_sample_processed).
#
# Requires scripts/run_dedup_datatrove.py to have already been run against
# $PROJECT_STORAGE/data/raw, writing deduped output to
# $PROJECT_STORAGE/data/processed/{pt,es,hi}/*.parquet.
#
# Must run as a compute job, not on the shared login node: this trains 6
# candidate tokenizers back to back (tokenizers' Rust BPE trainer spins up
# ~1 thread/core), and the login node is shared across every user on the
# cluster -- CPU contention there measured a ~15x slowdown (a run that
# should take under a minute took 17+ minutes with 100% swap in use from
# other users' jobs).
#
# Run scripts/generate_report.py --sweep-csv <out-dir>/results.csv
# --out-dir <out-dir> afterwards (fast, no data to read -- fine to run on
# the login node) to get the Kneedle-picked vocab size + charts.
#
# Usage: sbatch scripts/slurm/05_sweep_vocab_size_fineweb2.sh
#SBATCH --job-name=llm-und-sweep-vocab
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # tokenizers' Rust BPE trainer parallelizes merge counting
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%j-05_sweep_vocab.out
#SBATCH --error=runs/%j-05_sweep_vocab.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/sweep_vocab_size.py \
    --processed-dir "$PROJECT_STORAGE/data/processed" \
    --out-dir "$PROJECT_STORAGE/artifacts/tokenizer_sweep_deduped" \
    2>&1 | tee "$LOG_DIR/sweep_vocab_size.log"

echo "Sweep results under: $PROJECT_STORAGE/artifacts/tokenizer_sweep_deduped"
