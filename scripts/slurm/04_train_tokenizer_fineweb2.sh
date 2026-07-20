#!/bin/bash
# Task 2 -- train the final tokenizer on the deduped local corpus
# (scripts/run_dedup_datatrove.py's output), instead of the finished
# frank-rg/LLM-Assignment dataset. Uses scripts/train_tokenizer.py's
# --processed-dir path (src/tokenizer/data.py::iter_texts_processed /
# stratified_sample_processed).
#
# Requires scripts/run_dedup_datatrove.py to have already been run against
# $PROJECT_STORAGE/data/raw, writing deduped output to
# $PROJECT_STORAGE/data/processed/{pt,es,hi}/*.parquet.
#
# vocab_size=32000 was chosen by scripts/sweep_vocab_size.py + generate_report.py
# (Kneedle elbow on the compression curve), confirmed stable across three
# separate sweeps: the pilot corpus, fineweb-2/Sangraha pre-dedup, and now
# the real deduped output of scripts/run_dedup_datatrove.py (see
# artifacts/tokenizer_sweep_deduped/report.md) -- numbers barely moved
# post-dedup (1.5-7.8% dedup rates weren't enough to shift BPE merge stats).
#
# --limit-docs 500000 caps *each* language at 500k docs -- BPE
# merge-frequency stats converge well before the full corpus, so this is
# deliberately a small multiple of the sweep's sample, not "as much as we
# can fit." Re-run with a larger cap if the resulting tokenizer's
# fertility/compression snapshot looks undertrained.
#
# Usage: sbatch scripts/slurm/04_train_tokenizer_fineweb2.sh
#SBATCH --job-name=llm-und-train-tokenizer
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # tokenizers' Rust BPE trainer parallelizes merge counting
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%j-04_train_tokenizer.out
#SBATCH --error=runs/%j-04_train_tokenizer.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/train_tokenizer.py \
    --processed-dir "$PROJECT_STORAGE/data/processed" \
    --vocab-size 32000 \
    --limit-docs 500000 \
    --out-dir "$PROJECT_STORAGE/artifacts/tokenizer" \
    2>&1 | tee "$LOG_DIR/train_tokenizer.log"

echo "Tokenizer saved under: $PROJECT_STORAGE/artifacts/tokenizer"
