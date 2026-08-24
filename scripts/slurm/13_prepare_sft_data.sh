#!/bin/bash
# Task 5 -- tokenize + label-mask the registered SFT sources
# (src/sft/sources/: EuroBlocks pt/es, smoltalk2PT, indic-instruct hh-rlhf/hi,
# aya_collection hi) into $PROJECT_STORAGE/data/sft/{train,val}{,_labels}.bin,
# via scripts/prepare_sft_data.py.
#
# CPU-only (tokenization + packing, no model involved), hence
# --partition=berzelius-cpu -- this must NOT be run on the login node:
# EuroBlocks alone is a 1.09M-row single stream that has to be scanned end
# to end just to filter the pt/es rows out of it.
#
# --cpus-per-task=16: unlike scripts/slurm/06_prepare_pretrain_full.sh
# (which feeds tokenizers' Rust backend whole batches via
# batch_encode_plus and so scales with cores), prepare_sft_data.py renders
# one conversation at a time in Python (src/sft/render.py has to interleave
# per-turn label masks with the token ids, which the batch API can't
# express), so the tokenization itself is effectively single-threaded.
# The extra cores are for the HF datasets/parquet reader underneath it,
# not the tokenizer -- don't expect this to speed up with more.
#
# --mem=64G: prepare_sft_data.py holds every rendered example in a Python
# list (token ids + labels) before packing, so peak RSS scales with the
# whole SFT corpus rather than with a streaming window. Measured, not
# estimated: the full run (316,124 examples, 504M tokens across both
# splits) peaked at 24.6GB (`sacct -j <id> --format=MaxRSS`). 64G is ~2.5x
# that, which leaves room for the corpus to grow as sources are added to
# src/sft/sources/ -- raise it if a new source pushes MaxRSS close.
#
# --time=02:00:00: the full run took 23m49s wall clock (316,124 examples ->
# 504M tokens), so this is ~5x measured. The generous multiple is because
# most of that time is HF dataset download/scan, which is only that fast
# with the sources already in $HF_HOME -- a cold cache on a fresh
# $PROJECT_STORAGE re-downloads EuroBlocks' full 1.09M-row stream.
#
# Smoke-test the whole path (sources reachable, tokenizer loads, output
# shape sane) in a couple of minutes before committing to the full pull:
#   sbatch --export=ALL,SFT_LIMIT=200 scripts/slurm/13_prepare_sft_data.sh
# SFT_LIMIT caps examples *per source* and writes to a separate
# data/sft_smoke directory, so a smoke run can never be mistaken for real
# training data by scripts/slurm/14_train_sft.sh.
#
# Chain before the training job:
#   JOB1=$(sbatch --parsable scripts/slurm/13_prepare_sft_data.sh)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/slurm/14_train_sft.sh)
#
# Usage: sbatch scripts/slurm/13_prepare_sft_data.sh
#SBATCH --job-name=llm-und-prepare-sft
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00              # ~5x the measured 23m49s -- see comment above
#SBATCH --output=runs/%j-13_prepare_sft.out
#SBATCH --error=runs/%j-13_prepare_sft.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

# --tokenizer-dir points at x_andaf's project directory (world-readable,
# read-only for this job), NOT the repo's checked-in artifacts/tokenizer:
# the vocabulary written into data/sft/meta.json has to be the same one the
# pretrained checkpoint in 14_train_sft.sh was trained with, or train_sft.py
# refuses to start (vocab_size mismatch).
SOURCE_STORAGE=/proj/assert-berzelius/users/x_andaf/llm-und

LIMIT_ARGS=()
OUT_DIR="$PROJECT_STORAGE/data/sft"
if [ -n "${SFT_LIMIT:-}" ]; then
    LIMIT_ARGS=(--limit "$SFT_LIMIT")
    OUT_DIR="$PROJECT_STORAGE/data/sft_smoke"
    echo "SMOKE RUN: --limit $SFT_LIMIT, writing to $OUT_DIR (not the real data/sft)"
fi

uv run scripts/prepare_sft_data.py \
    --tokenizer-dir "$SOURCE_STORAGE/artifacts/tokenizer" \
    --out-dir "$OUT_DIR" \
    "${LIMIT_ARGS[@]}"

echo "Packed SFT data under: $OUT_DIR"
