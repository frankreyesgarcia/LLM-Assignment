#!/bin/bash
# Task 3 -- pretrain a real (not toy-config) GPT on the tokenized full
# corpus from scripts/slurm/06_prepare_pretrain_full.sh, via scripts/train.py
# (thin CLI over src/model/train.py::train_model -- same code path as the
# scaling-law sweeps, just pointed at real data/model sizes instead of the
# tiny debug defaults).
#
# Model size: 12 layer / 12 head / 768 dim, block_size 1024 -- a GPT-2-small
# equivalent (~124M params), picked directly rather than from a scaling-law
# fit (see scripts/fit_scaling_law.py if you want to redo this properly
# against a stated compute budget instead).
#
# batch_size=8 (not the more standard 64): a 100-iter dry run at
# batch_size=64 OOM'd during the training forward/backward pass -- the
# lm_head logits tensor alone is (batch, block_size, vocab_size) =
# (64, 1024, 32000) fp32 = ~8.4GB, and with no mixed precision (see KNOWN
# GAP below) several such buffers must coexist for backprop, which
# exceeded this GPU's 40GB. batch_size=8 fits comfortably, but does mean
# a noisier gradient per step than batch_size=64 would give; if that
# matters, prefer adding gradient accumulation instead (see KNOWN GAP
# below) so per-step memory stays low while the effective batch size seen
# by the optimizer stays at 64.
#
# max_iters=320000 at batch_size=8 * block_size=1024 = ~2.6B tokens seen --
# deliberately far short of one epoch over the full corpus (a byte-level
# 32k-vocab tokenizer over 1.3TB of text is on the order of 300B+ tokens;
# one epoch over that for a 124M model is ~100x past the Chinchilla
# compute-optimal ratio of ~20 tokens/param). max_iters/warmup_iters/
# eval_interval below are scaled by the same 8x as batch_size shrank by,
# so the total token budget and warmup/eval-interval *fractions* of the
# run match what they were at batch_size=64 -- only the per-step batch
# shrank, not the intended amount of training.
#
# KNOWN GAP: src/model/train.py has no gradient accumulation, no mixed
# precision (fp32 throughout despite GPT using
# F.scaled_dot_product_attention, which would get Flash Attention for free
# under autocast), and no checkpoint-resume (only saves the best-val-loss
# ckpt, can't continue a run past a job's walltime). For a single job that
# fits in one submission this is fine; if a real run needs to span
# multiple jobs or go faster, those would need to be added to
# src/model/train.py first -- not done here since it wasn't asked for.
#
# --partition/--gpus: Berzelius' CPU partition is explicitly named
# "berzelius-cpu" elsewhere in this repo (see _common.sh, 01-05); by
# convention that implies plain "berzelius" for GPU nodes, requested via
# --gpus rather than a separate GPU-specific partition name. Not verified
# against real cluster access -- confirm with `sinfo` / your cluster's
# docs once you have an account, and adjust --partition/--gpus/--time
# below accordingly.
#
# --time is an unverified placeholder (no cluster access to calibrate
# against, unlike the CPU-stage numbers elsewhere in this directory which
# were timed on real runs -- see scripts/slurm/02_build_final.sh). Do a
# short dry run first (--max-iters 100) and extrapolate from its
# tokens/sec before trusting this for a real submission.
#
# Usage: sbatch scripts/slurm/07_pretrain.sh
#SBATCH --job-name=llm-und-pretrain
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius        # unverified -- see comment above
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # unverified placeholder -- calibrate from a short dry run
#SBATCH --output=runs/%j-07_pretrain.out
#SBATCH --error=runs/%j-07_pretrain.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

uv run scripts/train.py \
    --data-dir "$PROJECT_STORAGE/data/pretrain_full" \
    --out-dir "$PROJECT_STORAGE/runs/pretrain_full" \
    --block-size 1024 \
    --batch-size 8 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --max-iters 320000 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-iters 8000 \
    --eval-interval 4000 \
    --eval-iters 50 \
    --device auto \
    2>&1 | tee "$LOG_DIR/pretrain.log"

echo "Checkpoint + logs under: $PROJECT_STORAGE/runs/pretrain_full"
