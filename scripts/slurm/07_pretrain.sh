#!/bin/bash
# Task 3 -- pretrain a real (not toy-config) GPT on the tokenized full
# corpus from scripts/slurm/06_prepare_pretrain_full.sh, via scripts/train.py
# (thin CLI over src/model/train.py::train_model).
#
# Model shape (n_layer/n_head/n_embd below) is hand-picked, not derived:
# the shape below (n_layer=8, n_head=8, n_embd=512) is the one already
# validated by a real dry run on this GPU (see below) rather than a fresh,
# unvalidated guess.
#
# --flops-budget only derives *how long to train that shape* (max_iters),
# from the FLOPs accounting identity FLOPs(N, D) = 6*N*D (src/model/
# scaling.py) -- N here is this shape's actual total (embeddings included)
# param count, not a target the flag searches over. --max-iters below would
# be ignored if also given.
#
# 1e17 FLOPs was picked to land in roughly a ~3h wall-clock window, using a
# real dry run's measured throughput as the conversion factor: 819,200
# tokens in 22.6s (36,248 tokens/sec) on this exact shape (n_layer=8,
# n_head=8, n_embd=512, vocab_size=32000, block_size=1024, batch_size=8;
# 42,128,384 total params) translates to an *achieved* compute rate of
# 6 * 42,128,384 * 36,248 ~= 9.162 TFLOP/s (see also: this is ~47% of an
# A100-SXM4-80GB's 19.5 TFLOP/s fp32 peak, unsurprising since
# src/model/train.py has no mixed precision -- see KNOWN GAP below).
# 1e17 FLOPs / 9.162 TFLOP/s ~= 10,914s ~= 3.03h. Recompute this conversion
# factor from your own dry run before trusting it on different hardware --
# e.g.
#   uv run scripts/train.py --data-dir ... --out-dir ... --max-iters 100 \
#       --n-layer 8 --n-head 8 --n-embd 512 --batch-size 8 --block-size 1024
# then achieved_flops_per_sec = 6 * n_params_total * measured_tokens_per_sec,
# and --flops-budget = achieved_flops_per_sec * (desired wall-clock seconds).
# At 1e17 FLOPs and this shape, that currently resolves to D~=395.6M tokens
# (max_iters ~=48,293 at batch_size=8 * block_size=1024).
# Raise --flops-budget (and --time proportionally, using the same
# TFLOP/s conversion factor) for a longer run on this same shape once
# there's a larger time budget available; raising the shape itself needs a
# separate, deliberate choice, not just a bigger --flops-budget.
#
# batch_size=8 (not the more standard 64): a 100-iter dry run of an earlier,
# larger 124M-param config at batch_size=64 OOM'd during the training
# forward/backward pass -- the lm_head logits tensor alone is (batch,
# block_size, vocab_size) = (64, 1024, 32000) fp32 = ~8.4GB, and with no
# mixed precision (see KNOWN GAP below) several such buffers must coexist
# for backprop, which exceeded this GPU's 40GB. batch_size=8 fits
# comfortably at the smaller shape used here too, but does mean a noisier
# gradient per step than batch_size=64 would give; if that matters, prefer
# adding gradient accumulation instead (see KNOWN GAP below) so per-step
# memory stays low while the effective batch size seen by the optimizer
# stays at 64.
#
# --warmup-iters/--eval-interval are still set by hand as ~2.5%/1.25% of
# max_iters (the same fractions an earlier hand-picked config used) --
# --flops-budget only derives max_iters, not these.
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
# --time=04:00:00 is sized off the ~3.03h estimate above (see --flops-budget
# comment) plus a ~20% buffer -- unlike the CPU-stage numbers elsewhere in
# this directory (timed on real runs, see scripts/slurm/02_build_final.sh),
# this is extrapolated from a short 100-iter dry run, not a full timed
# run, so treat it as approximate.
#
# Usage: sbatch scripts/slurm/07_pretrain.sh
#SBATCH --job-name=llm-und-pretrain
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius        # unverified -- see comment above
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00              # sized off measured dry-run throughput, see comment above
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
    --n-layer 8 --n-head 8 --n-embd 512 \
    --flops-budget 1e17 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-iters 1207 \
    --eval-interval 604 \
    --eval-iters 50 \
    --device auto \
    2>&1 | tee "$LOG_DIR/pretrain.log"

echo "Checkpoint + logs under: $PROJECT_STORAGE/runs/pretrain_full"
