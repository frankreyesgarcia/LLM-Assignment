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
# max_iters=60000 at batch_size=8 * block_size=1024 = ~492M tokens seen,
# targeting a ~12h wall-clock budget: that same dry run measured
# ~14,300 tokens/sec on this GPU, so 60000 * 8 * 1024 / 14300 ~= 34,400s
# ~= 9.6h, leaving buffer under --time below. This is a deliberately
# reduced token budget (the original sizing -- 320000 iters, ~2.6B tokens,
# picked to roughly match Chinchilla's ~20-tokens/param ratio for this
# 124M model -- would take ~51h at this throughput). Raise --max-iters
# (and --time proportionally) for a longer/less-undertrained run once
# there's a bigger time budget available; rescale --warmup-iters/
# --eval-interval with it to keep the same ~2.5%/1.25%-of-max_iters
# fractions used here.
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
# --time=12:00:00 is sized off the ~9.6h estimate above (see max_iters
# comment) plus buffer -- unlike the CPU-stage numbers elsewhere in this
# directory (timed on real runs, see scripts/slurm/02_build_final.sh),
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
#SBATCH --time=12:00:00              # sized off measured dry-run throughput, see comment above
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
    --max-iters 60000 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-iters 1500 \
    --eval-interval 750 \
    --eval-iters 50 \
    --device auto \
    2>&1 | tee "$LOG_DIR/pretrain.log"

echo "Checkpoint + logs under: $PROJECT_STORAGE/runs/pretrain_full"
