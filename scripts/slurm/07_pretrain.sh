#!/bin/bash
# Task 3 -- pretrain a real (not toy-config) GPT on the tokenized full
# corpus from scripts/slurm/06_prepare_pretrain_full.sh, via scripts/train.py
# (thin CLI over src/model/train.py::train_model).
#
# --flops-budget sizes both the model and the token count from a stated
# compute budget instead of picking them by hand: it applies the Kaplan et
# al. (2020) compute-optimal allocation (src/model/scaling.py, Eq 6.1/6.7)
# to derive n_layer/n_head/n_embd (targeting N_opt non-embedding params)
# and max_iters (targeting D_opt tokens), then --n-layer/--n-head/--n-embd/
# --max-iters are ignored. This replaces an earlier version of this script
# that hand-picked a 12-layer/768-dim (~124M param) shape and a wall-clock-
# fitted max_iters (60000, ~492M tokens) -- both now fall out of the budget
# below instead of being guessed.
#
# 1e17 FLOPs was picked to land in roughly a ~5h wall-clock window, using a
# real dry run's measured throughput as the conversion factor: 819,200
# tokens in 22.6s (36,248 tokens/sec) on a 25,220,096-non-embedding-param
# config (n_layer=8, n_head=8, n_embd=512, batch_size=8, block_size=1024)
# translates to an *achieved* compute rate of 6 * 25,220,096 * 36,248 ~=
# 5.485 TFLOP/s (see also: this is ~28% of an A100-SXM4-80GB's 19.5
# TFLOP/s fp32 peak, unsurprising since src/model/train.py has no mixed
# precision -- see KNOWN GAP below). 1e17 FLOPs / 5.485 TFLOP/s ~= 18,232s
# ~= 5.06h. Recompute this conversion factor from your own dry run before
# trusting it on different hardware -- e.g.
#   uv run scripts/train.py --data-dir ... --out-dir ... --max-iters 100 \
#       --n-layer 8 --n-head 8 --n-embd 512 --batch-size 8 --block-size 1024
# then achieved_flops_per_sec = 6 * n_params_non_embed * measured_tokens_per_sec,
# and --flops-budget = achieved_flops_per_sec * (desired wall-clock seconds).
# At 1e17 FLOPs this currently resolves to N_opt~=9.34M non-embedding params
# (n_layer=6, n_head=6, n_embd=384) and D_opt~=6.90B tokens (max_iters
# ~=841,737 at batch_size=8 * block_size=1024) -- verify with:
#   uv run scripts/plan_compute_budget.py --flops-budget 1e17 \
#       --batch-size 8 --block-size 1024
# Raise --flops-budget (and --time proportionally, using the same
# TFLOP/s conversion factor) for a bigger run once there's a larger time
# budget available.
#
# batch_size=8 (not the more standard 64): a 100-iter dry run of the
# earlier 124M-param config at batch_size=64 OOM'd during the training
# forward/backward pass -- the lm_head logits tensor alone is (batch,
# block_size, vocab_size) = (64, 1024, 32000) fp32 = ~8.4GB, and with no
# mixed precision (see KNOWN GAP below) several such buffers must coexist
# for backprop, which exceeded this GPU's 40GB. batch_size=8 fits
# comfortably, but does mean a noisier gradient per step than batch_size=64
# would give; if that matters, prefer adding gradient accumulation instead
# (see KNOWN GAP below) so per-step memory stays low while the effective
# batch size seen by the optimizer stays at 64. The Kaplan-derived model
# above is smaller than that 124M config, so batch_size=8 fits with margin
# to spare here -- raising it is an option, just not required.
#
# --warmup-iters/--eval-interval are still set by hand as ~2.5%/1.25% of
# max_iters (the same fractions the earlier hand-picked config used) --
# --flops-budget only derives model shape and max_iters, not these.
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
# --time=06:00:00 is sized off the ~5.06h estimate above (see --flops-budget
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
#SBATCH --time=06:00:00              # sized off measured dry-run throughput, see comment above
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
    --flops-budget 1e17 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-iters 21043 \
    --eval-interval 10522 \
    --eval-iters 50 \
    --device auto \
    2>&1 | tee "$LOG_DIR/pretrain.log"

echo "Checkpoint + logs under: $PROJECT_STORAGE/runs/pretrain_full"
