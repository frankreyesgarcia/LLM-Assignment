#!/bin/bash
# Task 5 -- supervised fine-tuning of the Task 3 pretrained checkpoint on
# the packed SFT data from scripts/slurm/13_prepare_sft_data.sh, via
# scripts/train_sft.py (thin CLI over src/model/train_sft.py).
#
# --init-from is the real pretraining run's checkpoint, not a pilot:
# pretrain_2.2e18_bf16_qknorm/ckpt.pt (GPTConfig vocab_size=32000,
# block_size=1024, n_layer=11, n_head=16, n_embd=1024, bias=True,
# qk_norm=True, at iter 240,204). train_sft.py reads model_cfg straight
# out of that file, so the architecture -- qk_norm included -- carries over
# without being restated here; only --block-size is asserted against it
# (<=1024). It lives in x_amash's project directory, which is
# world-readable and only read from here.
#
# --amp-dtype defaults to auto = bf16 on an A100, matching what that
# checkpoint was pretrained under (the "bf16" in its name). Left implicit
# rather than pinned so this script stays correct on a GPU without bf16.
#
# --max-iters is derived from the data rather than hardcoded: SFT is
# epoch-shaped (a small, fixed, high-quality corpus you want to see a
# couple of times), unlike pretraining's FLOPs budget over an effectively
# unbounded stream. SFT_EPOCHS below sets how many passes over
# data/sft/train.bin that works out to; 2 is the conventional default for
# instruction tuning (1 often underfits the chat format, 3+ starts
# overfitting a corpus this size). One row of a batch is one whole
# conversation, so an epoch is train_examples/batch_size steps and really
# is one visit to each conversation (bar the per-megabatch remainder that
# length grouping drops).
#
# --lr 5e-5 / --min-lr 5e-6 are scripts/train_sft.py's defaults, i.e. the
# conventional ~10x-below-pretraining starting point -- NOT tuned for this
# checkpoint. Check the first few hundred iterations of the W&B run
# actually show train/loss coming down before trusting a long job.
#
# --batch-size 16 at --block-size 1024 was verified to fit on a 40GB A100
# (-C "thin") by a 300-iteration run at these exact settings -- this shape
# (11 layers / n_embd=1024) is larger than 07_pretrain.sh's, but bf16
# rather than fp32 halves the lm_head logits buffer that OOM'd there. The
# A100-80GB nodes (-C "fat", see `sinfo -o "%P %G"`) have room for more if
# throughput ever needs it. --block-size is now an upper bound rather than
# the size of every batch: rows are padded to the longest conversation in
# their batch, so a batch of short conversations is a smaller tensor.
#
# Do a short dry run first -- into a *separate* --out-dir, since resuming
# is on by default and a real run pointed at a directory with a dry run's
# ckpt_last.pt in it would silently continue from there:
#   sbatch --export=ALL,SFT_DRYRUN=1 scripts/slurm/14_train_sft.sh
# That runs 20 iterations into $PROJECT_STORAGE/runs/sft_dryrun and logs a
# separate W&B run tagged "dryrun".
#
# Resubmitting this exact script (same --out-dir) auto-resumes from
# out_dir/ckpt_last.pt -- same mechanism as 07_pretrain.sh.
#
# --time=08:00:00 is sized from a measured 42,610 tokens/sec (300-iteration
# run, job 17363870, at exactly the shape/batch/block below on a 40GB
# A100 -- evals at the real cadence included, so it is not a
# training-steps-only number). 13_prepare_sft_data.sh's full output is
# 494,086,666 train tokens, so one epoch is ~3.2h and the default
# SFT_EPOCHS=2 is ~6.4h; 8h leaves ~25% slack. Re-derive this whenever the
# corpus, batch size, or GPU changes: max_iters prints at job start, and
# tokens_per_sec is logged to W&B at every eval (as are padded_tokens_seen
# and batch_fill_rate -- real tokens over tokens the GPU actually ran,
# which length-grouped batching should keep near 1.0).
#
# Note the *first* dry run of this script measured only 4,599 tokens/sec --
# at 20 iterations, torch.compile warmup plus an eval every 5 iterations
# (eval_iters=20 over two splits each) is nearly the whole wall clock. Do
# not size a real job from a run that short; SFT_DRYRUN_ITERS exists for
# this.
#
# Usage: sbatch scripts/slurm/14_train_sft.sh
#SBATCH --job-name=llm-und-sft
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00              # placeholder -- size from the dry run, see comment above
#SBATCH --output=runs/%j-14_train_sft.out
#SBATCH --error=runs/%j-14_train_sft.err

# NOTE: `runs/` must already exist before submission -- see pilot.sh for why
# a per-job-id subdirectory wouldn't work here.
set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

PRETRAIN_CKPT=${PRETRAIN_CKPT:-/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt.pt}
# SFT_DATA_DIR override exists so a dry run can validate the GPU path
# against 13_prepare_sft_data.sh's smoke output (data/sft_smoke) while the
# real prep job is still running.
DATA_DIR="${SFT_DATA_DIR:-$PROJECT_STORAGE/data/sft}"
OUT_DIR="$PROJECT_STORAGE/runs/sft"
BLOCK_SIZE=1024
BATCH_SIZE=16
SFT_EPOCHS=${SFT_EPOCHS:-2}

# Derive max_iters from the packed corpus size (see comment above).
MAX_ITERS=$(python3 - "$DATA_DIR/meta.json" "$BATCH_SIZE" "$SFT_EPOCHS" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
batch, epochs = int(sys.argv[2]), float(sys.argv[3])
print(max(1, round(meta["train_examples"] * epochs / batch)))
PY
)
EVAL_INTERVAL=$(( MAX_ITERS / 50 > 0 ? MAX_ITERS / 50 : 1 ))   # ~50 eval points per run
WARMUP_ITERS=$(( MAX_ITERS / 50 > 0 ? MAX_ITERS / 50 : 1 ))    # ~2% of the run, as in 07_pretrain.sh
WANDB_TAGS=(sft pretrain_2.2e18_bf16_qknorm)

if [ -n "${SFT_DRYRUN:-}" ]; then
    # Deliberately a different out-dir: --resume is on by default, so
    # sharing one with the real run would poison it (see comment above).
    OUT_DIR="$PROJECT_STORAGE/runs/sft_dryrun"
    # SFT_DRYRUN_ITERS: 20 is enough to prove the path works end to end, but
    # far too few to measure throughput from -- at that length torch.compile
    # warmup and the eval passes dominate the wall clock. Use a few hundred
    # iterations with the eval cadence backed off (as below) when the point
    # of the run is to size --time for the real job.
    MAX_ITERS=${SFT_DRYRUN_ITERS:-20}
    EVAL_INTERVAL=$(( MAX_ITERS / 4 > 0 ? MAX_ITERS / 4 : 1 ))
    WARMUP_ITERS=2
    WANDB_TAGS+=(dryrun)
    echo "DRY RUN: $MAX_ITERS iters into $OUT_DIR"
fi

echo "max_iters=$MAX_ITERS (epochs=$SFT_EPOCHS, block=$BLOCK_SIZE, batch=$BATCH_SIZE)"

# W&B logging is on unconditionally, not opt-in as in 07_pretrain.sh: every
# training run here should be recoverable after the fact from the run page
# rather than only from runs/%j-*.out. Needs WANDB_API_KEY in the
# environment or a `wandb login` in $HOME/.netrc -- sbatch propagates the
# submitting shell's environment by default, so exporting WANDB_API_KEY
# before submitting is enough. src/model/train_sft.py logs the full
# SFTTrainConfig as the run config, per-step loss/grad_norm/lr, and
# train+val loss/perplexity, tokens_seen, tokens_per_sec and elapsed_s at
# every eval. If a compute node ever turns out to have no outbound network,
# add --wandb-mode offline and `wandb sync` the run directory afterwards.
uv run scripts/train_sft.py \
    --init-from "$PRETRAIN_CKPT" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --block-size "$BLOCK_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --max-iters "$MAX_ITERS" \
    --lr 5e-5 \
    --min-lr 5e-6 \
    --warmup-iters "$WARMUP_ITERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-iters 20 \
    --device auto \
    --wandb \
    --wandb-project llm-und-sft \
    --wandb-run-name "$(basename "$OUT_DIR")_${SLURM_JOB_ID:-local}" \
    --wandb-tags "${WANDB_TAGS[@]}"

echo "Checkpoint + logs under: $OUT_DIR"
