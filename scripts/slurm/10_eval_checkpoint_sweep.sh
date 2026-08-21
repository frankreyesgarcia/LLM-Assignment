#!/bin/bash
# Task 4 -- evaluate a spaced-out subset of a pretrain run's periodic
# checkpoints (scripts/slurm/07_pretrain.sh's --checkpoint-interval
# output, ckpt_iter*.pt), one scripts/run_eval.py call per checkpoint, so
# scripts/aggregate_eval_results.py can build a benchmark-score-vs-
# training-iteration curve afterward instead of only ever seeing the
# final checkpoint's numbers.
#
# EVERY=5 (default) means "every 5th checkpoint by save order", not "every
# 5th iteration" -- spacing follows --checkpoint-interval, whatever it was
# for this run. ckpt_final.pt is always included on top of that subset,
# since it's the one number everyone actually cares about.
#
# Resumable: skips any iter<N> that already has a results.json, so a
# killed/requeued job picks up where it left off instead of re-running
# checkpoints already done.
#
# Usage:
#   CKPT_DIR=/proj/.../runs/pretrain_2.2e18_bf16_qknorm \
#   TOKENIZER_DIR=/proj/.../artifacts/tokenizer \
#   sbatch scripts/slurm/10_eval_checkpoint_sweep.sh
#
# Optional overrides (defaults shown): EVERY=5 TASKS=calame_pt,portugal_basic_qa,spanish_bench
# LIMIT=50 MODE=pretrain OUT_DIR="$PROJECT_STORAGE/runs/eval/<CKPT_DIR's basename>"
#SBATCH --job-name=llm-und-eval-sweep
#SBATCH --account=berzelius-2026-167
# GPU partition, capped under 1h per submission (MAX_CKPTS below) -- NSC
# auto-kills DGX jobs whose GPU stays under ~90W for sustained periods
# (job 17280247 got killed this way at 83.87W), but explicitly exempts
# "jobs that have not yet run for one hour". berzelius-cpu was tried as
# an alternative (job 17282415) and made 3 checkpoints in 10h -- that
# partition is heavily shared (node load average >100 observed) and
# ends up far slower than GPU despite the extra --cpus-per-task, so GPU
# + short submissions is the actual fix, not a fallback.
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=runs/%j-10_eval_checkpoint_sweep.out
#SBATCH --error=runs/%j-10_eval_checkpoint_sweep.err

set -euo pipefail
# sbatch copies this script into a spool dir before running it, so
# $(dirname "${BASH_SOURCE[0]}") no longer points at scripts/slurm/ --
# use SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) instead.
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the checkpoint directory, e.g. /proj/.../x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm}"
TOKENIZER_DIR="${TOKENIZER_DIR:?set TOKENIZER_DIR, e.g. /proj/.../x_andaf/llm-und/artifacts/tokenizer}"
OUT_DIR="${OUT_DIR:-$PROJECT_STORAGE/runs/eval/$(basename "$CKPT_DIR")}"
EVERY="${EVERY:-5}"
TASKS="${TASKS:-calame_pt,portugal_basic_qa,spanish_bench}"
LIMIT="${LIMIT:-50}"
MODE="${MODE:-pretrain}"
# Caps how many NOT-yet-evaluated checkpoints this single invocation
# processes, so one sbatch submission finishes comfortably inside the
# --time=00:55:00 above (measured ~13min/checkpoint on GPU -- 4 fits with
# margin). Unset/0 means no cap (process everything selected by EVERY).
# Driving several capped invocations back-to-back (see the sbatch --wait
# loop in the submission usage) covers all checkpoints without ever
# running long enough on a GPU node to trip NSC's low-power auto-kill.
MAX_CKPTS="${MAX_CKPTS:-0}"

# ckpt_iter*.pt is zero-padded to a fixed width (see scripts/train.py), so
# plain lexicographic sort == numeric sort by iteration.
mapfile -t CKPTS < <(ls "$CKPT_DIR"/ckpt_iter*.pt 2>/dev/null | sort | awk -v n="$EVERY" 'NR % n == 0')
if [ -f "$CKPT_DIR/ckpt_final.pt" ]; then
    CKPTS+=("$CKPT_DIR/ckpt_final.pt")
fi

echo "Evaluating ${#CKPTS[@]} checkpoints (every ${EVERY}th + final) into $OUT_DIR"
echo "tasks=$TASKS limit=$LIMIT mode=$MODE"

done_this_run=0
for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt" .pt)              # ckpt_iter0032460 or ckpt_final
    # ckpt_final.pt has no iteration number in its filename -- read the real
    # one from the checkpoint's own iter_num field instead of labeling its
    # directory "iterfinal" (not sortable/plottable alongside the numbered ones).
    if [[ "$name" == "ckpt_final" ]] || [[ "$name" == "ckpt_last" ]]; then
        iter=$(uv run python3 -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iter_num'])" "$ckpt")
    else
        iter=$(echo "$name" | grep -oE '[0-9]+')
    fi
    dest="$OUT_DIR/iter${iter}"
    if [ -f "$dest/results.json" ]; then
        echo "=== iter${iter}: already done, skipping ==="
        continue
    fi
    if [ "$MAX_CKPTS" != "0" ] && [ "$done_this_run" -ge "$MAX_CKPTS" ]; then
        echo "MAX_CKPTS=$MAX_CKPTS reached for this submission, stopping here (resume with another sbatch call)"
        break
    fi
    echo "=== iter${iter} ($ckpt) ==="
    uv run scripts/run_eval.py \
        --ckpt "$ckpt" \
        --tokenizer-dir "$TOKENIZER_DIR" \
        --mode "$MODE" \
        --tasks "$TASKS" \
        --num-fewshot 0 \
        --limit "$LIMIT" \
        --device auto \
        --out-dir "$dest"
    done_this_run=$((done_this_run + 1))
done

uv run scripts/aggregate_eval_results.py "$OUT_DIR"
echo "Curve CSV: $OUT_DIR/curve.csv"
