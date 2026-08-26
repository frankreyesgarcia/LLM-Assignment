#!/bin/bash
# Task 5 -- run the translated MMLU / GSM8K benchmarks (added on the
# eval-harness branch: mmlu_pt/mmlu_es/mmlu_hi, gsm8k_pt/gsm8k_hi) on the
# SFT checkpoint and the base checkpoint it was fine-tuned from.
#
# Why this is a separate job from 15_eval_sft.sh rather than more tasks in
# it: these five have a much heavier per-example cost. MMLU scores four
# candidate continuations per question instead of one generation, and
# GSM8K generates up to 256 tokens per question against 64 elsewhere.
#
# Why it is also separate from the numbers already on
# docs/eval_curve.html: that page reports these benchmarks for the
# *pretrain* final checkpoint (iteration 259,653) at full dataset size.
# This job runs the base checkpoint SFT actually started from (ckpt.pt,
# iteration 240,204) at --limit 500, so base and SFT are measured
# identically here. Do not difference a number from this job against one
# from that page.
#
# EXPECTED RESULT, stated up front so a floor is not mistaken for a bug:
# near-chance on MMLU (25% for 4-way choice) and near-zero on GSM8K. The
# pretrain sweep already lands there (see the "New benchmarks" section of
# docs/eval_curve.html), and SFT teaches response format, not multi-step
# arithmetic -- a 138M-param model has no reliable chain-of-thought. The
# point of running it on the SFT checkpoint is to confirm instruction
# tuning did not *degrade* these, and to see whether chat formatting lets
# the model at least emit a parseable final answer more often.
#
# Usage: sbatch scripts/slurm/16_eval_new_benchmarks.sh
#SBATCH --job-name=llm-und-eval-newbench
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=runs/%j-16_eval_newbench.out
#SBATCH --error=runs/%j-16_eval_newbench.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

BASE_CKPT=${BASE_CKPT:-/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt.pt}
SFT_CKPT=${SFT_CKPT:-$PROJECT_STORAGE/runs/sft/ckpt.pt}
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=${EVAL_OUT:-$PROJECT_STORAGE/runs/eval/newbench_sft_vs_base}
LIMIT=${EVAL_LIMIT:-500}
TASKS=${EVAL_TASKS:-mmlu_pt,mmlu_es,mmlu_hi,gsm8k_pt,gsm8k_hi}

# --log-samples: at an expected floor, the per-example generations are the
# only way to tell "cannot do the arithmetic" from "never emits a parseable
# final number", and those are different failures with different fixes.
for entry in "base:$BASE_CKPT" "sft:$SFT_CKPT"; do
    name=${entry%%:*}
    ckpt=${entry#*:}
    if [ ! -f "$ckpt" ]; then
        echo "MISSING checkpoint: $ckpt" >&2
        exit 1
    fi
    for mode in pretrain posttrain; do
        echo "=== $name / $mode start $(date -Iseconds) ==="
        uv run scripts/run_eval.py \
            --ckpt "$ckpt" --tokenizer-dir "$TOK" --mode "$mode" \
            --tasks "$TASKS" --num-fewshot 0 --limit "$LIMIT" --device auto \
            --log-samples --out-dir "$OUT/${name}_${mode}"
        echo "=== $name / $mode done $(date -Iseconds) ==="
    done
done

echo "=== ALL DONE $(date -Iseconds) ==="
echo "Results under: $OUT"
