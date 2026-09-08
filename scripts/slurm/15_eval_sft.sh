#!/bin/bash
# Task 5 -- evaluate the SFT checkpoint (scripts/slurm/14_train_sft.sh)
# against the pretrained checkpoint it was fine-tuned from, in both prompt
# formats, via scripts/run_eval.py.
#
# Four runs, because a single SFT-vs-base number in one mode can't
# separate "instruction tuning helped" from "the chat template helped":
#
#   base + pretrain    reference: what the checkpoint could already do
#   base + posttrain   control: the chat template on a model that has
#                      never seen it -- isolates how much of any posttrain
#                      gain is the template rather than the tuning
#   sft  + pretrain    regression check: did SFT damage raw language
#                      modelling (catastrophic forgetting)?
#   sft  + posttrain   the headline result
#
# --ckpt is the base run's ckpt.pt, not ckpt_final.pt: SFT was initialized
# from ckpt.pt (see 14_train_sft.sh), so this compares against exactly the
# weights it started from.
#
# calame_pt runs in pretrain mode only, for both checkpoints. It is
# last-word sentence completion -- doc_to_text is a bare sentence
# fragment, max_gen_toks=6, stop sequences are punctuation. Wrapping a
# fragment in a user turn asks an instruction-tuned model to *reply to*
# it rather than continue it, so a posttrain score would mostly measure
# prompt mismatch. It stays a pure language-modelling probe, which is
# also what makes it the useful forgetting check above.
#
# ALBA needs ANTHROPIC_API_KEY for judge scoring; without it
# scripts/run_eval.py still generates and saves the outputs, just with no
# judge_score (see its --judge help). Export the key before submitting to
# get scores in the same run:
#   ANTHROPIC_API_KEY=... sbatch scripts/slurm/15_eval_sft.sh
#
# --limit 500 keeps four checkpoint x mode combinations inside one
# submission: scripts/slurm/11_eval_full_validation.sh needed 6h for a
# *single* checkpoint in a single mode at full dataset size. Treat this as
# the comparison run that says which differences are worth measuring
# precisely, then re-run the interesting subset without --limit.
#
# Usage: sbatch scripts/slurm/15_eval_sft.sh
#SBATCH --job-name=llm-und-eval-sft
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%j-15_eval_sft.out
#SBATCH --error=runs/%j-15_eval_sft.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

BASE_CKPT=${BASE_CKPT:-/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt.pt}
SFT_CKPT=${SFT_CKPT:-$PROJECT_STORAGE/runs/sft/ckpt.pt}
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=${EVAL_OUT:-$PROJECT_STORAGE/runs/eval/sft_vs_base}
LIMIT=${EVAL_LIMIT:-500}

# Native PT/HI tasks plus the same Spanish subset 11_eval_full_validation.sh
# used. calame_pt is handled separately below (pretrain only).
TASKS=portugal_basic_qa,pt_culture,alba,chatrag_hi,belebele_spa_Latn,copa_es,escola,openbookqa_es,xstorycloze_es
JUDGE_ARGS=()
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    JUDGE_ARGS=(--judge anthropic)
    echo "ALBA: judge enabled"
else
    echo "ALBA: no ANTHROPIC_API_KEY -- generations will be saved unscored"
fi

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
            --out-dir "$OUT/${name}_${mode}" "${JUDGE_ARGS[@]}"
        echo "=== $name / $mode done $(date -Iseconds) ==="
    done

    echo "=== $name / calame_pt (pretrain only) start $(date -Iseconds) ==="
    uv run scripts/run_eval.py \
        --ckpt "$ckpt" --tokenizer-dir "$TOK" --mode pretrain \
        --tasks calame_pt --num-fewshot 0 --limit "$LIMIT" --device auto \
        --out-dir "$OUT/${name}_calame_pretrain"
    echo "=== $name / calame_pt done $(date -Iseconds) ==="
done

echo "=== ALL DONE $(date -Iseconds) ==="
echo "Results under: $OUT"
