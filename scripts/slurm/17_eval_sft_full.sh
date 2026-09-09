#!/bin/bash
# Task 5 -- full-dataset evaluation of the SFT checkpoint.
#
# scripts/slurm/15_eval_sft.sh and 16_eval_new_benchmarks.sh both ran at
# --limit 500, which was enough to compare base vs SFT but leaves every
# number with a sampling error the published base figures do not have
# (+/-2 points on a 4-way multiple-choice task at n=500). This re-runs
# the SFT checkpoint uncapped, so its numbers sit on the same footing as
# the base checkpoint's in scripts/slurm/11_eval_full_validation.sh and
# 13_eval_new_benchmarks_final.sh. The base is not re-run -- it already
# has full-dataset numbers.
#
# Task list and caps are copied from those two base scripts on purpose,
# so every base figure gets an SFT counterpart measured the same way.
# That includes pt_culture staying at --limit 2000: the base was capped
# there (2,000 of 105,380 rows -- true-full was measured as impractical
# even for one checkpoint), and an uncapped SFT number would not be
# comparable to it.
#
# WHICH BASE TO COMPARE AGAINST: ckpt.pt (iteration 240,204), the
# checkpoint SFT was actually initialised from -- not ckpt_final.pt
# (iteration 259,653), which is what 11_eval_full_validation.sh and
# 13_eval_new_benchmarks_final.sh published. The two are ~19k steps apart
# and it is not a small difference: EsCoLA reads 0.3200 at 240,204 against
# 0.6971 at 259,653 (the degenerate-class flip), and Portugal Basic QA
# 0.5600 against 0.6200. Differencing SFT against the wrong one invents a
# delta that is really just those 19k steps.
#
# Full-dataset base numbers at 240,204 already exist for every task in
# the `main` and `ptculture` groups: the 80-checkpoint full sweep
# (12_eval_checkpoint_sweep_full.sh) evaluated all of them at that exact
# iteration, and the values are embedded in docs/eval_curve.html's PANELS
# seriesFull. Nothing needs re-running for those.
#
# The gap the sweep does not cover is the `newbench` group (MMLU x3,
# GSM8K x2, MATH), which only ever ran against ckpt_final.pt. Run the
# base for that group at ckpt.pt with:
#   sbatch -t 08:00:00 --export=ALL,EVAL_MODE=pretrain,EVAL_GROUP=newbench,\
#       EVAL_CKPT=/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt.pt,\
#       EVAL_OUT=$PROJECT_STORAGE/runs/eval/base_ckptpt_full \   # took 1:50:58
#       scripts/slurm/17_eval_sft_full.sh
#
# The sweep is pretrain-mode only, so full-dataset base numbers in chat
# mode do not exist for any task; the report keeps using the n=500
# base-chat figures from 15/16_eval_*.sh for that column and labels the
# mixed sample sizes.
#
# Split into three groups x two prompt modes = 6 independent jobs rather
# than one long one, so a walltime kill loses one group instead of
# everything, and so the short groups are not stuck behind MMLU:
#
#   main       13 benchmarks, ~16,600 examples   (-t 02:00:00)
#   ptculture  pt_culture at the base's 2,000    (-t 01:00:00)
#   newbench   MMLU x3 (42,126) + GSM8K + MATH   (-t 03:00:00)
#
# Walltimes are measured, not extrapolated -- from the first full run
# (jobs 17490932-44), with roughly 60% headroom:
#
#   main       1:12:19 raw / 0:34:49 chat
#   ptculture  0:11:33 raw / 0:16:07 chat
#   newbench   1:11:13 raw / 0:50:22 chat, and 1:50:58 for the base
#              checkpoint, which is slower because it does not stop
#              generating the way the SFT checkpoint learned to
#
# The first submission asked for 6h/3h/8h, i.e. 3-8x what the jobs
# actually needed. That is not free: an oversized request is harder to
# backfill, and this project has already lost hours to exactly that
# (see the --cpus-per-task note in 13_prepare_sft_data.sh).
#
# Both modes are run: posttrain is how an SFT checkpoint is meant to be
# prompted, pretrain matches the protocol every published base number
# used and doubles as the catastrophic-forgetting check.
#
# Usage -- submit all six:
#   for m in pretrain posttrain; do
#     sbatch -t 02:00:00 --export=ALL,EVAL_MODE=$m,EVAL_GROUP=main      scripts/slurm/17_eval_sft_full.sh
#     sbatch -t 01:00:00 --export=ALL,EVAL_MODE=$m,EVAL_GROUP=ptculture scripts/slurm/17_eval_sft_full.sh
#     sbatch -t 03:00:00 --export=ALL,EVAL_MODE=$m,EVAL_GROUP=newbench  scripts/slurm/17_eval_sft_full.sh
#   done
#
#SBATCH --job-name=llm-und-eval-sft-full
#SBATCH --account=CHANGE_ME          # -A <PROJECT_ACCOUNT>, see _common.sh
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00              # overridden per group by the -t above
#SBATCH --output=runs/%j-17_eval_sft_full.out
#SBATCH --error=runs/%j-17_eval_sft_full.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

MODE=${EVAL_MODE:?set EVAL_MODE to pretrain or posttrain}
GROUP=${EVAL_GROUP:?set EVAL_GROUP to main, ptculture or newbench}
# EVAL_CKPT lets this same script evaluate the base checkpoint for the
# one group the sweep does not cover -- see the header.
SFT_CKPT=${EVAL_CKPT:-${SFT_CKPT:-$PROJECT_STORAGE/runs/sft/ckpt.pt}}
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=${EVAL_OUT:-$PROJECT_STORAGE/runs/eval/sft_full}

if [ ! -f "$SFT_CKPT" ]; then
    echo "MISSING checkpoint: $SFT_CKPT" >&2
    exit 1
fi

# No --limit anywhere except pt_culture, which keeps the base's cap.
LIMIT_ARGS=()
case "$GROUP" in
    main)
        TASKS=calame_pt,portugal_basic_qa,alba,chatrag_hi,belebele_spa_Latn,copa_es,escola,openbookqa_es,xstorycloze_es,mgsm_direct_es_spanish_bench,eqbench_es,cocoteros_es,phrases_es
        ;;
    ptculture)
        TASKS=pt_culture
        LIMIT_ARGS=(--limit 2000)   # matches the base run; see header
        ;;
    newbench)
        TASKS=mmlu_pt,mmlu_es,mmlu_hi,gsm8k_hi,gsm8k_pt,math_en
        ;;
    *)
        echo "unknown EVAL_GROUP: $GROUP (expected main, ptculture or newbench)" >&2
        exit 1
        ;;
esac

# ALBA needs ANTHROPIC_API_KEY to be scored; without it run_eval.py still
# generates and saves the outputs, just with no judge_score. Export the
# key before submitting to get a scored ALBA in the same run.
JUDGE_ARGS=()
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    JUDGE_ARGS=(--judge anthropic)
    echo "ALBA: judge enabled"
else
    echo "ALBA: no ANTHROPIC_API_KEY -- generations saved unscored"
fi

# --log-samples throughout: on the --limit 500 runs the per-example
# generations were what distinguished "answers wrongly" from "emits
# nothing at all" (the base checkpoint turned out to be mute on Hindi
# chat prompts), and the aggregate score cannot show that.
echo "=== SFT full eval: group=$GROUP mode=$MODE start $(date -Iseconds) ==="
echo "    ckpt=$SFT_CKPT"
echo "    tasks=$TASKS"
uv run scripts/run_eval.py \
    --ckpt "$SFT_CKPT" --tokenizer-dir "$TOK" --mode "$MODE" \
    --tasks "$TASKS" --num-fewshot 0 --device auto --log-samples \
    "${LIMIT_ARGS[@]}" "${JUDGE_ARGS[@]}" \
    --out-dir "$OUT/${GROUP}_${MODE}"
echo "=== SFT full eval: group=$GROUP mode=$MODE done $(date -Iseconds) ==="
echo "Results under: $OUT/${GROUP}_${MODE}"
