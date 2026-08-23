#!/bin/bash
# Task 4 -- run the newer benchmarks (mmlu_pt/es/hi, gsm8k_hi/pt, math_en)
# against just the final checkpoint, full dataset, no cap -- unlike the
# earlier 80-checkpoint sweep, this one was deliberately scoped to a
# single checkpoint (see chat: MMLU alone is 14,042 x 3 languages =
# 42,126 examples/checkpoint, ~80x more than any other single benchmark
# here -- fine for one checkpoint, not for 80 of them).
#SBATCH --job-name=llm-und-eval-newbench
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%j-13_eval_new_benchmarks_final.out
#SBATCH --error=runs/%j-13_eval_new_benchmarks_final.err

set -euo pipefail
source scripts/slurm/_common.sh

CKPT=/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt_final.pt
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=runs/eval/pretrain_2.2e18_bf16_qknorm_full/iter0259653_newbench

echo "=== new benchmarks (full dataset) start $(date -Iseconds) ==="
uv run scripts/run_eval.py \
    --ckpt "$CKPT" --tokenizer-dir "$TOK" --mode pretrain \
    --tasks mmlu_pt,mmlu_es,mmlu_hi,gsm8k_hi,gsm8k_pt,math_en \
    --num-fewshot 0 --device auto --log-samples \
    --out-dir "$OUT"
echo "=== done $(date -Iseconds) ==="
