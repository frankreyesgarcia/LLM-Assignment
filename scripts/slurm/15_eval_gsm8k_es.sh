#!/bin/bash
#SBATCH --job-name=llm-und-eval-gsm8kes
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%j-15_eval_gsm8k_es.out
#SBATCH --error=runs/%j-15_eval_gsm8k_es.err

set -euo pipefail
source scripts/slurm/_common.sh

CKPT=/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt_final.pt
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=runs/eval/pretrain_2.2e18_bf16_qknorm_full/iter0259653_gsm8kes

echo "=== gsm8k_es eval start $(date -Iseconds) ==="
uv run scripts/run_eval.py \
    --ckpt "$CKPT" --tokenizer-dir "$TOK" --mode pretrain \
    --tasks gsm8k_es \
    --num-fewshot 0 --device auto --log-samples \
    --out-dir "$OUT"
echo "=== done $(date -Iseconds) ==="
