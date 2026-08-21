#!/bin/bash
#SBATCH --job-name=llm-und-eval-full-val
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/%j-full_validation.out
#SBATCH --error=runs/%j-full_validation.err

set -euo pipefail
source scripts/slurm/_common.sh

CKPT=/proj/assert-berzelius/users/x_amash/llm-und/runs/pretrain_2.2e18_bf16_qknorm/ckpt_final.pt
TOK=/proj/assert-berzelius/users/x_andaf/llm-und/artifacts/tokenizer
OUT=runs/eval/full_validation

echo "=== main tasks (no --limit = full dataset) start $(date -Iseconds) ==="
uv run scripts/run_eval.py \
    --ckpt "$CKPT" --tokenizer-dir "$TOK" --mode pretrain \
    --tasks calame_pt,portugal_basic_qa,alba,chatrag_hi,belebele_spa_Latn,copa_es,escola,openbookqa_es,xstorycloze_es,mgsm_direct_es_spanish_bench,eqbench_es,cocoteros_es,phrases_es \
    --num-fewshot 0 --device auto --out-dir "$OUT/iter0259653_main"
echo "=== main tasks done $(date -Iseconds) ==="

echo "=== pt_culture (--limit 2000) start $(date -Iseconds) ==="
uv run scripts/run_eval.py \
    --ckpt "$CKPT" --tokenizer-dir "$TOK" --mode pretrain \
    --tasks pt_culture --num-fewshot 0 --limit 2000 --device auto --out-dir "$OUT/iter0259653_ptculture"
echo "=== pt_culture done $(date -Iseconds) ==="

echo "=== ALL DONE $(date -Iseconds) ==="
