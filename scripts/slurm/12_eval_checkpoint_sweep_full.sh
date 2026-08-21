#!/bin/bash
# Task 4 -- like 10_eval_checkpoint_sweep.sh, but at FULL dataset size
# (no --limit) instead of --limit 50, with --log-samples so every
# individual prediction (context/prediction/reference) is captured once,
# and sharded across SHARD_COUNT parallel invocations instead of one
# sequential chain -- at ~2h/checkpoint measured on a full-dataset run
# (job 17293026), all 80 checkpoints sequentially would be ~6.5 days;
# SHARD_COUNT=4 running concurrently cuts that to ~1.5-2 days without
# hogging the whole account's fair-share (see README/chat: cluster has
# ~300 jobs queued from ~60 users at any given time).
#
# pt_culture is capped at --limit 2000 (of its 105,380 rows) regardless
# of this script's "full dataset" framing -- true-full there was measured
# as impractical even for a single checkpoint. Every other task runs
# genuinely uncapped. Both calls' results.json get merged into one per
# checkpoint (run_eval.py itself only ever writes one results.json per
# --out-dir, so pt_culture needs a separate --out-dir + a merge step, same
# as scripts/slurm/11_eval_full_validation.sh's approach for the single
# validation checkpoint this script generalizes to the full 80).
#
# Usage (run once per shard, SHARD_INDEX 0..SHARD_COUNT-1, e.g. via a
# `for i in 0 1 2 3; do SHARD_INDEX=$i SHARD_COUNT=4 sbatch ...; done` loop
# -- see the driver this is launched from):
#   CKPT_DIR=... TOKENIZER_DIR=... SHARD_INDEX=0 SHARD_COUNT=4 \
#   sbatch scripts/slurm/12_eval_checkpoint_sweep_full.sh
#SBATCH --job-name=llm-und-eval-sweep-full
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=runs/%j-12_eval_checkpoint_sweep_full.out
#SBATCH --error=runs/%j-12_eval_checkpoint_sweep_full.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/scripts/slurm/_common.sh"

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the checkpoint directory}"
TOKENIZER_DIR="${TOKENIZER_DIR:?set TOKENIZER_DIR}"
OUT_DIR="${OUT_DIR:-$PROJECT_STORAGE/runs/eval/$(basename "$CKPT_DIR")_full}"
MODE="${MODE:-pretrain}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX (0-based)}"
SHARD_COUNT="${SHARD_COUNT:?set SHARD_COUNT}"
MAIN_TASKS="${MAIN_TASKS:-calame_pt,portugal_basic_qa,alba,chatrag_hi,belebele_spa_Latn,copa_es,escola,openbookqa_es,xstorycloze_es,mgsm_direct_es_spanish_bench,eqbench_es,cocoteros_es,phrases_es}"
PT_CULTURE_LIMIT="${PT_CULTURE_LIMIT:-2000}"
MAX_CKPTS="${MAX_CKPTS:-1}"

mapfile -t ALL_CKPTS < <(ls "$CKPT_DIR"/ckpt_iter*.pt 2>/dev/null | sort)
if [ -f "$CKPT_DIR/ckpt_final.pt" ]; then
    ALL_CKPTS+=("$CKPT_DIR/ckpt_final.pt")
fi

CKPTS=()
for i in "${!ALL_CKPTS[@]}"; do
    if [ $((i % SHARD_COUNT)) -eq "$SHARD_INDEX" ]; then
        CKPTS+=("${ALL_CKPTS[$i]}")
    fi
done

echo "Shard $SHARD_INDEX/$SHARD_COUNT: ${#CKPTS[@]} of ${#ALL_CKPTS[@]} checkpoints into $OUT_DIR"

done_this_run=0
for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt" .pt)
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
    if [ "$done_this_run" -ge "$MAX_CKPTS" ]; then
        echo "MAX_CKPTS=$MAX_CKPTS reached for this submission, stopping here"
        break
    fi

    echo "=== iter${iter} main tasks (full dataset) start $(date -Iseconds) ==="
    uv run scripts/run_eval.py \
        --ckpt "$ckpt" --tokenizer-dir "$TOKENIZER_DIR" --mode "$MODE" \
        --tasks "$MAIN_TASKS" --num-fewshot 0 --device auto --log-samples \
        --out-dir "$dest"

    echo "=== iter${iter} pt_culture (--limit $PT_CULTURE_LIMIT) start $(date -Iseconds) ==="
    uv run scripts/run_eval.py \
        --ckpt "$ckpt" --tokenizer-dir "$TOKENIZER_DIR" --mode "$MODE" \
        --tasks pt_culture --num-fewshot 0 --limit "$PT_CULTURE_LIMIT" --device auto --log-samples \
        --out-dir "$dest/_ptculture"

    uv run python3 -c "
import json
from pathlib import Path
dest = Path('$dest')
main = json.loads((dest / 'results.json').read_text())
pt = json.loads((dest / '_ptculture' / 'results.json').read_text())
main['results'].update(pt['results'])
(dest / 'results.json').write_text(json.dumps(main, indent=2, ensure_ascii=False))
"
    mv "$dest/_ptculture/pt_culture.samples.jsonl" "$dest/pt_culture.samples.jsonl"
    rm -rf "$dest/_ptculture"
    echo "=== iter${iter} done $(date -Iseconds) ==="
    done_this_run=$((done_this_run + 1))
done

echo "Shard $SHARD_INDEX/$SHARD_COUNT: submission done ($done_this_run checkpoints this run)"
