#!/bin/bash
# Translate openai/gsm8k's test split to Spanish via a locally-served
# GLM-4.7-Flash (vLLM), filling the one gap in src/eval/tasks/gsm8k.py
# (see the docstring there: gsm8k_hi/gsm8k_pt exist, gsm8k_es didn't
# because no maintained Spanish translation exists on HF Hub).
#
# GLM-4.7-Flash lives under a different project's model store
# (~/llm-local-models/glm-4.7-flash, served elsewhere via vLLM for
# "backtrack") -- this job serves its own throwaway vLLM instance on a
# fresh port instead of touching that project's host/port state, per
# explicit instruction (isolate from backtrack).
#SBATCH --job-name=llm-und-translate-gsm8k-es
#SBATCH --account=berzelius-2026-167
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%j-14_translate_gsm8k_es.out
#SBATCH --error=runs/%j-14_translate_gsm8k_es.err

set -euo pipefail
source scripts/slurm/_common.sh

MODEL_DIR=/proj/assert-berzelius/users/x_frrey/llm-local-models/glm-4.7-flash
PORT=8010
SERVED_NAME=glm-4.7-flash

echo "=== translate gsm8k -> es, start $(date -Iseconds), node $(hostname) ==="

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-agent
module load buildenv-gcccuda/12.4.1-gcc13.3.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export VLLM_CACHE_ROOT=/proj/assert-berzelius/users/x_frrey/vllm-cache
mkdir -p "$VLLM_CACHE_ROOT"

vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser glm45 \
    > runs/${SLURM_JOB_ID}-vllm-server.log 2>&1 &
VLLM_PID=$!

echo "vllm server pid=$VLLM_PID, waiting for readiness..."
ready=0
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        echo "vllm ready after $((i * 10))s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vllm server process died during startup -- see runs/${SLURM_JOB_ID}-vllm-server.log" >&2
        exit 1
    fi
    sleep 10
done
if [ "$ready" -ne 1 ]; then
    echo "vllm server never became ready after 900s" >&2
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

conda deactivate

uv run scripts/translate_gsm8k_es.py \
    --server-url "http://127.0.0.1:${PORT}/v1" \
    --model "$SERVED_NAME" \
    --out data/gsm8k_es/test.jsonl \
    --concurrency 12 \
    ${LIMIT:+--limit "$LIMIT"}

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "=== done $(date -Iseconds) ==="
