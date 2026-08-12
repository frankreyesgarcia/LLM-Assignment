# Shared preamble for the scripts in this directory. Source it, don't run it
# directly: `source "$(dirname "$0")/_common.sh"`.
#
# Fill in the two placeholders below before submitting anything:
#   PROJECT_ACCOUNT  -- your Naiss/SUPR compute allocation (sbatch -A)
#   PROJECT_STORAGE  -- persistent project storage (/proj/<your-project>/...),
#                       NOT node-local $TMPDIR. scripts/run_dedup_datatrove.py's
#                       intermediate signatures/buckets/clusters (--work-dir)
#                       and final deduped output (--out-dir) both need to
#                       survive across separate job submissions -- $TMPDIR is
#                       wiped per-job on most Naiss clusters and would
#                       silently break a multi-stage/resumed run.
export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:?set PROJECT_ACCOUNT to your Naiss allocation, e.g. naiss2026-x-y}"
export PROJECT_STORAGE="${PROJECT_STORAGE:?set PROJECT_STORAGE to persistent project storage, e.g. /proj/your-project/llm-und}"

# Triton compiles a small CUDA helper at runtime (torch.compile, reached via
# FlexAttention in src/model/gpt.py) by shelling out to a C compiler. The gcc
# on PATH here is an NSC wrapper that exits 1 with "you have not loaded the
# appropriate build environment module", killing any job that compiles a
# kernel. Triton reads CC before falling back to `which gcc`, so point it at
# the system compiler. Overridable: a cluster with a working module-provided
# gcc can export CC itself.
export CC="${CC:-/usr/bin/gcc}"

# ...and it needs Python.h, which comes from python3.12-devel -- installed on
# the login node but NOT on the compute nodes, so sysconfig's
# /usr/include/python3.12 resolves on the node you test from and is missing on
# the node the job runs on. Point gcc at a copy kept in project storage
# (CPATH is appended to, not replaced: the cluster already sets it).
# Stage it once from a login node:
#   mkdir -p "$PROJECT_STORAGE/pyheaders" && cp -r /usr/include/python3.12 "$PROJECT_STORAGE/pyheaders/"
PY_HEADERS="$PROJECT_STORAGE/pyheaders/python3.12"
if [ -f "$PY_HEADERS/Python.h" ]; then
    export CPATH="$PY_HEADERS${CPATH:+:$CPATH}"
elif [ ! -f /usr/include/python3.12/Python.h ]; then
    echo "WARNING: no Python.h at $PY_HEADERS or /usr/include/python3.12 -- any torch.compile/FlexAttention job will fail." >&2
    echo "         Stage it from a login node: mkdir -p \"\$PROJECT_STORAGE/pyheaders\" && cp -r /usr/include/python3.12 \"\$PROJECT_STORAGE/pyheaders/\"" >&2
fi

# huggingface_hub/datasets default to caching downloads under ~/.cache/huggingface,
# i.e. $HOME -- which on Berzelius has only a 20GB quota (see scripts/slurm/README.md)
# and would fill up almost immediately given the size of these sources (see
# src/sources.py). Redirect every HF cache location into project storage
# instead. HF_HOME alone would cover this (HF_HUB_CACHE/HF_DATASETS_CACHE both
# default to under it), but setting all three explicitly avoids relying on
# that fallback across library versions.
export HF_HOME="$PROJECT_STORAGE/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# huggingface_hub derives its cached-login token path from HF_HOME too, so
# redirecting HF_HOME above (necessarily) also moves where it looks for the
# token written by `hf auth login` -- it won't find one saved under the
# default $HOME/.cache/huggingface. Read the real default location once
# here so jobs authenticate without every script remembering to do this:
# unauthenticated requests were observed both throttled (~8 MB/s vs. ~65
# MB/s aggregate authenticated/parallel, see scripts/download_sources.py)
# and occasionally 403/500 from HF's Xet CDN.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Adjust to whatever this cluster provides -- Naiss clusters differ (Alvis,
# Dardel, Tetralith, ...). uv itself is a single static binary (see
# https://docs.astral.sh/uv/getting-started/installation/); a `module load`
# is usually not needed for uv, only if the cluster requires an explicit
# Python module before uv can find an interpreter to manage.
command -v uv >/dev/null || { echo "uv not found on PATH -- install it first (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

uv sync --frozen
