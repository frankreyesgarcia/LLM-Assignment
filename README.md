# LLM-UND — pre-training data pipeline + tokenizer

Task 1: dedup pipeline for PT/ES/HI pre-training data into a single HF
dataset. Scope is deliberately narrow -- 4 sources only:

- `pt-fineweb2`, `es-fineweb2`, `hi-fineweb2` (HuggingFaceFW/fineweb-2)
- `hi-sangraha` (`ai4bharat/sangraha`'s "verified" split -- human-verified
  sites + OCR'd PDFs + transcribed speech, the one hi source that isn't
  itself another Common-Crawl derivative)

No language/quality filtering or cleaning is applied -- sources are used
as-is; cross-source MinHash dedup is the only processing step (see
below for why hi-fineweb2/hi-sangraha specifically needed checking: they
measured ~2% raw overlap by hand before this pipeline existed).

Task 2: a shared multilingual byte-level BPE tokenizer trained on that
corpus (see "Task 2 — Tokenizer" below).

## Status

- `src/sources.py` — the source-of-truth for the 4 supported rows:
  `VALID_LANGUAGES`, `SOURCE_SPECS` (repo_id + glob pattern + text/id
  column names, confirmed against each repo's real file layout), and
  `ROWS_BY_LANGUAGE` (pt/es get one row each; hi gets two, since
  hi-fineweb2 and hi-sangraha need to dedup *against each other*).
- `scripts/download_sources.py` — bulk-downloads the 4 sources' raw files
  to local disk with many concurrent connections
  (`huggingface_hub.snapshot_download`). A single `datasets(streaming=True)`
  connection measured ~8 MB/s regardless of link speed and occasionally
  403/500'd from HF's Xet CDN under sustained load; many concurrent
  connections measured ~61-66 MB/s aggregate on the same repos (~7-8x).
  Resumable for free (`snapshot_download` skips files already present and
  up to date).
- `scripts/run_dedup_datatrove.py` — the main pipeline: cross-source
  MinHash dedup using [`datatrove`](https://github.com/huggingface/datatrove)
  (HuggingFace's own pipeline library -- literally what built fineweb-2
  itself), not a from-scratch implementation. Runs the standard 4-stage
  flow (`MinhashDedupSignature` -> `MinhashDedupBuckets` ->
  `MinhashDedupCluster` -> `MinhashDedupFilter`) once per language, writing
  deduped output to `data/processed/{lang}/*.parquet`. For hi, the
  hi-fineweb2 and hi-sangraha readers are chained back-to-back in the same
  pipeline (datatrove's `BaseDiskReader.run()` does `if data: yield from
  data` before yielding its own docs, so two readers concatenate into one
  shared stream) -- that's what makes the two sources dedup against each
  other, not just internally.

  Uses a custom `WhitespaceWordTokenizer` for MinHash shingling instead of
  datatrove's language-routed tokenizers: pt/es route to `SpaCyTokenizer`
  (needs `spacy` + a downloaded model) and hi routes to
  `StanzaTokenizer("kmr")` (needs `stanza`, which pulls in a full PyTorch +
  CUDA stack just to split words) -- confirmed by hitting both import
  errors directly. A plain `str.split()` tokenizer (same approach the
  removed from-scratch dedup always used) avoids both, with zero
  correctness cost for shingling purposes.

  `--executor {local,slurm}`: `local` uses `LocalPipelineExecutor`
  (single-node, for smoke-testing with `--limit`); `slurm` uses
  `SlurmPipelineExecutor`, which shards each stage across a real SLURM job
  array (`--tasks`) -- this is the actual fix for the parallelism ceiling
  the old pipeline had (measured ~30 docs/sec for fineweb2 and ~4.25
  docs/sec for hi-sangraha under the old single-worker-per-source design,
  putting a full run at ~46+ hours; `SlurmPipelineExecutor` scales
  horizontally across nodes instead of being capped at one core per
  source).
- `configs/dedup.yaml` — `MinhashConfig` fields (`n_grams`, `num_buckets`,
  `hashes_per_bucket`, `seed`), starting from datatrove's own documented
  defaults.
- `src/aggregate.py` — `shuffle_into_shards`: two-pass streaming shuffle
  (stream input -> randomly partition into K shards sized off on-disk
  bytes -> locally shuffle each bounded-size shard) so Etapa 6 never loads
  a whole language into memory and writes ~500MB-1GB output shards instead
  of one giant file per config. Unchanged by the datatrove rewrite --
  fully schema-agnostic, just reads whatever's in `data/processed/{lang}/`.
- `scripts/build_final_dataset.py` — Etapa 6: aggregates the per-language
  parts into the `pt`/`es`/`hi`/`all` config shape via `shuffle_into_shards`.
- `scripts/slurm/` — SLURM scripts for a Naiss/SUPR run (download, dedup,
  Etapa 6 aggregation, HF upload, plus the fineweb2/sangraha-scoped
  tokenizer sweep/train jobs).

**Deliberately out of scope, not "not yet implemented":** the other 15
sources this pipeline used to support (EuroWeb-2512, CulturaX, HPLT,
corpus-carolina, Portuguese-PD, corpus-ptbr-v2, fineweb2-bagaco2,
Spanish-PD-Books/Newspapers) and all language/quality filtering -- both
were removed, not left unfinished, when scope was narrowed to the 4
sources above.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv,
then sync the environment (this resolves and installs everything from
`uv.lock`, including the `pytest` dev group):

```bash
uv sync
```

## Usage

```bash
# 1. Bulk-download the 4 sources' raw files -> data/raw/. A single
# `datasets(streaming=True)` connection is a single throttled connection
# per source -- measured ~8 MB/s regardless of link speed, and occasionally
# 403/500s from HF's Xet CDN under sustained load. snapshot_download
# instead measured ~61-66 MB/s aggregate on the same repos (~7-8x).
uv run scripts/download_sources.py

# 2. Smoke-test the dedup pipeline locally on a tiny sample first --
# LocalPipelineExecutor, no SLURM needed.
uv run scripts/run_dedup_datatrove.py --executor local --limit 500 --tasks 2

# 3. Real run, sharded across a SLURM job array (datatrove submits its own
# sbatch jobs -- see scripts/slurm/01_dedup_datatrove.sh for a driver-job
# wrapper). --tasks should scale with source size (es-fineweb2 alone is
# ~441M docs).
uv run scripts/run_dedup_datatrove.py --executor slurm --tasks 200 \
    --account <your-naiss-account> --partition berzelius-cpu --time 24:00:00

# Etapa 6: aggregate into the pt/es/hi/all config shape -> data/final/
uv run scripts/build_final_dataset.py

# tests
uv run pytest tests/ -v
```

## Task 2 — Tokenizer

Byte-level BPE (GPT-2/Llama-3 style), trained with Hugging Face
`tokenizers` on the `all` config of `frank-rg/LLM-Assignment` — one
shared multilingual vocab, not per-language, since it feeds one
multilingual model. Byte-level pre-tokenization means there's never an
OOV/`<unk>` fallback needed, which matters since the corpus mixes
Devanagari (hi) and Latin (pt/es) scripts. Special tokens follow the
Llama-3 chat convention (`<|begin_of_text|>`, `<|end_of_text|>`,
`<|pad|>`, `<|start_header_id|>`/`<|end_header_id|>`/`<|eot_id|>`) so the
base tokenizer is chat-fine-tuning-ready without resizing the embedding
table later — see `src/tokenizer/train.py`.

Vocab size isn't picked by vibes: `scripts/sweep_vocab_size.py` trains
several candidate sizes on a fixed-size, per-language-stratified sample
and evaluates each on a disjoint held-out sample; `scripts/generate_report.py`
then picks the vocab size with the **Kneedle algorithm** (point of
maximum curvature on the compression-vs-vocab-size curve; `kneed`
package) rather than an asserted "gain < X%" threshold, and reports
embedding-param cost as context against real Llama-3.2 1B/3B precedent
rather than an asserted % ceiling — both of those were arbitrary magic
numbers in an earlier version of this pipeline, replaced after review.
See `src/tokenizer/report.py::pick_chosen_vocab_size` for the exact
rule.

```bash
# 1. Sweep candidate vocab sizes -> artifacts/tokenizer_sweep/results.csv
uv run scripts/sweep_vocab_size.py

# 2. Train + save the final tokenizer at the chosen size
uv run scripts/train_tokenizer.py --vocab-size 32000 --out-dir artifacts/tokenizer

# 3. Compare against GPT-2 / EuroLLM / Sarvam-1 / poolside Laguna-M.1
# (trains a fresh same-vocab-size tokenizer on the disjoint train bucket
# for a fair "ours" row -- the shipped artifact is trained on the full
# corpus, so evaluating *it* on this held-out set would be in-sample)
uv run scripts/compare_baselines.py --vocab-size 32000 --tokenizer-dir artifacts/tokenizer

# 4. Assemble a report from the two CSVs above: methodology, numbers, and
# polished charts (compression elbow via Kneedle, marginal-gain-per-doubling,
# cost/benefit scatter against real Llama precedent, baseline comparison)
# -> artifacts/tokenizer_sweep/report.md. Auto-picks vocab size unless
# --chosen-vocab-size is passed.
uv run scripts/generate_report.py

# tests (offline, no network)
uv run pytest tests/test_tokenizer.py tests/test_report.py -v
```

**Pilot-scale result** (2,494-doc corpus, ~16.8 MB): Kneedle's elbow on
the compression curve lands at **vocab_size=32,000** (embedding-param
share ~19.8% of a GPT-2-small-sized target model — between the real
Llama-3.2-3B and 1B tied-embedding shares, ~12% and ~21% respectively,
so within precedent for a small model, not an outlier). An earlier
version of this pipeline used a hand-picked "~7% marginal gain"
threshold and landed on 16,000 instead; that number turned out to be an
unprincipled magic number sensitive to exactly where a candidate's gain
fell relative to the cutoff, so it was replaced with Kneedle. Hindi's
fertility barely moved across the *entire* sweep (3.45 at 8k vocab to
3.31 at 100k) while pt/es kept improving -- suggesting hi is
data-starved rather than vocab-starved at this pilot scale (only ~700
hi docs), a Task-1 corpus-collection problem, not something a bigger
tokenizer vocab fixes. Note: this vocab-size choice should be re-run
once the real (non-pilot) corpus is available -- this is a pilot-data
artifact, not a final recommendation.

Baseline comparison (32k tokenizer trained fresh on the sweep's train
bucket, evaluated zero-shot alongside existing tokenizers on the
disjoint held-out bucket): our tokenizer beats **every** baseline,
including EuroLLM-1.7B, on pt (4.34 vs. 4.05 bytes/token) and es (4.41
vs. 4.25), and beats GPT-2 badly on hi (3.89 vs. 1.70 tokens/word)
despite training on ~1000x less data than the multilingual baselines.
It still trails Sarvam-1 and EuroLLM on hi -- consistent with the
"hi is data-starved" finding above. Full numbers in
`artifacts/tokenizer_sweep/baseline_comparison.csv`.

## Known findings (2026-07-17)

- Measured hi-fineweb2/hi-sangraha overlap by hand (custom exact + MinHash
  dedup, before this pipeline existed) at 500k docs/source: **~2%** of
  hi-sangraha's docs were exact or near duplicates of hi-fineweb2 --
  mostly exact matches (8,923 of 10,013 total), a modest but real overlap,
  not the two sources being mostly redundant.
- The old from-scratch pipeline (`src/filters/`, `src/dedup/`,
  `scripts/run_all_sources.py`, all since removed) processed one source
  with exactly one single-threaded worker regardless of available
  cores/nodes -- measured ~30 docs/sec for fineweb2 and ~4.25 docs/sec for
  hi-sangraha's OCR'd PDF text (slower per-doc processing cost), putting a
  full run at ~46+ hours even with filtering skipped entirely. This is why
  the pipeline was rebuilt around `datatrove`'s `SlurmPipelineExecutor`
  instead of patching the old design -- see "Status" above.
- datatrove's per-source-different-schema trap: `ParquetReader`'s default
  `read_metadata=True` captures every extra column into a per-row
  `metadata` struct; hi-fineweb2 and hi-sangraha have different extra
  columns (fineweb-2: `dump`/`url`/`date`/...; sangraha: `type`), so
  writing their combined output to one file crashed with a pyarrow schema
  mismatch. Fixed at the row-count level, not globally: `build_readers`
  only disables `read_metadata` for languages with more than one source
  row (currently just hi) -- pt/es are single-source and keep their real
  provenance columns in the final output.
- datatrove's language-routed word tokenizers (used for MinHash shingling)
  both pull in unexpectedly heavy dependencies: hi ("hin") routes to
  `StanzaTokenizer("kmr")`, needing `stanza` and a full PyTorch/CUDA stack;
  pt/es route to `SpaCyTokenizer`, needing `spacy` + a downloaded model.
  Even datatrove's own `WhitespaceTokenizer` "fallback" secretly delegates
  to spaCy's multilingual "xx" model. Replaced with a minimal custom
  `WordTokenizer` subclass doing plain `str.split()` -- the same approach
  the removed from-scratch dedup always used, with no extra dependencies.
