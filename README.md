# LLM-UND — pre-training data pipeline + tokenizer

Task 1: pipeline for filtering, cleaning, and deduplicating PT/ES/HI
pre-training data into a single HF dataset (`frank-rg/LLM-Assignment`).
See `TASK1-PLAN.md` for the full design.

Task 2: a shared multilingual byte-level BPE tokenizer trained on that
dataset (see "Task 2 — Tokenizer" below).

## Status

Implemented so far (Fase 0 through most of Fase 5 of the plan):

- `src/ingest/` — unified schema (`Document`), `SourceAdapter` interface,
  `GenericTextAdapter` (covers 7 of the 12 sources as-is), `HPLTAdapter`
  (HPLT's `lang`/`prob` columns are lists, not scalars), `CarolinaAdapter`
  (bespoke loader for corpus-carolina's gzipped TEI-XML shards -- its HF
  loading script is no longer supported, so this reads the repo's raw
  `corpus/{taxonomy}/**/*.xml.gz` files directly via streaming download +
  incremental XML parse), and `registry.py` — a factory (`build_adapter(row)`)
  mapping all 10 currently ingestable sources to a configured adapter, plus
  a hard guard that fails fast if EuroWeb Hindi is ever requested with a
  split other than `high`.
- `src/filters/` — language hard/soft filter, text cleaning, quality heuristics.
- `src/dedup/` — exact dedup (SHA256, in-memory) and near dedup (MinHash +
  LSH), two near-dedup backends: `NearDeduper` (in-memory, for
  `run_pilot.py`/tests) and `SqliteNearDeduper` (LSH buckets + signatures
  in a SQLite file instead of a Python dict, so the index doesn't have to
  fit in RAM -- what `run_all_sources.py` uses).
- `src/aggregate.py` — `shuffle_into_shards`: two-pass streaming shuffle
  (stream input -> randomly partition into K shards sized off on-disk
  bytes -> locally shuffle each bounded-size shard) so Etapa 6 never loads
  a whole language into memory and writes ~500MB-1GB output shards instead
  of one giant file per config.
- `scripts/inspect_sources.py` — queries HF `datasets-server` for size/schema/license
  on all 12 sources from the assignment; writes `configs/sources.yaml`.
  (3 of the largest sources' exact sizes time out server-side on HF's end --
  `configs/sources.yaml` falls back to `estimated_via_hub_listing` for
  those, summing file sizes from the Hub repo listing directly.)
- `scripts/run_pilot.py` — full Etapas 1-5 pipeline on one small source (`corpus-ptbr-v2`).
- `scripts/run_all_sources.py` — same pipeline across all 10 available
  sources, with **per-language** exact + near dedup (as required by the
  plan for cross-source overlap). Streams kept docs to Parquet part files
  in batches (`data/processed/{pt,es,hi}/<row>__part####.parquet`) instead
  of holding a whole run in memory, and is resumable: `checkpoint.json`
  tracks which sources finished, so a crash/timeout only costs re-streaming
  the one source that was in flight, not the whole run.
- `scripts/build_final_dataset.py` — Etapa 6: aggregates the per-language
  parts into the `pt`/`es`/`hi`/`all` config shape via `shuffle_into_shards`.
- `scripts/slurm/` — SLURM scripts for a Naiss/SUPR run (pilot, full
  ingest+filter+dedup, Etapa 6 aggregation, HF upload); see its README for
  how this maps onto (and deviates from) the plan's original 4-stage sketch.

**Blocked** (see `src/ingest/registry.py::BLOCKED_SOURCES`):
- `CulturaX` — gated dataset, needs an HF token with accepted terms. This
  is the only remaining blocked source; `corpus-carolina` was unblocked
  via `CarolinaAdapter` (see above).

Not yet implemented: `compute_stats.py` (Etapa 7's schema/language/dedup
validation checklist -- `funnel_stats.json` from `run_all_sources.py`
covers part of this already, but not the full checklist), and the actual
HF Hub upload of a real (non-pilot-scale) corpus.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv,
then sync the environment (this resolves and installs everything from
`uv.lock`, including the `pytest` dev group):

```bash
uv sync
```

## Usage

```bash
# Fase 0: inspect all 12 sources (size, schema, license) -> configs/sources.yaml
uv run scripts/inspect_sources.py

# smoke-test a single source without touching configs/sources.yaml
uv run scripts/inspect_sources.py --only hi-euroweb

# Fase 1: run the pilot pipeline (streams up to --limit docs)
uv run scripts/run_pilot.py --limit 2000

# Fase 2: run all 10 available sources -> data/processed/{pt,es,hi}/*.parquet
# (resumable via data/processed/checkpoint.json). --limit-per-source and
# --full are mutually exclusive and one is required -- no silent default,
# since a real full run is a multi-TB pull (see configs/sources.yaml).
uv run scripts/run_all_sources.py --limit-per-source 500
# uv run scripts/run_all_sources.py --full   # real corpus -- see scripts/slurm/ for Naiss/SUPR

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

## Known findings from the pilot (corpus-ptbr-v2, 2026-07-08)

These already fed back into `configs/filters.yaml` / `src/filters/quality.py`:

- A naive "any non-word character" symbol-ratio rule drops ~97% of normal
  prose (punctuation alone exceeds a 0.1 ratio). Narrowed to a curated set
  of code/template noise characters.
- `*` and `_` (markdown emphasis, snake_case identifiers) are not noise —
  excluding them took the pilot's false-positive rate from ~52% to ~5%.
- The plan's suggested `max_ngram_repetitions: 3` (4-grams) drops legitimate
  long-form articles that repeat a topical phrase (e.g. "um poste de
  energia" 7x in one article). Raised to 10.
- `corpus-carolina`'s HF loading script is no longer supported by
  `datasets`/`datasets-server` (`trust_remote_code` was removed for it
  entirely). Its real structure turned out to be gzip-compressed TEI XML
  (`corpus/{taxonomy}/**/*.xml.gz`, one `<TEI>` element per document) --
  confirmed by inspecting the repo's raw file list and one shard, then
  written as `CarolinaAdapter` (streaming download + gunzip + incremental
  `xml.etree.ElementTree` parse, no full-file materialization).
- `CulturaX` is gated (needs an HF token with accepted terms) -- this is
  now the only source still blocked.
- `Portuguese-PD`'s `/size` API returns 0 bytes/rows despite having real
  data — its parquet index isn't computed on HF's side; don't trust that
  field blindly, cross-check against the schema/sample response.
