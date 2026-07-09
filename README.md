# LLM-UND — Task 1: pre-training data pipeline

Pipeline for filtering, cleaning, and deduplicating PT/ES/HI pre-training
data into a single HF dataset. See `TASK1-PLAN.md` for the full design.

## Status

Implemented so far (Fase 0 + Fase 1 + most of Fase 2 of the plan):

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
- `src/dedup/` — exact dedup (SHA256) and near dedup (MinHash + LSH).
- `scripts/inspect_sources.py` — queries HF `datasets-server` for size/schema/license
  on all 12 sources from the assignment; writes `configs/sources.yaml`.
- `scripts/run_pilot.py` — full Etapas 1-5 pipeline on one small source (`corpus-ptbr-v2`).
- `scripts/run_all_sources.py` — same pipeline across all 10 available
  sources, with **per-language** exact dedup (as required by the plan for
  cross-source overlap), writing `data/processed/{pt,es,hi}.parquet`.

**Blocked** (see `src/ingest/registry.py::BLOCKED_SOURCES`):
- `CulturaX` — gated dataset, needs an HF token with accepted terms. This
  is the only remaining blocked source; `corpus-carolina` was unblocked
  via `CarolinaAdapter` (see above).

Not yet implemented: the `pt`/`es`/`hi`/`all` config aggregation + HF Hub
upload (Etapa 6), sharded MinHash dedup at scale, `compute_stats.py`, and
the SLURM orchestration scripts.

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

# Fase 2: run all 10 available sources -> data/processed/{pt,es,hi}.parquet
uv run scripts/run_all_sources.py --limit-per-source 500

# tests
uv run pytest tests/ -v
```

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
