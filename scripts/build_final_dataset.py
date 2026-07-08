#!/usr/bin/env python3
"""Fase 2 test run of Etapa 6 (TASK1-PLAN.md): build the final dataset shape
from the pilot-scale per-language shards in data/processed/*.parquet
(produced by run_all_sources.py -- NOT the full corpus, just to see how the
final output would look).

Writes a local folder shaped exactly like the real HF dataset repo would
be: one config per language (pt/es/hi) plus a combined `all` config, with a
README.md declaring the configs the same way HF multi-config datasets do
(e.g. HuggingFaceFW/fineweb-2). Does NOT upload anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregate import build_all_config, dedup_stats, shuffle_table

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ["pt", "es", "hi"]
PER_LANG_SEED = 42
ALL_CONFIG_SEED = 1337

README_TEMPLATE = """\
---
configs:
- config_name: pt
  data_files:
  - split: train
    path: pt/train-*.parquet
- config_name: es
  data_files:
  - split: train
    path: es/train-*.parquet
- config_name: hi
  data_files:
  - split: train
    path: hi/train-*.parquet
- config_name: all
  data_files:
  - split: train
    path: all/train-*.parquet
---

# llm-und/pretrain-pt-es-hi (PILOT-SCALE TEST BUILD -- not the final corpus)

Built from `data/processed/{{pt,es,hi}}.parquet`, themselves produced by a
300-docs-per-source pilot run across the 10 currently ingestable sources
(see TASK1-PLAN.md sec 2.1 and README.md for the 2 blocked ones). This is
here to validate the Etapa 6 aggregation shape end-to-end, not to publish.

| config | docs |
|--------|-----:|
{stats_rows}
"""


def main(in_dir: Path, out_dir: Path) -> None:
    lang_tables = {}
    for lang in LANGUAGES:
        path = in_dir / f"{lang}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/run_all_sources.py first")
        lang_tables[lang] = shuffle_table(pq.read_table(path), seed=PER_LANG_SEED)

    all_table = build_all_config(lang_tables, seed=ALL_CONFIG_SEED)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = dedup_stats(lang_tables)
    stats["all"] = all_table.num_rows

    for lang, table in lang_tables.items():
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, lang_dir / "train-00000-of-00001.parquet")

    all_dir = out_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(all_table, all_dir / "train-00000-of-00001.parquet")

    stats_rows = "\n".join(f"| {cfg} | {n:,} |" for cfg, n in stats.items())
    (out_dir / "README.md").write_text(README_TEMPLATE.format(stats_rows=stats_rows))

    print("Built final dataset shape at:", out_dir)
    for cfg, n in stats.items():
        print(f"  config={cfg:5s} docs={n:,}")


if __name__ == "__main__":
    main(in_dir=REPO_ROOT / "data" / "processed", out_dir=REPO_ROOT / "data" / "final")
