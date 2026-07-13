#!/usr/bin/env python3
"""Etapa 6 (TASK1-PLAN.md): build the final dataset shape from the
per-language Parquet part files in data/processed/{pt,es,hi}/*.parquet
(produced by run_all_sources.py, one or more part files per source row).

Writes a local folder shaped exactly like the real HF dataset repo would
be: one config per language (pt/es/hi) plus a combined `all` config, with a
README.md declaring the configs the same way HF multi-config datasets do
(e.g. HuggingFaceFW/fineweb-2). Does NOT upload anywhere.

Uses src.aggregate.shuffle_into_shards: a two-pass streaming shuffle that
never loads a whole config into memory and writes ~500MB-1GB output shards
(plan sec 6, Etapa 6 step 4) instead of one giant Parquet file per config.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aggregate import shuffle_into_shards

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

# llm-und/pretrain-pt-es-hi

Built from `data/processed/{{pt,es,hi}}/*.parquet`, produced by
`scripts/run_all_sources.py` across the currently ingestable sources (see
TASK1-PLAN.md sec 2.1 and README.md for blocked ones). Doc counts below
reflect whatever run produced the input -- check `data/processed/funnel_stats.json`
for the ingest/filter/dedup breakdown behind these numbers.

| config | docs |
|--------|-----:|
{stats_rows}
"""


def main(in_dir: Path, out_dir: Path) -> None:
    lang_dirs = {lang: in_dir / lang for lang in LANGUAGES}
    for lang, lang_dir in lang_dirs.items():
        if not lang_dir.exists() or not any(lang_dir.glob("*.parquet")):
            raise FileNotFoundError(
                f"{lang_dir} has no Parquet part files -- run scripts/run_all_sources.py first"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    for lang, lang_dir in lang_dirs.items():
        stats[lang] = shuffle_into_shards([lang_dir], out_dir / lang, seed=PER_LANG_SEED)

    stats["all"] = shuffle_into_shards(list(lang_dirs.values()), out_dir / "all", seed=ALL_CONFIG_SEED)

    stats_rows = "\n".join(f"| {cfg} | {n:,} |" for cfg, n in stats.items())
    (out_dir / "README.md").write_text(README_TEMPLATE.format(stats_rows=stats_rows))

    print("Built final dataset shape at:", out_dir)
    for cfg, n in stats.items():
        print(f"  config={cfg:5s} docs={n:,}")


if __name__ == "__main__":
    main(in_dir=REPO_ROOT / "data" / "processed", out_dir=REPO_ROOT / "data" / "final")
