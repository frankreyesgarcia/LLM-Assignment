#!/usr/bin/env python3
"""Etapa 6 (TASK1-PLAN.md): build the final dataset shape from the
per-language Parquet part files in data/processed/{pt,es,hi}/*.parquet
(produced by scripts/run_dedup_datatrove.py).

Writes a local folder shaped exactly like the real HF dataset repo would
be: one config per language (pt/es/hi), with a README.md declaring the
configs the same way HF multi-config datasets do (e.g. HuggingFaceFW/
fineweb-2). Does NOT upload anywhere.

Deliberately does NOT also build a combined `all` config (a physical 4th
copy of the entire corpus): pt/es/hi/any combination is already one line
away via `datasets.concatenate_datasets` (see the README template below),
and `all` was also the single riskiest stage in practice -- it needs the
most shards of any config (pt+es+hi combined) and was never actually
reached before this was cut -- pt alone needed 600GB+ to get through its
498-shard pass 1, and `all` would need substantially more.

Uses src.aggregate.shuffle_into_shards: a two-pass streaming shuffle that
never loads a whole config into memory and writes ~500MB-1GB output shards
(plan sec 6, Etapa 6 step 4) instead of one giant Parquet file per config.

No validation/test split is built here (each config is `train` only, by
design, not an omission): evaluation uses fineweb-2's own existing
test/validation splits for pt/es/hi rather than holding out a slice of
this pipeline's output.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from src.aggregate import shuffle_into_shards

REPO_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ["pt", "es", "hi"]
PER_LANG_SEED = 42
# Drop data/processed/{lang}'s `metadata` struct (crawl provenance -- dump/
# url/date/language_score/...) here: it varies in shape across languages
# (pt/es carry 9 fields, hi carries 1, since only hi's dedup needed
# read_metadata=False to avoid a cross-source schema clash -- see
# scripts/run_dedup_datatrove.py), and its size was the direct cause of an
# OOM on a real pt run (498 concurrently-open shard writers, each buffering
# the full 9-field struct per row).
OUTPUT_COLUMNS = ["text", "id"]

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
---

# llm-und/pretrain-pt-es-hi

Built from `data/processed/{{pt,es,hi}}/*.parquet`, produced by
`scripts/run_dedup_datatrove.py` (MinHash dedup across pt-fineweb2,
es-fineweb2, hi-fineweb2, and hi-sangraha -- see README.md). Doc counts
below reflect whatever run produced the input.

| config | docs |
|--------|-----:|
{stats_rows}

Load a single language:

```python
from datasets import load_dataset
ds = load_dataset("{{repo_id}}", "pt")  # or "es", "hi"
```

Load any combination (e.g. all three) by concatenating -- this is a
one-line operation at load time, not a separate config baked into the
dataset:

```python
from datasets import load_dataset, concatenate_datasets
ds = concatenate_datasets([load_dataset("{{repo_id}}", lang, split="train") for lang in ["pt", "es", "hi"]])
```
"""


def _completed_row_count(lang_out_dir: Path) -> int | None:
    """None if `lang_out_dir` isn't a clean completed run (no output at all,
    a `.work/` dir left behind by an interrupted shuffle_into_shards call,
    or leftover `.tmp-shard-*` files from the older flat-partition design --
    shuffle_into_shards has no partial-resume logic, so either marker means
    the whole language needs to be redone, not just picked up). Otherwise
    the total row count already on disk, for the stats table.
    """
    if not lang_out_dir.exists():
        return None
    if (lang_out_dir / ".work").exists() or any(lang_out_dir.glob(".tmp-shard-*.parquet")):
        return None
    train_files = list(lang_out_dir.glob("train-*.parquet"))
    if not train_files:
        return None
    return sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)


def main(in_dir: Path, out_dir: Path) -> None:
    lang_dirs = {lang: in_dir / lang for lang in LANGUAGES}
    for lang, lang_dir in lang_dirs.items():
        if not lang_dir.exists() or not any(lang_dir.glob("*.parquet")):
            raise FileNotFoundError(
                f"{lang_dir} has no Parquet part files -- run scripts/run_dedup_datatrove.py first"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    for lang, lang_dir in lang_dirs.items():
        lang_out_dir = out_dir / lang
        completed_rows = _completed_row_count(lang_out_dir)
        if completed_rows is not None:
            print(f"--- {lang}: already built ({completed_rows:,} rows), skipping ---")
            stats[lang] = completed_rows
            continue
        if lang_out_dir.exists():
            print(f"--- {lang}: clearing stale/partial output before rebuilding ---")
            shutil.rmtree(lang_out_dir)
        stats[lang] = shuffle_into_shards(
            [lang_dir], lang_out_dir, seed=PER_LANG_SEED, columns=OUTPUT_COLUMNS
        )

    stats_rows = "\n".join(f"| {cfg} | {n:,} |" for cfg, n in stats.items())
    (out_dir / "README.md").write_text(README_TEMPLATE.format(stats_rows=stats_rows))

    print("Built final dataset shape at:", out_dir)
    for cfg, n in stats.items():
        print(f"  config={cfg:5s} docs={n:,}")


if __name__ == "__main__":
    main(in_dir=REPO_ROOT / "data" / "processed", out_dir=REPO_ROOT / "data" / "final")
