"""Etapa 6 — Aggregation and shuffle (TASK1-PLAN.md).

Turns the per-language deduped shards (data/processed/{pt,es,hi}.parquet,
produced by scripts/run_all_sources.py) into the final dataset shape: one
HF repo with 4 configs -- pt, es, hi, all -- satisfying "a single HF
dataset" (minutas 12 jun) while still letting each language be loaded on
its own.
"""

from __future__ import annotations

import pyarrow as pa


def shuffle_table(table: pa.Table, seed: int) -> pa.Table:
    """Deterministic shuffle of all rows in `table`."""
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.permutation(table.num_rows)
    return table.take(pa.array(indices))


def build_all_config(lang_tables: dict[str, pa.Table], seed: int) -> pa.Table:
    """Concatenate all per-language tables and shuffle globally.

    Uses a different seed than the per-language shuffle so language blocks
    don't stay contiguous after concatenation.
    """
    combined = pa.concat_tables(list(lang_tables.values()), promote_options="default")
    return shuffle_table(combined, seed)


def dedup_stats(lang_tables: dict[str, pa.Table]) -> dict[str, int]:
    return {lang: table.num_rows for lang, table in lang_tables.items()}
