"""Etapa 6 — Aggregation and shuffle (TASK1-PLAN.md).

Turns the per-language deduped shards (data/processed/{pt,es,hi}/*.parquet,
produced by scripts/run_all_sources.py) into the final dataset shape: one
HF repo with 4 configs -- pt, es, hi, all -- satisfying "a single HF
dataset" (minutas 12 jun) while still letting each language be loaded on
its own.

`shuffle_into_shards` is the scale-safe entry point (streams input, never
loads a whole config into RAM, writes ~500MB-1GB output shards per the
plan sec 6 Etapa 6 step 4). `shuffle_table` is kept as a small in-memory
helper it uses internally for the bounded-size per-shard pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DEFAULT_TARGET_SHARD_BYTES = 750 * 1024 * 1024  # midpoint of the plan's 500MB-1GB target


def shuffle_table(table: pa.Table, seed: int) -> pa.Table:
    """Deterministic shuffle of all rows in `table`. Only safe for
    already-bounded-size tables (e.g. one output shard) -- see
    shuffle_into_shards for the streaming, unbounded-input path.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(table.num_rows)
    return table.take(pa.array(indices))


def estimate_shard_count(input_dirs: list[Path], target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES) -> int:
    """Ballpark shard count from on-disk input size (same schema/encoding as
    the output, so it's a reasonable proxy without reading any row data).
    """
    total_bytes = sum(f.stat().st_size for d in input_dirs for f in d.glob("*.parquet"))
    return max(1, round(total_bytes / target_shard_bytes))


def shuffle_into_shards(
    input_dirs: list[Path],
    out_dir: Path,
    seed: int,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
) -> int:
    """Two-pass streaming shuffle across one or more input directories of
    Parquet part files (pass a single dir for a per-language config, or all
    three language dirs together for the combined `all` config).

    Pass 1 (streaming partition): stream input in record batches and assign
    each row to one of K output shards via a seeded RNG, writing directly to
    K open ParquetWriters -- the full dataset never has to fit in memory,
    only one batch at a time. K is sized off on-disk input bytes so each
    shard lands near target_shard_bytes.

    Pass 2 (local shuffle): each shard file, now bounded to ~target_shard_bytes,
    is read back, shuffled in memory (shuffle_table), and rewritten -- this
    is what actually randomizes row order; pass 1 only randomizes which
    shard a row lands in; rows keep their relative stream order within a
    shard until this pass.

    Returns the total row count written. Writes out_dir/train-NNNNN-of-MMMMM.parquet.
    """
    # ds.dataset() accepts a single directory or a list of *files* -- not a
    # list of directories -- so glob each dir down to its part files first.
    input_files = [str(f) for d in input_dirs for f in sorted(d.glob("*.parquet"))]
    dataset = ds.dataset(input_files, format="parquet")
    num_shards = estimate_shard_count(input_dirs, target_shard_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths = [out_dir / f".tmp-shard-{i:05d}.parquet" for i in range(num_shards)]
    writers = [pq.ParquetWriter(tmp_paths[i], dataset.schema) for i in range(num_shards)]
    rows_per_shard = [0] * num_shards
    rng = np.random.default_rng(seed)

    try:
        for batch in dataset.to_batches():
            if batch.num_rows == 0:
                continue
            shard_idx = rng.integers(0, num_shards, size=batch.num_rows)
            table = pa.Table.from_batches([batch])
            for i in range(num_shards):
                mask = shard_idx == i
                if not mask.any():
                    continue
                sub = table.filter(pa.array(mask))
                writers[i].write_table(sub)
                rows_per_shard[i] += sub.num_rows
    finally:
        for w in writers:
            w.close()

    # Drop any shard that ended up empty (possible if num_shards is large
    # relative to the actual row count) and renumber the rest sequentially
    # so output filenames have no gaps.
    nonempty = [i for i in range(num_shards) if rows_per_shard[i] > 0]
    total_rows = 0
    for out_idx, i in enumerate(nonempty):
        shard_table = pq.read_table(tmp_paths[i])
        # Distinct sub-seed per shard (still fully deterministic from `seed`)
        # so shards don't all reproduce the same permutation pattern.
        shuffled = shuffle_table(shard_table, seed=seed + i + 1)
        final_path = out_dir / f"train-{out_idx:05d}-of-{len(nonempty):05d}.parquet"
        pq.write_table(shuffled, final_path)
        total_rows += shuffled.num_rows

    for tmp_path in tmp_paths:
        tmp_path.unlink(missing_ok=True)

    return total_rows
