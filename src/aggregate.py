"""Etapa 6 — Aggregation and shuffle (TASK1-PLAN.md).

Turns the per-language deduped shards (data/processed/{pt,es,hi}/*.parquet,
produced by scripts/run_dedup_datatrove.py) into the final dataset shape: one
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
# Hard cap on concurrently-open ParquetWriters, independent of corpus size --
# see shuffle_into_shards's docstring for why this exists. Confirmed at real
# scale: 498 concurrent writers (pt) needed 600GB; 1158 (es) OOM'd twice
# (600G, then 900G) even after fixing the two prior memory bugs (metadata
# schema size, dictionary encoding). A small fixed cap keeps memory bounded
# regardless of how large a future source/language gets.
DEFAULT_MAX_CONCURRENT_WRITERS = 64


def shuffle_table(table: pa.Table, seed: int) -> pa.Table:
    """Deterministic shuffle of all rows in `table`. Only safe for
    already-bounded-size tables (e.g. one output shard) -- see
    shuffle_into_shards for the streaming, unbounded-input path.

    Promotes string/binary columns to their 64-bit-offset (large_*) variants
    before `.take()`: `.take()` gathers by building one contiguous buffer per
    column internally, and plain string/binary columns use 32-bit offsets
    (2GB limit) -- hit for real on a hi shard whose on-disk/compressed size
    was ~1.5GB (under target_shard_bytes) but whose decompressed text
    exceeded 2GB, since use_dictionary=False (see
    _partition_by_random_bucket's docstring) makes actual output larger than
    what estimate_shard_count's input-size-based target assumed.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(table.num_rows)
    large_schema = pa.schema(
        [
            f.with_type(pa.large_string())
            if pa.types.is_string(f.type)
            else f.with_type(pa.large_binary())
            if pa.types.is_binary(f.type)
            else f
            for f in table.schema
        ]
    )
    table = table.cast(large_schema)
    return table.take(pa.array(indices))


def estimate_shard_count(input_dirs: list[Path], target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES) -> int:
    """Ballpark shard count from on-disk input size (same schema/encoding as
    the output, so it's a reasonable proxy without reading any row data).
    """
    total_bytes = sum(f.stat().st_size for d in input_dirs for f in d.glob("*.parquet"))
    return max(1, round(total_bytes / target_shard_bytes))


def _partition_by_random_bucket(
    batches,
    schema: pa.Schema,
    out_paths: list[Path],
    num_buckets: int,
    rng: np.random.Generator,
) -> list[int]:
    """Streams `batches`, assigns each row to one of `num_buckets` via `rng`,
    and writes to `out_paths` (one ParquetWriter per bucket, all open for the
    whole call). Returns rows written per bucket.

    Shared by both levels of shuffle_into_shards's partitioning (the coarse
    pass and each fine pass) -- structurally identical either way: read
    batches, randomly assign, write grouped.

    use_dictionary=False: pq.ParquetWriter defaults to True, which makes a
    writer accumulate a dictionary of every unique value it's ever written
    for its whole open lifetime. Free text is essentially all-unique, so
    that dictionary is functionally a second copy of everything routed to
    that bucket, growing for as long as the writer stays open -- confirmed
    as a real OOM cause that a naive "just add more --mem" didn't fix, since
    the footprint scaled with total progress, not a bounded per-batch
    working set.
    """
    writers = [pq.ParquetWriter(out_paths[i], schema, use_dictionary=False) for i in range(num_buckets)]
    rows_per_bucket = [0] * num_buckets
    try:
        for batch in batches:
            if batch.num_rows == 0:
                continue
            bucket_idx = rng.integers(0, num_buckets, size=batch.num_rows)
            table = pa.Table.from_batches([batch])

            # Group rows by bucket via a sort instead of scanning every
            # bucket against every batch (`for i in range(num_buckets): mask
            # = bucket_idx == i`, an O(batch_size) comparison repeated
            # num_buckets times -- O(batches * buckets) total, and since
            # both grow with corpus size this scaled close to O(docs^2) in
            # practice, confirmed as the dominant cost on a real
            # ~75GB/102-bucket run before this fix). Sorting once per batch
            # (O(batch_size log batch_size)) and slicing contiguous runs
            # makes each bucket's share of a batch an O(1) lookup instead of
            # an O(batch_size) scan.
            order = np.argsort(bucket_idx, kind="stable")
            sorted_table = table.take(pa.array(order))
            sorted_bucket_idx = bucket_idx[order]
            boundaries = np.searchsorted(sorted_bucket_idx, np.arange(num_buckets + 1))
            for i in range(num_buckets):
                start, end = int(boundaries[i]), int(boundaries[i + 1])
                if start == end:
                    continue
                sub = sorted_table.slice(start, end - start)
                writers[i].write_table(sub)
                rows_per_bucket[i] += sub.num_rows
    finally:
        for w in writers:
            w.close()
    return rows_per_bucket


def _finalize_shard(tmp_path: Path, seed: int, final_path: Path) -> int:
    """Read a closed, bounded-size shard file back, shuffle its row order in
    memory (the partitioning pass only randomizes which shard a row lands
    in, not its order within the shard), write it to its final name, and
    remove the tmp file. Returns the row count.
    """
    table = pq.read_table(tmp_path)
    shuffled = shuffle_table(table, seed=seed)
    pq.write_table(shuffled, final_path)
    tmp_path.unlink()
    return shuffled.num_rows


def shuffle_into_shards(
    input_dirs: list[Path],
    out_dir: Path,
    seed: int,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    columns: list[str] | None = None,
    max_concurrent_writers: int = DEFAULT_MAX_CONCURRENT_WRITERS,
) -> int:
    """Streaming shuffle across one or more input directories of Parquet
    part files (pass a single dir for a per-language config, or all three
    language dirs together for the combined `all` config), into
    ~target_shard_bytes-sized output shards.

    Never holds more than `max_concurrent_writers` ParquetWriters open at
    once, regardless of how many final output shards the corpus needs
    (`estimate_shard_count`) -- memory scales with `max_concurrent_writers`,
    not corpus size. This matters because a single flat partition (open all
    num_shards writers, stream once) is what OOM'd repeatedly at real scale:
    pt (498 shards) needed 600GB; es (1158 shards) OOM'd at both 600G and
    900G even after fixing the two prior memory bugs (metadata schema size,
    dictionary encoding) -- 1158 concurrently-open writers was itself the
    problem, not something either of those fixes could solve.

    If `num_shards <= max_concurrent_writers`, this reduces to a single
    partition pass followed by finalization (equivalent to the old flat
    design). Otherwise:

    1. Coarse pass: partition the whole input into `max_concurrent_writers`
       coarse buckets via one streaming pass (bounded writers).
    2. Fine pass, one coarse bucket at a time: read a single (now closed,
       so already bounded to ~1/max_concurrent_writers of the corpus)
       coarse bucket back, and partition *it* into its share of the final
       shard count (also bounded to <= max_concurrent_writers, since shards
       are split evenly across coarse buckets). Only one coarse bucket's
       data is ever in flight at a time.

    Row assignment is still uniformly random overall: each row's final
    shard is (coarse bucket, its random slot within that bucket's share
    of shards), both independently random -- equivalent in distribution to
    a single flat random assignment across all final shards, just computed
    in two bounded-memory steps instead of one unbounded one.

    Finalization (shuffling each bounded shard's row order in memory and
    writing its final `train-NNNNN-of-MMMMM.parquet` name) is identical
    either way -- see `_finalize_shard`.

    `columns`, if given, projects the input down to just those columns at
    read time (never materializing the rest) -- e.g. dropping a `metadata`
    struct that (a) varies in shape across per-language inputs (pt/es carry
    9 provenance fields, hi carries 1, since only hi's dedup pass needed
    read_metadata=False to avoid a cross-source schema clash -- see
    scripts/run_dedup_datatrove.py) and would otherwise make combining them
    for the `all` config hit that same kind of mismatch one level up, and
    (b) multiplies the memory each concurrently-open writer buffers.

    Returns the total row count written. Writes out_dir/train-NNNNN-of-MMMMM.parquet.
    """
    # ds.dataset() accepts a single directory or a list of *files* -- not a
    # list of directories -- so glob each dir down to its part files first.
    input_files = [str(f) for d in input_dirs for f in sorted(d.glob("*.parquet"))]
    dataset = ds.dataset(input_files, format="parquet")
    num_shards = estimate_shard_count(input_dirs, target_shard_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = dataset.schema if columns is None else pa.schema([dataset.schema.field(c) for c in columns])
    rng = np.random.default_rng(seed)

    # Intermediate coarse/fine bucket files live here -- kept separate from
    # out_dir's final `train-*.parquet` names so a caller (e.g.
    # scripts/build_final_dataset.py) can tell "still in progress" (this
    # directory exists) apart from "done" (only train-* files exist) after
    # a crash.
    work_dir = out_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # (shard_index, tmp_path) for every non-empty shard, in no particular
    # order -- final numbering is assigned below once we know how many
    # actually got any rows (a shard can end up empty if num_shards is
    # large relative to the real row count).
    nonempty: list[tuple[int, Path]] = []

    if num_shards <= max_concurrent_writers:
        tmp_paths = [work_dir / f"shard-{i:05d}.parquet" for i in range(num_shards)]
        rows_per_shard = _partition_by_random_bucket(dataset.to_batches(columns=columns), schema, tmp_paths, num_shards, rng)
        for i, path in enumerate(tmp_paths):
            if rows_per_shard[i] > 0:
                nonempty.append((i, path))
            else:
                path.unlink(missing_ok=True)
    else:
        num_coarse = max_concurrent_writers
        coarse_paths = [work_dir / f"coarse-{i:05d}.parquet" for i in range(num_coarse)]
        coarse_rows = _partition_by_random_bucket(dataset.to_batches(columns=columns), schema, coarse_paths, num_coarse, rng)

        # Split num_shards as evenly as possible across the num_coarse
        # buckets -- each bucket's fine pass then opens at most
        # ceil(num_shards / num_coarse) <= max_concurrent_writers writers.
        base, extra = divmod(num_shards, num_coarse)
        global_idx = 0
        for i, coarse_path in enumerate(coarse_paths):
            n_fine = base + (1 if i < extra else 0)
            if coarse_rows[i] == 0 or n_fine == 0:
                coarse_path.unlink(missing_ok=True)
                continue
            fine_dataset = ds.dataset([str(coarse_path)], format="parquet")
            fine_paths = [work_dir / f"fine-{global_idx + j:05d}.parquet" for j in range(n_fine)]
            # Distinct, deterministic sub-seed per coarse bucket so re-runs
            # with the same `seed` reproduce the same output.
            fine_rng = np.random.default_rng(seed + i + 1)
            fine_rows = _partition_by_random_bucket(fine_dataset.to_batches(), schema, fine_paths, n_fine, fine_rng)
            coarse_path.unlink()
            for j, fine_path in enumerate(fine_paths):
                if fine_rows[j] > 0:
                    nonempty.append((global_idx + j, fine_path))
                else:
                    fine_path.unlink(missing_ok=True)
            global_idx += n_fine

    total_rows = 0
    for out_idx, (shard_idx, tmp_path) in enumerate(nonempty):
        final_path = out_dir / f"train-{out_idx:05d}-of-{len(nonempty):05d}.parquet"
        # Distinct sub-seed per shard (still fully deterministic from `seed`)
        # so shards don't all reproduce the same permutation pattern.
        total_rows += _finalize_shard(tmp_path, seed=seed + shard_idx + 1, final_path=final_path)

    work_dir.rmdir()
    return total_rows
