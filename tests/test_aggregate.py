import pyarrow as pa
import pyarrow.parquet as pq

from src.aggregate import estimate_shard_count, shuffle_into_shards, shuffle_table

_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("language", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("url", pa.string()),
        ("metadata", pa.string()),
    ]
)


def _write_part(path, n, prefix="doc"):
    rows = [
        {
            "text": f"{prefix}-{i}",
            "language": "pt",
            "source": "test-source",
            "source_id": f"{prefix}-{i}",
            "url": None,
            "metadata": "{}",
        }
        for i in range(n)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), path)


def _read_all_texts(out_dir):
    texts = []
    for path in sorted(out_dir.glob("*.parquet")):
        texts.extend(pq.read_table(path, columns=["text"]).column("text").to_pylist())
    return texts


# --- shuffle_table ---


def test_shuffle_table_is_a_permutation():
    table = pa.Table.from_pylist([{"x": i} for i in range(50)])
    shuffled = shuffle_table(table, seed=1)
    assert sorted(shuffled.column("x").to_pylist()) == list(range(50))


def test_shuffle_table_deterministic_for_same_seed():
    table = pa.Table.from_pylist([{"x": i} for i in range(50)])
    a = shuffle_table(table, seed=7).column("x").to_pylist()
    b = shuffle_table(table, seed=7).column("x").to_pylist()
    assert a == b


# --- estimate_shard_count ---


def test_estimate_shard_count_small_input_is_one_shard(tmp_path):
    part_dir = tmp_path / "pt"
    part_dir.mkdir()
    _write_part(part_dir / "part0.parquet", 100)
    assert estimate_shard_count([part_dir], target_shard_bytes=750 * 1024 * 1024) == 1


def test_estimate_shard_count_scales_with_target_size(tmp_path):
    part_dir = tmp_path / "pt"
    part_dir.mkdir()
    _write_part(part_dir / "part0.parquet", 1000)
    total_bytes = (part_dir / "part0.parquet").stat().st_size
    # Force ~4 shards by setting the target to a quarter of the actual input size.
    assert estimate_shard_count([part_dir], target_shard_bytes=max(1, total_bytes // 4)) == 4


# --- shuffle_into_shards ---


def test_shuffle_into_shards_small_input_single_shard(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 30)

    out_dir = tmp_path / "out" / "pt"
    n = shuffle_into_shards([part_dir], out_dir, seed=42)

    assert n == 30
    shards = sorted(out_dir.glob("*.parquet"))
    assert [p.name for p in shards] == ["train-00000-of-00001.parquet"]
    assert sorted(_read_all_texts(out_dir)) == sorted(f"doc-{i}" for i in range(30))


def test_shuffle_into_shards_preserves_all_rows_no_loss_or_duplication(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 40, prefix="a")
    _write_part(part_dir / "part1.parquet", 25, prefix="b")

    out_dir = tmp_path / "out" / "pt"
    n = shuffle_into_shards([part_dir], out_dir, seed=1)

    assert n == 65
    texts = _read_all_texts(out_dir)
    assert len(texts) == 65
    assert sorted(texts) == sorted([f"a-{i}" for i in range(40)] + [f"b-{i}" for i in range(25)])


def test_shuffle_into_shards_splits_into_multiple_shards_at_scale(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 2000)
    total_bytes = (part_dir / "part0.parquet").stat().st_size

    out_dir = tmp_path / "out" / "pt"
    n = shuffle_into_shards([part_dir], out_dir, seed=1, target_shard_bytes=max(1, total_bytes // 5))

    assert n == 2000
    shards = sorted(out_dir.glob("*.parquet"))
    assert len(shards) > 1
    # filenames are gap-free and agree on the total shard count
    expected_names = {f"train-{i:05d}-of-{len(shards):05d}.parquet" for i in range(len(shards))}
    assert {p.name for p in shards} == expected_names
    assert sorted(_read_all_texts(out_dir)) == sorted(f"doc-{i}" for i in range(2000))


def test_shuffle_into_shards_no_duplicate_rows_across_shards(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 500)
    total_bytes = (part_dir / "part0.parquet").stat().st_size

    out_dir = tmp_path / "out" / "pt"
    shuffle_into_shards([part_dir], out_dir, seed=3, target_shard_bytes=max(1, total_bytes // 6))

    texts = _read_all_texts(out_dir)
    assert len(texts) == len(set(texts)) == 500


def test_shuffle_into_shards_deterministic_for_same_seed(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 300)
    total_bytes = (part_dir / "part0.parquet").stat().st_size

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    shuffle_into_shards([part_dir], out_a, seed=99, target_shard_bytes=max(1, total_bytes // 3))
    shuffle_into_shards([part_dir], out_b, seed=99, target_shard_bytes=max(1, total_bytes // 3))

    assert _read_all_texts(out_a) == _read_all_texts(out_b)


def test_shuffle_into_shards_columns_projects_and_tolerates_schema_mismatch(tmp_path):
    # Mirrors the real pt/hi situation: two input dirs whose non-projected
    # columns disagree in type (string vs. struct) -- combining them without
    # `columns=` would hit a schema mismatch; projecting down to the shared
    # columns should work regardless, and the output shouldn't carry the
    # dropped column at all.
    pt_schema = pa.schema([("text", pa.string()), ("id", pa.string()), ("metadata", pa.string())])
    hi_schema = pa.schema(
        [("text", pa.string()), ("id", pa.string()), ("metadata", pa.struct([("file_path", pa.string())]))]
    )
    pt_dir = tmp_path / "in" / "pt"
    hi_dir = tmp_path / "in" / "hi"
    pt_dir.mkdir(parents=True)
    hi_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"text": f"pt-{i}", "id": f"pt-{i}", "metadata": "{}"} for i in range(20)], schema=pt_schema
        ),
        pt_dir / "part0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"text": f"hi-{i}", "id": f"hi-{i}", "metadata": {"file_path": "x"}} for i in range(15)],
            schema=hi_schema,
        ),
        hi_dir / "part0.parquet",
    )

    out_dir = tmp_path / "out" / "all"
    n = shuffle_into_shards([pt_dir, hi_dir], out_dir, seed=5, columns=["text", "id"])

    assert n == 35
    shard = pq.read_table(sorted(out_dir.glob("*.parquet"))[0])
    assert shard.column_names == ["text", "id"]
    texts = shard.column("text").to_pylist()
    assert sorted(t for t in texts if t.startswith("pt-")) == sorted(f"pt-{i}" for i in range(20))
    assert sorted(t for t in texts if t.startswith("hi-")) == sorted(f"hi-{i}" for i in range(15))


def test_shuffle_into_shards_two_level_preserves_all_rows_no_loss_or_duplication(tmp_path):
    # Force the two-level coarse/fine path (see shuffle_into_shards's
    # docstring): num_shards > max_concurrent_writers. This is the path
    # that keeps concurrently-open ParquetWriters bounded regardless of
    # corpus size -- confirmed necessary at real scale (es's 1158 shards
    # OOM'd at both 600G and 900G with the old flat design, which opened
    # all num_shards writers at once).
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 2000)
    total_bytes = (part_dir / "part0.parquet").stat().st_size

    out_dir = tmp_path / "out" / "pt"
    n = shuffle_into_shards(
        [part_dir],
        out_dir,
        seed=1,
        target_shard_bytes=max(1, total_bytes // 20),  # ~20 shards
        max_concurrent_writers=3,  # force num_shards (~20) > cap (3)
    )

    assert n == 2000
    shards = sorted(out_dir.glob("*.parquet"))
    assert len(shards) > 3  # confirms the two-level path actually engaged
    expected_names = {f"train-{i:05d}-of-{len(shards):05d}.parquet" for i in range(len(shards))}
    assert {p.name for p in shards} == expected_names
    texts = _read_all_texts(out_dir)
    assert sorted(texts) == sorted(f"doc-{i}" for i in range(2000))
    assert len(texts) == len(set(texts))
    # No leftover intermediate state.
    assert not (out_dir / ".work").exists()


def test_shuffle_into_shards_two_level_deterministic_for_same_seed(tmp_path):
    part_dir = tmp_path / "in" / "pt"
    part_dir.mkdir(parents=True)
    _write_part(part_dir / "part0.parquet", 500)
    total_bytes = (part_dir / "part0.parquet").stat().st_size

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    kwargs = dict(seed=99, target_shard_bytes=max(1, total_bytes // 10), max_concurrent_writers=3)
    shuffle_into_shards([part_dir], out_a, **kwargs)
    shuffle_into_shards([part_dir], out_b, **kwargs)

    assert _read_all_texts(out_a) == _read_all_texts(out_b)


def test_shuffle_into_shards_combines_multiple_input_dirs(tmp_path):
    pt_dir = tmp_path / "in" / "pt"
    es_dir = tmp_path / "in" / "es"
    pt_dir.mkdir(parents=True)
    es_dir.mkdir(parents=True)
    _write_part(pt_dir / "part0.parquet", 20, prefix="pt")
    _write_part(es_dir / "part0.parquet", 15, prefix="es")

    out_dir = tmp_path / "out" / "all"
    n = shuffle_into_shards([pt_dir, es_dir], out_dir, seed=5)

    assert n == 35
    texts = _read_all_texts(out_dir)
    assert sorted(t for t in texts if t.startswith("pt-")) == sorted(f"pt-{i}" for i in range(20))
    assert sorted(t for t in texts if t.startswith("es-")) == sorted(f"es-{i}" for i in range(15))
