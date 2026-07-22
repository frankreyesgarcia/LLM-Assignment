"""Tests for scripts/build_final_dataset.py's resume/skip behavior.

shuffle_into_shards has no partial-resume logic (a mid-run OOM/timeout
means redoing that language from scratch) -- these tests cover main()'s
own skip-if-already-built check, which exists specifically so a rerun
after one language OOMs (e.g. es, after pt already finished) doesn't waste
hours redoing languages that already completed cleanly.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_final_dataset import main

_SCHEMA = pa.schema([("text", pa.string()), ("id", pa.string())])


def _write_processed(lang_dir, n, prefix):
    lang_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"text": f"{prefix}-{i}", "id": f"{prefix}-{i}"} for i in range(n)]
    pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), lang_dir / "part0.parquet")


def _set_languages(monkeypatch, languages):
    monkeypatch.setattr("scripts.build_final_dataset.LANGUAGES", languages)


def test_skips_already_completed_language(tmp_path, monkeypatch):
    _set_languages(monkeypatch, ["pt"])
    in_dir = tmp_path / "processed"
    out_dir = tmp_path / "final"
    _write_processed(in_dir / "pt", 20, "pt")

    main(in_dir, out_dir)
    first_run_mtimes = {p: p.stat().st_mtime for p in (out_dir / "pt").glob("train-*.parquet")}
    assert first_run_mtimes

    # Rerun: should detect pt is already built and not rewrite its shards.
    main(in_dir, out_dir)
    second_run_mtimes = {p: p.stat().st_mtime for p in (out_dir / "pt").glob("train-*.parquet")}
    assert second_run_mtimes == first_run_mtimes


def test_rebuilds_stale_partial_output(tmp_path, monkeypatch):
    _set_languages(monkeypatch, ["pt"])
    in_dir = tmp_path / "processed"
    out_dir = tmp_path / "final"
    _write_processed(in_dir / "pt", 20, "pt")

    # Simulate a run that died mid pass-1: only a .tmp-shard file, no
    # finished train-*.parquet -- must be treated as incomplete, not skipped.
    pt_out = out_dir / "pt"
    pt_out.mkdir(parents=True)
    (pt_out / ".tmp-shard-00000.parquet").write_bytes(b"not a real parquet file")

    main(in_dir, out_dir)

    assert not list(pt_out.glob(".tmp-shard-*.parquet"))
    train_files = list(pt_out.glob("train-*.parquet"))
    assert train_files
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)
    assert total_rows == 20


def test_only_rebuilds_the_failed_language(tmp_path, monkeypatch):
    _set_languages(monkeypatch, ["pt", "es"])
    in_dir = tmp_path / "processed"
    out_dir = tmp_path / "final"
    _write_processed(in_dir / "pt", 20, "pt")
    _write_processed(in_dir / "es", 15, "es")

    main(in_dir, out_dir)
    pt_mtimes_before = {p: p.stat().st_mtime for p in (out_dir / "pt").glob("train-*.parquet")}

    # Simulate es having OOM'd on a second attempt (stale tmp shard) while
    # pt's earlier successful output is untouched.
    for f in (out_dir / "es").glob("train-*.parquet"):
        f.unlink()
    (out_dir / "es" / ".tmp-shard-00000.parquet").write_bytes(b"partial")

    main(in_dir, out_dir)

    pt_mtimes_after = {p: p.stat().st_mtime for p in (out_dir / "pt").glob("train-*.parquet")}
    assert pt_mtimes_after == pt_mtimes_before
    es_rows = sum(
        pq.ParquetFile(f).metadata.num_rows for f in (out_dir / "es").glob("train-*.parquet")
    )
    assert es_rows == 15
