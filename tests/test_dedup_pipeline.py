"""Integration test for scripts/run_dedup_datatrove.py's MinHash wiring.

Runs the real 4-stage datatrove pipeline (via LocalPipelineExecutor, no
SLURM needed) over tiny synthetic parquet fixtures standing in for two
sources sharing one language -- the same shape as hi-fineweb2/hi-sangraha,
but small enough to run in a test. This is the only place a wiring mistake
(wrong reader order, wrong stage dependency, wrong id/text column) would
get caught before a real multi-hour run.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.run_dedup_datatrove import build_language_stages

# Long enough (>= MinhashConfig's default n_grams=5 words) for shingling to
# be meaningful, distinct enough per doc to not accidentally collide.
_DUPLICATE_TEXT = "this exact duplicate document appears in both sources for the dedup test"
_ROW_A_DOCS = [
    ("a1", "unique document number one from row a with enough words to shingle"),
    ("a2", _DUPLICATE_TEXT),
    ("a3", "unique document number three from row a with enough words to shingle"),
]
_ROW_B_DOCS = [
    ("b1", _DUPLICATE_TEXT),
    ("b2", "unique document number two from row b with enough words to shingle"),
]


def _write_parquet(path: Path, id_key: str, docs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({id_key: [d[0] for d in docs], "text": [d[1] for d in docs]})
    pq.write_table(table, path)


@pytest.fixture
def fixture_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    _write_parquet(raw_dir / "test-row-a" / "data.parquet", "id", _ROW_A_DOCS)
    _write_parquet(raw_dir / "test-row-b" / "data.parquet", "doc_id", _ROW_B_DOCS)

    fake_specs = {
        "test-row-a": {"pattern": "data.parquet", "text_key": "text", "id_key": "id"},
        "test-row-b": {"pattern": "data.parquet", "text_key": "text", "id_key": "doc_id"},
    }
    fake_rows_by_language = {"hi": ["test-row-a", "test-row-b"]}

    monkeypatch.setattr("scripts.run_dedup_datatrove.SOURCE_SPECS", fake_specs)
    monkeypatch.setattr("scripts.run_dedup_datatrove.ROWS_BY_LANGUAGE", fake_rows_by_language)

    return raw_dir, tmp_path


def test_dedup_removes_cross_source_duplicate(fixture_dirs: tuple[Path, Path]) -> None:
    from datatrove.pipeline.dedup.minhash import MinhashConfig

    raw_dir, tmp_path = fixture_dirs
    out_dir = tmp_path / "processed"
    work_dir = tmp_path / "work"

    _, _, _, stage4 = build_language_stages(
        lang="hi",
        raw_dir=raw_dir,
        work_dir=work_dir,
        out_dir=out_dir,
        minhash_config=MinhashConfig(),
        limit=-1,
        tasks=1,
        executor_kind="local",
        slurm_kwargs={},
    )
    stage4.run()

    out_files = sorted((out_dir / "hi").glob("*.parquet"))
    assert out_files, "expected deduped output parquet under out_dir/hi/"
    table = pa.concat_tables([pq.read_table(f, columns=["text"]) for f in out_files])
    texts = table.column("text").to_pylist()

    # 5 input docs, 1 exact duplicate across the two sources -> 4 survive.
    assert len(texts) == 4
    assert texts.count(_DUPLICATE_TEXT) == 1
    assert set(texts) == {
        _DUPLICATE_TEXT,
        "unique document number one from row a with enough words to shingle",
        "unique document number three from row a with enough words to shingle",
        "unique document number two from row b with enough words to shingle",
    }
