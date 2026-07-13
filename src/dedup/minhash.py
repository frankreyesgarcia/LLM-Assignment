"""Etapa 5b — Near dedup via MinHash + LSH (TASK1-PLAN.md).

Granularity: full document (simplest starting point per the plan). Default
params: 128 permutations, Jaccard threshold ~0.85. Dedup is run per language
to reduce cross-language false positives.

Two implementations:
- `NearDeduper`: pure in-memory `datasketch.MinHashLSH`. Fast, simple, but
  its index (buckets + signatures for every kept doc) has to fit in RAM --
  fine for run_pilot.py and tests, not for a full-scale run.
- `SqliteNearDeduper`: same LSH banding math (reuses datasketch's own
  `_optimal_param` to pick (b, r) for the given threshold/num_perm, so it
  has the same false-positive/negative behavior as `NearDeduper`), but
  buckets and signatures live in a SQLite file instead of Python dicts --
  "usar LSH buckets" per the plan's Etapa 5b scale note (sec 5). Memory use
  no longer grows with corpus size; disk I/O trades off against that. This
  is what scripts/run_all_sources.py uses.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from datasketch.lsh import _optimal_param

_WORD_RE = re.compile(r"\S+")
DEFAULT_NUM_PERMUTATIONS = 128
DEFAULT_JACCARD_THRESHOLD = 0.85
DEFAULT_SHINGLE_SIZE = 5  # word n-gram size for shingling


def _shingles(text: str, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < shingle_size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}


def compute_minhash(
    text: str,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> MinHash:
    mh = MinHash(num_perm=num_permutations)
    for shingle in _shingles(text, shingle_size):
        mh.update(shingle.encode("utf-8"))
    return mh


class NearDeduper:
    """Stateful near-dedup index; instantiate one per language."""

    def __init__(
        self,
        num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
        jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
        shingle_size: int = DEFAULT_SHINGLE_SIZE,
    ) -> None:
        self.num_permutations = num_permutations
        self.shingle_size = shingle_size
        self.lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_permutations)
        self._next_id = 0

    def is_duplicate(self, text: str) -> bool:
        mh = compute_minhash(text, self.num_permutations, self.shingle_size)
        if self.lsh.query(mh):
            return True
        key = f"doc-{self._next_id}"
        self._next_id += 1
        self.lsh.insert(key, mh)
        return False

    def __len__(self) -> int:
        return self._next_id


class SqliteNearDeduper:
    """Disk-backed near-dedup index: LSH buckets + MinHash signatures live in
    a SQLite file, not in a Python dict, so RAM use doesn't grow with corpus
    size. Instantiate one per language, pointed at a per-language DB file.

    Writes within one `is_duplicate` batch aren't committed per-call -- call
    `commit()` at a natural checkpoint boundary (after a source finishes).
    An uncommitted write is rolled back if the process dies first, which is
    exactly the behavior scripts/run_all_sources.py wants: a source that
    crashed mid-way leaves no trace here either, matching how its partial
    Parquet output gets discarded on resume.
    """

    def __init__(
        self,
        db_path: str | Path,
        num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
        jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
        shingle_size: int = DEFAULT_SHINGLE_SIZE,
        false_positive_weight: float = 0.5,
        false_negative_weight: float = 0.5,
    ) -> None:
        self.num_permutations = num_permutations
        self.shingle_size = shingle_size
        self.jaccard_threshold = jaccard_threshold
        # Same (b, r) banding datasketch.MinHashLSH would pick for this
        # threshold/num_perm -- keeps false-positive/negative rates in line
        # with NearDeduper for the same params.
        self.b, self.r = _optimal_param(
            jaccard_threshold, num_permutations, false_positive_weight, false_negative_weight
        )
        # Don't hardcode the hashvalues dtype/scheme (datasketch could change
        # its internal representation) -- derive them once from a real
        # MinHash so reconstructing one from stored bytes in is_duplicate()
        # always matches what compute_minhash() actually produced. `scheme`
        # must be passed explicitly when rebuilding a MinHash from raw
        # hashvalues (datasketch >=2.0 refuses to guess it).
        _reference = MinHash(num_perm=num_permutations)
        self._dtype = _reference.hashvalues.dtype
        self._scheme = _reference.scheme
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS signatures ("
            "doc_id INTEGER PRIMARY KEY, hashvalues BLOB NOT NULL, source_row TEXT)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS buckets (band INTEGER NOT NULL, bucket_key BLOB NOT NULL, doc_id INTEGER NOT NULL)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_buckets_band_key ON buckets(band, bucket_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_signatures_source_row ON signatures(source_row)")
        self.conn.commit()
        row = self.conn.execute("SELECT MAX(doc_id) FROM signatures").fetchone()
        self._next_id = (row[0] + 1) if row and row[0] is not None else 0

    def _band_keys(self, hashvalues: np.ndarray) -> list[bytes]:
        return [hashvalues[band * self.r : (band + 1) * self.r].tobytes() for band in range(self.b)]

    def is_duplicate(self, text: str, source_row: str | None = None) -> bool:
        mh = compute_minhash(text, self.num_permutations, self.shingle_size)
        band_keys = self._band_keys(mh.hashvalues)

        candidate_ids: set[int] = set()
        for band, key in enumerate(band_keys):
            rows = self.conn.execute(
                "SELECT doc_id FROM buckets WHERE band = ? AND bucket_key = ?", (band, key)
            ).fetchall()
            candidate_ids.update(r[0] for r in rows)

        for doc_id in candidate_ids:
            row = self.conn.execute("SELECT hashvalues FROM signatures WHERE doc_id = ?", (doc_id,)).fetchone()
            if row is None:
                continue
            other = MinHash(
                num_perm=self.num_permutations,
                hashvalues=np.frombuffer(row[0], dtype=self._dtype),
                scheme=self._scheme,
            )
            if mh.jaccard(other) >= self.jaccard_threshold:
                return True

        doc_id = self._next_id
        self._next_id += 1
        self.conn.execute("INSERT INTO signatures VALUES (?, ?, ?)", (doc_id, mh.hashvalues.tobytes(), source_row))
        self.conn.executemany(
            "INSERT INTO buckets VALUES (?, ?, ?)", [(band, key, doc_id) for band, key in enumerate(band_keys)]
        )
        return False

    def delete_by_source(self, source_row: str) -> None:
        """Purge entries tagged with `source_row` and commit immediately.

        Needed to close a narrow crash window: if the process dies after
        `commit()` but before checkpoint.json records the source as done,
        the next run treats the source as incomplete and re-streams it --
        but this index would already hold its docs from the prior attempt,
        so every re-processed doc would look like a near-duplicate of
        itself. Call this before reprocessing any source that isn't
        checkpoint-complete (harmless no-op if there's nothing to delete).
        """
        doc_ids = [r[0] for r in self.conn.execute("SELECT doc_id FROM signatures WHERE source_row = ?", (source_row,))]
        if not doc_ids:
            return
        placeholders = ",".join("?" * len(doc_ids))
        self.conn.execute(f"DELETE FROM buckets WHERE doc_id IN ({placeholders})", doc_ids)
        self.conn.execute(f"DELETE FROM signatures WHERE doc_id IN ({placeholders})", doc_ids)
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.close()

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]
