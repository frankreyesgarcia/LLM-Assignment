from src.dedup.exact import ExactDeduper, normalize_for_hash
from src.dedup.minhash import NearDeduper, SqliteNearDeduper


# --- exact.py ---


def test_exact_dedup_normalizes_case_and_whitespace():
    assert normalize_for_hash("Hello   World") == normalize_for_hash("hello world")


def test_exact_deduper_flags_second_identical_doc():
    deduper = ExactDeduper()
    assert deduper.is_duplicate("some document text") is False
    assert deduper.is_duplicate("some document text") is True
    assert len(deduper) == 1


def test_exact_deduper_ignores_case_and_whitespace_differences():
    deduper = ExactDeduper()
    assert deduper.is_duplicate("Some   Document   Text") is False
    assert deduper.is_duplicate("some document text") is True


def test_exact_deduper_keeps_distinct_docs():
    deduper = ExactDeduper()
    assert deduper.is_duplicate("document A") is False
    assert deduper.is_duplicate("document B") is False
    assert len(deduper) == 2


# --- minhash.py ---

# Long, non-repetitive text (a repeated short phrase would collapse into a
# tiny shingle *set*, since shingles() dedupes -- that starves MinHash/LSH
# of signal near the threshold). LSH also has real false negatives close to
# its threshold (it's a probabilistic band/row scheme, not a hard cutoff),
# so the near-copy below only changes one word to keep true Jaccard ~0.98.
_BASE_TEXT = (
    "este es un documento de prueba suficientemente largo para generar muchos shingles distintos "
    "y poder comparar la similitud de jaccard con un margen razonable frente al ruido propio "
    "de la aproximacion por minhash cuando el conjunto de shingles es pequeno o repetitivo "
    "y ademas incluye mas palabras variadas para que el conjunto total sea considerablemente mas grande todavia"
)


def test_near_deduper_flags_near_identical_text():
    deduper = NearDeduper(jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    near_copy = _BASE_TEXT + " mas."
    assert deduper.is_duplicate(near_copy) is True


def test_near_deduper_keeps_distinct_docs():
    deduper = NearDeduper(jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    unrelated = "completely different content about a totally unrelated topic in another domain"
    assert deduper.is_duplicate(unrelated) is False
    assert len(deduper) == 2


# --- SqliteNearDeduper: same behavior as NearDeduper, disk-backed instead of in-memory ---


def test_sqlite_near_deduper_flags_near_identical_text(tmp_path):
    deduper = SqliteNearDeduper(tmp_path / "near.sqlite3", jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    near_copy = _BASE_TEXT + " mas."
    assert deduper.is_duplicate(near_copy) is True
    deduper.close()


def test_sqlite_near_deduper_keeps_distinct_docs(tmp_path):
    deduper = SqliteNearDeduper(tmp_path / "near.sqlite3", jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    unrelated = "completely different content about a totally unrelated topic in another domain"
    assert deduper.is_duplicate(unrelated) is False
    assert len(deduper) == 2
    deduper.close()


def test_sqlite_near_deduper_uncommitted_writes_roll_back_on_crash(tmp_path):
    db_path = tmp_path / "near.sqlite3"
    deduper = SqliteNearDeduper(db_path, jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    # No commit() -- close the raw connection directly (bypassing our
    # close(), which commits) to simulate an abrupt crash. sqlite3 discards
    # an uncommitted transaction on close, same as it would on process
    # death; relying on `del`+GC here would be a timing-dependent test, not
    # a real crash simulation.
    deduper.conn.close()

    reopened = SqliteNearDeduper(db_path, jaccard_threshold=0.8)
    assert len(reopened) == 0
    assert reopened.is_duplicate(_BASE_TEXT) is False
    reopened.commit()
    reopened.close()


def test_sqlite_near_deduper_committed_writes_persist_across_reopen(tmp_path):
    db_path = tmp_path / "near.sqlite3"
    deduper = SqliteNearDeduper(db_path, jaccard_threshold=0.8)
    assert deduper.is_duplicate(_BASE_TEXT) is False
    deduper.commit()
    del deduper

    reopened = SqliteNearDeduper(db_path, jaccard_threshold=0.8)
    assert len(reopened) == 1
    near_copy = _BASE_TEXT + " mas."
    assert reopened.is_duplicate(near_copy) is True
    reopened.close()


def test_sqlite_near_deduper_matches_optimal_param_banding(tmp_path):
    deduper = SqliteNearDeduper(tmp_path / "near.sqlite3", num_permutations=128, jaccard_threshold=0.85)
    assert (deduper.b, deduper.r) == (8, 16)
    deduper.close()


def test_sqlite_near_deduper_delete_by_source_purges_only_that_source(tmp_path):
    deduper = SqliteNearDeduper(tmp_path / "near.sqlite3", jaccard_threshold=0.8)
    unrelated = "completely different content about a totally unrelated topic in another domain"
    assert deduper.is_duplicate(_BASE_TEXT, source_row="row-a") is False
    assert deduper.is_duplicate(unrelated, source_row="row-b") is False
    deduper.commit()
    assert len(deduper) == 2

    deduper.delete_by_source("row-a")
    assert len(deduper) == 1
    # row-a's doc is gone from the index -- inserting it again is not a duplicate.
    assert deduper.is_duplicate(_BASE_TEXT, source_row="row-a") is False
    # row-b's doc is untouched -- its near-copy is still flagged as a duplicate.
    near_copy_of_unrelated = unrelated + " extra"
    assert deduper.is_duplicate(near_copy_of_unrelated, source_row="row-b") is True
    deduper.close()


def test_sqlite_near_deduper_delete_by_source_on_empty_source_is_noop(tmp_path):
    deduper = SqliteNearDeduper(tmp_path / "near.sqlite3", jaccard_threshold=0.8)
    deduper.delete_by_source("never-seen-row")  # must not raise
    assert len(deduper) == 0
    deduper.close()
