from src.dedup.exact import ExactDeduper, normalize_for_hash
from src.dedup.minhash import NearDeduper


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
