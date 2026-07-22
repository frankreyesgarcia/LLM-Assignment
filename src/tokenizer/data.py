"""Stratified, byte-capped sampling from the pt/es/hi pre-training dataset.

Shared by `scripts/sweep_vocab_size.py` (train + held-out buckets) and
`scripts/compare_baselines.py` (held-out bucket only), so both use the
exact same sampling logic and never accidentally evaluate on data a
tokenizer was trained on.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.sources import VALID_LANGUAGES


def parse_lang_ratios(spec: str) -> dict[str, float]:
    """Parse a `"hi:0.5,pt:0.25,es:0.25"`-style CLI value into a dict.

    Values don't need to sum to 1 -- `oversample_by_ratio` normalizes
    them -- so `"hi:2,pt:1,es:1"` (a 2:1:1 ratio) works just as well.
    """
    ratios: dict[str, float] = {}
    for part in spec.split(","):
        lang, _, weight = part.partition(":")
        lang = lang.strip()
        if not lang or not weight:
            raise ValueError(f"Malformed lang ratio entry {part!r}, expected 'lang:weight'")
        if lang not in VALID_LANGUAGES:
            raise ValueError(f"Unknown language {lang!r} in lang ratio spec, expected one of {VALID_LANGUAGES}")
        ratios[lang] = float(weight)
    return ratios


def oversample_by_ratio(
    docs_by_lang: dict[str, list[str]],
    lang_ratios: dict[str, float],
    target_total_bytes: float | None = None,
) -> dict[str, list[str]]:
    """Build a per-language training doc list at an exact byte ratio by
    *repeating* a language's documents when it doesn't have enough
    unique data to reach its target share, instead of truncating the
    other languages down to match the scarcest one.

    This is the standard multilingual-tokenizer trick (mBERT/XLM-R-style
    upsampling of low-resource languages): duplicating documents just
    linearly scales that language's word-frequency mass for BPE's
    merge-frequency counting. That's safe for a *tokenizer* specifically
    -- BPE training is pure frequency counting with no gradient descent,
    so repeated text carries no memorization/overfitting risk the way it
    would for a model.

    If `target_total_bytes` isn't given, it defaults to the largest
    total for which every language can be built from its *own* full
    natural data at least once (`max` over languages of
    `natural_bytes[lang] / ratio[lang]`) -- i.e. the language that's
    naturally closest to its target share needs zero repeats or
    truncation, and every other language is oversampled (never
    truncated) up to match it.
    """
    total = sum(lang_ratios.values())
    if total <= 0:
        raise ValueError(f"lang_ratios must have a positive sum, got {lang_ratios!r}")
    ratios = {lang: weight / total for lang, weight in lang_ratios.items() if weight > 0}

    natural_bytes: dict[str, int] = {}
    for lang in ratios:
        available = docs_by_lang.get(lang) or []
        if not available:
            raise ValueError(f"No documents available for language {lang!r}, cannot honor lang_ratios")
        natural_bytes[lang] = sum(len(t.encode("utf-8")) for t in available)

    if target_total_bytes is None:
        target_total_bytes = max(natural_bytes[lang] / ratios[lang] for lang in ratios)

    oversampled: dict[str, list[str]] = {}
    for lang, ratio in ratios.items():
        available = docs_by_lang[lang]
        target_bytes = target_total_bytes * ratio
        collected: list[str] = []
        collected_bytes = 0
        i = 0
        while collected_bytes < target_bytes:
            doc = available[i % len(available)]
            collected.append(doc)
            collected_bytes += len(doc.encode("utf-8"))
            i += 1
        oversampled[lang] = collected
    return oversampled


def stratified_sample(
    repo_id: str,
    config: str,
    split: str,
    buckets: list[tuple[str, float]],
    limit_docs: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Stream `repo_id`/`config` and fill named buckets proportionally.

    `buckets` is a list of `(name, size_mb)` pairs, each capped at
    `size_mb / len(VALID_LANGUAGES)` bytes per language. Documents are
    assigned, per language, to whichever *non-full* bucket is currently
    furthest below its target fill ratio (bytes_so_far / cap) -- e.g.
    `[("train", 500), ("heldout", 50)]` fills both in ~10:1 proportion
    in lockstep, rather than draining "train" to completion first. That
    matters because the pilot-scale dataset (~17 MB total) is *smaller*
    than the 500 MB train cap: a strict fill-in-order scheme would starve
    "heldout" completely (all data goes to "train", nothing left over),
    silently producing an empty/NaN eval sample. Proportional filling
    keeps both buckets non-empty and disjoint regardless of corpus size.
    """
    from datasets import load_dataset

    caps = {name: mb * 1_000_000 / len(VALID_LANGUAGES) for name, mb in buckets}
    bytes_so_far: dict[str, dict[str, int]] = {name: defaultdict(int) for name, _ in buckets}
    docs: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name, _ in buckets}

    ds = load_dataset(repo_id, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit_docs is not None and i >= limit_docs:
            break
        lang = row.get("language")
        text = row.get("text")
        if lang not in VALID_LANGUAGES or not text:
            continue
        nbytes = len(text.encode("utf-8"))

        candidates = [name for name, _ in buckets if bytes_so_far[name][lang] < caps[name]]
        if not candidates:
            continue
        target = min(candidates, key=lambda name: bytes_so_far[name][lang] / caps[name])

        docs[target][lang].append(text)
        bytes_so_far[target][lang] += nbytes

    return {name: dict(d) for name, d in docs.items()}


def iter_texts_from_parquet_files(files: list[Path], limit_docs: int | None = None):
    """Lazily yield the `text` column from a list of parquet files, in
    order, stopping once `limit_docs` non-empty texts have been yielded
    (no cap if `limit_docs` is None). Shared by `stratified_sample_processed`
    (bucketed sampling below) and `scripts/train_tokenizer.py`'s
    `iter_texts_processed` (flat streaming for BPE training) -- both read
    local `scripts/run_dedup_datatrove.py` output the same way, so the
    glob/batch-iteration logic lives in exactly one place.
    """
    import pyarrow.parquet as pq

    n = 0
    for file in files:
        if limit_docs is not None and n >= limit_docs:
            return
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(columns=["text"], batch_size=1000):
            for text in batch.column("text").to_pylist():
                if limit_docs is not None and n >= limit_docs:
                    return
                n += 1
                if text:
                    yield text


def stratified_sample_processed(
    processed_dir: Path,
    buckets: list[tuple[str, float]],
    limit_docs_per_lang: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Like `stratified_sample`, but reads the already-deduped local corpus
    at `processed_dir/{lang}/*.parquet` (scripts/run_dedup_datatrove.py's
    output) instead of streaming a HF dataset -- same proportional
    bucket-filling logic, just over `pyarrow.parquet` batches per language
    instead of one combined `load_dataset` stream with a `language` column.

    No per-source row-splitting is needed here (unlike the pre-dedup raw
    downloads, where hi pulled from two separate physical directories):
    the datatrove pipeline already merges and dedups each language's
    sources into one flat directory, so this just reads it directly.
    """
    caps = {name: mb * 1_000_000 / len(VALID_LANGUAGES) for name, mb in buckets}
    bytes_so_far: dict[str, dict[str, int]] = {name: defaultdict(int) for name, _ in buckets}
    docs: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name, _ in buckets}

    for lang in VALID_LANGUAGES:
        lang_dir = processed_dir / lang
        files = sorted(lang_dir.glob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files found under {lang_dir}")

        for text in iter_texts_from_parquet_files(files, limit_docs_per_lang):
            candidates = [name for name, _ in buckets if bytes_so_far[name][lang] < caps[name]]
            if not candidates:
                break
            target = min(candidates, key=lambda name: bytes_so_far[name][lang] / caps[name])
            docs[target][lang].append(text)
            bytes_so_far[target][lang] += len(text.encode("utf-8"))

    return {name: dict(d) for name, d in docs.items()}
