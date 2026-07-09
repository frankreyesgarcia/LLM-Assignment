"""Stratified, byte-capped sampling from the pt/es/hi pre-training dataset.

Shared by `scripts/sweep_vocab_size.py` (train + held-out buckets) and
`scripts/compare_baselines.py` (held-out bucket only), so both use the
exact same sampling logic and never accidentally evaluate on data a
tokenizer was trained on.
"""

from __future__ import annotations

from collections import defaultdict

from src.ingest.base import VALID_LANGUAGES


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
