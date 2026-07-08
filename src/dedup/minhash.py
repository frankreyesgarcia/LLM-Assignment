"""Etapa 5b — Near dedup via MinHash + LSH (TASK1-PLAN.md).

Granularity: full document (simplest starting point per the plan). Default
params: 128 permutations, Jaccard threshold ~0.85. Dedup is run per language
to reduce cross-language false positives. This single-process LSH index is
meant for the pilot (Fase 1); sharded/distributed dedup for the full run is
a later phase ("Dedup a escala", sec 7).
"""

from __future__ import annotations

import re

from datasketch import MinHash, MinHashLSH

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
