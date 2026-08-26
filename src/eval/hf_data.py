"""Shared HF `datasets` loading helper for task modules (Task 4).

Streams when `limit` is given so a small `--limit` doesn't pull an entire
large split to disk first (matters for duarteocarmo/PT-Culture_Data at
105k rows and nvidia/ChatRAG-Hi's 8 configs) -- for small benchmarks
(calame-pt, portugal-basic-qa-ptcore) streaming vs. not makes no real
difference, so one code path handles both.
"""

from __future__ import annotations

import itertools

from datasets import load_dataset


def load_rows(path: str, config: str | None, split: str, limit: int | None) -> list[dict]:
    if limit is not None:
        ds = load_dataset(path, config, split=split, streaming=True)
        return list(itertools.islice(ds, limit))
    ds = load_dataset(path, config, split=split)
    return list(ds)
