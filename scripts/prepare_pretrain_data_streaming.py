#!/usr/bin/env python3
"""Streaming variant of scripts/prepare_pretrain_data.py, for corpora too
large to load into a single Python list of strings (that script's approach
-- fine for the few-thousand-doc pilot dataset it was written for, not for
the ~1.3TB andre15silva/pretrain-pt-es-hi final corpus).

Two differences from prepare_pretrain_data.py:

1. Reuses an already-materialized local copy of the corpus before ever
   contacting HF. andre15silva/pretrain-pt-es-hi is itself a copy of this
   repo's own scripts/build_final_dataset.py output (same shard layout,
   same pt/es/hi sizes) -- so if $PROJECT_STORAGE/data/final already has
   it (produced earlier in this same pipeline, before ever uploading),
   there's no reason to re-download it. Only falls back to
   huggingface_hub.snapshot_download (itself idempotent/resumable, see
   scripts/download_sources.py) if that local copy isn't present.

2. Splits train/val by *shard*, not by shuffling every doc into one
   in-memory list first. build_final_dataset.py's shuffle_into_shards
   already globally shuffles documents across shards before writing them
   (see its docstring), so reserving whole shards for val is equivalent to
   a random per-doc split without needing the full corpus in RAM.
   Tokenization is two-pass per split (count tokens, then allocate the
   memmap and write) -- the same shape nanoGPT's own data-prep scripts
   use, since the total token count isn't known until it's tokenized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "andre15silva/pretrain-pt-es-hi"
DEFAULT_LANGUAGES = ["pt", "es", "hi"]


def ensure_local_corpus(
    repo_id: str,
    languages: list[str],
    local_dir: Path,
    project_final_dir: Path | None,
    max_workers: int,
) -> Path:
    if project_final_dir is not None and all(
        any((project_final_dir / lang).glob("*.parquet")) for lang in languages
    ):
        print(f"Using existing local corpus at {project_final_dir} (skipping HF entirely)")
        return project_final_dir

    from huggingface_hub import snapshot_download

    print(f"No usable local corpus found under {project_final_dir} -- fetching from {repo_id}")
    for lang in languages:
        print(f"--- {lang} ---", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{lang}/*.parquet",
            local_dir=str(local_dir),
            max_workers=max_workers,
        )
    return local_dir


def list_shards(corpus_dir: Path, lang: str) -> list[Path]:
    shards = sorted((corpus_dir / lang).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {corpus_dir / lang}")
    return shards


def reserve_val_shards(shards: list[Path], val_shards: int) -> tuple[list[Path], list[Path]]:
    val_shards = min(val_shards, max(1, len(shards) // 10))  # never reserve more than ~10%
    return shards[val_shards:], shards[:val_shards]


def iter_texts(shards: list[Path], batch_rows: int):
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text:
                    yield text


def count_tokens(shards: list[Path], tokenizer, batch_rows: int) -> tuple[int, int]:
    n_docs, n_tokens = 0, 0
    texts: list[str] = []
    for text in iter_texts(shards, batch_rows):
        texts.append(text)
        if len(texts) >= batch_rows:
            for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
                n_docs += 1
                n_tokens += len(ids) + 1  # +1 for the EOS separator
            texts.clear()
    if texts:
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            n_docs += 1
            n_tokens += len(ids) + 1
        texts.clear()
    return n_docs, n_tokens


def write_tokens(shards: list[Path], tokenizer, eos_id: int, batch_rows: int, out_path: Path, n_tokens: int) -> None:
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    pos = 0
    texts: list[str] = []

    def flush() -> None:
        nonlocal pos
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            n = len(ids)
            arr[pos : pos + n] = ids
            arr[pos + n] = eos_id
            pos += n + 1
        texts.clear()

    for text in iter_texts(shards, batch_rows):
        texts.append(text)
        if len(texts) >= batch_rows:
            flush()
    if texts:
        flush()
    assert pos == n_tokens, f"token count drifted between passes ({pos} written vs {n_tokens} counted)"
    arr.flush()


def run(
    repo_id: str,
    languages: list[str],
    local_dir: Path,
    project_final_dir: Path | None,
    tokenizer_dir: Path,
    out_dir: Path,
    val_shards_per_lang: int,
    batch_rows: int,
    max_workers: int,
) -> None:
    from transformers import AutoTokenizer

    corpus_dir = ensure_local_corpus(repo_id, languages, local_dir, project_final_dir, max_workers)

    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    assert tokenizer.vocab_size < 2**16, "uint16 packing assumes vocab_size < 65536"
    eos_id = tokenizer.eos_token_id

    train_shards, val_shards = [], []
    for lang in languages:
        shards = list_shards(corpus_dir, lang)
        lang_train, lang_val = reserve_val_shards(shards, val_shards_per_lang)
        print(f"{lang}: {len(shards)} shards -> {len(lang_train)} train / {len(lang_val)} val")
        train_shards += lang_train
        val_shards += lang_val

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": tokenizer.vocab_size, "tokenizer_dir": str(tokenizer_dir), "eos_token_id": eos_id}
    for split_name, shards in [("train", train_shards), ("val", val_shards)]:
        print(f"Counting {split_name} tokens across {len(shards)} shards...")
        n_docs, n_tokens = count_tokens(shards, tokenizer, batch_rows)
        print(f"{split_name}: {n_docs:,} docs, {n_tokens:,} tokens -- writing {out_dir / f'{split_name}.bin'}")
        write_tokens(shards, tokenizer, eos_id, batch_rows, out_dir / f"{split_name}.bin", n_tokens)
        meta[f"{split_name}_docs"] = n_docs
        meta[f"{split_name}_tokens"] = n_tokens

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--languages", nargs="*", default=DEFAULT_LANGUAGES)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=REPO_ROOT / "data" / "pretrain_source",
        help="Download target if no usable local corpus is found (snapshot_download, resumable)",
    )
    parser.add_argument(
        "--project-final-dir",
        type=Path,
        default=None,
        help="Check here first for an already-materialized copy (e.g. $PROJECT_STORAGE/data/final "
        "from this repo's own scripts/build_final_dataset.py) before touching HF at all",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "pretrain_full")
    parser.add_argument(
        "--val-shards-per-lang",
        type=int,
        default=2,
        help="Whole shards reserved for val per language (shards are already globally shuffled, so no need to re-shuffle docs)",
    )
    parser.add_argument("--batch-rows", type=int, default=2000, help="Rows per tokenizer batch call")
    parser.add_argument("--max-workers", type=int, default=16, help="snapshot_download concurrency, if a download is needed")
    args = parser.parse_args()

    run(
        repo_id=args.repo_id,
        languages=args.languages,
        local_dir=args.local_dir,
        project_final_dir=args.project_final_dir,
        tokenizer_dir=args.tokenizer_dir,
        out_dir=args.out_dir,
        val_shards_per_lang=args.val_shards_per_lang,
        batch_rows=args.batch_rows,
        max_workers=args.max_workers,
    )
