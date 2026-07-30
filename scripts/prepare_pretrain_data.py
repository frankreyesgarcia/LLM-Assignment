#!/usr/bin/env python3
"""Task 3 — tokenize the frank-rg/LLM-Assignment HF dataset into a flat
binary token stream for pretraining.

Why this shape (see the script's PR description / README for the full
explanation): documents are shuffled and split train/val *before*
tokenizing, so no document straddles the split; each doc's tokens are
separated by an EOS token so the model learns "text can restart here"
instead of splicing unrelated articles into one continuous story; the
whole split is then concatenated into one long token stream and written
via `numpy.memmap` as `uint16` (this tokenizer's 32,000 vocab fits, at
half the bytes/token of int32) so training can slice random fixed-length
chunks out of a file that never has to fit in RAM, even once this points
at a larger (non-pilot) corpus.

Note: frank-rg/LLM-Assignment is a *gated* dataset -- pulling it requires
`hf auth login` with an account that's been granted access first (see
`hf auth login`'s device-code flow; this repo's other HF-hitting scripts,
e.g. scripts/train_tokenizer.py, already rely on the same cached token at
~/.cache/huggingface/token).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_docs(repo_id: str, config: str, split: str) -> list[str]:
    from datasets import load_dataset

    # Not streamed: the whole point here is to shuffle + doc-split before
    # tokenizing (see module docstring), which needs the full doc list in
    # memory anyway -- and at this dataset's size (a few thousand docs,
    # tens of MB) that's cheap.
    ds = load_dataset(repo_id, config, split=split)
    return [t for t in ds["text"] if t]


def tokenize_split(texts: list[str], tokenizer, eos_id: int) -> np.ndarray:
    # batch_encode_plus is much faster than calling the tokenizer once per
    # doc -- one Rust-side call over the whole list instead of thousands
    # of Python-level round trips.
    encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]
    n_tokens = sum(len(ids) + 1 for ids in encodings)  # +1 per doc for the EOS separator
    arr = np.empty(n_tokens, dtype=np.uint16)
    pos = 0
    for ids in encodings:
        n = len(ids)
        arr[pos : pos + n] = ids
        arr[pos + n] = eos_id
        pos += n + 1
    return arr


def run(repo_id: str, config: str, split: str, tokenizer_dir: Path, out_dir: Path, val_fraction: float, seed: int) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    assert tokenizer.vocab_size < 2**16, "uint16 packing assumes vocab_size < 65536"
    eos_id = tokenizer.eos_token_id

    print(f"Loading docs from {repo_id}/{config} ({split})...")
    texts = load_docs(repo_id, config, split)
    print(f"{len(texts):,} docs loaded")

    # Shuffle + split at the *document* level, before tokenizing/concatenating
    # -- splitting the flat token stream instead could cut a document in
    # half across train/val and leak its second half into "unseen" eval data.
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": tokenizer.vocab_size, "tokenizer_dir": str(tokenizer_dir), "eos_token_id": eos_id}
    for split_name, idx in [("train", train_idx), ("val", val_idx)]:
        split_texts = [texts[i] for i in idx]
        arr = tokenize_split(split_texts, tokenizer, eos_id)
        out_path = out_dir / f"{split_name}.bin"
        arr.tofile(out_path)
        meta[f"{split_name}_docs"] = len(split_texts)
        meta[f"{split_name}_tokens"] = int(arr.size)
        print(f"{split_name}: {len(split_texts):,} docs -> {arr.size:,} tokens -> {out_path}")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="frank-rg/LLM-Assignment")
    parser.add_argument("--config", default="all", help="HF dataset config to tokenize (pt/es/hi/all)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "pretrain")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        repo_id=args.repo_id,
        config=args.config,
        split=args.split,
        tokenizer_dir=args.tokenizer_dir,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
