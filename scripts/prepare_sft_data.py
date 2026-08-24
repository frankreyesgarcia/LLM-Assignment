#!/usr/bin/env python3
"""SFT data prep: tokenize+mask registered src/sft/sources/*.py datasets
into a packed binary format for scripts/train_sft.py.

Mirrors scripts/prepare_pretrain_data.py's flat-memmap-stream shape (see
that script's docstring), with two differences:

1. Two parallel streams are written per split, not one: `{split}.bin`
   (uint16 token ids, same as pretraining) and `{split}_labels.bin`
   (int16 -- signed, unlike the token stream, so it can hold the -1
   "ignore this position" sentinel `F.cross_entropy(..., ignore_index=-1)`
   already expects in src/model/gpt.py). A position's label is -1 unless
   it's part of an assistant turn (src/sft/render.py), so training only
   computes loss on what the model actually needs to learn to produce.

2. A third stream, `{split}_starts.bin` (int64), holds the offset of every
   conversation's first token. src/model/train_sft.py samples training
   windows starting at those offsets rather than at an arbitrary token
   index: with uniform offsets, 99.8% of windows begin part-way into a
   conversation, so its assistant tokens get supervised with the user turn
   that prompted them sitting outside the window (measured on this
   corpus: 33.8% of the average window). Sampling on boundaries makes
   that 0% -- the only conversation a window can cut is the last one, and
   that loses the tail of an answer, not the prompt.

3. Each conversation is separated by the tokenizer's EOS token (exactly
   like pretraining separates documents), NOT the EOT token that also
   appears after every individual turn inside a conversation --
   src/model/train.py's doc_masking treats every occurrence of
   meta["eos_token_id"] as a document boundary, and a conversation's
   internal turns must stay in one attention document (the model needs to
   see the user's question while attending over the assistant's answer),
   so only the outermost boundary may use that sentinel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sft.render import render_messages  # noqa: E402 (after sys.path insert)
from src.sft.sources import *  # noqa: F401,F403,E402 -- populates the registry
from src.sft.registry import get_source, list_sources  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_all_examples(source_names: list[str], languages: set[str] | None, limit: int | None):
    for name in source_names:
        source = get_source(name)()
        if languages is not None and not (set(source.languages) & languages):
            continue
        yield from source.load_examples(limit)


def run(
    sources: list[str],
    languages: set[str] | None,
    tokenizer_dir: Path,
    out_dir: Path,
    val_fraction: float,
    limit: int | None,
    seed: int,
    max_example_tokens: int | None = None,
) -> None:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    assert tokenizer.vocab_size < 2**16, "uint16 packing assumes vocab_size < 65536"
    eos_id = tokenizer.eos_token_id

    print(f"Loading + rendering examples from {sources} (languages={languages or 'all'})...")
    examples = []  # list of (token_ids, labels, source, language)
    dropped_too_long: dict[str, int] = {}
    for ex in load_all_examples(sources, languages, limit):
        token_ids, labels = render_messages(tokenizer, ex.messages)
        if not token_ids:
            continue
        # A conversation longer than the window the model will be trained at
        # can never be seen whole, and nothing downstream truncates it: it
        # would be split across windows, so the fragments after the first
        # supervise assistant tokens whose prompt is out of view. Drop it
        # here instead, where it can be counted, rather than silently
        # fragmenting it at training time.
        if max_example_tokens is not None and len(token_ids) > max_example_tokens:
            dropped_too_long[ex.source] = dropped_too_long.get(ex.source, 0) + 1
            continue
        examples.append((token_ids, labels, ex.source, ex.language))
    n_dropped = sum(dropped_too_long.values())
    print(f"{len(examples):,} examples rendered")
    if max_example_tokens is None:
        print(
            "WARNING: no --max-example-tokens set. Conversations longer than the training block_size "
            "will be split across windows, supervising assistant tokens whose prompt is outside the "
            "window. Pass --max-example-tokens <block_size> to drop them instead."
        )
    else:
        print(f"{n_dropped:,} examples dropped as longer than --max-example-tokens {max_example_tokens:,}")

    # Shuffle + split at the example level, before packing -- same reasoning
    # as prepare_pretrain_data.py's doc-level split: packing first could
    # straddle a conversation across train/val.
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, dict] = {}
    per_language: dict[str, dict] = {}
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_dir": str(tokenizer_dir),
        "eos_token_id": eos_id,
        "ignore_index": -1,
        "sources": sources,
        "languages": sorted(languages) if languages else "all",
        "max_example_tokens": max_example_tokens,
        "dropped_too_long": dict(sorted(dropped_too_long.items())),
        "dropped_too_long_total": n_dropped,
    }
    for split_name, idx in [("train", train_idx), ("val", val_idx)]:
        n_tokens = sum(len(examples[i][0]) + 1 for i in idx)  # +1 per example for the EOS separator
        tokens_arr = np.empty(n_tokens, dtype=np.uint16)
        labels_arr = np.empty(n_tokens, dtype=np.int16)
        starts_arr = np.empty(len(idx), dtype=np.int64)
        pos = 0
        # Per-source/per-language accounting: which datasets actually made it
        # into this corpus, and in what proportion. Without it the output is
        # just one flat token stream and a source *name* list -- there'd be
        # nothing to distinguish a source that contributed 200k examples from
        # one that contributed 200, or silently yielded nothing at all.
        # Counted here rather than at load time so it reflects what was
        # really written, after empty renders are dropped.
        for n_written, i in enumerate(idx):
            token_ids, labels, source, language = examples[i]
            n = len(token_ids)
            starts_arr[n_written] = pos
            n_sup = sum(1 for label in labels if label != -1)
            for bucket, key in ((per_source, source), (per_language, language)):
                entry = bucket.setdefault(
                    key, {"train_examples": 0, "val_examples": 0, "train_tokens": 0, "val_tokens": 0,
                          "train_supervised_tokens": 0, "val_supervised_tokens": 0}
                )
                entry[f"{split_name}_examples"] += 1
                entry[f"{split_name}_tokens"] += n
                entry[f"{split_name}_supervised_tokens"] += n_sup
            tokens_arr[pos : pos + n] = token_ids
            labels_arr[pos : pos + n] = labels
            tokens_arr[pos + n] = eos_id
            labels_arr[pos + n] = -1  # the separator itself is never a training target
            pos += n + 1
        tokens_arr.tofile(out_dir / f"{split_name}.bin")
        labels_arr.tofile(out_dir / f"{split_name}_labels.bin")
        starts_arr.tofile(out_dir / f"{split_name}_starts.bin")
        meta[f"{split_name}_examples"] = len(idx)
        meta[f"{split_name}_tokens"] = int(n_tokens)
        n_supervised = int((labels_arr != -1).sum())
        # Conversation lengths, so the caller can see how many examples are
        # longer than the block_size they intend to train at. Nothing here
        # truncates or drops those -- get_sft_batch samples random windows
        # out of the packed stream, so an over-length conversation is split
        # across windows rather than discarded (see this module's docstring).
        lengths = np.array([len(examples[i][0]) for i in idx])
        meta[f"{split_name}_length_percentiles"] = {
            f"p{q}": int(np.percentile(lengths, q)) for q in (50, 75, 90, 95, 99)
        }
        meta[f"{split_name}_length_max"] = int(lengths.max())
        print(
            f"{split_name}: {len(idx):,} examples -> {n_tokens:,} tokens "
            f"({n_supervised:,} supervised, {n_supervised / n_tokens:.1%}) -> {out_dir}/{split_name}.bin"
        )
        print(
            f"{split_name}: conversation length p50={np.percentile(lengths, 50):.0f} "
            f"p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} "
            f"max={lengths.max():,} tokens"
        )

    meta["per_source"] = dict(sorted(per_source.items()))
    meta["per_language"] = dict(sorted(per_language.items()))
    print("\nPer source (train split):")
    for name, entry in meta["per_source"].items():
        share = entry["train_tokens"] / meta["train_tokens"] if meta["train_tokens"] else 0.0
        print(f"  {name:<45} {entry['train_examples']:>9,} examples  {entry['train_tokens']:>13,} tokens  ({share:5.1%})")
    if dropped_too_long:
        print("Dropped as too long, per source:")
        for name, count in sorted(dropped_too_long.items()):
            print(f"  {name:<45} {count:>9,}")
    print("Per language (train split):")
    for name, entry in meta["per_language"].items():
        share = entry["train_tokens"] / meta["train_tokens"] if meta["train_tokens"] else 0.0
        print(f"  {name:<45} {entry['train_examples']:>9,} examples  {entry['train_tokens']:>13,} tokens  ({share:5.1%})")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=f"Registered src/sft/sources names to use (default: all). Available: {list_sources()}",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Restrict to these language codes (e.g. pt es hi); default: no filter.",
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sft")
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument(
        "--max-example-tokens",
        type=int,
        default=None,
        help="Drop conversations longer than this many tokens. Set it to the block_size the model will be "
        "trained at (the base checkpoint's, e.g. 1024): a longer conversation cannot be seen whole and would "
        "instead be split across training windows. Default: no filter, with a warning.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples loaded per source (before language filtering within a source), for smoke-testing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        sources=args.sources or list_sources(),
        languages=set(args.languages) if args.languages else None,
        tokenizer_dir=args.tokenizer_dir,
        out_dir=args.out_dir,
        val_fraction=args.val_fraction,
        limit=args.limit,
        seed=args.seed,
        max_example_tokens=args.max_example_tokens,
    )
