#!/usr/bin/env python3
"""Task 2 — train and save the final byte-level BPE tokenizer.

Run `scripts/sweep_vocab_size.py` first to pick `--vocab-size` from data
(see that script's docstring for the decision rule) rather than
guessing. This script trains on the full stream (bounded by
`--limit-docs` for debugging), saves a `transformers`-loadable artifact,
then re-streams a small sample per language and prints a fertility/
compression report so the run log doubles as a quality snapshot.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sources import VALID_LANGUAGES
from src.tokenizer.data import iter_texts_from_parquet_files, stratified_sample, stratified_sample_processed
from src.tokenizer.eval import per_language_report
from src.tokenizer.logging_utils import tee_to_log
from src.tokenizer.train import save_pretrained, train_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def iter_texts(repo_id: str, config: str, split: str, limit_docs: int | None):
    from datasets import load_dataset

    ds = load_dataset(repo_id, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit_docs is not None and i >= limit_docs:
            break
        text = row.get("text")
        if text:
            yield text


def iter_texts_processed(processed_dir: Path, limit_docs_per_lang: int | None):
    """Same shape as `iter_texts`, but reads the already-deduped local
    corpus at `processed_dir/{lang}/*.parquet` (scripts/run_dedup_datatrove.py's
    output) instead of streaming a HF dataset (see `--processed-dir`)."""
    for lang in sorted(VALID_LANGUAGES):
        lang_dir = processed_dir / lang
        files = sorted(lang_dir.glob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files found under {lang_dir}")
        yield from iter_texts_from_parquet_files(files, limit_docs_per_lang)


def run(
    repo_id: str,
    config: str,
    split: str,
    vocab_size: int,
    limit_docs: int | None,
    eval_sample_mb: float,
    out_dir: Path,
    processed_dir: Path | None = None,
) -> None:
    if processed_dir is not None:
        print(
            f"Training vocab_size={vocab_size:,} on the deduped corpus under {processed_dir} "
            f"(limit_docs_per_lang={limit_docs})..."
        )
        tokenizer = train_tokenizer(iter_texts_processed(processed_dir, limit_docs), vocab_size=vocab_size)
    else:
        print(f"Training vocab_size={vocab_size:,} on {repo_id}/{config} (limit_docs={limit_docs})...")
        tokenizer = train_tokenizer(iter_texts(repo_id, config, split, limit_docs), vocab_size=vocab_size)
    fast_tokenizer = save_pretrained(tokenizer, out_dir)
    print(f"Saved tokenizer to {out_dir} (vocab_size={len(fast_tokenizer):,})")

    print(f"Re-streaming a {eval_sample_mb}MB sample per language for a quality snapshot...")
    if processed_dir is not None:
        buckets = stratified_sample_processed(processed_dir, [("eval", eval_sample_mb)])
    else:
        buckets = stratified_sample(repo_id, config, split, [("eval", eval_sample_mb)])
    report = per_language_report(fast_tokenizer, buckets["eval"])
    print()
    print(f"{'lang':>8} {'fertility':>10} {'compression':>12}")
    for lang, metrics in report.items():
        print(f"{lang:>8} {metrics['fertility']:>10.3f} {metrics['compression_ratio']:>12.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="frank-rg/LLM-Assignment")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug cap on docs streamed for training")
    parser.add_argument("--eval-sample-mb", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Train on the deduped local corpus under this dir "
        "(scripts/run_dedup_datatrove.py's --out-dir, {lang}/*.parquet) instead of streaming "
        "--repo-id from the Hub. With this set, --limit-docs caps docs *per language* rather than total.",
    )
    args = parser.parse_args()

    with tee_to_log(args.out_dir, "train_tokenizer"):
        run(
            repo_id=args.repo_id,
            config=args.config,
            split=args.split,
            vocab_size=args.vocab_size,
            limit_docs=args.limit_docs,
            eval_sample_mb=args.eval_sample_mb,
            out_dir=args.out_dir,
            processed_dir=args.processed_dir,
        )

    # See scripts/run_pilot.py for why: `datasets` streaming leaves
    # background threads that crash normal interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
