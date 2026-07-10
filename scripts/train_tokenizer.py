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

from src.tokenizer.data import stratified_sample
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


def run(
    repo_id: str,
    config: str,
    split: str,
    vocab_size: int,
    limit_docs: int | None,
    eval_sample_mb: float,
    out_dir: Path,
) -> None:
    print(f"Training vocab_size={vocab_size:,} on {repo_id}/{config} (limit_docs={limit_docs})...")
    tokenizer = train_tokenizer(iter_texts(repo_id, config, split, limit_docs), vocab_size=vocab_size)
    fast_tokenizer = save_pretrained(tokenizer, out_dir)
    print(f"Saved tokenizer to {out_dir} (vocab_size={len(fast_tokenizer):,})")

    print(f"Re-streaming a {eval_sample_mb}MB sample per language for a quality snapshot...")
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
        )

    # See scripts/run_pilot.py for why: `datasets` streaming leaves
    # background threads that crash normal interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
