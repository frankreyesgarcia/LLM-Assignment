#!/usr/bin/env python3
"""Task 2 — how does the trained tokenizer stack up against existing ones?

One-shot comparison (not part of the vocab-size sweep loop, which trains
many candidate sizes of *our* tokenizer). All rows are evaluated on the
*same disjoint held-out bucket*, produced the same way as
`scripts/sweep_vocab_size.py`'s split. This matters: the baselines
(gpt2, EuroLLM, ...) were never trained on this corpus, so comparing them
against a version of "ours" that *was* trained on the eval text would be
an apples-to-oranges, in-sample-vs-zero-shot comparison biased in our
favor. So the "ours" row here is a fresh tokenizer trained only on the
train bucket at `--vocab-size`, not the final shipped artifact in
`artifacts/tokenizer/` (which is trained on the full corpus, by design,
to squeeze out every byte of the real training data). Pass
`--tokenizer-dir` to *additionally* report the shipped artifact, clearly
labeled as in-sample/not-comparable, for informational purposes only.

Baselines (all verified ungated on the HF Hub):
- openai-community/gpt2 -- classic byte-level BPE, English-centric;
  expect it to lose on hi, useful pt/es reference.
- utter-project/EuroLLM-1.7B -- European-multilingual, same org as the
  EuroWeb dataset source already used in this pipeline; pt/es reference,
  hi likely weak/absent.
- sarvamai/sarvam-1 -- reputable Hindi-capable multilingual tokenizer
  (11 languages incl. Hindi); the Hindi-quality reference point.
- poolside/Laguna-M.1 -- frontier-model-scale tokenizer, as a "what do
  the big labs' vocabularies buy you" reference point.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.data import stratified_sample
from src.tokenizer.eval import per_language_report
from src.tokenizer.train import save_pretrained, train_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

BASELINES = {
    "gpt2": "openai-community/gpt2",
    "eurollm-1.7b": "utter-project/EuroLLM-1.7B",
    "sarvam-1": "sarvamai/sarvam-1",
    "laguna-m1": "poolside/Laguna-M.1",
}


def run_comparison(
    repo_id: str,
    config: str,
    split: str,
    train_mb: float,
    heldout_mb: float,
    vocab_size: int,
    limit_docs: int | None,
    tokenizer_dir: Path | None,
    out_dir: Path,
) -> list[dict]:
    from transformers import AutoTokenizer

    print(f"Streaming train+held-out samples from {repo_id}/{config} (train<={train_mb}MB, heldout<={heldout_mb}MB)...")
    buckets = stratified_sample(repo_id, config, split, [("train", train_mb), ("heldout", heldout_mb)], limit_docs)
    train_docs, docs = buckets["train"], buckets["heldout"]
    for lang, texts in docs.items():
        print(f"  {lang}: train={len(train_docs.get(lang, [])):,} docs, heldout={len(texts):,} docs")

    tokenizers_to_eval: dict[str, tuple[str, Any]] = {}
    for name, repo in BASELINES.items():
        print(f"Loading baseline '{name}' ({repo})...")
        tokenizers_to_eval[name] = (repo, AutoTokenizer.from_pretrained(repo))

    print(f"Training a fresh vocab_size={vocab_size:,} tokenizer on the train bucket for a fair 'ours' row...")
    train_texts = [text for texts in train_docs.values() for text in texts]
    ours = save_pretrained(train_tokenizer(train_texts, vocab_size=vocab_size), out_dir / "_comparison_only_tokenizer")
    tokenizers_to_eval["ours"] = (f"trained here, vocab_size={vocab_size}", ours)

    if tokenizer_dir is not None:
        print(f"Loading shipped artifact ({tokenizer_dir}) -- NOTE: trained on this whole corpus, IN-SAMPLE, not comparable...")
        tokenizers_to_eval["ours-shipped (in-sample, NOT comparable)"] = (
            str(tokenizer_dir),
            AutoTokenizer.from_pretrained(str(tokenizer_dir)),
        )

    rows: list[dict] = []
    for name, (repo, tokenizer) in tokenizers_to_eval.items():
        report = per_language_report(tokenizer, docs)
        for lang, metrics in report.items():
            rows.append(
                {
                    "tokenizer": name,
                    "repo_id": repo,
                    "vocab_size": len(tokenizer),
                    "lang": lang,
                    "fertility": metrics["fertility"],
                    "compression_ratio": metrics["compression_ratio"],
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tokenizer", "repo_id", "vocab_size", "lang", "fertility", "compression_ratio"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    _print_markdown_table(rows)
    return rows


def _print_markdown_table(rows: list[dict]) -> None:
    print()
    print("| tokenizer | vocab_size | lang | fertility | compression_ratio |")
    print("|---|---:|---|---:|---:|")
    for r in rows:
        print(
            f"| {r['tokenizer']} | {r['vocab_size']:,} | {r['lang']} | "
            f"{r['fertility']:.3f} | {r['compression_ratio']:.3f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="frank-rg/LLM-Assignment")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-sample-mb", type=float, default=500.0, help="Train bucket for the fresh 'ours' tokenizer")
    parser.add_argument("--heldout-sample-mb", type=float, default=50.0, help="Held-out bucket all tokenizers are evaluated on")
    parser.add_argument("--vocab-size", type=int, default=16_000, help="Vocab size for the fresh 'ours' tokenizer (match your sweep-chosen size)")
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug cap on docs streamed from the source")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional: also report the shipped artifact (from train_tokenizer.py), clearly labeled as in-sample/not comparable",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer_sweep")
    args = parser.parse_args()

    run_comparison(
        repo_id=args.repo_id,
        config=args.config,
        split=args.split,
        train_mb=args.train_sample_mb,
        heldout_mb=args.heldout_sample_mb,
        vocab_size=args.vocab_size,
        limit_docs=args.limit_docs,
        tokenizer_dir=args.tokenizer_dir,
        out_dir=args.out_dir,
    )

    # See scripts/run_pilot.py for why: `datasets` streaming leaves
    # background threads that crash normal interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
