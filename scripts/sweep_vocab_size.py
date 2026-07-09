#!/usr/bin/env python3
"""Task 2 — decide tokenizer vocab size from data, not vibes.

Fertility and compression ratio are inverses of each other and both
improve monotonically with vocab size, so maximizing compression alone
just pushes toward an unbounded vocab -- not a real objective. The
actual tradeoff is compression *per unit of added cost*, and there are
two costs, both computed against an assumed target model `--hidden-dim`
/ `--target-total-params` (swap in the real numbers once the
pre-training model's architecture is decided):

- Param cost: the (weight-tied) embedding/LM-head matrix scales as
  vocab_size * hidden_dim. For a small pre-training model this can be a
  large fraction of total params (GPT-2-small: 768*50257 ~= 38.6M of
  ~124M total, ~30%) -- not negligible like it is for
  Llama/GPT-4-scale models.
- Inference cost: a bigger vocab shortens token sequences (fewer decode
  steps) but makes the LM head softmax matmul (hidden_dim * vocab_size)
  more expensive per step. Both curves are reported, not collapsed into
  one score.

Trains one tokenizer per candidate vocab size on a fixed-size,
per-language-stratified training sample (BPE merge statistics converge
well before you need the full corpus -- this is why GPT-2/Llama train
tokenizers on a subsample, not the whole corpus), evaluates each on a
disjoint held-out sample, and writes a CSV. Run
`scripts/generate_report.py` afterwards to pick a vocab size and render
the charts: it finds the compression curve's elbow with the Kneedle
algorithm (`src/tokenizer/report.py::pick_chosen_vocab_size`) rather
than an asserted "gain < X%" threshold, and reports embedding_param_share
as context against real Llama-3.2 1B/3B precedent rather than an
asserted cost ceiling -- both of those were arbitrary magic numbers in
an earlier version of this script.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.base import VALID_LANGUAGES
from src.tokenizer.data import stratified_sample
from src.tokenizer.eval import per_language_report
from src.tokenizer.train import train_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCAB_SIZES = [8_000, 16_000, 32_000, 50_000, 65_536, 100_000]


def embedding_param_share(vocab_size: int, hidden_dim: int, target_total_params: int) -> float:
    """Fraction of an assumed target model's params spent on the
    (weight-tied) embedding/LM-head matrix."""
    embedding_params = vocab_size * hidden_dim
    return embedding_params / target_total_params


def run_sweep(
    repo_id: str,
    config: str,
    split: str,
    vocab_sizes: list[int],
    train_mb: float,
    heldout_mb: float,
    hidden_dim: int,
    target_total_params: int,
    limit_docs: int | None,
    out_dir: Path,
) -> list[dict]:
    print(f"Streaming samples from {repo_id}/{config} (train<={train_mb}MB, heldout<={heldout_mb}MB)...")
    buckets = stratified_sample(
        repo_id, config, split, [("train", train_mb), ("heldout", heldout_mb)], limit_docs
    )
    train_docs, heldout_docs = buckets["train"], buckets["heldout"]
    for lang in VALID_LANGUAGES:
        print(
            f"  {lang}: train={len(train_docs.get(lang, [])):>6,} docs, "
            f"heldout={len(heldout_docs.get(lang, [])):>6,} docs"
        )
    train_texts = [text for texts in train_docs.values() for text in texts]

    rows: list[dict] = []
    for vocab_size in vocab_sizes:
        print(f"Training vocab_size={vocab_size:,} ...")
        tokenizer = train_tokenizer(train_texts, vocab_size=vocab_size)
        report = per_language_report(tokenizer, heldout_docs)
        share = embedding_param_share(vocab_size, hidden_dim, target_total_params)
        for lang, metrics in report.items():
            rows.append(
                {
                    "vocab_size": vocab_size,
                    "lang": lang,
                    "fertility": metrics["fertility"],
                    "compression_ratio": metrics["compression_ratio"],
                    "embedding_param_share": share,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["vocab_size", "lang", "fertility", "compression_ratio", "embedding_param_share"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    _write_plot(rows, out_dir / "vocab_size_tradeoff.png")
    _print_table(rows)
    return rows


def _write_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    vocab_sizes = sorted({r["vocab_size"] for r in rows})
    langs = sorted({r["lang"] for r in rows if r["lang"] != "overall"})

    fig, (ax_compression, ax_cost) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for lang in langs:
        ys = [next(r["compression_ratio"] for r in rows if r["vocab_size"] == v and r["lang"] == lang) for v in vocab_sizes]
        ax_compression.plot(vocab_sizes, ys, marker="o", label=lang)
    ax_compression.set_ylabel("compression ratio (bytes/token)")
    ax_compression.set_title("Compression vs. vocab size (higher = better, look for the elbow)")
    ax_compression.legend()

    shares = [next(r["embedding_param_share"] for r in rows if r["vocab_size"] == v) for v in vocab_sizes]
    ax_cost.plot(vocab_sizes, shares, marker="o", color="black")
    ax_cost.set_ylabel("embedding param share")
    ax_cost.set_xlabel("vocab size")
    ax_cost.set_title("Embedding+LM-head share of assumed target model params (cost curve)")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Wrote {out_path}")


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"{'vocab_size':>10} {'lang':>8} {'fertility':>10} {'compression':>12} {'embed_share':>12}")
    for r in rows:
        print(
            f"{r['vocab_size']:>10,} {r['lang']:>8} {r['fertility']:>10.3f} "
            f"{r['compression_ratio']:>12.3f} {r['embedding_param_share']:>12.2%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="frank-rg/LLM-Assignment")
    parser.add_argument("--config", default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--vocab-sizes", default=",".join(str(v) for v in DEFAULT_VOCAB_SIZES))
    parser.add_argument("--train-sample-mb", type=float, default=500.0)
    parser.add_argument("--heldout-sample-mb", type=float, default=50.0)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=768,
        help="ONE illustrative anchor for embedding_param_share (default: GPT-2-Small's hidden_dim) "
        "-- no target pre-training model size is decided yet, so this is not a real target; "
        "see scripts/generate_report.py's cost_vs_model_scale chart for the sensitivity across "
        "several real published model scales instead of trusting this single number.",
    )
    parser.add_argument(
        "--target-total-params",
        type=int,
        default=124_000_000,
        help="Paired with --hidden-dim (default: GPT-2-Small's 124M params). Same caveat: illustrative anchor, not a decision.",
    )
    parser.add_argument("--limit-docs", type=int, default=None, help="Debug cap on docs streamed from the source")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer_sweep")
    args = parser.parse_args()

    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    run_sweep(
        repo_id=args.repo_id,
        config=args.config,
        split=args.split,
        vocab_sizes=vocab_sizes,
        train_mb=args.train_sample_mb,
        heldout_mb=args.heldout_sample_mb,
        hidden_dim=args.hidden_dim,
        target_total_params=args.target_total_params,
        limit_docs=args.limit_docs,
        out_dir=args.out_dir,
    )

    # See scripts/run_pilot.py for why: `datasets` streaming leaves
    # background threads that crash normal interpreter teardown.
    sys.stdout.flush()
    os._exit(0)
