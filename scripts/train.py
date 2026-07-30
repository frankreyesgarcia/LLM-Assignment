#!/usr/bin/env python3
"""Task 3 — pretrain a GPT-style model on data/pretrain (see
scripts/prepare_pretrain_data.py). Thin CLI wrapper around
src/model/train.py::train_model, which scripts/run_scaling_sweep.py also
calls directly for its grid of runs.

Auto-detects cuda vs. cpu (--device auto, the default) -- the same
command runs unchanged on this CPU-only machine or on a GPU node later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import TrainConfig, train_model
from src.tokenizer.logging_utils import tee_to_log

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "pretrain")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "train")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        max_iters=args.max_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        device=args.device,
        seed=args.seed,
    )
    with tee_to_log(args.out_dir, "train"):
        result = train_model(cfg)
        print(
            f"\nDone: {result['n_params_non_embed']:,} non-embedding params, "
            f"{result['tokens_seen']:,} tokens seen, "
            f"final val_loss={result['final_val_loss']:.4f} "
            f"({result['elapsed_s']:.1f}s)"
        )
