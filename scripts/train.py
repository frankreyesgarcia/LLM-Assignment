#!/usr/bin/env python3
"""Task 3 — pretrain a GPT-style model on data/pretrain (see
scripts/prepare_pretrain_data.py). Thin CLI wrapper around
src/model/train.py::train_model.

Auto-detects cuda vs. cpu (--device auto, the default) -- the same
command runs unchanged on this CPU-only machine or on a GPU node later.

--flops-budget sizes the run for you: given a compute budget in FLOPs, it
derives model shape (n_layer/n_head/n_embd) and token count (-> max_iters)
from the Kaplan et al. (2020) compute-optimal allocation (see
src/model/scaling.py) instead of reading those from the flags below --
this replaces picking max_iters by hand to fit a wall-clock budget (see
scripts/slurm/07_pretrain.sh's git history) with picking it from a stated
compute budget instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.scaling import describe_plan, kaplan_compute_optimal_run, max_iters_for
from src.model.train import TrainConfig, train_model
from src.tokenizer.logging_utils import tee_to_log

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "pretrain")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "train")
    parser.add_argument(
        "--flops-budget",
        type=float,
        default=None,
        help=(
            "Total training compute budget in FLOPs. When set, --n-layer/--n-head/--n-embd/"
            "--max-iters below are ignored -- model shape and token count are instead derived "
            "from the Kaplan et al. (2020) compute-optimal allocation (src/model/scaling.py), "
            "removing any separate token limit."
        ),
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=None, help="Ignored if --flops-budget is set. Default 4.")
    parser.add_argument("--n-head", type=int, default=None, help="Ignored if --flops-budget is set. Default 4.")
    parser.add_argument("--n-embd", type=int, default=None, help="Ignored if --flops-budget is set. Default 128.")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--max-iters", type=int, default=None, help="Ignored if --flops-budget is set. Default 1000."
    )
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

    if args.flops_budget is not None:
        overridden = [
            name
            for name, val in (("--n-layer", args.n_layer), ("--n-head", args.n_head), ("--n-embd", args.n_embd), ("--max-iters", args.max_iters))
            if val is not None
        ]
        if overridden:
            print(f"--flops-budget given: ignoring {', '.join(overridden)} (deriving them from the Kaplan compute-optimal allocation instead)")
        plan = kaplan_compute_optimal_run(args.flops_budget)
        n_layer, n_head, n_embd = plan["n_layer"], plan["n_head"], plan["n_embd"]
        max_iters = max_iters_for(plan, args.batch_size, args.block_size)
        print("Kaplan-optimal sizing: " + describe_plan(plan, args.batch_size, args.block_size))
    else:
        n_layer = args.n_layer if args.n_layer is not None else 4
        n_head = args.n_head if args.n_head is not None else 4
        n_embd = args.n_embd if args.n_embd is not None else 128
        max_iters = args.max_iters if args.max_iters is not None else 1000

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=args.dropout,
        max_iters=max_iters,
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
