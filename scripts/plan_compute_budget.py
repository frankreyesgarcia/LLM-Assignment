#!/usr/bin/env python3
"""Task 3 — print the Kaplan et al. (2020) compute-optimal sizing (model
shape + token count) for one or more FLOPs budgets, without training
anything.

This is the same computation scripts/train.py's --flops-budget runs
internally (see src/model/scaling.py) -- exposed standalone so a few
candidate budgets can be compared before committing a GPU job to any of
them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.scaling import describe_plan, kaplan_compute_optimal_run

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--flops-budget",
        type=float,
        nargs="+",
        required=True,
        help="One or more compute budgets in FLOPs, e.g. --flops-budget 1e19 1e20 1e21.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Only affects the reported max_iters (tokens -> steps conversion); matches scripts/train.py's default.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Only affects the reported max_iters (tokens -> steps conversion); matches scripts/train.py's default.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    for flops in args.flops_budget:
        plan = kaplan_compute_optimal_run(flops)
        print(describe_plan(plan, args.batch_size, args.block_size))
