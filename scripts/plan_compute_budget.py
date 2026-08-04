#!/usr/bin/env python3
"""Task 3 — print how many tokens / training steps fit inside one or more
FLOPs budgets for a given model shape, without training anything.

This is the same computation scripts/train.py's --flops-budget runs
internally (see src/model/scaling.py) -- exposed standalone so a few
candidate budgets can be compared before committing a GPU job to any of
them. Model shape is an input here, not an output -- src/model/scaling.py's
module docstring explains why this doesn't pick a shape for you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.gpt import GPT, GPTConfig
from src.model.scaling import describe_budget

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
        "--vocab-size",
        type=int,
        default=32000,
        help="Tokenizer vocab size (affects total param count). Default matches this repo's chosen tokenizer.",
    )
    parser.add_argument("--n-layer", type=int, default=4, help="Matches scripts/train.py's default.")
    parser.add_argument("--n-head", type=int, default=4, help="Matches scripts/train.py's default.")
    parser.add_argument("--n-embd", type=int, default=128, help="Matches scripts/train.py's default.")
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
        help="Affects both the model's positional embedding size and the reported max_iters; matches scripts/train.py's default.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    n_params = GPT(
        GPTConfig(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )
    ).num_params(non_embedding=False)
    for flops in args.flops_budget:
        print(describe_budget(flops, n_params, args.n_layer, args.n_head, args.n_embd, args.batch_size, args.block_size))
