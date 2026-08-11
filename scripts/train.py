#!/usr/bin/env python3
"""Task 3 — pretrain a GPT-style model on data/pretrain (see
scripts/prepare_pretrain_data.py). Thin CLI wrapper around
src/model/train.py::train_model.

Auto-detects cuda vs. cpu (--device auto, the default) -- the same
command runs unchanged on this CPU-only machine or on a GPU node later.

--flops-budget sizes *how long to train* the model shape given by
--n-layer/--n-head/--n-embd (unaffected by this flag -- see those flags'
own help): given a compute budget in FLOPs and that shape's actual
parameter count, it derives a token count and --max-iters from the FLOPs
accounting identity FLOPs(N, D) = 6*N*D (src/model/scaling.py), instead of
picking max_iters by hand to fit a wall-clock budget (see
scripts/slurm/07_pretrain.sh's git history). It does not pick model shape
for you -- src/model/scaling.py's module docstring explains why (in short:
that needs a compute-optimal N-vs-D scaling law fit to this repo's own
setup, which is out of scope for this flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.gpt import GPT, GPTConfig
from src.model.scaling import describe_budget, max_iters_for_budget
from src.model.train import DEFAULT_VAL_BIN, TrainConfig, train_model
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
            "Total training compute budget in FLOPs, for the model shape given by --n-layer/"
            "--n-head/--n-embd. When set, --max-iters below is ignored -- token count and "
            "max_iters are instead derived from that shape's actual param count and this budget "
            "(src/model/scaling.py)."
        ),
    )
    parser.add_argument(
        "--val-bin",
        default=DEFAULT_VAL_BIN,
        help=f"Validation token file inside --data-dir (default {DEFAULT_VAL_BIN}; see src/model/train.py).",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save a checkpoint (out_dir/ckpt_iter{N}.pt) every this many iterations, on top of "
        "the best-val-loss ckpt.pt above -- for crash recovery and inspecting training dynamics "
        "on a long run. Defaults to --eval-interval (already a per-run-tuned cadence) rather than "
        "a fixed number, which wouldn't generalize across different --max-iters budgets.",
    )
    parser.add_argument(
        "--no-doc-masking",
        dest="doc_masking",
        action="store_false",
        help="Disable intra-document attention masking (plain causal attention across the "
        "whole packed window, the GPT-2/nanoGPT default), for ablation at matched FLOPs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "Log this run to Weights & Biases (https://wandb.ai): train/val loss + perplexity, "
            "lr, grad norm, tokens seen/sec, elapsed time, per-step loss, plus this run's full "
            "config (model shape, hyperparameters, --flops-budget) and param counts. Requires "
            "`wandb login` (or WANDB_API_KEY) unless --wandb-mode offline."
        ),
    )
    parser.add_argument("--wandb-project", default="llm-und-pretrain")
    parser.add_argument("--wandb-entity", default=None, help="W&B team/username; defaults to your account.")
    parser.add_argument("--wandb-run-name", default=None, help="Defaults to a wandb-generated name.")
    parser.add_argument("--wandb-tags", nargs="+", default=None)
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Passed to wandb.init(mode=...). Use 'offline' on cluster nodes without outbound "
        "internet access, then `wandb sync <run-dir>` from a node that has it.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    if args.flops_budget is not None:
        if args.max_iters is not None:
            print("--flops-budget given: ignoring --max-iters (deriving it from the FLOPs budget instead)")
        vocab_size = json.loads((args.data_dir / "meta.json").read_text())["vocab_size"]
        n_params = GPT(
            GPTConfig(
                vocab_size=vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
            )
        ).num_params(non_embedding=False)
        max_iters = max_iters_for_budget(args.flops_budget, n_params, args.batch_size, args.block_size)
        print(
            describe_budget(
                args.flops_budget, n_params, args.n_layer, args.n_head, args.n_embd, args.batch_size, args.block_size
            )
        )
    else:
        max_iters = args.max_iters if args.max_iters is not None else 1000

    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else args.eval_interval

    cfg = TrainConfig(
        data_dir=args.data_dir,
        val_bin=args.val_bin,
        out_dir=args.out_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        max_iters=max_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_interval=checkpoint_interval,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tuple(args.wandb_tags) if args.wandb_tags else None,
        wandb_mode=args.wandb_mode,
        doc_masking=args.doc_masking,
    )
    with tee_to_log(args.out_dir, "train"):
        result = train_model(cfg)
        print(
            f"\nDone: {result['n_params_total']:,} total params, "
            f"{result['tokens_seen']:,} tokens seen, "
            f"final val_loss={result['final_val_loss']:.4f} "
            f"({result['elapsed_s']:.1f}s)"
        )
