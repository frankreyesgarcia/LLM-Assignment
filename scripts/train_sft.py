#!/usr/bin/env python3
"""SFT — fine-tune a pretrained checkpoint on data/sft (see
scripts/prepare_sft_data.py). Thin CLI wrapper around
src/model/train_sft.py::train_sft_model, mirroring scripts/train.py's shape.

--init-from is required: SFT always continues from a pretrained checkpoint
(scripts/train.py's ckpt.pt/ckpt_final.pt/ckpt_last.pt), never random init.
--lr defaults an order of magnitude below pretraining's (5e-5 vs. 3e-4) --
the conventional starting point for continuing training on a much smaller,
higher-quality dataset without forgetting what pretraining learned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.train import AMP_DTYPE_CHOICES
from src.model.train_sft import SFTTrainConfig, train_sft_model
from src.tokenizer.logging_utils import tee_to_log

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "sft")
    parser.add_argument("--init-from", type=Path, required=True, help="Pretrained checkpoint to fine-tune (required).")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "sft")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Defaults to --eval-interval.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--no-doc-masking",
        dest="doc_masking",
        action="store_false",
        help="Disable per-conversation attention masking -- see src/model/train_sft.py's docstring for why this "
        "is normally required, not just a performance nicety, given how prepare_sft_data.py packs conversations.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", default="auto", choices=list(AMP_DTYPE_CHOICES))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="llm-und-sft")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    cfg = SFTTrainConfig(
        data_dir=args.data_dir,
        init_from=args.init_from,
        out_dir=args.out_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else args.eval_interval,
        resume=args.resume,
        doc_masking=args.doc_masking,
        device=args.device,
        amp_dtype=args.amp_dtype,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tuple(args.wandb_tags) if args.wandb_tags else None,
        wandb_mode=args.wandb_mode,
    )
    with tee_to_log(args.out_dir, "train_sft"):
        result = train_sft_model(cfg)
        print(f"\nDone: final val_loss={result['final_val_loss']:.4f} ({result['elapsed_s']:.1f}s)")
