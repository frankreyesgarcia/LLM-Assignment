#!/usr/bin/env python3
"""Task 4 — evaluate a checkpoint across the PT/ES/HI benchmarks below.

Two prompt formats, selected by --mode:

- pretrain:  plain ICL/zero-shot text prompts, for checkpoints that have
  never seen the chat template (scripts/train.py output straight off
  data/pretrain).
- posttrain: chat-template prompts (src/tokenizer/train.py::CHAT_TEMPLATE),
  answers read back out of the generated assistant turn -- for checkpoints
  that have been instruction-tuned.

Benchmarks are modular (src/eval/tasks/): PT/HI ones are implemented
natively against a shared Task interface (src/eval/tasks/base.py), and
Spanish delegates to the real lm-evaluation-harness spanish_bench suite
(src/eval/tasks/spanish_bench.py) via an adapter (src/eval/lm_eval_adapter.py)
instead of reimplementing its ~30 task variants. --list-tasks shows the
native set; any other --tasks name is passed straight through to lm-eval,
so e.g. --tasks xnli_es or --tasks spanish_bench both work.

Examples:

  # Native PT/HI benchmarks, zero-shot ICL, on a pretrain checkpoint
  python scripts/run_eval.py --ckpt runs/train/ckpt.pt --mode pretrain \\
      --tasks calame_pt,portugal_basic_qa,pt_culture,chatrag_hi --num-fewshot 0

  # Full Spanish suite, chat-template mode, on an SFT checkpoint
  python scripts/run_eval.py --ckpt runs/sft/ckpt.pt --mode posttrain \\
      --tasks spanish_bench --num-fewshot 0 --limit 50

  # ALBA with an LLM judge (needs ANTHROPIC_API_KEY); omit --judge to still
  # generate + save outputs without a judge_score.
  python scripts/run_eval.py --ckpt runs/train/ckpt.pt --mode pretrain \\
      --tasks alba --judge anthropic --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.eval.tasks  # noqa: F401 -- import side effect populates the registry
from src.eval.core import AMP_DTYPE_CHOICES
from src.eval.judge import build_judge
from src.eval.model_adapter import EvalModel
from src.eval.registry import get_task, list_tasks
from src.eval.runner import run_task
from src.eval.types import EvalMode, TaskResult

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, help="Checkpoint saved by scripts/train.py (e.g. runs/train/ckpt.pt)")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Native tasks (--list-tasks) run through this repo's own "
        "runner; any other name is passed through to lm-evaluation-harness (e.g. spanish_bench, xnli_es).",
    )
    parser.add_argument("--mode", choices=[m.value for m in EvalMode], help="pretrain (ICL) or posttrain (chat template)")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Cap examples per task (for smoke-testing).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", default="auto", choices=list(AMP_DTYPE_CHOICES))
    parser.add_argument(
        "--judge",
        default=None,
        help="ALBA scoring backend: omit for none (generations saved, unscored) or 'anthropic[:model]' "
        "(needs ANTHROPIC_API_KEY) -- see src/eval/judge.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "eval")
    parser.add_argument("--log-samples", action="store_true", help="Include per-example predictions in the report.")
    parser.add_argument("--list-tasks", action="store_true", help="Print native task names and exit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.list_tasks:
        print("Native tasks:", ", ".join(list_tasks()))
        print(
            "Any other --tasks name is passed through to lm-evaluation-harness "
            "(e.g. spanish_bench, or an individual sub-task like xnli_es)."
        )
        return

    if args.ckpt is None or args.mode is None or args.tasks is None:
        raise SystemExit("--ckpt, --mode, and --tasks are required (see --list-tasks / --help).")

    mode = EvalMode(args.mode)
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    native = list_tasks()
    native_names = [t for t in task_names if t in native]
    lm_eval_names = [t for t in task_names if t not in native]

    eval_model = EvalModel(args.ckpt, args.tokenizer_dir, device=args.device, amp_dtype=args.amp_dtype)
    print(f"Loaded checkpoint ({eval_model.n_params:,} non-embedding params) on {eval_model.device}")
    judge = build_judge(args.judge)

    results: dict[str, TaskResult] = {}

    for name in native_names:
        task = get_task(name)()
        print(f"\n[{name}] {mode.value}, {args.num_fewshot}-shot" + (f", limit={args.limit}" if args.limit else ""))
        result = run_task(
            task, eval_model, mode, args.num_fewshot, args.limit, args.seed, judge, args.log_samples
        )
        results[name] = result
        print(f"[{name}] n={result.n_examples} {result.metrics}")

    if lm_eval_names:
        from src.eval.tasks.spanish_bench import run_lm_eval_tasks

        print(f"\n[lm-eval] {lm_eval_names} -- {mode.value}, {args.num_fewshot}-shot")
        results.update(
            run_lm_eval_tasks(eval_model, mode, lm_eval_names, args.num_fewshot, args.limit, args.seed)
        )
        for name in lm_eval_names:
            for task_name, result in results.items():
                if task_name == name or task_name.startswith(f"{name}_"):
                    print(f"[{task_name}] n={result.n_examples} {result.metrics}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ckpt": str(args.ckpt),
        "mode": mode.value,
        "n_params": eval_model.n_params,
        "results": {
            name: {k: v for k, v in asdict(r).items() if k != "samples" or args.log_samples}
            for name, r in results.items()
        },
    }
    out_path = args.out_dir / "results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
