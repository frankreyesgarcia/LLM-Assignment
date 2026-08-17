"""spanish_bench (Task 4): delegates to lm-evaluation-harness rather than
reimplementing its ~30 task variants (Belebele_es, XQuAD_es, XNLI_es,
COPA-es, PAWS-X_es, XStoryCloze_es, FLORES_es, MGSM_es, ...).

lm-eval already has real, maintained implementations of the whole suite
(github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/spanish_bench);
this module is just the glue that hands it our GPT checkpoint (via
lm_eval_adapter.LMEvalAdapter) and normalizes its results dict into this
harness's TaskResult, so the CLI/report code doesn't need to know spanish
tasks came from a different code path than the native PT/HI ones.

Not registered in the native Task registry -- scripts/run_eval.py
dispatches to run_lm_eval_tasks() for any task name lm-eval recognizes
that isn't a native task (see its --list-tasks / dispatch logic).
"""

from __future__ import annotations

import json
from pathlib import Path

import lm_eval

from src.eval.lm_eval_adapter import LMEvalAdapter
from src.eval.model_adapter import EvalModel
from src.eval.types import EvalMode, TaskResult

# Metric keys come back as "metric_name,filter_name" (e.g. "acc,none");
# stderr entries ("acc_stderr,none") and non-numeric bookkeeping ("alias")
# are dropped so TaskResult.metrics is a clean metric-name -> float dict.
_SKIP_SUFFIXES = ("_stderr",)


def _clean_metrics(raw: dict) -> dict[str, float]:
    metrics = {}
    for key, value in raw.items():
        if key == "alias" or not isinstance(value, (int, float)):
            continue
        name = key.split(",", 1)[0]
        if name.endswith(_SKIP_SUFFIXES):
            continue
        metrics[name] = float(value)
    return metrics


def run_lm_eval_tasks(
    eval_model: EvalModel,
    mode: EvalMode,
    task_names: list[str],
    num_fewshot: int = 0,
    limit: int | None = None,
    seed: int = 1234,
    log_samples: bool = False,
    samples_dir: Path | None = None,
) -> dict[str, TaskResult]:
    """samples_dir: when given (with log_samples=True), each task's
    per-document results (lm-eval's own SampleResult shape -- doc,
    target, raw+filtered model responses) are streamed to
    <samples_dir>/<task_name>.samples.jsonl, mirroring how native tasks'
    samples are logged in runner.py, for the same reason: full-dataset
    runs are too big to hold in memory or embed in results.json.
    """
    adapter = LMEvalAdapter(eval_model)
    output = lm_eval.simple_evaluate(
        model=adapter,
        tasks=task_names,
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=(mode == EvalMode.POSTTRAIN),
        fewshot_as_multiturn=(mode == EvalMode.POSTTRAIN),
        bootstrap_iters=0,
        log_samples=log_samples,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        confirm_run_unsafe_code=True,
    )
    if output is None:
        return {}

    if log_samples and samples_dir is not None:
        samples_dir.mkdir(parents=True, exist_ok=True)
        for task_name, docs in output.get("samples", {}).items():
            path = samples_dir / f"{task_name}.samples.jsonl"
            with path.open("a") as f:
                for doc in docs:
                    f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")

    n_samples = output.get("n-samples", {})
    results = {}
    for task_name, raw_metrics in output["results"].items():
        n = n_samples.get(task_name, {})
        results[task_name] = TaskResult(
            task_name=task_name,
            language="es",
            mode=mode,
            num_fewshot=output.get("n-shot", {}).get(task_name, num_fewshot),
            n_examples=n.get("effective", n.get("original", 0)) if isinstance(n, dict) else 0,
            metrics=_clean_metrics(raw_metrics),
        )
    return results
