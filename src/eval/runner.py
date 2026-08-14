"""Drives one Task against one EvalModel (Task 4).

Fewshot sampling, prompt assembly for both EvalModes, and per-request-type
dispatch (LOGLIKELIHOOD vs GENERATE) all live here -- once, instead of once
per task module -- so every benchmark is scored the same way and adding one
means writing a Task subclass, not touching this file.

Fewshot examples are drawn leave-one-out from the same doc list being
evaluated, not a separate held-out pool: several of these benchmarks ship
a single split with no train/dev section (portugal-basic-qa-ptcore is 50
rows total), so "the eval set minus the current example" is the only
fewshot source that exists for them, and it works fine for the larger
benchmarks too.
"""

from __future__ import annotations

import random

from src.eval.judge import Judge
from src.eval.model_adapter import EvalModel
from src.eval.tasks.base import Task
from src.eval.types import Doc, EvalMode, RequestType, TaskResult


def _sample_fewshot(docs: list[Doc], doc: Doc, k: int, rng: random.Random) -> list[Doc]:
    if k <= 0:
        return []
    pool = [d for d in docs if d.idx != doc.idx]
    return rng.sample(pool, min(k, len(pool)))


def _join_choice(context: str, choice: str) -> str:
    if context == "" or context[-1].isspace():
        return choice
    return " " + choice


def _build_icl_context(task: Task, doc: Doc, shots: list[Doc]) -> str:
    parts = []
    if task.description:
        parts.append(task.description)
    for shot in shots:
        target = task.doc_to_target(shot)
        parts.append(task.doc_to_text(shot) + _join_choice(task.doc_to_text(shot), target))
    parts.append(task.doc_to_text(doc))
    return "\n\n".join(parts)


def _build_chat_context(task: Task, model: EvalModel, doc: Doc, shots: list[Doc]) -> str:
    messages = []
    if task.description:
        messages.append({"role": "system", "content": task.description})
    for shot in shots:
        messages.extend(task.doc_to_messages(shot))
        messages.append({"role": "assistant", "content": task.doc_to_target(shot)})
    messages.extend(task.doc_to_messages(doc))
    return model.apply_chat_template(messages, add_generation_prompt=True)


def run_task(
    task: Task,
    model: EvalModel,
    mode: EvalMode,
    num_fewshot: int = 0,
    limit: int | None = None,
    seed: int = 1234,
    judge: Judge | None = None,
    log_samples: bool = False,
) -> TaskResult:
    docs = task.load_docs(limit)
    if not docs:
        return TaskResult(task.name, task.language, mode, num_fewshot, 0, {})

    rng = random.Random(seed)
    metric_totals: dict[str, list[float]] = {}
    samples = []

    for doc in docs:
        shots = _sample_fewshot(docs, doc, num_fewshot, rng)
        context = (
            _build_icl_context(task, doc, shots)
            if mode == EvalMode.PRETRAIN
            else _build_chat_context(task, model, doc, shots)
        )

        if task.request_type == RequestType.LOGLIKELIHOOD:
            choices = task.doc_to_choices(doc)
            logprobs = [model.loglikelihood(context, _join_choice(context, c))[0] for c in choices]
            pred_idx = max(range(len(choices)), key=lambda i: logprobs[i])
            metrics = {"accuracy": 1.0 if pred_idx == task.gold_index(doc) else 0.0}
            prediction = choices[pred_idx]
        else:
            prediction = model.generate_until(context, task.stop_sequences, task.max_gen_toks)
            if getattr(task, "needs_judge", False):
                metrics = task.process_result(doc, prediction, judge=judge)
            else:
                metrics = task.process_result(doc, prediction)

        for k, v in metrics.items():
            if v is not None:
                metric_totals.setdefault(k, []).append(v)
        if log_samples:
            samples.append({"idx": doc.idx, "prediction": prediction, "metrics": metrics})

    aggregated = {k: sum(v) / len(v) for k, v in metric_totals.items() if v}
    return TaskResult(task.name, task.language, mode, num_fewshot, len(docs), aggregated, samples)
