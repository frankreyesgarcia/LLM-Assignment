"""One-off: translate openai/gsm8k's 'main'/'test' split (1,319 rows) to
Spanish using a locally-served model (vLLM OpenAI-compatible endpoint), so
gsm8k_es can join gsm8k_hi/gsm8k_pt (src/eval/tasks/gsm8k.py) -- no
comparably-maintained Spanish GSM8K translation exists on HF Hub (checked
before writing this; only small unofficial community copies).

Preserves the exact final numeric answer regardless of what the model
outputs: GSM8K's own "#### <number>" marker is a machine-checkable grading
target, not prose, so it's re-appended verbatim from the English source
after translation rather than trusted to the translation model's output --
a mistranslated/hallucinated number there would silently corrupt every
future eval run against this dataset.

Usage (see scripts/slurm/14_translate_gsm8k_es.sh for the full job):
    uv run scripts/translate_gsm8k_es.py \\
        --server-url http://127.0.0.1:8010/v1 --model glm-4.7-flash \\
        --out data/gsm8k_es/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = (
    "Eres un traductor profesional de ingles a espanol, especializado en "
    "problemas matematicos escolares. Traduces el enunciado y la solucion "
    "paso a paso, preservando exactamente todos los numeros y las "
    "anotaciones de calculo entre << >> tal cual aparecen (son notacion "
    "de calculadora, no texto -- no se traducen ni se alteran). Responde "
    "EXCLUSIVAMENTE con un objeto JSON de una sola linea: "
    '{"question": "...", "answer": "..."}. Sin texto adicional, sin '
    "bloques de codigo, sin explicaciones."
)


def _final_number(answer: str) -> str:
    return answer.split("####")[-1].strip()


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        # strict=False: GSM8K answers are multi-line CoT, and the model
        # routinely emits a literal newline inside the JSON string value
        # instead of an escaped "\n" -- strict json.loads rejects that as
        # an invalid control character even though the JSON is otherwise
        # well-formed, which was silently discarding ~1/4 of translations.
        return json.loads(text[start : end + 1], strict=False)
    except json.JSONDecodeError:
        return None


def translate_row(server_url: str, model: str, row: dict) -> tuple[dict, bool]:
    gold_number = _final_number(row["answer"])
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"question: {row['question']}\nanswer: {row['answer']}"},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        # GLM-4.7-Flash thinks by default (wraps a <think>...</think> block
        # before the actual content); a straight translation doesn't need
        # extended reasoning, and leaving thinking on burned the whole
        # max_tokens budget on the reasoning trace, truncating the JSON
        # answer to nothing on ~86% of the first run. Off + a reasoning
        # parser server-side (belt and suspenders) fixes it.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{server_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    parsed = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read())
            content = body["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if parsed and "question" in parsed and "answer" in parsed:
                break
        except (urllib.error.URLError, KeyError, IndexError, TimeoutError):
            parsed = None
    # A malformed reply can produce a truthy dict missing "question" or
    # "answer" on the final retry (the loop only breaks on a *complete*
    # parse, but leaves the last incomplete `parsed` assigned otherwise) --
    # checking `is None` here missed that case and crashed the whole run
    # with a KeyError three rows from the end of a 1,319-row job.
    fell_back = not parsed or "question" not in parsed or "answer" not in parsed
    if fell_back:
        # fallback: keep the English pair rather than drop the row
        parsed = {"question": row["question"], "answer": row["answer"]}

    es_answer = str(parsed["answer"])
    if "####" in es_answer:
        es_answer = es_answer.split("####")[0].rstrip() + f"\n#### {gold_number}"
    else:
        es_answer = es_answer.rstrip() + f"\n#### {gold_number}"
    return {"question": str(parsed["question"]), "answer": es_answer}, fell_back


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[translate_gsm8k_es] translating {len(rows)} rows")

    results: list[dict | None] = [None] * len(rows)
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(translate_row, args.server_url, args.model, row): i
            for i, row in enumerate(rows)
        }
        done = 0
        for fut in futures:
            i = futures[fut]
            try:
                results[i], fell_back = fut.result()
            except Exception as e:  # noqa: BLE001 -- one bad row must never sink the batch
                print(f"[translate_gsm8k_es] row {i} raised {e!r}, falling back to English")
                results[i] = {"question": rows[i]["question"], "answer": rows[i]["answer"]}
                fell_back = True
            fallbacks += fell_back
            done += 1
            if done % 50 == 0:
                print(f"[translate_gsm8k_es] {done}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pct = 100 * fallbacks / len(results) if results else 0
    print(f"[translate_gsm8k_es] wrote {len(results)} rows to {args.out}")
    print(f"[translate_gsm8k_es] {fallbacks}/{len(results)} ({pct:.1f}%) fell back to untranslated English")


if __name__ == "__main__":
    main()
