"""One-off: push the locally-translated data/gsm8k_es/test.jsonl (see
scripts/translate_gsm8k_es.py) to HF Hub as frank-rg/gsm8k-es, "test"
split -- matching the DATASET string registered in
src/eval/tasks/gsm8k.py::GSM8KSpanish.

Usage:
    HF_TOKEN=... uv run scripts/push_gsm8k_es_to_hub.py
"""

from __future__ import annotations

import os

from datasets import Dataset, load_dataset

REPO_ID = "frank-rg/gsm8k-es"
LOCAL_PATH = "data/gsm8k_es/test.jsonl"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN before running this script")

    ds = load_dataset("json", data_files=LOCAL_PATH, split="train")
    ds = Dataset.from_dict({"question": ds["question"], "answer": ds["answer"]})
    ds.push_to_hub(REPO_ID, split="test", token=token, private=False)
    print(f"[push_gsm8k_es_to_hub] pushed {len(ds)} rows to {REPO_ID} (split=test)")


if __name__ == "__main__":
    main()
