#!/usr/bin/env python3
"""Fase 0 — Source reconnaissance (TASK1-PLAN.md, sec 2.3).

For each of the 12 pre-training sources listed in the assignment, query the
HuggingFace `datasets-server` API to get:
  - size (bytes / rows) per config/split, with retries for cold caches
  - schema + 3 sample rows (first-rows API)
  - license (from the HF repo API)

Writes the aggregated inventory to configs/sources.yaml and prints a summary
table. Safe to re-run: large datasets may need a warm-up pass before their
parquet index is ready (see status "timeout_retry_needed" below).
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import requests
import yaml

DATASETS_SERVER = "https://datasets-server.huggingface.co"
HF_API = "https://huggingface.co/api/datasets"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
SAMPLE_ROWS = 3

# The 12 pre-training sources from `LLM Assignment.pdf` / TASK1-PLAN.md sec 2.1.
# EuroWeb-2512 physically backs 3 rows (multi/es/hi) — see notes.
SOURCES: list[dict[str, Any]] = [
    dict(
        row="multi-euroweb", group="multi", name="EuroWeb-2512",
        repo_id="utter-project/EuroWeb-2512", language="multi",
        hf_config=None, hf_split=None,
        notes="Filter by language config (pt/es/hi). Hindi MUST use split "
              "'high' only (other splits contain sexual content).",
    ),
    dict(
        row="multi-culturax", group="multi", name="CulturaX",
        repo_id="uonlp/CulturaX", language="multi",
        hf_config=None, hf_split=None,
        notes="Gated dataset — requires an HF token with accepted terms. "
              "Filter by its language column after loading.",
    ),
    dict(
        row="pt-fineweb2-bagaco2", group="pt", name="fineweb2-bagaco2",
        repo_id="duarteocarmo/fineweb2-bagaco2", language="pt",
        hf_config="all", hf_split="train",
    ),
    dict(
        row="pt-corpus-carolina", group="pt", name="corpus-carolina",
        repo_id="carolina-c4ai/corpus-carolina", language="pt",
        hf_config=None, hf_split=None,
        notes="Uses a custom HF loading script; datasets-server API is not "
              "supported for it (needs load_dataset(..., trust_remote_code=True)).",
    ),
    dict(
        row="pt-portuguese-pd", group="pt", name="Portuguese-PD",
        repo_id="PleIAs/Portuguese-PD", language="pt",
        hf_config=None, hf_split=None,
    ),
    dict(
        row="pt-corpus-ptbr-v2", group="pt", name="corpus-ptbr-v2",
        repo_id="Madras1/corpus-ptbr-v2", language="pt",
        hf_config=None, hf_split="train",
    ),
    dict(
        row="es-fineweb2", group="es", name="fineweb-2 (spa_Latn)",
        repo_id="HuggingFaceFW/fineweb-2", language="es",
        hf_config="spa_Latn", hf_split="train",
    ),
    dict(
        row="es-euroweb", group="es", name="EuroWeb-2512 (es)",
        repo_id="utter-project/EuroWeb-2512", language="es",
        hf_config="es", hf_split=None,
        notes="Same physical dataset as the 'multi-euroweb' row above — "
              "do not ingest twice.",
    ),
    dict(
        row="es-hplt2", group="es", name="HPLT2.0_cleaned (spa_Latn)",
        repo_id="HPLT/HPLT2.0_cleaned", language="es",
        hf_config="spa_Latn", hf_split="train",
    ),
    dict(
        row="hi-fineweb2", group="hi", name="fineweb-2 (hin_Deva)",
        repo_id="HuggingFaceFW/fineweb-2", language="hi",
        hf_config="hin_Deva", hf_split="train",
    ),
    dict(
        row="hi-euroweb", group="hi", name="EuroWeb-2512 (hi/high)",
        repo_id="utter-project/EuroWeb-2512", language="hi",
        hf_config="hi", hf_split="high",
        notes="Same physical dataset as the 'multi-euroweb' row above — "
              "do not ingest twice. MUST use split 'high' (others contain "
              "sexual content).",
    ),
    dict(
        row="hi-hplt2", group="hi", name="HPLT2.0_cleaned (hin_Deva)",
        repo_id="HPLT/HPLT2.0_cleaned", language="hi",
        hf_config="hin_Deva", hf_split="train",
    ),
]

_RETRYABLE_MARKERS = ("busier", "not ready", "time-out", "timeout", "gateway", "read timed out")
_GATED_MARKERS = ("gated", "authentication")
_SCRIPT_MARKERS = ("no longer supported",)


def _get_json(url: str, retries: int = MAX_RETRIES) -> dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(3 * (2**attempt))
    return {"error": last_error}


def _classify_error(data: dict[str, Any]) -> str:
    # The distinguishing text is sometimes only in `cause_message`, not `error`
    # (e.g. corpus-carolina's "Dataset scripts are no longer supported" lives
    # there, not in the top-level `error` field) — search the whole payload.
    haystack = " ".join(str(v) for v in data.values()).lower()
    if any(marker in haystack for marker in _GATED_MARKERS):
        return "gated"
    if any(marker in haystack for marker in _SCRIPT_MARKERS):
        return "script_based"
    if any(marker in haystack for marker in _RETRYABLE_MARKERS):
        return "timeout_retry_needed"
    return "error"


def get_size(repo_id: str, config: str | None) -> dict[str, Any]:
    url = f"{DATASETS_SERVER}/size?dataset={repo_id}"
    if config:
        url += f"&config={config}"

    for attempt in range(MAX_RETRIES + 1):
        data = _get_json(url, retries=0)
        if "error" not in data:
            node = data["size"].get("config") or data["size"].get("dataset")
            num_rows = node.get("num_rows")
            num_bytes = node.get("num_bytes_parquet_files")
            if not num_rows and not num_bytes:
                # HF sometimes returns a structurally valid but empty size
                # response (no error) for datasets whose parquet index isn't
                # computed yet, even though the dataset has real data (seen
                # with Portuguese-PD) — don't report this as "0 GB".
                return {"status": "size_unavailable", "note": "size API returned 0 rows/bytes with no error; check schema/samples instead"}
            return {
                "status": "ok",
                "num_bytes_parquet": num_bytes,
                "num_rows": num_rows,
                "splits": [
                    {
                        "split": s["split"],
                        "num_bytes_parquet": s.get("num_bytes_parquet_files"),
                        "num_rows": s.get("num_rows"),
                    }
                    for s in data["size"].get("splits", [])
                ],
            }
        status = _classify_error(data)
        if status == "timeout_retry_needed" and attempt < MAX_RETRIES:
            time.sleep(3 * (2**attempt))
            continue
        return {"status": status, "error": data["error"]}
    return {"status": "unknown"}


def discover_split(repo_id: str, config: str | None) -> str | None:
    url = f"{DATASETS_SERVER}/splits?dataset={repo_id}"
    if config:
        url += f"&config={config}"
    data = _get_json(url)
    if "error" in data:
        return None
    splits = [s["split"] for s in data.get("splits", [])]
    if not splits:
        return None
    return "train" if "train" in splits else splits[0]


def get_first_rows(repo_id: str, config: str | None, split: str | None) -> dict[str, Any]:
    if split is None:
        split = discover_split(repo_id, config)
    if split is None:
        return {"status": "no_split_found"}

    url = f"{DATASETS_SERVER}/first-rows?dataset={repo_id}&split={split}"
    url += f"&config={config}" if config else "&config=default"
    data = _get_json(url)
    if "error" in data:
        return {"status": _classify_error(data), "error": data["error"], "split_used": split}

    columns = [f["name"] for f in data.get("features", [])]
    samples = [row["row"] for row in data.get("rows", [])[:SAMPLE_ROWS]]
    return {"status": "ok", "split_used": split, "columns": columns, "samples": samples}


def get_license(repo_id: str) -> str | None:
    data = _get_json(f"{HF_API}/{repo_id}")
    if "error" in data:
        return None
    card = data.get("cardData") or {}
    if card.get("license"):
        return card["license"]
    for tag in data.get("tags", []):
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def inspect_source(source: dict[str, Any]) -> dict[str, Any]:
    print(f"Inspecting {source['name']} ({source['repo_id']}, config={source['hf_config']})...")
    size = get_size(source["repo_id"], source["hf_config"])
    first_rows = get_first_rows(source["repo_id"], source["hf_config"], source["hf_split"])
    license_id = get_license(source["repo_id"])
    return {
        **{k: v for k, v in source.items()},
        "license": license_id,
        "size": size,
        "schema": first_rows,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print(f"{'source':30s} {'lang':5s} {'status':22s} {'size_gb':>10s} {'rows':>14s}")
    print("-" * 100)
    for r in results:
        size = r["size"]
        status = size.get("status", "ok")
        gb = f"{(size.get('num_bytes_parquet') or 0) / 1e9:.1f}" if status == "ok" else "-"
        rows = f"{size.get('num_rows'):,}" if status == "ok" and size.get("num_rows") else "-"
        print(f"{r['name']:30s} {r['language']:5s} {status:22s} {gb:>10s} {rows:>14s}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", help="Inspect a single source by its 'row' key (see SOURCES), for quick smoke tests."
    )
    parser.add_argument(
        "--out", default="configs/sources.yaml", help="Output YAML path (skipped with --only unless --write)."
    )
    parser.add_argument("--write", action="store_true", help="Write output even when using --only.")
    args = parser.parse_args()

    sources = SOURCES
    if args.only:
        sources = [s for s in SOURCES if s["row"] == args.only]
        if not sources:
            valid = ", ".join(s["row"] for s in SOURCES)
            raise SystemExit(f"Unknown --only={args.only!r}. Valid values: {valid}")

    results = [inspect_source(s) for s in sources]
    print_summary(results)

    if args.only and not args.write:
        print("\n(--only used without --write: not touching configs/sources.yaml)")
        return

    with open(args.out, "w") as f:
        yaml.safe_dump({"sources": results}, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote inventory for {len(results)} sources to {args.out}")


if __name__ == "__main__":
    main()
