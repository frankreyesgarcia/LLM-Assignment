#!/usr/bin/env python3
"""Upload the built dataset folder (data/final/, see build_final_dataset.py)
to the HF Hub as a dataset repo -- data only, no pipeline code.

Uses `huggingface_hub.upload_folder`, which pushes the local folder as-is
(README.md with its `configs:` frontmatter + the per-config parquet files),
so the repo keeps exactly the pt/es/hi/all structure already validated
locally with `datasets.load_dataset("data/final", config)`.

Requires an HF token with write access, either via `huggingface-cli login`
or the HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(repo_id: str, local_dir: Path, private: bool) -> None:
    api = HfApi()
    if "/" not in repo_id:
        # create_repo resolves a bare name to "<username>/<name>" automatically,
        # but upload_folder does not -- pass the fully-qualified id to both,
        # or upload_folder 404s looking up a literal user/org named `repo_id`.
        repo_id = f"{api.whoami()['name']}/{repo_id}"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        commit_message=f"Upload dataset from {local_dir.name}",
    )
    visibility = "private" if private else "public"
    print(f"Uploaded {local_dir} to https://huggingface.co/datasets/{repo_id} ({visibility})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help='e.g. "frank-rg/pretrain-pt-es-hi"')
    parser.add_argument("--local-dir", type=Path, default=REPO_ROOT / "data" / "final")
    parser.add_argument("--public", action="store_true", help="Upload as public (default: private)")
    args = parser.parse_args()
    main(args.repo_id, args.local_dir, private=not args.public)
