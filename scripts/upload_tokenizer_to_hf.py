#!/usr/bin/env python3
"""Upload the trained tokenizer artifact (artifacts/tokenizer/, see
scripts/train_tokenizer.py) to the HF Hub as a model repo -- tokenizers
don't have their own repo type, they're hosted the same way a model's
tokenizer would be, loadable via `AutoTokenizer.from_pretrained(repo_id)`.

Uses `huggingface_hub.upload_folder`, which pushes the local folder as-is
(tokenizer.json/tokenizer_config.json/chat_template.jinja + README.md model
card). Excludes `logs/` (training run logs, not part of the artifact itself).

Requires an HF token with write access, either via `hf auth login` or the
HF_TOKEN env var.
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
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        ignore_patterns=["logs/*", "logs"],
        commit_message=f"Upload tokenizer from {local_dir.name}",
    )
    visibility = "private" if private else "public"
    print(f"Uploaded {local_dir} to https://huggingface.co/{repo_id} ({visibility})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help='e.g. "andre15silva/pt-es-hi-tokenizer"')
    parser.add_argument("--local-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--public", action="store_true", help="Upload as public (default: private)")
    args = parser.parse_args()
    main(args.repo_id, args.local_dir, private=not args.public)
