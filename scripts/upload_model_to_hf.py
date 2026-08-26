#!/usr/bin/env python3
"""Upload a trained checkpoint (scripts/train.py / scripts/train_sft.py) to
the HF Hub as a model repo, alongside the tokenizer it was trained with.

NOT a `transformers` model. This repo's GPT (src/model/gpt.py) has learned
positional embeddings *and* qk-norm, which no stock HF architecture
matches, so `AutoModel.from_pretrained` cannot load it -- a faithful
conversion would need a custom modeling file and `trust_remote_code`.
What gets uploaded instead is the plain state dict as safetensors plus a
config.json of the GPTConfig fields, which is enough to rebuild the model
with three lines against this repo (the model card spells them out).

Safetensors rather than the original .pt for two reasons: it loads
without `pickle` (a .pt of this repo's checkpoints can only be read with
`weights_only=False`, which requires `src.model.gpt` importable just to
unpickle the GPTConfig dataclass), and it is the format the Hub can
introspect.

Requires an HF token with write access, either via `huggingface-cli
login` or the HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_upload_dir(ckpt_path: Path, tokenizer_dir: Path, card_path: Path | None, staging: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    state = dict(ckpt["model_state_dict"])

    # lm_head is tied to the token embedding (src/model/gpt.py), and
    # safetensors refuses to serialize two names pointing at one storage.
    # Drop it and record the tying in config.json so the loader can
    # restore it -- GPT.__init__ already ties them, so a plain
    # load_state_dict on a fresh model puts it back.
    tied = "lm_head.weight" in state and state["lm_head.weight"].data_ptr() == state["transformer.wte.weight"].data_ptr()
    if tied:
        del state["lm_head.weight"]

    staging.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, staging / "model.safetensors")

    config = asdict(model_cfg) if hasattr(model_cfg, "__dataclass_fields__") else dict(vars(model_cfg))
    config.update({
        "model_type": "llm-und-gpt",
        "architecture": "src/model/gpt.py::GPT",
        "tie_word_embeddings": bool(tied),
        "torch_dtype": str(next(iter(state.values())).dtype).replace("torch.", ""),
        "iter_num": int(ckpt.get("iter_num", -1)),
    })
    for key in ("final_train_loss", "final_val_loss"):
        if key in ckpt:
            config[key] = float(ckpt[key])
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # The tokenizer travels with the weights: a checkpoint is unusable
    # without the exact vocabulary it was trained on, and pointing at a
    # separate repo is one more thing to get wrong.
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        src = tokenizer_dir / name
        if src.exists():
            shutil.copy2(src, staging / name)

    if card_path is not None:
        shutil.copy2(card_path, staging / "README.md")

    return config


def main(
    repo_id: str,
    ckpt_path: Path,
    tokenizer_dir: Path,
    card_path: Path | None,
    private: bool,
    dry_run: bool,
    staging_root: Path | None = None,
) -> None:
    api = HfApi()
    if "/" not in repo_id:
        # create_repo resolves a bare name to "<username>/<name>"
        # automatically, but upload_folder does not -- pass the
        # fully-qualified id to both, or upload_folder 404s looking up a
        # literal user/org named `repo_id`.
        repo_id = f"{api.whoami()['name']}/{repo_id}"

    # Stage beside the checkpoint by default, not in $TMPDIR: /tmp on a
    # Berzelius node is 100G shared and was observed with <1MB free, while
    # the safetensors copy of these weights alone is ~820MB.
    staging_root = staging_root or ckpt_path.parent
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(staging_root)) as tmp:
        staging = Path(tmp) / "upload"
        config = build_upload_dir(ckpt_path, tokenizer_dir, card_path, staging)
        files = sorted(p.name for p in staging.iterdir())
        total_mb = sum(p.stat().st_size for p in staging.iterdir()) / 1e6
        print(f"Staged {len(files)} files ({total_mb:.0f} MB): {', '.join(files)}")
        print(f"config: {json.dumps(config)}")
        if dry_run:
            print(f"--dry-run: not uploading. Would push to {repo_id} ({'private' if private else 'public'}).")
            return
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(staging),
            commit_message=f"Upload {ckpt_path.parent.name}/{ckpt_path.name} (iter {config.get('iter_num')})",
        )
    print(f"Uploaded to https://huggingface.co/{repo_id} ({'private' if private else 'public'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help='e.g. "andre15silva/pt-es-hi-sft"')
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint to upload (e.g. runs/sft/ckpt.pt)")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--model-card", type=Path, default=None, help="Markdown file to upload as README.md")
    parser.add_argument("--public", action="store_true", help="Upload as public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Stage and report, but do not create or push.")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Where to stage the upload folder (default: next to --ckpt). Not $TMPDIR -- /tmp on a compute node "
        "is too small for the weights.",
    )
    args = parser.parse_args()

    main(
        args.repo_id,
        args.ckpt,
        args.tokenizer_dir,
        args.model_card,
        private=not args.public,
        dry_run=args.dry_run,
        staging_root=args.staging_dir,
    )
