#!/usr/bin/env python3
"""Task 3 — prompt a trained checkpoint interactively (or once via --prompt).

Loading a checkpoint back is the mirror image of training: the tokenizer
turns your prompt text into token IDs, `GPT.generate()` autoregressively
samples a continuation (feed model -> take its predicted next-token
distribution -> sample one -> append -> repeat), and the tokenizer
decodes the resulting ID sequence back into text.

Note: any checkpoint trained on this repo's pilot-scale data
(data/final, ~15MB) for a short, cheap run is going to produce
disfluent/incoherent continuations -- that's expected. This script is
for exercising the trained model and the generation path, not for
producing good text; that needs the real (non-pilot) corpus and a much
longer, larger-scale training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model.gpt import GPT

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from iter {ckpt['iter_num']} ({model.num_params():,} non-embedding params)")
    return model


def generate_text(
    model: GPT, tokenizer, prompt: str, device: torch.device,
    max_new_tokens: int, temperature: float, top_k: int | None,
) -> str:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs" / "train" / "ckpt.pt")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--prompt", default=None, help="If omitted, drops into an interactive prompt loop")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    model = load_model(args.ckpt, device)

    def run_once(prompt: str) -> None:
        text = generate_text(model, tokenizer, prompt, device, args.max_new_tokens, args.temperature, args.top_k)
        print(text)

    if args.prompt is not None:
        run_once(args.prompt)
    else:
        print("Interactive mode -- type a prompt and press enter (Ctrl+D to quit).")
        while True:
            try:
                prompt = input("\n> ")
            except EOFError:
                break
            if prompt.strip():
                run_once(prompt)
