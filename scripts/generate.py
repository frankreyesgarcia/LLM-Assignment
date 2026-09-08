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
    chat: bool = False,
) -> str:
    """--chat wraps the prompt in the chat template an SFT checkpoint was
    tuned on (scripts/prepare_sft_data.py renders training data with the
    same one), and returns only the assistant turn. Without it the prompt
    is fed as raw text and the model just continues it -- which is what a
    pretrained-only checkpoint expects, and what an SFT checkpoint will
    handle badly, since it never saw bare text after fine-tuning.
    """
    if chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    if not chat:
        return tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
    # Only the generated assistant turn, cut at the first turn-end token.
    # GPT.generate has no stop-token support (it always runs the full
    # max_new_tokens), so the truncation happens here -- by token id, since
    # decoding with skip_special_tokens never surfaces <|eot_id|> as text
    # for a string match to find. Same reasoning as
    # src/eval/model_adapter.py::EvalModel._stop_token_ids.
    generated = out[0].tolist()[prompt_len:]
    stop_ids = {
        tid
        for tid in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        if tid is not None and tid != tokenizer.unk_token_id
    }
    for i, tid in enumerate(generated):
        if tid in stop_ids:
            generated = generated[:i]
            break
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs" / "train" / "ckpt.pt")
    # AutoTokenizer.from_pretrained("andre15silva/pt-es-hi-tokenizer").save_pretrained("artifacts/tokenizer")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO_ROOT / "artifacts" / "tokenizer")
    parser.add_argument("--prompt", default=None, help="If omitted, drops into an interactive prompt loop")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Wrap the prompt in the chat template and return only the assistant turn -- for SFT checkpoints "
        "(scripts/train_sft.py). Without it the prompt is continued as raw text.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    model = load_model(args.ckpt, device)

    def run_once(prompt: str) -> None:
        text = generate_text(
            model, tokenizer, prompt, device, args.max_new_tokens, args.temperature, args.top_k, args.chat
        )
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
