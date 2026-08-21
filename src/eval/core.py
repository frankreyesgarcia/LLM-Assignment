"""Low-level model primitives the eval harness is built on (Task 4).

Two operations cover every benchmark here: scoring a fixed continuation's
log-probability (multiple-choice/cloze tasks) and greedily generating text
until a stop sequence (open-ended tasks). Both take one example at a time
rather than a padded batch -- this repo's checkpoints are pilot-scale and
eval sets are small (tens to low thousands of examples), so the extra
complexity of attention-mask-aware batching isn't worth it, and it sidesteps
a real correctness risk: GPT.forward has no attention-mask input, so
padding same-batch sequences of different lengths without masking would
silently corrupt logits.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.model.gpt import GPT
from src.model.train import AMP_DTYPE_CHOICES, autocast_for, resolve_amp_dtype


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def score_continuation(
    model: GPT,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    block_size: int,
    context_ids: list[int],
    continuation_ids: list[int],
) -> tuple[float, bool]:
    """Sum log-probability the model assigns `continuation_ids` right after
    `context_ids`, plus whether that continuation is exactly what greedy
    (argmax) decoding would have produced.

    Mirrors lm-evaluation-harness's HFLM._loglikelihood_tokens (same
    context+continuation -> logits -> gather -> sum recipe), simplified to
    one example at a time. `context_ids` must be non-empty (callers own
    the empty-context/BOS-prefix special case).
    """
    assert context_ids, "context_ids must be non-empty"
    assert continuation_ids, "continuation_ids must be non-empty"

    full = context_ids + continuation_ids
    # Left-truncate to fit block_size + 1 (GPT.forward asserts T <= block_size,
    # and we need one extra token of context to predict the first continuation
    # token) -- same truncation GPT.generate applies to its growing context.
    full = full[-(block_size + 1) :]
    cont_len = min(len(continuation_ids), len(full) - 1)
    assert cont_len > 0, "block_size too small to fit any continuation tokens"

    idx = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    with autocast_for(device, amp_dtype):
        logits, _ = model(idx)
    logits = logits[0].float()  # (T, vocab)
    cont_logits = logits[-cont_len:]  # predictions for the last cont_len positions
    cont_targets = torch.tensor(full[-cont_len:], dtype=torch.long, device=device)

    log_probs = F.log_softmax(cont_logits, dim=-1)
    token_logprobs = log_probs.gather(1, cont_targets.unsqueeze(-1)).squeeze(-1)
    greedy = log_probs.argmax(dim=-1)
    is_greedy = bool(torch.equal(greedy, cont_targets))
    return float(token_logprobs.sum()), is_greedy


@torch.no_grad()
def greedy_generate(
    model: GPT,
    tokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    block_size: int,
    context_ids: list[int],
    stop_sequences: list[str],
    stop_token_ids: list[int],
    max_new_tokens: int,
) -> str:
    """Greedily decode up to `max_new_tokens` tokens, stopping as soon as
    a generated token is in `stop_token_ids` (special tokens like EOS/EOT --
    checked by ID since skip_special_tokens=True decoding never shows them
    as text) or the decoded text-so-far contains one of `stop_sequences`
    (plain-text stop strings, e.g. punctuation for cloze tasks).

    Argmax rather than GPT.generate's sampling: eval tasks want
    deterministic, reproducible predictions, not exploratory continuations.
    """
    idx = torch.tensor([context_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        with autocast_for(device, amp_dtype):
            logits, _ = model(idx_cond)
        next_id = int(logits[0, -1].float().argmax())
        if next_id in stop_token_ids:
            break
        generated_ids.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(stop in text for stop in stop_sequences):
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    for stop in stop_sequences:
        pos = text.find(stop)
        if pos != -1:
            text = text[:pos]
    return text


__all__ = [
    "AMP_DTYPE_CHOICES",
    "resolve_device",
    "resolve_amp_dtype",
    "load_checkpoint",
    "load_tokenizer",
    "encode",
    "score_continuation",
    "greedy_generate",
]
