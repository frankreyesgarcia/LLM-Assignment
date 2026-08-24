"""Turn one SFTExample's messages into (token_ids, labels) for training.

Renders each message with the exact same per-turn wrapping
src/tokenizer/train.py::CHAT_TEMPLATE uses
(`<|start_header_id|>ROLE<|end_header_id|>\\n\\nCONTENT<|eot_id|>`), one
message at a time rather than one `apply_chat_template` call over the
whole conversation -- this project's template has no cross-message
conditioning (no system-prompt-dependent branching, no
add_generation_prompt lookback within a turn), so per-message rendering is
exactly equivalent to the template's output while giving an O(n) per-turn
token span for free instead of needing to diff cumulative-prefix
tokenizations to recover it.

`labels` mirrors `token_ids` on assistant turns (including their trailing
`<|eot_id|>`, so the model learns to stop) and is `-1` everywhere else
(ignored by `F.cross_entropy(..., ignore_index=-1)` in src/model/gpt.py,
already there for the loglikelihood eval path).
"""

from __future__ import annotations

from src.tokenizer.train import END_HEADER_TOKEN, EOT_TOKEN, START_HEADER_TOKEN

IGNORE_INDEX = -1


def render_messages(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    token_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        text = f"{START_HEADER_TOKEN}{role}{END_HEADER_TOKEN}\n\n{message['content']}{EOT_TOKEN}"
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        labels.extend(ids if role == "assistant" else [IGNORE_INDEX] * len(ids))
    return token_ids, labels
