"""Byte-level BPE tokenizer training (Task 2).

Trains a single shared multilingual (pt/es/hi) tokenizer, GPT-2/Llama-3
style: byte-level pre-tokenization means every input string is
representable (no OOV/`<unk>` fallback needed, unlike word- or
character-level vocabularies), which matters here since the corpus mixes
Devanagari (hi) and Latin (pt/es) scripts in one vocab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

# Llama-3-style special tokens: bos/eos/pad plus a minimal chat-turn set
# (header wrapper + end-of-turn) so the base tokenizer is ready for later
# chat fine-tuning without resizing the embedding table. Role names
# (system/user/assistant) are written as plain text between the header
# tokens rather than given their own tokens, keeping the set small and
# roles open-ended. No `<unk>` — byte-level BPE has full coverage by
# construction.
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"
START_HEADER_TOKEN = "<|start_header_id|>"
END_HEADER_TOKEN = "<|end_header_id|>"
EOT_TOKEN = "<|eot_id|>"

SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    START_HEADER_TOKEN,
    END_HEADER_TOKEN,
    EOT_TOKEN,
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' "
    "+ message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


def build_tokenizer() -> Tokenizer:
    """Construct an (untrained) byte-level BPE tokenizer.

    Uses the modern `tokenizers` API (Tokenizer + explicit components),
    not the removed `ByteLevelBPETokenizer` convenience class.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def train_tokenizer(text_iterator: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer on `text_iterator`.

    `text_iterator` is consumed lazily (`train_from_iterator`), so this
    scales past in-memory corpus size without materializing it all at
    once.
    """
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer


def save_pretrained(tokenizer: Tokenizer, out_dir: Path) -> PreTrainedTokenizerFast:
    """Wrap `tokenizer` as a `PreTrainedTokenizerFast` and save it to `out_dir`.

    Produces `tokenizer.json`, `tokenizer_config.json`,
    `special_tokens_map.json` -- loadable via
    `AutoTokenizer.from_pretrained(out_dir)` like any HF tokenizer.
    """
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        additional_special_tokens=[START_HEADER_TOKEN, END_HEADER_TOKEN, EOT_TOKEN],
        chat_template=CHAT_TEMPLATE,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(str(out_dir))
    return fast_tokenizer
