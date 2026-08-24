"""Shared types for the SFT data pipeline.

Kept dependency-free (no torch/transformers imports), same reasoning as
src/eval/types.py::Doc -- source modules and tests can import this cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTExample:
    """One normalized SFT example: a chat, not yet tokenized.

    `messages` is a list of `{"role": ..., "content": ...}` dicts in the
    same shape src/tokenizer/train.py::CHAT_TEMPLATE renders -- the
    abstraction every source adapter converts into, and the only thing
    scripts/prepare_sft_data.py needs to know about downstream. `raw`
    keeps the original row for debugging/inspection, mirroring
    src/eval/types.py::Doc.
    """

    messages: list[dict]
    language: str
    source: str
    raw: dict
