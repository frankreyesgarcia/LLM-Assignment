"""smoltalk2PT (duarteocarmo/smoltalk2PT): European Portuguese SFT data,
4 configs (everyday_conversations, magpie_ultra, smol_rewrite,
tulu_personas), each already a `messages: [{role, content}]` column --
HuggingFaceTB/smoltalk2 machine-translated to Portuguese. Only
`chat_template_kwargs` (thinking/tool-call flags this project's chat
template doesn't use) and `source` are dropped; everything else passes
through unchanged.
"""

from __future__ import annotations

from typing import Iterator

from src.eval.hf_data import load_rows
from src.sft.registry import register
from src.sft.sources.base import SFTSource
from src.sft.types import SFTExample

DATASET = "duarteocarmo/smoltalk2PT"
CONFIGS = ("everyday_conversations", "magpie_ultra", "smol_rewrite", "tulu_personas")


@register("smoltalk2pt")
class Smoltalk2PT(SFTSource):
    languages = ("pt",)

    def load_examples(self, limit: int | None = None) -> Iterator[SFTExample]:
        n = 0
        for config in CONFIGS:
            remaining = None if limit is None else limit - n
            if remaining is not None and remaining <= 0:
                return
            for row in load_rows(DATASET, config, "train", remaining):
                yield SFTExample(messages=row["messages"], language="pt", source=f"smoltalk2pt/{config}", raw=row)
                n += 1
