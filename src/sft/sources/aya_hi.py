"""aya_collection (CohereLabs/aya_collection), `templated_hindi_headline`
and `templated_hindi_news` configs: unlike the other 3 sources, this one is
NOT already messages-shaped -- it's `inputs`/`targets` prompt/completion
string pairs (template-generated: headline -> article, article -> summary),
so this is the one adapter that actually builds the messages list rather
than just renaming/passing a column through.

Narrow in task coverage (2 templated generation tasks, not open-ended
instruction-following) -- currently the only Hindi source alongside
indic_instruct_hi.py; a broader general-purpose Hindi SFT source is still
an open gap (see README).
"""

from __future__ import annotations

from typing import Iterator

from src.eval.hf_data import load_rows
from src.sft.registry import register
from src.sft.sources.base import SFTSource
from src.sft.types import SFTExample

DATASET = "CohereLabs/aya_collection"
CONFIGS = ("templated_hindi_headline", "templated_hindi_news")


@register("aya_hi")
class AyaHi(SFTSource):
    languages = ("hi",)

    def load_examples(self, limit: int | None = None) -> Iterator[SFTExample]:
        n = 0
        for config in CONFIGS:
            remaining = None if limit is None else limit - n
            if remaining is not None and remaining <= 0:
                return
            for row in load_rows(DATASET, config, "train", remaining):
                messages = [
                    {"role": "user", "content": row["inputs"]},
                    {"role": "assistant", "content": row["targets"]},
                ]
                yield SFTExample(messages=messages, language="hi", source=f"aya_hi/{config}", raw=row)
                n += 1
