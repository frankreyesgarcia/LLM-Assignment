"""indic-instruct-data-v0.1 (ai4bharat/indic-instruct-data-v0.1), `hh-rlhf`
config, `hi` split: despite the "hh-rlhf" name (Anthropic's helpfulness/
harmlessness *preference* dataset), this config has already been converted
to plain multi-turn SFT data -- a `messages: [{role, content}]` column, not
chosen/rejected pairs. `quality_metrics` (chrF/chrF++/sacreBLEU against the
original English) is machine-translation quality, not response quality;
exposed as an optional filter since a caller may want to drop badly
translated rows, but not applied by default.
"""

from __future__ import annotations

from typing import Iterator

from src.eval.hf_data import load_rows
from src.sft.registry import register
from src.sft.sources.base import SFTSource
from src.sft.types import SFTExample

DATASET = "ai4bharat/indic-instruct-data-v0.1"
CONFIG = "hh-rlhf"
SPLIT = "hi"


@register("indic_instruct_hi")
class IndicInstructHi(SFTSource):
    languages = ("hi",)

    def load_examples(self, limit: int | None = None) -> Iterator[SFTExample]:
        for row in load_rows(DATASET, CONFIG, SPLIT, limit):
            yield SFTExample(messages=row["messages"], language="hi", source="indic_instruct_hi", raw=row)
