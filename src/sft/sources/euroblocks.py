"""EuroBlocks-SFT-2512 (utter-project/EuroBlocks-SFT-2512): one `default`
config/`train` split covering 114 languages in a single `language` column
(spelled-out English names, e.g. "Portuguese"/"Spanish" -- confirmed
against real rows, not guessed) plus a `conversations` column already
`[{role, content}]`. One class parameterized by language rather than one
per language, since the only difference between euroblocks_pt/euroblocks_es
is which `language` value to filter on.
"""

from __future__ import annotations

import itertools
from typing import Iterator

from datasets import load_dataset

from src.sft.registry import register
from src.sft.sources.base import SFTSource
from src.sft.types import SFTExample

DATASET = "utter-project/EuroBlocks-SFT-2512"
LANGUAGE_VALUES = {"pt": "Portuguese", "es": "Spanish"}


class EuroBlocks(SFTSource):
    # Always streamed: this is a single 114-language split, so pulling the
    # whole thing to disk just to filter down to one language would waste
    # most of the download. islice(..., limit) short-circuits once enough
    # matching rows are found.
    def load_examples(self, limit: int | None = None) -> Iterator[SFTExample]:
        language = self.languages[0]
        target = LANGUAGE_VALUES[language]
        ds = load_dataset(DATASET, split="train", streaming=True)
        matches = (row for row in ds if row["language"] == target)
        for row in itertools.islice(matches, limit):
            yield SFTExample(messages=row["conversations"], language=language, source="euroblocks", raw=row)


@register("euroblocks_pt")
class EuroBlocksPt(EuroBlocks):
    languages = ("pt",)


@register("euroblocks_es")
class EuroBlocksEs(EuroBlocks):
    languages = ("es",)
