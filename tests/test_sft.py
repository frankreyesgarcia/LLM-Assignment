"""Tests for the SFT data pipeline (src/sft/).

Offline throughout -- no network. Source adapters are tested by
monkeypatching the HF loading call each module makes (load_rows /
load_dataset) with synthetic rows matching the real schemas confirmed
against the live datasets-server API while building these adapters
(src/sft/sources/*.py docstrings), the same approach tests/test_eval.py
uses for benchmark row shapes.
"""

from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerFast

from src.sft import registry as sft_registry
from src.sft.render import render_messages
from src.sft.sources.aya_hi import AyaHi
from src.sft.sources.base import SFTSource
from src.sft.sources.euroblocks import LANGUAGE_VALUES, EuroBlocksPt
from src.sft.types import SFTExample
from src.tokenizer.train import EOT_TOKEN, SPECIAL_TOKENS, train_tokenizer

# ---------------------------------------------------------------------------
# render.py
# ---------------------------------------------------------------------------

CORPUS = ["Olá, tudo bem?", "Sim, tudo ótimo, obrigado!", "system prompt aqui"] * 20


def _tokenizer():
    # render_messages is always called with an AutoTokenizer.from_pretrained
    # PreTrainedTokenizerFast in real usage (scripts/prepare_sft_data.py) --
    # wrap the same way src/tokenizer/train.py::save_pretrained does so this
    # test tokenizer is callable the same way.
    tk = train_tokenizer(CORPUS, vocab_size=600)
    return PreTrainedTokenizerFast(tokenizer_object=tk, additional_special_tokens=list(SPECIAL_TOKENS))


def test_render_messages_masks_non_assistant_turns():
    tokenizer = _tokenizer()
    messages = [
        {"role": "user", "content": "Olá, tudo bem?"},
        {"role": "assistant", "content": "Sim, tudo ótimo, obrigado!"},
    ]
    token_ids, labels = render_messages(tokenizer, messages)
    assert len(token_ids) == len(labels)

    eot_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
    # Split token_ids/labels at the first EOT (end of the user turn).
    split = token_ids.index(eot_id) + 1
    user_labels, assistant_labels = labels[:split], labels[split:]
    assistant_ids = token_ids[split:]

    assert all(label == -1 for label in user_labels)
    assert assistant_labels == assistant_ids  # assistant turn (incl. its own EOT) is unmasked


def test_render_messages_all_masked_gives_no_supervision():
    tokenizer = _tokenizer()
    token_ids, labels = render_messages(tokenizer, [{"role": "system", "content": "system prompt aqui"}])
    assert token_ids  # still produces tokens...
    assert all(label == -1 for label in labels)  # ...but nothing to learn from


# ---------------------------------------------------------------------------
# registry.py
# ---------------------------------------------------------------------------


def test_register_and_get_source():
    @sft_registry.register("_test_dummy_source")
    class _Dummy(SFTSource):
        languages = ("pt",)

        def load_examples(self, limit=None):
            return iter([])

    assert sft_registry.get_source("_test_dummy_source") is _Dummy
    assert "_test_dummy_source" in sft_registry.list_sources()


def test_register_duplicate_name_raises():
    @sft_registry.register("_test_dummy_source_2")
    class _DummyA(SFTSource):
        languages = ("pt",)

        def load_examples(self, limit=None):
            return iter([])

    with pytest.raises(ValueError):

        @sft_registry.register("_test_dummy_source_2")
        class _DummyB(SFTSource):
            languages = ("es",)

            def load_examples(self, limit=None):
                return iter([])


def test_get_unknown_source_raises():
    with pytest.raises(KeyError):
        sft_registry.get_source("_not_a_real_source")


# ---------------------------------------------------------------------------
# sources/aya_hi.py: inputs/targets -> messages conversion (the one adapter
# that isn't just a column rename)
# ---------------------------------------------------------------------------


def test_aya_hi_converts_inputs_targets_to_messages(monkeypatch):
    synthetic_row = {
        "id": 1,
        "inputs": "यह शीर्षक है, इसके लिए एक लेख लिखें: <headline>",
        "targets": "यह लेख है: ...",
        "language": "hin",
    }

    def fake_load_rows(dataset, config, split, limit):
        return [synthetic_row]

    monkeypatch.setattr("src.sft.sources.aya_hi.load_rows", fake_load_rows)

    # limit=1 is consumed entirely by the first config
    # (templated_hindi_headline) -- see AyaHi.load_examples's remaining-
    # budget bookkeeping.
    examples = list(AyaHi().load_examples(limit=1))
    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, SFTExample)
    assert ex.language == "hi"
    assert ex.messages == [
        {"role": "user", "content": synthetic_row["inputs"]},
        {"role": "assistant", "content": synthetic_row["targets"]},
    ]


# ---------------------------------------------------------------------------
# sources/euroblocks.py: language-column filter
# ---------------------------------------------------------------------------


def test_euroblocks_filters_by_language_value(monkeypatch):
    rows = [
        {"language": "English", "conversations": [{"role": "user", "content": "hi"}]},
        {"language": "Portuguese", "conversations": [{"role": "user", "content": "olá"}]},
        {"language": "Spanish", "conversations": [{"role": "user", "content": "hola"}]},
        {"language": "Portuguese", "conversations": [{"role": "user", "content": "bom dia"}]},
    ]

    def fake_load_dataset(dataset, split, streaming):
        return iter(rows)

    monkeypatch.setattr("src.sft.sources.euroblocks.load_dataset", fake_load_dataset)

    examples = list(EuroBlocksPt().load_examples())
    assert len(examples) == 2
    assert all(ex.language == "pt" for ex in examples)
    assert LANGUAGE_VALUES["pt"] == "Portuguese"
