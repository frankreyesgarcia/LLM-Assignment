"""Tests for the SFT data pipeline (src/sft/).

Offline throughout -- no network. Source adapters are tested by
monkeypatching the HF loading call each module makes (load_rows /
load_dataset) with synthetic rows matching the real schemas confirmed
against the live datasets-server API while building these adapters
(src/sft/sources/*.py docstrings), the same approach tests/test_eval.py
uses for benchmark row shapes.
"""

from __future__ import annotations

import json

import pytest
from transformers import PreTrainedTokenizerFast

from src.sft import registry as sft_registry
from src.sft.render import render_messages
from src.sft.sources.aya_hi import AyaHi
from src.sft.sources.base import SFTSource
from src.sft.sources.euroblocks import LANGUAGE_VALUES, EuroBlocksPt
from src.sft.types import SFTExample
from src.tokenizer.train import EOS_TOKEN, EOT_TOKEN, SPECIAL_TOKENS, train_tokenizer

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


# ---------------------------------------------------------------------------
# prepare_sft_data.py: per-source accounting
# ---------------------------------------------------------------------------


def test_prepare_sft_data_records_per_source_counts(tmp_path):
    """meta.json has to say how much each dataset actually contributed --
    a flat token total can't distinguish a source that landed 200k examples
    from one that silently yielded nothing."""
    import scripts.prepare_sft_data as prep

    @sft_registry.register("fake_pt")
    class FakePt(SFTSource):
        languages = ("pt",)

        def load_examples(self, limit=None):
            for i in range(20):
                yield SFTExample(
                    messages=[
                        {"role": "user", "content": "Olá, tudo bem?"},
                        {"role": "assistant", "content": f"Sim, tudo ótimo, obrigado! {i}"},
                    ],
                    language="pt",
                    source="fake_pt",
                    raw={},
                )

    @sft_registry.register("fake_hi")
    class FakeHi(SFTSource):
        languages = ("hi",)

        def load_examples(self, limit=None):
            for i in range(5):
                yield SFTExample(
                    messages=[
                        {"role": "user", "content": "system prompt aqui"},
                        {"role": "assistant", "content": f"Sim, tudo ótimo, obrigado! {i}"},
                    ],
                    language="hi",
                    source="fake_hi/config_a",
                    raw={},
                )

    try:
        # eos_token, unlike in _tokenizer(): prepare_sft_data.py writes the
        # EOS id between conversations as the document-boundary sentinel, so
        # a tokenizer without one can't be packed at all.
        tokenizer_dir = tmp_path / "tokenizer"
        tk = train_tokenizer(CORPUS, vocab_size=600)
        PreTrainedTokenizerFast(
            tokenizer_object=tk, additional_special_tokens=list(SPECIAL_TOKENS), eos_token=EOS_TOKEN
        ).save_pretrained(str(tokenizer_dir))

        out_dir = tmp_path / "sft"
        prep.run(
            sources=["fake_pt", "fake_hi"],
            languages=None,
            tokenizer_dir=tokenizer_dir,
            out_dir=out_dir,
            val_fraction=0.2,
            limit=None,
            seed=0,
        )

        meta = json.loads((out_dir / "meta.json").read_text())
        assert set(meta["per_source"]) == {"fake_pt", "fake_hi/config_a"}
        assert set(meta["per_language"]) == {"pt", "hi"}

        # Sub-source granularity is preserved (fake_hi/config_a, not fake_hi),
        # and every example lands in exactly one split.
        for key, expected in (("per_source", 25), ("per_language", 25)):
            counted = sum(e["train_examples"] + e["val_examples"] for e in meta[key].values())
            assert counted == expected
        assert sum(e["train_tokens"] for e in meta["per_source"].values()) == meta["train_tokens"] - meta["train_examples"]

        # Supervised counts are a strict subset of total tokens: the user
        # turns are masked to -1, so they can never be equal.
        for entry in meta["per_source"].values():
            assert 0 < entry["train_supervised_tokens"] < entry["train_tokens"]
    finally:
        for name in ("fake_pt", "fake_hi"):
            sft_registry._REGISTRY.pop(name, None)


def test_prepare_sft_data_drops_over_length_examples(tmp_path):
    """A conversation longer than the training block_size can't be seen
    whole; --max-example-tokens drops it (counted, per source) rather than
    letting it be split across windows."""
    import scripts.prepare_sft_data as prep

    @sft_registry.register("fake_long")
    class FakeLong(SFTSource):
        languages = ("pt",)

        def load_examples(self, limit=None):
            for i, content in enumerate(["Olá, tudo bem?"] * 5 + ["Sim, tudo ótimo, obrigado! " * 200] * 5):
                yield SFTExample(
                    messages=[
                        {"role": "user", "content": "Olá, tudo bem?"},
                        {"role": "assistant", "content": content},
                    ],
                    language="pt",
                    source="fake_long",
                    raw={},
                )

    try:
        tokenizer_dir = tmp_path / "tokenizer"
        tk = train_tokenizer(CORPUS, vocab_size=600)
        PreTrainedTokenizerFast(
            tokenizer_object=tk, additional_special_tokens=list(SPECIAL_TOKENS), eos_token=EOS_TOKEN
        ).save_pretrained(str(tokenizer_dir))

        out_dir = tmp_path / "sft"
        prep.run(
            sources=["fake_long"],
            languages=None,
            tokenizer_dir=tokenizer_dir,
            out_dir=out_dir,
            val_fraction=0.2,
            limit=None,
            seed=0,
            max_example_tokens=64,
        )

        meta = json.loads((out_dir / "meta.json").read_text())
        assert meta["max_example_tokens"] == 64
        assert meta["dropped_too_long"] == {"fake_long": 5}
        assert meta["dropped_too_long_total"] == 5
        assert meta["train_examples"] + meta["val_examples"] == 5
        assert meta["train_length_max"] <= 64
    finally:
        sft_registry._REGISTRY.pop("fake_long", None)


def test_sft_batches_start_on_conversation_boundaries(tmp_path):
    """get_sft_batch must sample windows at conversation starts: a uniform
    offset into the packed stream lands mid-conversation nearly always,
    supervising assistant tokens whose prompt is outside the window."""
    import numpy as np

    from src.model.train_sft import get_sft_batch, load_sft_data
    import scripts.prepare_sft_data as prep

    @sft_registry.register("fake_boundary")
    class FakeBoundary(SFTSource):
        languages = ("pt",)

        def load_examples(self, limit=None):
            for i in range(200):
                yield SFTExample(
                    messages=[
                        {"role": "user", "content": "Olá, tudo bem?"},
                        {"role": "assistant", "content": f"Sim, tudo ótimo, obrigado! {i}"},
                    ],
                    language="pt",
                    source="fake_boundary",
                    raw={},
                )

    try:
        tokenizer_dir = tmp_path / "tokenizer"
        tk = train_tokenizer(CORPUS, vocab_size=600)
        PreTrainedTokenizerFast(
            tokenizer_object=tk, additional_special_tokens=list(SPECIAL_TOKENS), eos_token=EOS_TOKEN
        ).save_pretrained(str(tokenizer_dir))

        out_dir = tmp_path / "sft"
        prep.run(
            sources=["fake_boundary"],
            languages=None,
            tokenizer_dir=tokenizer_dir,
            out_dir=out_dir,
            val_fraction=0.1,
            limit=None,
            seed=0,
        )

        block_size = 16
        tokens, labels, starts, meta = load_sft_data(out_dir, block_size)
        assert starts["train"] is not None

        import torch

        x, y = get_sft_batch(
            tokens["train"], labels["train"], block_size, 32, torch.device("cpu"), starts["train"]
        )
        assert x.shape == (32, block_size)

        # Every sampled window must begin at a recorded conversation start.
        stream = np.asarray(tokens["train"])
        valid = set(starts["train"].tolist())
        for row in x:
            matches = np.flatnonzero(stream[: len(stream) - block_size] == int(row[0]))
            assert any(int(m) in valid for m in matches)

        # And the first token of a window is never mid-assistant-turn: the
        # conversation's opening tokens are always masked (the user turn).
        assert (y[:, 0] == -1).all()
    finally:
        for name in ("fake_boundary",):
            sft_registry._REGISTRY.pop(name, None)
