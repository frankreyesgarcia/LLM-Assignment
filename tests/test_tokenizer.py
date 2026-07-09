from pathlib import Path

from transformers import PreTrainedTokenizerFast

from src.tokenizer.train import (
    EOT_TOKEN,
    SPECIAL_TOKENS,
    START_HEADER_TOKEN,
    save_pretrained,
    train_tokenizer,
)

# Offline synthetic corpus -- no network, doesn't depend on the live HF dataset.
PT_TEXT = "Olá mundo, como vai você? Este é um texto de teste em português."
ES_TEXT = "Hola mundo, ¿cómo estás? Este es un texto de prueba en español."
HI_TEXT = "नमस्ते दुनिया, आप कैसे हैं? यह हिंदी में एक परीक्षण पाठ है।"
MIXED_TEXT = f"{PT_TEXT} {ES_TEXT} {HI_TEXT}"
CORPUS = [PT_TEXT, ES_TEXT, HI_TEXT] * 30
VOCAB_SIZE = 1000


def _train():
    return train_tokenizer(CORPUS, vocab_size=VOCAB_SIZE)


def test_vocab_size_within_bounds():
    tokenizer = _train()
    byte_alphabet_floor = 256 + len(SPECIAL_TOKENS)
    assert byte_alphabet_floor <= tokenizer.get_vocab_size() <= VOCAB_SIZE


def test_encode_decode_roundtrip():
    tokenizer = _train()
    for text in [PT_TEXT, ES_TEXT, HI_TEXT, MIXED_TEXT]:
        encoding = tokenizer.encode(text)
        assert tokenizer.decode(encoding.ids) == text


def test_no_unk_tokens_in_output():
    tokenizer = _train()
    assert "<unk>" not in tokenizer.get_vocab()
    for text in [PT_TEXT, ES_TEXT, HI_TEXT, MIXED_TEXT]:
        assert "�" not in tokenizer.decode(tokenizer.encode(text).ids)


def test_save_and_reload_roundtrip(tmp_path: Path):
    tokenizer = _train()
    saved = save_pretrained(tokenizer, tmp_path)

    reloaded = PreTrainedTokenizerFast.from_pretrained(str(tmp_path))
    for text in [PT_TEXT, ES_TEXT, HI_TEXT, MIXED_TEXT]:
        assert reloaded.encode(text) == saved.encode(text)


def test_chat_template_produces_expected_structure(tmp_path: Path):
    tokenizer = _train()
    saved = save_pretrained(tokenizer, tmp_path)

    rendered = saved.apply_chat_template(
        [
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá"},
        ],
        tokenize=False,
    )
    assert rendered.count(START_HEADER_TOKEN) == 2
    assert rendered.count(EOT_TOKEN) == 2
    assert "user" in rendered and "assistant" in rendered

    encoded = saved.apply_chat_template(
        [{"role": "user", "content": "oi"}], tokenize=True, add_generation_prompt=True
    )
    assert len(encoded["input_ids"]) > 0
