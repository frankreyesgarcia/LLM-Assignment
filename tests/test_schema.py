import json

import pytest

from src.ingest.base import Document, normalize_language_code


def test_document_to_dict_roundtrip():
    doc = Document(text="hola mundo", language="es", source="test", source_id="abc123")
    d = doc.to_dict()
    assert d == {
        "text": "hola mundo",
        "language": "es",
        "source": "test",
        "source_id": "abc123",
        "url": None,
        "metadata": "{}",
    }


def test_document_to_dict_serializes_metadata_as_json_string():
    # metadata must be a JSON string, not a nested struct: different sources
    # have incompatible field types for the same key (e.g. int vs float
    # `educational_score`), which breaks pyarrow.concat_tables when merging
    # pt+es+hi into the `all` config (Etapa 6).
    doc = Document(
        text="hola",
        language="es",
        source="test",
        source_id="x",
        metadata={"educational_score": 4.5, "nested": {"a": 1}},
    )
    d = doc.to_dict()
    assert isinstance(d["metadata"], str)
    assert json.loads(d["metadata"]) == {"educational_score": 4.5, "nested": {"a": 1}}


def test_document_generates_source_id_when_missing():
    doc = Document(text="hola mundo", language="es", source="test", source_id="")
    assert doc.source_id != ""
    assert len(doc.source_id) == 16


def test_document_rejects_invalid_language():
    with pytest.raises(ValueError):
        Document(text="hello", language="en", source="test", source_id="x")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("spa_Latn", "es"),
        ("SPA", "es"),
        ("pt-BR", "pt"),
        ("ptbr", "pt"),
        ("por_Latn", "pt"),
        ("hin_Deva", "hi"),
        ("fr", None),
        ("", None),
    ],
)
def test_normalize_language_code(raw, expected):
    assert normalize_language_code(raw) == expected
