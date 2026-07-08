from src.filters.clean import clean_text, collapse_whitespace, fix_mojibake, strip_control_chars
from src.filters.language import hard_filter
from src.filters.quality import QualityConfig, check_quality
from src.ingest.base import Document


def _doc(text: str, language: str = "pt") -> Document:
    return Document(text=text, language=language, source="test", source_id="x")


# --- language.hard_filter ---


def test_hard_filter_accepts_target_languages():
    assert hard_filter(_doc("texto", "pt"))
    assert hard_filter(_doc("texto", "es"))
    assert hard_filter(_doc("texto", "hi"))


# --- clean.py ---


def test_fix_mojibake():
    assert fix_mojibake("informaÃ§Ã£o") == "informação"


def test_collapse_whitespace():
    assert collapse_whitespace("a   b\n\n\n\nc") == "a b\n\nc"


def test_strip_control_chars():
    assert strip_control_chars("a\x00b\x01c") == "abc"
    assert strip_control_chars("line1\nline2\ttab") == "line1\nline2\ttab"  # keeps \n and \t


def test_clean_text_dedupes_consecutive_lines():
    text = "hello\nhello\nworld"
    assert clean_text(text) == "hello\nworld"


# --- quality.py ---


def test_check_quality_drops_short_docs():
    cfg = QualityConfig(min_words=50)
    assert check_quality("short text", cfg) == "too_short"


def test_check_quality_keeps_normal_prose():
    cfg = QualityConfig()
    sentences = [
        "Este é um texto de exemplo com várias frases normais.",
        "Contiene puntuación estándar, como comas y puntos.",
        "También puede tener **énfasis** en markdown sin ser ruido.",
        "Cada frase trae palabras distintas para evitar repeticiones de n-gramas.",
        "El pipeline de calidad no debería marcar esto como spam ni como ruido.",
        "Esta frase final agrega aún más variedad léxica al texto de prueba.",
    ]
    text = " ".join(sentences * 3)
    assert check_quality(text, cfg) is None


def test_check_quality_does_not_flag_markdown_emphasis():
    # Regression test: '*' (markdown bold/italic) and '_' (snake_case) must
    # not be treated as noise symbols -- found via the pilot run on
    # corpus-ptbr-v2, where this false-positived ~50% of legitimate articles.
    cfg = QualityConfig()
    sentences = [
        "Este artigo explica um tema com bastante detalhe e clareza.",
        "O uso de smtp_tls_policy_maps é comum em configurações de Postfix.",
        "Cada parágrafo traz uma ideia nova para evitar repetições excessivas.",
        "O texto continua com mais uma frase distinta sobre o assunto.",
    ]
    text = "**Título do Artigo**\n\n" + " ".join(sentences * 3)
    assert check_quality(text, cfg) is None


def test_check_quality_flags_code_like_noise():
    cfg = QualityConfig(min_words=5)
    text = "{foo: bar, baz: [1,2,3]}. " * 20
    assert check_quality(text, cfg) == "too_many_symbols"


def test_check_quality_flags_banned_substring():
    cfg = QualityConfig()
    text = ("Normal words here. " * 30) + "Lorem ipsum dolor sit amet."
    assert check_quality(text, cfg) == "banned_substring:lorem ipsum"


def test_check_quality_flags_ngram_repetition():
    cfg = QualityConfig(min_words=5, max_ngram_repetitions=3)
    text = "The exact same phrase repeats here. " * 10
    assert check_quality(text, cfg) == "ngram_repetition"


def test_check_quality_allows_moderate_topical_repetition():
    # Pilot finding: legitimate articles repeat a topical 4-gram several
    # times (e.g. "um poste de energia" appeared 7x in a real doc).
    cfg = QualityConfig(max_ngram_repetitions=10)
    topical = "Um poste de energia caiu na rua ontem à noite."
    filler = [
        "Isso é um tema importante para todos na cidade.",
        "Vizinhos relataram o barulho alto durante a madrugada.",
        "A equipe de manutenção chegou horas depois do ocorrido.",
    ]
    text = " ".join(topical if i % 3 == 0 else filler[i % len(filler)] for i in range(21))
    assert check_quality(text, cfg) is None
