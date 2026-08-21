"""Pluggable LLM-as-judge scoring (Task 4), for ALBA (tasks/alba.py).

ALBA's own eval (github.com/AMALIA-LLM/alba-benchmark) has no fixed
reference answer to compare against -- prompts like "give an example of a
Portuguese neologism" are scored 1-5 by an LLM judge against a rubric.
Reproducing their exact rubric/few-shot format so a judge here is scoring
the same thing their paper's judge scores; the judge backend itself is
pluggable so this harness doesn't hard-depend on any one LLM API.
"""

from __future__ import annotations

import csv
import io
import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import requests

FEWSHOT_URL = "https://raw.githubusercontent.com/AMALIA-LLM/alba-benchmark/main/evaluation/fewshot_samples.csv"

# Category -> 1-5 rubric, adapted verbatim from alba-benchmark's
# evaluation/scorer.py (INSTRUCTIONS_1_SCORE_PT), which is category-agnostic.
RUBRIC_PT = """\
1 - Muito Mau
    Muito Imprecisa: Quase nada está correto ou preciso, se é que algo está.
    Qualidade Muito Baixa: A resposta carece de formatação, a linguagem usada é confusa e difícil de seguir, e/ou a qualidade da linguagem é muito baixa.
    Muito Incompleta: Quase não há contexto e/ou justificação dada com a resposta ou é completamente irrelevante/mal justificada.
2 - Mau
    Imprecisa: Há várias imprecisões ao longo da resposta.
    Qualidade Baixa: A resposta não está bem estruturada, a linguagem carece de clareza e/ou tem vários erros.
    Incompleta: Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
3 - Algo Mau
    Algo Imprecisa: Há alguma informação imprecisa ou incorreta.
    Qualidade Aceitável: A resposta está razoavelmente bem formatada e clara, e tem apenas alguns erros.
    Algo Incompleta: Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
4 - Algo Bom
    Algo Precisa: Quase não há imprecisões ou erros.
    Boa Qualidade: A resposta está maioritariamente bem estruturada e clara. Quase não há erros.
    Algo Completa: A resposta fornece algum contexto/justificação/explicação que tem relevância.
5 - Muito Bom
    Muito Precisa: Não há imprecisões ou erros.
    Qualidade Muito Alta: A resposta está muito bem estruturada, a linguagem é clara e fácil de entender.
    Muito Completa: A resposta está completa, bem justificada e fornece contexto adicional.
"""

JUDGE_PROMPT_PT = """\
És um Classificador Profissional de Texto em Português Europeu.
Vais avaliar uma determinada interação na categoria de: {category}

Terás de pontuar de 1 a 5, aqui estão as instruções:
{rubric}

Para te ajudar na tua avaliação, aqui estão alguns exemplos:
{few_shots}

A tua resposta deve estar no seguinte formato:
Raciocínio: <raciocínio>
Pontuação Global: <pontuação de 1 a 5>

Agora é a tua vez, analisa esta interação e responde:
Prompt: {prompt}
Resposta: {answer}
"""

_SCORE_RE = re.compile(r"Pontuação Global:\s*(\d+)")


@lru_cache(maxsize=1)
def _load_fewshot_examples() -> dict[str, list[dict[str, str]]]:
    """category -> its 4 calibration rows (12 scored example answers: 3 per
    row, at ratings 5/4-3/2-1) from alba-benchmark's own fewshot_samples.csv
    -- fetched once per process. Mirrors evaluation/scorer.py's FEWSHOT_CSV
    exactly (same file, same grouping), since the judge's calibration is
    what keeps its 1-5 scale meaningful across categories.
    """
    response = requests.get(FEWSHOT_URL, timeout=30)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    by_category: dict[str, list[dict[str, str]]] = {}
    for row in reader:
        by_category.setdefault(row["category"], []).append(row)
    return by_category


def _format_fewshots(category: str) -> str:
    rows = _load_fewshot_examples().get(category, [])
    blocks = []
    for row in rows:
        for suffix, rating_key in (("A (5)", "Rating"), ("B (4-3)", "Rating.1"), ("C (2-1)", "Rating.2")):
            blocks.append(
                f"Prompt: {row['Prompt']}\n"
                f"Resposta: {row[f'Answer {suffix}']}\n"
                f"Pontuação Global: {row[rating_key]}"
            )
    return "\n\n".join(blocks)


def format_judge_prompt(prompt: str, answer: str, category: str) -> str:
    return JUDGE_PROMPT_PT.format(
        category=category, rubric=RUBRIC_PT, few_shots=_format_fewshots(category), prompt=prompt, answer=answer
    )


def extract_score(judge_response: str) -> float | None:
    match = _SCORE_RE.search(judge_response)
    return float(match.group(1)) if match else None


class Judge(ABC):
    @abstractmethod
    def score(self, prompt: str, answer: str, category: str) -> float | None:
        """Return a 1-5 score, or None if the judge's response couldn't be parsed."""


class NullJudge(Judge):
    """No-op judge: records that scoring was skipped rather than calling out
    to any API. The default when no judge backend is configured -- ALBA
    generations still get produced and saved, just left unscored.
    """

    def score(self, prompt: str, answer: str, category: str) -> float | None:
        return None


class AnthropicJudge(Judge):
    """Scores with the Claude API, using ALBA's own rubric/prompt format.

    Requires the `anthropic` package and `ANTHROPIC_API_KEY` -- both are
    optional (see requirements.txt / --judge cli flag), so importing this
    module never fails; only instantiating this class does.
    """

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def score(self, prompt: str, answer: str, category: str) -> float | None:
        judge_prompt = format_judge_prompt(prompt, answer, category)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return extract_score(text)


def build_judge(spec: str | None) -> Judge:
    """spec: None/"null" for NullJudge, "anthropic[:model]" for AnthropicJudge."""
    if not spec or spec == "null":
        return NullJudge()
    if spec.startswith("anthropic"):
        model = spec.split(":", 1)[1] if ":" in spec else "claude-sonnet-5"
        return AnthropicJudge(model=model)
    raise ValueError(f"unknown judge spec {spec!r}")
