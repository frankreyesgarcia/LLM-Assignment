"""Tests for the eval harness (Task 4).

Offline synthetic fixtures throughout -- no network, no downloaded
checkpoints. Per-task tests construct Doc objects directly with the exact
row shapes each real dataset returns (verified against the live
datasets-server API while building these tasks) instead of calling
load_docs, so they check the prompt/scoring logic without depending on
network access or the datasets library actually reaching HF.
"""

from __future__ import annotations

import torch

from src.eval import core, metrics
from src.eval.judge import NullJudge, extract_score, format_judge_prompt
from src.eval.model_adapter import EvalModel
from src.eval.runner import run_task
from src.eval.tasks.alba import ALBA
from src.eval.tasks.base import Task
from src.eval.tasks.calame_pt import CalamePT
from src.eval.tasks.chatrag_hi import ChatRAGHi
from src.eval.tasks.pt_culture import PTCulture
from src.eval.tasks.portugal_basic_qa import PortugalBasicQA
from src.eval.tasks.spanish_bench import _clean_metrics
from src.eval.types import Doc, EvalMode, RequestType, TaskResult

# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_normalize_text_strips_accents_case_and_punctuation():
    assert metrics.normalize_text("Lisboa!") == metrics.normalize_text("lisboa")
    assert metrics.normalize_text("competição") == "competicao"


def test_exact_match():
    assert metrics.exact_match("Lisboa", ["lisboa"]) == 1.0
    assert metrics.exact_match("Porto", ["Lisboa", "Coimbra"]) == 0.0


def test_first_word_match_ignores_trailing_generation():
    assert metrics.first_word_match(" reserva mais coisas depois", ["reserva"]) == 1.0
    assert metrics.first_word_match("outra coisa", ["reserva"]) == 0.0


def test_token_f1_partial_overlap():
    f1 = metrics.token_f1("a festa e em Braga", ["a festa acontece em Braga todos os anos"])
    assert 0.0 < f1 < 1.0
    assert metrics.token_f1("a festa e em Braga", ["a festa e em Braga"]) == 1.0


# ---------------------------------------------------------------------------
# core.py: score_continuation / greedy_generate against a hand-built model
# ---------------------------------------------------------------------------


class _FixedNextTokenModel:
    """Always predicts `next_id` regardless of input -- deterministic
    stand-in for GPT so stop-condition tests don't depend on trained
    weights ever producing a particular token.
    """

    def __init__(self, next_id: int, vocab_size: int):
        self.next_id = next_id
        self.vocab_size = vocab_size

    def __call__(self, idx: torch.Tensor):
        b, t = idx.shape
        logits = torch.full((b, t, self.vocab_size), -10.0)
        logits[:, :, self.next_id] = 10.0
        return logits, None


class _FakeTokenizer:
    """Minimal tokenizer stub: token id N decodes to the string "wN "."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"w{i} " for i in ids)


def test_greedy_generate_stops_on_stop_token_id():
    model = _FixedNextTokenModel(next_id=3, vocab_size=10)
    text = core.greedy_generate(
        model, _FakeTokenizer(), torch.device("cpu"), None, block_size=16,
        context_ids=[1, 2], stop_sequences=[], stop_token_ids=[3], max_new_tokens=5,
    )
    assert text == ""  # the very first predicted token is the stop token


def test_greedy_generate_stops_on_stop_sequence_text():
    model = _FixedNextTokenModel(next_id=7, vocab_size=10)  # decodes to "w7 w7 w7 ..."
    # "7 w7" only completes once 2 tokens ("w7 w7 ") have been generated, and
    # starts one character in -- so the truncated text keeps that one leading
    # character rather than being empty, unlike matching from position 0.
    text = core.greedy_generate(
        model, _FakeTokenizer(), torch.device("cpu"), None, block_size=16,
        context_ids=[1], stop_sequences=["7 w7"], stop_token_ids=[], max_new_tokens=10,
    )
    assert "7 w7" not in text
    assert text == "w"


def test_greedy_generate_respects_max_new_tokens():
    model = _FixedNextTokenModel(next_id=2, vocab_size=10)
    text = core.greedy_generate(
        model, _FakeTokenizer(), torch.device("cpu"), None, block_size=16,
        context_ids=[1], stop_sequences=[], stop_token_ids=[], max_new_tokens=4,
    )
    assert text == "w2 w2 w2 w2 "


def test_score_continuation_matches_greedy_when_continuation_is_argmax():
    model = _FixedNextTokenModel(next_id=5, vocab_size=10)
    logprob, is_greedy = core.score_continuation(
        model, torch.device("cpu"), None, block_size=16, context_ids=[1, 2], continuation_ids=[5, 5, 5],
    )
    assert is_greedy is True
    assert logprob > float("-inf")

    _, is_greedy_wrong = core.score_continuation(
        model, torch.device("cpu"), None, block_size=16, context_ids=[1, 2], continuation_ids=[5, 6],
    )
    assert is_greedy_wrong is False


# ---------------------------------------------------------------------------
# runner.py: prompt building + dispatch, against a fake EvalModel
# ---------------------------------------------------------------------------


class _FakeEvalModel:
    """Records what it was called with instead of running a real model, so
    tests can assert on the exact prompts the runner builds.
    """

    def __init__(self):
        self.loglikelihood_calls = []
        self.generate_calls = []

    def loglikelihood(self, context, continuation):
        self.loglikelihood_calls.append((context, continuation))
        # Prefer whichever continuation is alphabetically first, just to
        # get a deterministic, checkable "prediction".
        return (-float(ord(continuation.strip()[:1] or "z")), False)

    def generate_until(self, context, stop_sequences, max_new_tokens):
        self.generate_calls.append((context, stop_sequences, max_new_tokens))
        return "generated answer"

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "CHAT:" + "|".join(f"{m['role']}={m['content']}" for m in messages)


class _MCTask(Task):
    name = "mc_task"
    language = "pt"
    request_type = RequestType.LOGLIKELIHOOD
    description = "Escolhe a opção correta."

    def load_docs(self, limit=None):
        rows = [
            {"q": "Q1", "choices": ["A", "B"], "gold": 0},
            {"q": "Q2", "choices": ["C", "D"], "gold": 1},
            {"q": "Q3", "choices": ["E", "F"], "gold": 0},
        ]
        return [Doc(i, r) for i, r in enumerate(rows[:limit] if limit else rows)]

    def doc_to_text(self, doc):
        return f"Pergunta: {doc.raw['q']}\nResposta:"

    def doc_to_choices(self, doc):
        return doc.raw["choices"]

    def gold_index(self, doc):
        return doc.raw["gold"]


class _GenTask(Task):
    name = "gen_task"
    language = "hi"
    request_type = RequestType.GENERATE
    description = "Responde."
    stop_sequences = ["\n"]
    max_gen_toks = 32

    def load_docs(self, limit=None):
        rows = [{"q": "Q1", "ref": "gold1"}, {"q": "Q2", "ref": "gold2"}]
        return [Doc(i, r) for i, r in enumerate(rows[:limit] if limit else rows)]

    def doc_to_text(self, doc):
        return f"Q: {doc.raw['q']}\nA:"

    def doc_to_target(self, doc):
        return doc.raw["ref"]

    def doc_to_messages(self, doc):
        return [{"role": "user", "content": doc.raw["q"]}]

    def process_result(self, doc, prediction):
        return {"exact_match": metrics.exact_match(prediction, [doc.raw["ref"]])}


def test_run_task_loglikelihood_pretrain_builds_icl_context():
    model = _FakeEvalModel()
    result = run_task(_MCTask(), model, EvalMode.PRETRAIN, num_fewshot=0, limit=None)
    assert result.n_examples == 3
    assert "accuracy" in result.metrics
    # No fewshot: context is exactly description + doc_to_text.
    first_context = model.loglikelihood_calls[0][0]
    assert first_context == "Escolhe a opção correta.\n\nPergunta: Q1\nResposta:"


def test_run_task_loglikelihood_fewshot_excludes_current_doc():
    model = _FakeEvalModel()
    run_task(_MCTask(), model, EvalMode.PRETRAIN, num_fewshot=1, limit=None, seed=0)
    for context, _ in model.loglikelihood_calls:
        # The fewshot example's own gold answer text must appear in-context,
        # but a doc can never be its own fewshot shot -- with 3 docs and
        # k=1, the shot is always one of the *other* two docs' text.
        assert context.count("Pergunta:") == 2  # 1 fewshot + the real question


def test_run_task_posttrain_uses_chat_template():
    model = _FakeEvalModel()
    run_task(_MCTask(), model, EvalMode.POSTTRAIN, num_fewshot=0, limit=None)
    context = model.loglikelihood_calls[0][0]
    assert context.startswith("CHAT:")
    assert "system=Escolhe a opção correta." in context
    assert "user=Pergunta: Q1" in context


def test_run_task_generate_dispatches_to_generate_until_and_scores():
    model = _FakeEvalModel()
    result = run_task(_GenTask(), model, EvalMode.PRETRAIN, num_fewshot=0, limit=None)
    assert len(model.generate_calls) == 2
    assert result.metrics["exact_match"] == 0.0  # "generated answer" != "gold1"/"gold2"


def test_run_task_alba_without_judge_is_unscored_but_runs():
    model = _FakeEvalModel()
    task = ALBA()
    task.load_docs = lambda limit=None: [
        Doc(0, {"prompt_id": "p0", "category": "Lexicology", "prompt": "Dá um sinónimo de crise."})
    ]
    result = run_task(task, model, EvalMode.PRETRAIN, num_fewshot=0, limit=None, judge=NullJudge())
    assert result.n_examples == 1
    assert "judge_score" not in result.metrics  # None-valued -> dropped, not averaged as 0
    assert result.metrics["judged"] == 0.0


# ---------------------------------------------------------------------------
# Per-task schema handling, against real row shapes (no network)
# ---------------------------------------------------------------------------


def test_calame_pt_doc_to_text_and_scoring():
    task = CalamePT()
    doc = Doc(0, {"id": 0, "sentence": "...essa nova aventura nos", "last_word": "reserva"})
    assert task.doc_to_text(doc) == "...essa nova aventura nos"
    assert task.doc_to_target(doc) == "reserva"
    assert task.process_result(doc, " reserva.") == {"accuracy": 1.0}
    assert task.process_result(doc, " outra coisa") == {"accuracy": 0.0}


def test_portugal_basic_qa_gold_index_from_label():
    task = PortugalBasicQA()
    doc = Doc(0, {"question": "Qual é a capital de Portugal?", "choices": ["Porto", "Lisboa", "Coimbra"], "label": "b", "answer": "Lisboa"})
    assert task.doc_to_choices(doc) == ["Porto", "Lisboa", "Coimbra"]
    assert task.gold_index(doc) == 1
    assert task.doc_to_target(doc) == "Lisboa"


def test_pt_culture_uses_last_turn_as_target():
    task = PTCulture()
    doc = Doc(
        0,
        {
            "conversations": [
                {"role": "human", "content": "Qual a importância da Romaria?"},
                {"role": "gpt", "content": "É uma festa realizada em Soutelo, Vila Verde."},
            ],
            "category": "Festivals",
            "_task_type": "QA_Short",
            "_seed_id": "FEST_030",
        },
    )
    assert task.doc_to_target(doc) == "É uma festa realizada em Soutelo, Vila Verde."
    messages = task.doc_to_messages(doc)
    assert messages == [{"role": "user", "content": "Qual a importância da Romaria?"}]
    result = task.process_result(doc, "É uma festa realizada em Soutelo, Vila Verde.")
    assert result["exact_match"] == 1.0
    assert result["token_f1"] == 1.0


def test_chatrag_hi_context_block_and_f1():
    task = ChatRAGHi()
    doc = Doc(
        0,
        {
            "answers": ["क्या आप योजना बना रहे हैं?"],
            "document": "ssa",
            "messages": [{"role": "user", "content": "मुझे मदद चाहिए।"}],
            "ctxs": [{"title": "", "text": "लाभ योजनाकार: उत्तरजीवी सामाजिक सुरक्षा जानकारी।"}] * 2,
            "ground_truth_ctx": {"ctx": "लाभ योजनाकार", "index": 0},
        },
    )
    messages = task.doc_to_messages(doc)
    assert messages[0]["role"] == "system"
    assert "सामाजिक सुरक्षा" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "मुझे मदद चाहिए।"}
    assert task.doc_to_target(doc) == "क्या आप योजना बना रहे हैं?"
    exact = task.process_result(doc, "क्या आप योजना बना रहे हैं?")["token_f1"]
    assert exact == 1.0


# ---------------------------------------------------------------------------
# judge.py
# ---------------------------------------------------------------------------


def test_format_judge_prompt_includes_category_prompt_answer():
    prompt = format_judge_prompt("Dá um exemplo.", "Aqui está.", "Lexicology")
    assert "Lexicology" in prompt
    assert "Dá um exemplo." in prompt
    assert "Aqui está." in prompt


def test_extract_score_parses_score_and_handles_garbage():
    assert extract_score("Raciocínio: boa resposta\nPontuação Global: 4") == 4.0
    assert extract_score("no score here") is None


def test_null_judge_always_returns_none():
    assert NullJudge().score("p", "a", "cat") is None


# ---------------------------------------------------------------------------
# spanish_bench.py: lm-eval results normalization
# ---------------------------------------------------------------------------


def test_clean_metrics_drops_stderr_and_alias_keeps_numeric():
    raw = {"alias": "xnli_es", "acc,none": 0.42, "acc_stderr,none": 0.05, "acc_norm,none": 0.5}
    assert _clean_metrics(raw) == {"acc": 0.42, "acc_norm": 0.5}


# ---------------------------------------------------------------------------
# model_adapter.py: EvalModel against a real tiny GPT + trained tokenizer
# ---------------------------------------------------------------------------


def _build_tiny_eval_model(tmp_path) -> EvalModel:
    from src.model.gpt import GPT, GPTConfig
    from src.tokenizer.train import save_pretrained, train_tokenizer

    corpus = [
        "Ola mundo isto e um pequeno teste.",
        "Hola mundo esto es una prueba pequena.",
        "यह एक छोटा सा परीक्षण है।",
    ] * 5
    tokenizer = train_tokenizer(iter(corpus), vocab_size=280)
    tok_dir = tmp_path / "tokenizer"
    fast_tokenizer = save_pretrained(tokenizer, tok_dir)

    cfg = GPTConfig(vocab_size=len(fast_tokenizer), block_size=32, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_cfg": cfg, "model_state_dict": model.state_dict(), "iter_num": 0}, ckpt_path)

    return EvalModel(ckpt_path, tok_dir, device="cpu", amp_dtype="fp32")


def test_eval_model_loglikelihood_and_generate_smoke(tmp_path):
    eval_model = _build_tiny_eval_model(tmp_path)

    logprob, is_greedy = eval_model.loglikelihood("Ola mundo", " isto")
    assert isinstance(logprob, float)
    assert isinstance(is_greedy, bool)

    text = eval_model.generate_until("Ola mundo", stop_sequences=["\n"], max_new_tokens=4)
    assert isinstance(text, str)


def test_eval_model_apply_chat_template_renders_llama3_style_headers(tmp_path):
    eval_model = _build_tiny_eval_model(tmp_path)
    rendered = eval_model.apply_chat_template(
        [{"role": "user", "content": "Ola"}], add_generation_prompt=True
    )
    assert "<|start_header_id|>user<|end_header_id|>" in rendered
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_task_result_is_json_roundtrippable():
    import json
    from dataclasses import asdict

    result = TaskResult("calame_pt", "pt", EvalMode.PRETRAIN, 0, 10, {"accuracy": 0.5})
    json.dumps(asdict(result))  # must not raise
