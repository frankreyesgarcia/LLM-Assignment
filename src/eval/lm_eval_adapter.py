"""Adapts EvalModel to lm-evaluation-harness's `TemplateLM` interface
(Task 4), so spanish_bench.py can hand our GPT checkpoint straight to
`lm_eval.simple_evaluate` and get the real spanish_bench suite -- rather
than reimplementing a hand-picked subset of its ~30 task variants.

Only the required abstract methods are implemented (tok_encode,
eot_token_id, _loglikelihood_tokens, loglikelihood_rolling,
generate_until), each a thin loop over EvalModel/core primitives -- no
padded batching (see core.py's module docstring for why: this repo's
checkpoints and eval sets are small enough that per-example calls are
plenty fast, and it avoids implementing attention-mask-aware batching for
a model whose forward() doesn't take one).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lm_eval.api.model import TemplateLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from src.eval import core

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

    from src.eval.model_adapter import EvalModel


class LMEvalAdapter(TemplateLM):
    backend = "causal"

    def __init__(self, eval_model: "EvalModel") -> None:
        super().__init__()
        self.eval_model = eval_model
        self.tokenizer = eval_model.tokenizer

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "eval-model-tokenizer")

    def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
        return core.encode(self.tokenizer, string)

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        return self.eval_model.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        return [
            core.score_continuation(
                self.eval_model.model,
                self.eval_model.device,
                self.eval_model.amp_dtype,
                self.eval_model.block_size,
                context_enc,
                continuation_enc,
            )
            for _, context_enc, continuation_enc in requests
        ]

    def loglikelihood_rolling(self, requests: list["Instance"], disable_tqdm: bool = False) -> list[float]:
        results = []
        for (string,) in [req.args for req in requests]:
            windows = [
                make_disjoint_window(w)
                for w in get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.eval_model.block_size,
                    context_len=1,
                )
            ]
            total = sum(
                core.score_continuation(
                    self.eval_model.model,
                    self.eval_model.device,
                    self.eval_model.amp_dtype,
                    self.eval_model.block_size,
                    ctx or [self.eot_token_id],
                    cont,
                )[0]
                for ctx, cont in windows
            )
            results.append(total)
        return results

    def generate_until(self, requests: list["Instance"], disable_tqdm: bool = False) -> list[str]:
        results = []
        for context, gen_kwargs in [req.args for req in requests]:
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)
            results.append(self.eval_model.generate_until(context, until, max_gen_toks))
        return results
