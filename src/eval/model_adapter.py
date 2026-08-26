"""EvalModel (Task 4): the one thing every native task module talks to.

Wraps a loaded checkpoint + tokenizer behind three mode-agnostic
operations -- loglikelihood, generate_until, apply_chat_template. Tasks
never touch torch directly; they build a prompt string (or chat messages)
and hand it to EvalModel, so the same task code runs in both EvalModes.
"""

from __future__ import annotations

from pathlib import Path

from src.eval import core


class EvalModel:
    def __init__(
        self,
        ckpt_path: Path,
        tokenizer_dir: Path,
        device: str = "auto",
        amp_dtype: str = "auto",
    ) -> None:
        self.device = core.resolve_device(device)
        self.model = core.load_checkpoint(ckpt_path, self.device)
        self.tokenizer = core.load_tokenizer(tokenizer_dir)
        self.amp_dtype = core.resolve_amp_dtype(amp_dtype, self.device)
        self.block_size = self.model.config.block_size
        self.n_params = self.model.num_params()
        # Llama-3-style chat template (src/tokenizer/train.py) ends a turn
        # with <|eot_id|>, not tokenizer.eos_token (<|end_of_text|>) --
        # generate_until needs to stop there too in posttrain mode. Checked
        # by token ID, not text: skip_special_tokens=True decoding never
        # surfaces special tokens as text for a string-match stop to catch.
        self._stop_token_ids = [
            tid
            for tid in (self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            if tid is not None and tid != self.tokenizer.unk_token_id
        ]

    def encode(self, text: str) -> list[int]:
        return core.encode(self.tokenizer, text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Split (context, continuation) into token ID lists, keeping the
        word-boundary space (if any) with the continuation rather than the
        context -- otherwise byte-level BPE can merge "hello" + " world"
        differently than it would tokenize "hello world" as one string.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.encode(context + continuation)
        context_ids = self.encode(context) if context else []
        return context_ids, whole[len(context_ids) :]

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        """log P(continuation | context) under the model, and whether
        continuation is what greedy decoding would produce. Used by
        LOGLIKELIHOOD tasks to rank candidate answers -- see
        tasks/base.py's RequestType docstring.
        """
        if context == "":
            bos_id = self.tokenizer.bos_token_id
            cont_ids = self.encode(continuation)
            context_ids = [bos_id] if bos_id is not None else cont_ids[:1]
            if bos_id is None:
                cont_ids = cont_ids[1:]
        else:
            context_ids, cont_ids = self._encode_pair(context, continuation)
        return core.score_continuation(
            self.model, self.device, self.amp_dtype, self.block_size, context_ids, cont_ids
        )

    def generate_until(self, context: str, stop_sequences: list[str], max_new_tokens: int) -> str:
        context_ids = self.encode(context)
        return core.greedy_generate(
            self.model,
            self.tokenizer,
            self.device,
            self.amp_dtype,
            self.block_size,
            context_ids,
            stop_sequences,
            self._stop_token_ids,
            max_new_tokens,
        )

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
