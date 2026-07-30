"""Task 3 — a simple GPT-style decoder-only transformer.

nanoGPT-style architecture: learned positional embeddings, pre-norm
LayerNorm, GELU MLP, tied input/output embeddings. Deliberately the
"simple GPT" option the assignment allows (vs. a modern RoPE/RMSNorm/
SwiGLU stack) -- fewer moving parts, and it's the reference architecture
most scaling-law literature (Kaplan et al. 2020, GPT-2/GPT-3) itself
uses, which matters when this model is later used to reproduce that kind
of experiment.

A decoder-only transformer is trained on one task: given tokens
`[t_0, ..., t_{i-1}]`, predict `t_i`, for every position `i` at once
(teacher forcing). Causal self-attention is what makes this valid --
each position may only attend to itself and earlier positions, never
later ones, otherwise the model could "see the answer" during training
and would be useless at actual generation time (where future tokens
don't exist yet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # max context length (sequence length) the model can attend over
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True  # bias terms in Linear/LayerNorm; GPT-2 uses True, some newer models drop it


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Self-attention lets every position build a representation of "what do
    I need from the rest of the sequence to predict what's next", by
    comparing itself (a query) against every position's key and pooling
    their values by similarity. Splitting into `n_head` heads lets the
    model attend to several different kinds of relationships (e.g.
    syntax vs. topic) in parallel, each in a smaller subspace, instead of
    one big attention pattern.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # One fused matmul producing query, key, and value projections at
        # once (3x n_embd out-features) -- an efficiency trick, equivalent
        # to three separate Linear layers but one kernel launch instead of
        # three.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): split the embedding into
        # per-head chunks so each head's attention is computed independently.
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # scaled_dot_product_attention computes softmax(Q K^T / sqrt(d)) V
        # with the causal mask fused in (is_causal=True) -- position i's
        # query can only attend to keys/values at positions <= i. PyTorch
        # picks the fastest available kernel for the device (flash
        # attention on supported GPUs, a plain math fallback on CPU), so
        # this line doesn't change when this later runs on a GPU cluster.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network.

    Attention's job is gathering relevant information from other
    positions; the MLP's job is transforming what a position now knows
    into a better representation for predicting the next token. The
    4x expansion (n_embd -> 4*n_embd -> n_embd) is the standard
    "Transformer" (Vaswani et al. 2017) / GPT-2 ratio -- most of a GPT's
    parameters live here, not in attention.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One transformer block: attention sub-layer + MLP sub-layer.

    Pre-norm (LayerNorm applied *before* each sub-layer, not after) and
    residual ("skip") connections around each sub-layer -- `x = x +
    sublayer(norm(x))`. Both exist for the same reason: they keep
    gradients flowing cleanly through many stacked layers. Without the
    residual path, gradients have to travel through every layer's
    nonlinearity, which for a deep stack shrinks (vanishes) or blows up;
    the `+ x` gives gradients a direct shortcut back to earlier layers.
    Pre-norm (vs. GPT-1/original Transformer's post-norm) is the choice
    GPT-2 onward converged on because it trains more stably at depth.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """A decoder-only transformer language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the output projection (lm_head, "which vocab
        # token is this hidden state most like") reuses the *same*
        # weight matrix as the input token embedding (wte, "map this
        # vocab token to a hidden state"), rather than learning a second,
        # separate 32000 x n_embd matrix. Both weight matrices already
        # represent a token <-> vector mapping, so sharing them is a
        # long-standing GPT-2 trick that roughly halves the embedding
        # parameter count -- material here, since embeddings are ~20% of
        # total params at this tokenizer's 32,000 vocab (see the Task 2
        # tokenizer report).
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2's scaled init: residual-path projections get their std
        # scaled down by the number of layers, since their outputs sum
        # into the residual stream `n_layer` times -- without this,
        # activations grow with depth.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Total parameter count.

        `non_embedding=True` excludes the token+positional embedding
        tables. Scaling-law literature (Kaplan et al. 2020) reports model
        size this way, since embedding params scale with vocab size, not
        model "depth/width", and would otherwise distort the loss-vs-N
        power-law fit -- see scripts/run_scaling_sweep.py.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            # wte is excluded too, but wte and lm_head are tied (the same
            # tensor), so subtracting it once already removes it from the
            # total -- don't subtract twice.
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd), broadcasts over batch
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy between predicted next-token distribution and
            # the actual next token, at every position simultaneously.
            # Flattened to (B*T, vocab_size) vs. (B*T,) since
            # cross_entropy expects per-example logits, not per-sequence.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` continuation tokens.

        Not needed for training itself -- useful as a smoke test (if
        generation produces garbage-but-plausible-looking token
        sequences, the forward pass and weights are at least wired up
        correctly) and for eyeballing what a trained checkpoint learned.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
