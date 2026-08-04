"""Task 3 -- fit a training run's token count / step count to a stated FLOPs
budget for a *given* model shape.

Earlier revisions of this file derived both model shape and token count
from a FLOPs budget via a compute-optimal N-vs-D split borrowed directly
from Kaplan et al. (2020)'s or Hoffmann et al. (2022) "Chinchilla"'s own
fitted power-law exponents. Per review: that's invalid -- those exponents
are the output of a regression fit to *their* architecture, optimizer/LR
schedule, tokenizer and dataset, and there's no reason this repo's own
loss-vs-(N,D) surface shares them (Chinchilla's own Section 3.4 blames much
of Kaplan's exponents on schedule choices alone). Deriving our own exponents
means literally running Chinchilla's "Approach 2: IsoFLOP profiles" method on
our own setup: train several model sizes per compute budget on our own data,
find each budget's loss-minimizing N, then fit a scaling law across budgets.
That's real GPU/wall-clock cost and is scoped to a later branch -- this
branch only adds the `--flops-budget` plumbing itself, using a model shape
chosen by hand (same as before any scaling-law code existed) rather than
one derived from a law we haven't fit yet.

What *is* architecture/data-agnostic, and so is safe to keep now, is the
FLOPs *accounting* identity below (`FLOPS_PER_PARAM_TOKEN`): given how many
parameters a model has, how many FLOPs one training step costs is just
counting matmuls, not a regression fit -- both papers use this exact
identity (Kaplan 2020 Section 2.1; Chinchilla Section 3.3/Appendix F) as a
>=90%-accurate stand-in for their own exact per-architecture FLOP count
(Chinchilla Table A4). So this file answers a narrower question: given a
model shape you already chose, how many tokens / training steps fit inside
a stated FLOPs budget.
"""

from __future__ import annotations

# FLOPs(N, D) ~= 6*N*D -- 2 for the forward pass, 4 for the backward pass
# (twice the forward, since gradients are needed w.r.t. both inputs and
# weights), per parameter per token (Kaplan et al. 2020 Section 2.1;
# Chinchilla Section 3.3 cites the same identity, and Appendix F/Table A4
# check it against an exact per-architecture FLOP count, finding <=10%
# error). N here is the model's *total* parameter count, embeddings
# included, since a forward pass genuinely spends FLOPs on the embedding
# lookup and (mainly) the output logits matmul -- see
# GPT.num_params(non_embedding=False), src/model/gpt.py.
FLOPS_PER_PARAM_TOKEN = 6


def tokens_for_budget(flops: float, n_params: int) -> int:
    """D(C, N): how many training tokens fit in FLOPs budget `C` for a model
    with `n_params` (total) parameters, at FLOPs(N, D) = 6*N*D. Assumes a
    single epoch (no repeated tokens) -- the caller is responsible for
    having enough distinct training tokens available.

    This is pure FLOPs bookkeeping for a *given* N -- it does not choose N;
    see this module's docstring for why choosing N is out of scope here.
    """
    return round(flops / (FLOPS_PER_PARAM_TOKEN * n_params))


def max_iters_for_budget(flops: float, n_params: int, batch_size: int, block_size: int) -> int:
    """Convert `tokens_for_budget`'s token count into a training-loop
    `max_iters`, given how many tokens one step consumes
    (batch_size * block_size).
    """
    tokens = tokens_for_budget(flops, n_params)
    return max(1, round(tokens / (batch_size * block_size)))


def describe_budget(
    flops: float, n_params: int, n_layer: int, n_head: int, n_embd: int, batch_size: int, block_size: int
) -> str:
    """One-line human-readable summary, shared by scripts/train.py's
    --flops-budget and scripts/plan_compute_budget.py so the two don't
    duplicate this string.
    """
    tokens = tokens_for_budget(flops, n_params)
    max_iters = max_iters_for_budget(flops, n_params, batch_size, block_size)
    return (
        f"{flops:.3e} FLOPs budget for n_layer={n_layer}, n_head={n_head}, n_embd={n_embd} "
        f"({n_params:,} total params): D={tokens:,} tokens -> max_iters={max_iters:,} "
        f"(batch_size={batch_size} x block_size={block_size})"
    )
