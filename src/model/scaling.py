"""Task 3 -- two separate jobs that share one FLOPs-accounting identity.

1. Fit a training run's token count / step count to a stated FLOPs budget
   for a *given* model shape (tokens_for_budget / max_iters_for_budget /
   describe_budget below) -- doesn't choose n_layer/n_head/n_embd, just
   converts a budget into D/max_iters for a shape you already picked.

2. Choose that model shape in the first place, via the Chinchilla
   "IsoFLOP profiles" method (gpt_shape_for_width / fit_isoflop_parabola /
   parabola_vertex / fit_power_law / optimal_params_and_tokens /
   nearest_width_for_params below): train a grid of small models at
   several FLOP budgets (scripts/isoflop_sweep.py), fit a parabola per
   budget in log(N)-vs-loss space and read off each one's minimum
   (compute-optimal N and D for that budget), then fit a power law through
   those minima across budgets (scripts/fit_isoflop_scaling_law.py) and
   extrapolate it to the real training run's budget. Unlike Kaplan's
   original method, this holds compute fixed and varies the N-vs-D split
   directly, which is what makes it recover the actual compute-optimal
   trade-off instead of being biased toward oversized models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# FLOPs(N, D) ~= 6*N*D -- 2 for the forward pass, 4 for the backward pass
# (twice the forward, since gradients are needed w.r.t. both inputs and
# weights), per parameter per token. N here is the model's *total*
# parameter count, embeddings included, since a forward pass genuinely
# spends FLOPs on the embedding lookup and (mainly) the output logits
# matmul -- see GPT.num_params(non_embedding=False), src/model/gpt.py.
FLOPS_PER_PARAM_TOKEN = 6

# Target attention head size (n_embd / n_head) across the IsoFLOP width
# grid -- widths that can't hit it exactly get the closest achievable head
# size instead (see gpt_shape_for_width).
# Not a derived constant -- it's the convention all
# four official GPT-2 variants use (768/12, 1024/16, 1280/20, 1600/25 all
# = 64) and GPT-3's smaller variants keep too, though not held rigidly at
# the largest scale (full GPT-3-175B grows head size to 128 instead --
# see Levinstein, "A Conceptual Guide to Transformers, Part II":
# https://benlevinstein.substack.com/p/a-conceptual-guide-to-transformers-b70).
# Also already what this repo's own real run uses (scripts/slurm/
# 07_pretrain.sh: n_layer=8, n_head=8, n_embd=512 -> 512/8=64), so the
# sweep's models stay shape-consistent with it instead of introducing a
# new, unrelated convention.
WIDTH_HEAD_DIM = 64

# gpt_shape_for_width anchors its depth curve to the real run's own shape
# (scripts/slurm/07_pretrain.sh: n_layer=8, n_embd=512) rather than an
# arbitrary point, so the shape family the sweep trains passes exactly
# through the real run's own point at width=ANCHOR_WIDTH.
ANCHOR_WIDTH = 512
ANCHOR_LAYER = 8

# Depth grows *slower* than width (n_layer ~ n_embd**DEPTH_WIDTH_EXPONENT,
# DEPTH_WIDTH_EXPONENT < 1) instead of the two scaling in lockstep. Fit by
# log-log regression against the published GPT-2/GPT-3 model family's
# (n_layer, n_embd) pairs -- (12,768) (24,1024) (24,1536) (24,2048)
# (32,2560) (32,4096) (40,5140) (96,12288) -- which gives an exponent of
# ~0.61; 0.5 (plain sqrt) is used here as the closest simple fraction, on
# the same side of 1 as that fit. This replaces an earlier n_layer =
# n_embd // WIDTH_HEAD_DIM rule (depth and width scaling in lockstep,
# i.e. exponent exactly 1) that had no such grounding.
DEPTH_WIDTH_EXPONENT = 0.5


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
    """One-line human-readable summary printed by scripts/train.py's
    --flops-budget.
    """
    tokens = tokens_for_budget(flops, n_params)
    max_iters = max_iters_for_budget(flops, n_params, batch_size, block_size)
    return (
        f"{flops:.3e} FLOPs budget for n_layer={n_layer}, n_head={n_head}, n_embd={n_embd} "
        f"({n_params:,} total params): D={tokens:,} tokens -> max_iters={max_iters:,} "
        f"(batch_size={batch_size} x block_size={block_size})"
    )


def gpt_shape_for_width(n_embd: int) -> tuple[int, int, int]:
    """(n_layer, n_head, n_embd) for a GPT-shaped model of the given width.

    Width is the one knob scripts/isoflop_sweep.py varies to sweep model
    size; head count and depth are both derived from it, but on different
    curves: n_head keeps head *size* as close to WIDTH_HEAD_DIM as the
    width allows (standard GPT convention), while n_layer =
    ANCHOR_LAYER * (n_embd / ANCHOR_WIDTH) ** DEPTH_WIDTH_EXPONENT grows
    *slower* than width (DEPTH_WIDTH_EXPONENT < 1) and is anchored to pass
    exactly through the real run's own (n_layer=8, n_embd=512) point. So
    every point on an IsoFLOP curve is a scaled version of the same shape
    family (the real run's shape included), and that family's *aspect
    ratio* (width/depth) widens as it scales up, matching how the actual
    GPT-2/GPT-3 family scales (see DEPTH_WIDTH_EXPONENT) instead of
    holding it fixed.

    n_head is the divisor of n_embd whose head size lands nearest
    WIDTH_HEAD_DIM (ties go to the larger head count), which reduces to
    exactly n_embd // WIDTH_HEAD_DIM for multiples of it and to 1 below
    it. Widths in between are allowed rather than rejected so the sweep
    can sample the parabola vertex densely -- at 64-spacing the vertex
    band (widths 32-192) only gets a handful of points, and a coarse
    vertex is what destabilizes the fitted minimum.
    """
    if n_embd < 1:
        raise ValueError(f"n_embd={n_embd} must be >= 1")
    divisors = [h for h in range(1, n_embd + 1) if n_embd % h == 0]
    n_head = min(divisors, key=lambda h: (abs(n_embd / h - WIDTH_HEAD_DIM), -h))
    n_layer = max(1, round(ANCHOR_LAYER * (n_embd / ANCHOR_WIDTH) ** DEPTH_WIDTH_EXPONENT))
    return n_layer, n_head, n_embd


def nearest_width_for_params(target_n: float, vocab_size: int, block_size: int, max_width: int = 8192) -> int:
    """The width (multiple of WIDTH_HEAD_DIM) whose gpt_shape_for_width()
    model has a real parameter count closest to `target_n` -- how a fitted
    N_opt(C) (a continuous number) gets turned back into a concrete,
    trainable (n_layer, n_head, n_embd).

    num_params(width) is monotonically increasing in width (a wider/deeper
    model never has fewer parameters), so |diff| decreases then increases
    exactly once -- this stops at the first non-improving step instead of
    scanning up to `max_width` regardless of `target_n`. That early exit
    matters: without it, a modest `target_n` would still instantiate (and
    randomly-initialize) every width up to max_width, including e.g. a
    128-layer, 8192-wide candidate -- a >100B-parameter real nn.Module --
    just to rule it out.
    """
    from src.model.gpt import GPT, GPTConfig

    best_width, best_diff = WIDTH_HEAD_DIM, float("inf")
    for width in range(WIDTH_HEAD_DIM, max_width + 1, WIDTH_HEAD_DIM):
        n_layer, n_head, n_embd = gpt_shape_for_width(width)
        cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        diff = abs(GPT(cfg).num_params(non_embedding=False) - target_n)
        if diff >= best_diff:
            break
        best_diff, best_width = diff, width
    return best_width


def fit_isoflop_parabola(log_n: Sequence[float], loss: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares fit of loss = a*x^2 + b*x + c, x = log(N), across one
    IsoFLOP budget's (model size, final loss) points -- the "parabola in
    log-log space" the Chinchilla IsoFLOP method plots per budget. Returns
    (a, b, c); see parabola_vertex() for reading off the minimum.
    """
    a, b, c = np.polyfit(np.asarray(log_n, dtype=float), np.asarray(loss, dtype=float), 2)
    return float(a), float(b), float(c)


def parabola_vertex(a: float, b: float, c: float) -> tuple[float, float]:
    """(x_min, loss_min) at the vertex of a*x^2 + b*x + c.

    Raises ValueError if a <= 0: a non-positive leading coefficient means
    the fitted curve has no minimum (it's concave, or a straight line) --
    this happens when a budget's sampled model sizes didn't bracket the
    true optimum (e.g. loss was still falling at the widest model tried).
    Such a budget should be widened and re-run, not silently reported with
    a bogus vertex.
    """
    if a <= 0:
        raise ValueError(f"parabola has no minimum (a={a!r} <= 0) -- widen the model-size grid for this budget")
    x_min = -b / (2 * a)
    loss_min = a * x_min**2 + b * x_min + c
    return x_min, loss_min


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Fit y = coefficient * x**exponent by linear regression in log-log
    space (log y = log(coefficient) + exponent*log(x)) -- Chinchilla
    "Approach 2"'s fit of N_opt(C) or D_opt(C) across the per-budget
    parabola minima. Returns (coefficient, exponent).
    """
    log_x = np.log(np.asarray(x, dtype=float))
    log_y = np.log(np.asarray(y, dtype=float))
    exponent, log_coefficient = np.polyfit(log_x, log_y, 1)
    return float(np.exp(log_coefficient)), float(exponent)


def optimal_params_and_tokens(
    flops: float, n_fit: tuple[float, float], d_fit: tuple[float, float]
) -> tuple[float, float]:
    """N_opt(C), D_opt(C) for compute budget `flops`, from the (coefficient,
    exponent) power-law fits fit_power_law() already produced for N_opt(C)
    and D_opt(C) respectively (see scripts/fit_isoflop_scaling_law.py) --
    the extrapolation step that turns a small-scale sweep into a shape
    recommendation for a much larger real training run.
    """
    n_coeff, n_exp = n_fit
    d_coeff, d_exp = d_fit
    return n_coeff * flops**n_exp, d_coeff * flops**d_exp


def main() -> None:
    """CLI: print describe_budget() for one or more --flops-budget values,
    without training anything. Model shape is an input here, not an
    output -- see this module's docstring for why.

    Usage: uv run src/model/scaling.py --flops-budget 1e17 1e18 \
        --vocab-size 32000 --n-layer 8 --n-head 8 --n-embd 512 \
        --batch-size 8 --block-size 1024
    """
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.model.gpt import GPT, GPTConfig

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flops-budget", type=float, nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    n_params = GPT(GPTConfig(args.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd)).num_params(
        non_embedding=False
    )
    for flops in args.flops_budget:
        print(describe_budget(flops, n_params, args.n_layer, args.n_head, args.n_embd, args.batch_size, args.block_size))


if __name__ == "__main__":
    main()
