"""Task 3 -- compute-optimal sizing via Kaplan et al. (2020), "Scaling Laws
for Neural Language Models" (arXiv:2001.08361), Section 6, "Optimal
Allocation of the Compute Budget".

Uses the paper's own fitted constants directly rather than refitting them
on this repo's data/tokenizer/architecture (that would need its own sweep,
scoped out in favor of the published exponents). Every constant below is
cited to its exact equation so it can be checked against the paper.

Kaplan's `C` in this section means `Cmin`: the compute a *compute-optimal*
run (batch size well below critical, no data reuse) would need to reach a
given loss -- not literally whatever compute an arbitrary training run
happens to use. Treating a stated FLOPs budget as `Cmin` is the standard
way this formula gets used in practice (e.g. to size GPT-3), and is what
these functions do.
"""

from __future__ import annotations

# Section 1.3 "Notation": "we quote numerical values in PF-days, where one
# PF-day = 1e15 * 24 * 3600 = 8.64e19 floating point operations."
FLOPS_PER_PF_DAY = 8.64e19

# head_dim = 64 is the GPT-2/GPT-3 convention this repo's own hand-picked
# 124M-param config already uses (scripts/slurm/07_pretrain.sh: n_head=12,
# n_embd=768 -> head_dim=64). Fixing n_embd = _ASPECT_RATIO * n_layer (so
# n_head = n_layer, since head_dim stays 64) reproduces that exact config's
# aspect ratio, and Kaplan Section 2.1 finds performance is not very
# sensitive to shape ("aspect ratio") in the first place -- so this is a
# defensible fixed choice, not something worth fitting.
_ASPECT_RATIO = 64


def kaplan_optimal_params(flops: float) -> int:
    """N_opt(C): compute-optimal non-embedding parameter count.

    Kaplan et al. 2020, Eq (6.1) / Figure 14: N(Cmin) = (1.3e9) * Cmin^0.73,
    Cmin in PF-days, N in non-embedding parameters.
    """
    c_pf_days = flops / FLOPS_PER_PF_DAY
    return round(1.3e9 * c_pf_days**0.73)


def kaplan_optimal_tokens(flops: float) -> int:
    """D_opt(C): compute-optimal training dataset size, in tokens.

    Kaplan et al. 2020, Eq (6.7): D(Cmin) ~= (4e10 tokens) * Cmin^0.26,
    Cmin in PF-days. This is a single-epoch (no data reuse) figure -- the
    paper's stated maximum rate dataset size can productively grow with
    compute under compute-optimal training.
    """
    c_pf_days = flops / FLOPS_PER_PF_DAY
    return round(4e10 * c_pf_days**0.26)


def architecture_for_params(n_target: int) -> tuple[int, int, int]:
    """Pick (n_layer, n_head, n_embd) whose non-embedding param count is
    closest to `n_target`, at a fixed aspect ratio (see `_ASPECT_RATIO`).

    Solves Kaplan et al. 2020 Eq (2.1), the paper's own non-embedding
    parameter formula for a Transformer with d_attn = d_ff/4 = d_model
    (true of this repo's GPT -- see src/model/gpt.py):

        N ~= 12 * n_layer * n_embd^2

    for n_embd = _ASPECT_RATIO * n_layer, i.e. N ~= 12 * _ASPECT_RATIO^2 *
    n_layer^3, giving n_layer = cbrt(N / (12 * _ASPECT_RATIO^2)). Rounded
    to the nearest layer count (minimum 1).
    """
    n_layer = max(1, round((n_target / (12 * _ASPECT_RATIO**2)) ** (1 / 3)))
    n_embd = _ASPECT_RATIO * n_layer
    n_head = n_layer
    return n_layer, n_head, n_embd


def kaplan_compute_optimal_run(flops: float) -> dict:
    """Full compute-optimal sizing for a FLOPs budget: model shape + tokens.

    Combines `kaplan_optimal_params`, `architecture_for_params`, and
    `kaplan_optimal_tokens` into the one call scripts/train.py needs for
    `--flops-budget`.
    """
    n_opt = kaplan_optimal_params(flops)
    n_layer, n_head, n_embd = architecture_for_params(n_opt)
    d_opt = kaplan_optimal_tokens(flops)
    return {
        "flops_budget": flops,
        "pf_days": flops / FLOPS_PER_PF_DAY,
        "n_opt": n_opt,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "d_opt_tokens": d_opt,
    }


def max_iters_for(plan: dict, batch_size: int, block_size: int) -> int:
    """Convert a plan's `d_opt_tokens` into a training-loop `max_iters`.

    Kept separate from `kaplan_compute_optimal_run` since batch/block size
    are a training-loop detail, not part of the compute-optimal allocation
    itself -- `d_opt_tokens` only becomes a step count once you know how
    many tokens one training step consumes.
    """
    tokens_per_iter = batch_size * block_size
    return max(1, round(plan["d_opt_tokens"] / tokens_per_iter))


def describe_plan(plan: dict, batch_size: int, block_size: int) -> str:
    """One-line human-readable summary of a `kaplan_compute_optimal_run` plan.

    Shared by scripts/train.py's --flops-budget and
    scripts/plan_compute_budget.py so the two don't duplicate this string.
    """
    max_iters = max_iters_for(plan, batch_size, block_size)
    return (
        f"{plan['flops_budget']:.3e} FLOPs ({plan['pf_days']:.3g} PF-days): "
        f"N_opt={plan['n_opt']:,} non-embed params -> n_layer={plan['n_layer']}, "
        f"n_head={plan['n_head']}, n_embd={plan['n_embd']}; "
        f"D_opt={plan['d_opt_tokens']:,} tokens -> max_iters={max_iters:,} "
        f"(batch_size={batch_size} x block_size={block_size})"
    )
