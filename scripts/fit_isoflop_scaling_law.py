#!/usr/bin/env python3
"""Task 3 -- fit a compute-optimal scaling law from scripts/isoflop_sweep.py's
results.csv (the Chinchilla "IsoFLOP profiles" method, "Approach 2" in the
paper), and use it to recommend a model shape for a real training run.

For each FLOP budget in the sweep, the swept models' (n_params, loss)
points trace a U-shaped curve in log(N)-vs-loss space -- too small a model
underfits, too big a model barely trains before exhausting that budget's
token allowance. Fitting a parabola to each budget's points and reading
off its minimum gives that budget's compute-optimal (N_opt, D_opt, L_opt --
shape and the loss it achieves). Fitting a power law across budgets' minima
(N_opt(C) = a*C^alpha, D_opt(C) = b*C^beta, L_opt(C) = e*C^-p) turns that
into a law that extrapolates to a real run's much larger
--final-flops-budget, without ever training a model that large. L_opt(C)'s
fit is a plain 2-parameter power law, not Chinchilla's own 3-parameter
L(C) = E + A*C^-alpha (irreducible-loss floor) -- see fit_per_budget_minima's
call site for why -- so treat its extrapolation as a rough check, not a
precise prediction.

Renders the four standard plots (training loss vs. params per budget;
FLOPs vs. optimal params; FLOPs vs. optimal tokens; FLOPs vs. optimal loss)
plus a fit_summary.json with the fitted coefficients and the recommended
shape.

Usage:
    uv run scripts/fit_isoflop_scaling_law.py \
        --results-csv "$PROJECT_STORAGE/runs/isoflop_sweep/results.csv" \
        --final-flops-budget 1e17
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.model.scaling import (
    FLOPS_PER_PARAM_TOKEN,
    fit_isoflop_parabola,
    fit_power_law,
    gpt_shape_for_width,
    nearest_width_for_params,
    optimal_params_and_tokens,
    parabola_vertex,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Categorical palette slots 1-5 (blue/orange/aqua/yellow/magenta), fixed
# order -- validated colorblind-safe for adjacent-pair use (lines/legend
# entries in a fixed sequence) by this repo's dataviz skill; see
# references/palette.md there. Not cycled -- if the sweep ever has more
# than 5 budgets, add slots 6-8 in the same fixed order rather than
# wrapping back to slot 1.
BUDGET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FIT_LINE_COLOR = "#3a3a38"


def load_results(csv_path: Path) -> dict[float, list[dict]]:
    by_budget: dict[float, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_budget[float(row["flops_budget"])].append(
                {
                    "width": int(row["width"]),
                    # n_params is total (embedding included) -- Chinchilla's
                    # own convention (module docstring; not Kaplan's, which
                    # excludes embeddings) -- and is what the parabola/
                    # power-law fit below is run on. n_params_non_embedding
                    # is carried along only to report the embedding
                    # fraction (see main()), not used in any fit.
                    "n_params": int(row["n_params"]),
                    "n_params_non_embedding": int(row["n_params_non_embedding"]),
                    "train_loss": float(row["final_train_loss"]),
                    "val_loss": float(row["final_val_loss"]),
                }
            )
    for rows in by_budget.values():
        rows.sort(key=lambda r: r["n_params"])
    return dict(sorted(by_budget.items()))


def fit_per_budget_minima(
    by_budget: dict[float, list[dict]],
) -> tuple[list[float], list[float], list[float], list[float], dict]:
    """Fit a parabola per budget and return the valid (C, N_opt, D_opt,
    L_opt) quadruples plus each budget's (a, b, c) fit for plotting --
    budgets whose grid didn't bracket a real minimum (parabola_vertex
    raising ValueError) are dropped from the power-law fits, not silently
    included. L_opt is the vertex's loss value -- the loss a compute-optimal
    model at that budget actually achieves, as opposed to N_opt/D_opt which
    describe its shape.
    """
    budgets, n_opts, d_opts, l_opts, parabolas = [], [], [], [], {}
    for c, rows in by_budget.items():
        log_n = [np.log(r["n_params"]) for r in rows]
        loss = [r["train_loss"] for r in rows]
        a, b, coeff_c = fit_isoflop_parabola(log_n, loss)
        parabolas[c] = (a, b, coeff_c)
        try:
            x_min, y_min = parabola_vertex(a, b, coeff_c)
        except ValueError as e:
            print(f"  budget C={c:.3e}: {e} -- excluded from the power-law fit")
            continue
        # A parabola fit through points that are all on the same (rising or
        # falling) side of the true curve can still have a>0 -- its vertex
        # then falls outside the sampled N range, i.e. it's extrapolated,
        # not interpolated, from data that never actually bracketed a
        # minimum. That extrapolation isn't trustworthy: away from the
        # data, nothing constrains the fitted parabola to track the real
        # (only approximately parabolic near its own minimum) loss curve.
        # Same fix as an unbracketed minimum -- widen this budget's --widths.
        if not (min(log_n) <= x_min <= max(log_n)):
            print(
                f"  budget C={c:.3e}: fitted vertex N={np.exp(x_min):,.0f} falls outside the sampled "
                f"[{np.exp(min(log_n)):,.0f}, {np.exp(max(log_n)):,.0f}] param range (extrapolated, not "
                "bracketed) -- excluded from the power-law fit"
            )
            continue
        n_opt = float(np.exp(x_min))
        d_opt = c / (FLOPS_PER_PARAM_TOKEN * n_opt)
        budgets.append(c)
        n_opts.append(n_opt)
        d_opts.append(d_opt)
        l_opts.append(float(y_min))
    return budgets, n_opts, d_opts, l_opts, parabolas


def budget_color_map(all_budgets: list[float]) -> dict[float, str]:
    """One fixed color per FLOP budget, keyed by the budget's value -- not
    its position in whatever (possibly filtered) list is being plotted.
    Colors must follow the *entity* (a specific budget) across all three
    plots so the same budget is recognizable in each one; deriving color
    from list position instead would silently reassign colors whenever a
    budget gets excluded from the power-law fit (fit_per_budget_minima),
    desynchronizing plot_frontier's colors from plot_loss_vs_params's.
    """
    return {c: BUDGET_COLORS[i % len(BUDGET_COLORS)] for i, c in enumerate(all_budgets)}


def plot_loss_vs_params(
    by_budget: dict[float, list[dict]], parabolas: dict, bracketed_budgets: set[float], out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    colors = budget_color_map(list(by_budget))
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, rows in by_budget.items():
        color = colors[c]
        ns = [r["n_params"] for r in rows]
        losses = [r["train_loss"] for r in rows]
        ax.scatter(ns, losses, color=color, s=36, zorder=3, label=f"C = {c:.2e} FLOPs")

        a, b, coeff_c = parabolas[c]
        x_smooth = np.linspace(np.log(min(ns)), np.log(max(ns)), 200)
        y_smooth = a * x_smooth**2 + b * x_smooth + coeff_c
        ax.plot(np.exp(x_smooth), y_smooth, color=color, linewidth=1.5, alpha=0.8, zorder=2)
        # Only mark the vertex when it's bracketed by this budget's own
        # sampled widths (see fit_per_budget_minima) -- an unbracketed
        # vertex is extrapolated past the data and would be misleading to
        # draw as if it were a found minimum.
        if a > 0 and c in bracketed_budgets:
            x_min, y_min = parabola_vertex(a, b, coeff_c)
            ax.scatter([np.exp(x_min)], [y_min], color=color, s=90, marker="*", edgecolor="black", linewidth=0.6, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("model parameters N (log scale)")
    ax.set_ylabel("final training loss")
    ax.set_title("IsoFLOP profiles: training loss vs. model size, per FLOP budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def plot_frontier(
    budgets: list[float],
    values: list[float],
    colors: dict[float, str],
    fit: tuple[float, float],
    final_flops: float,
    final_value: float,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    coeff, exponent = fit
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, v in zip(budgets, values):
        ax.scatter([c], [v], color=colors[c], s=70, zorder=3)

    x_line = np.geomspace(min(budgets), max(final_flops, max(budgets)) * 1.2, 200)
    y_line = coeff * x_line**exponent
    ax.plot(x_line, y_line, color=FIT_LINE_COLOR, linestyle="--", linewidth=1.5, zorder=2, label=f"fit: {exponent:.3f} power law")

    ax.scatter([final_flops], [final_value], color=FIT_LINE_COLOR, s=140, marker="*", edgecolor="black", linewidth=0.8, zorder=4, label=f"final run (C={final_flops:.2e})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute budget C (FLOPs, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "isoflop_sweep")
    parser.add_argument(
        "--min-flops-budget",
        type=float,
        default=None,
        help=(
            "Drop budgets below this from the analysis. Not the same as the per-budget bracketing "
            "check (fit_per_budget_minima): a budget can produce an in-range vertex purely by curve-fit "
            "coincidence while still being systematically unreliable -- e.g. a budget so small that no "
            "width gets enough gradient steps for model capacity to matter, so every width scores close "
            "to its near-random-init loss and the 'minimum' is fit noise, not a real capacity trade-off. "
            "Use this to exclude a known-bad low-budget range identified by eye (see README's Task 3 "
            "section) rather than relying on bracketing alone to catch it."
        ),
    )
    parser.add_argument("--final-flops-budget", type=float, default=1e17)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--block-size", type=int, default=1024)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_budget = load_results(args.results_csv)
    all_rows = [r for rows in by_budget.values() for r in rows]
    print(f"Loaded {len(all_rows)} cells across {len(by_budget)} FLOP budgets from {args.results_csv}")
    embed_fracs = [1 - r["n_params_non_embedding"] / r["n_params"] for r in all_rows]
    print(
        f"  embedding share of total N: {min(embed_fracs):.1%}-{max(embed_fracs):.1%} across cells "
        "(narrowest widths are embedding-dominated; total N -- not non-embedding N -- is what's fit below, "
        "per Chinchilla's convention)"
    )

    if args.min_flops_budget is not None:
        dropped = [c for c in by_budget if c < args.min_flops_budget]
        for c in dropped:
            del by_budget[c]
        if dropped:
            print(f"Dropped {len(dropped)} budget(s) below --min-flops-budget={args.min_flops_budget:.3e}: {dropped}")

    budgets, n_opts, d_opts, l_opts, parabolas = fit_per_budget_minima(by_budget)
    if len(budgets) < 2:
        raise SystemExit(
            f"Only {len(budgets)} budget(s) yielded a valid parabola minimum -- need at least 2 to fit a "
            "power law. Widen the model-size grid (scripts/isoflop_sweep.py --widths) for the excluded "
            "budget(s) and re-run the sweep."
        )

    n_fit = fit_power_law(budgets, n_opts)
    d_fit = fit_power_law(budgets, d_opts)
    # L_opt(C) = l_coeff * C^l_exp -- same log-log linear regression as
    # N_opt/D_opt (fit_power_law), through each budget's parabola-vertex
    # *loss* rather than its shape. Unlike N_opt/D_opt this isn't the
    # Chinchilla paper's own parameterization: their loss-vs-compute fit is
    # L(C) = E + A*C^-alpha, a 3-parameter form with an irreducible-loss
    # floor E (loss can't sensibly go to 0 as C grows) fit by nonlinear
    # least squares. A plain 2-parameter power law needs only log-log linear
    # regression, matching this script's existing N_opt/D_opt machinery
    # exactly, but will extrapolate toward 0 loss at large C and shouldn't
    # be trusted far past the sampled budgets -- treat l_final below as a
    # rough, order-of-magnitude check, not a precise prediction.
    l_coeff, l_exp = fit_power_law(budgets, l_opts)
    n_final, d_final = optimal_params_and_tokens(args.final_flops_budget, n_fit, d_fit)
    l_final = l_coeff * args.final_flops_budget**l_exp
    recommended_width = nearest_width_for_params(n_final, args.vocab_size, args.block_size)
    n_layer, n_head, n_embd = gpt_shape_for_width(recommended_width)

    print(f"\nFitted N_opt(C) = {n_fit[0]:.4g} * C^{n_fit[1]:.4f}")
    print(f"Fitted D_opt(C) = {d_fit[0]:.4g} * C^{d_fit[1]:.4f}")
    print(f"Fitted L_opt(C) = {l_coeff:.4g} * C^{l_exp:.4f}  (no irreducible-loss floor -- rough extrapolation only)")
    print(f"\nAt --final-flops-budget={args.final_flops_budget:.3e}:")
    print(f"  N_opt ~= {n_final:,.0f} params, D_opt ~= {d_final:,.0f} tokens")
    print(f"  L_opt ~= {l_final:.4f} (final training loss)")
    print(f"  nearest feasible shape: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    summary = {
        "results_csv": str(args.results_csv),
        "n_budgets_used": len(budgets),
        "n_budgets_total": len(by_budget),
        "n_cells": len(all_rows),
        "embedding_share_of_n": {"min": min(embed_fracs), "max": max(embed_fracs)},
        "n_opt_fit": {"coefficient": n_fit[0], "exponent": n_fit[1]},
        "d_opt_fit": {"coefficient": d_fit[0], "exponent": d_fit[1]},
        "l_opt_fit": {"coefficient": l_coeff, "exponent": l_exp},
        "final_flops_budget": args.final_flops_budget,
        "n_opt_at_final_budget": n_final,
        "d_opt_at_final_budget": d_final,
        "l_opt_at_final_budget": l_final,
        "recommended_shape": {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embd},
    }
    summary_path = args.out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    colors = budget_color_map(list(by_budget))
    plot_loss_vs_params(by_budget, parabolas, set(budgets), args.out_dir / "isoflop_loss_curves.png")
    plot_frontier(
        budgets,
        n_opts,
        colors,
        n_fit,
        args.final_flops_budget,
        n_final,
        ylabel="optimal model size N_opt (params, log scale)",
        title="Compute-optimal model size vs. FLOPs budget",
        out_path=args.out_dir / "isoflop_params_vs_flops.png",
    )
    plot_frontier(
        budgets,
        d_opts,
        colors,
        d_fit,
        args.final_flops_budget,
        d_final,
        ylabel="optimal token count D_opt (log scale)",
        title="Compute-optimal token count vs. FLOPs budget",
        out_path=args.out_dir / "isoflop_tokens_vs_flops.png",
    )
    plot_frontier(
        budgets,
        l_opts,
        colors,
        (l_coeff, l_exp),
        args.final_flops_budget,
        l_final,
        ylabel="optimal training loss L_opt (log scale)",
        title="Compute-optimal training loss vs. FLOPs budget",
        out_path=args.out_dir / "isoflop_loss_vs_flops.png",
    )


if __name__ == "__main__":
    main()
