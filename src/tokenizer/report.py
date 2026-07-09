"""Chart rendering for the tokenizer sweep/comparison report.

Follows the `dataviz` skill's method: form picked by the data's job
(trend -> line, magnitude -> bar, "one series is the point" -> emphasis),
color assigned last from the validated palette in `src/tokenizer/palette.py`
(fixed hue order, never recolored on sort/filter), one axis per panel,
hairline hairline gridlines, legends/direct labels so identity is never
color-alone. Static PNGs (matplotlib), so the interaction-layer parts of
the skill (hover/tooltip) don't apply -- every value that would live in a
tooltip is either a direct label here or in the companion CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless report generation, never interactive

from src.tokenizer.palette import (
    BASELINE_AXIS,
    CATEGORICAL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SURFACE,
)

LANGUAGE_ORDER = ["pt", "es", "hi"]
TOKENIZER_ORDER = ["gpt2", "eurollm-1.7b", "sarvam-1", "laguna-m1", "ours"]
TOKENIZER_LABELS = {
    "gpt2": "GPT-2",
    "eurollm-1.7b": "EuroLLM-1.7B",
    "sarvam-1": "Sarvam-1",
    "laguna-m1": "Laguna-M.1",
    "ours": "Ours",
}

# Real published (hidden_dim, total_params) for six well-known models,
# ordered small -> large: GPT-2 family (Small/Medium/Large -- OpenAI's
# published specs, hidden_dim 768/1024/1280, params 124M/355M/774M) and
# Llama family (hidden_dim from "The Llama 3 Herd of Models",
# arxiv.org/abs/2407.21783, and the Llama 3.2 1B/3B architecture
# writeups; params 1.23B/3.21B/8.03B). This project has NOT decided a
# target pre-training model size -- these are real anchors borrowed from
# known architectures, not a guess at what this project's model will be.
# Used to show embedding_param_share as a function of model scale
# (plot_cost_vs_model_scale) instead of computing it against one
# arbitrary assumed size, since the earlier single-assumption version of
# this chart (a fixed --hidden-dim/--target-total-params default of
# GPT-2-small) was exactly the kind of unexamined number this whole
# exercise is trying to avoid.
REFERENCE_MODELS = [
    ("GPT-2-Small", 768, 124_000_000),
    ("GPT-2-Large", 1280, 774_000_000),
    ("Llama-3.2-1B", 2048, 1_230_000_000),
    ("Llama-3.2-3B", 3072, 3_210_000_000),
    ("Llama-3.1-8B", 4096, 8_030_000_000),
]

# Ordinal ramp (one hue, monotone lightness, validated with --ordinal)
# for REFERENCE_MODELS -- these are ordered tiers (small -> large), not
# distinct categorical identities, so they get the ordinal ramp rather
# than a categorical hue per slot.
_ORDINAL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Real small-model embedding-param-share precedent, used on the cost/benefit
# chart in place of an asserted "% ceiling". Computed from Meta's published
# Llama 3 architecture specs (vocab_size=128,256 across the family;
# hidden_dim from "The Llama 3 Herd of Models", arxiv.org/abs/2407.21783,
# and the Llama 3.2 1B/3B architecture writeups) as
# (tied embedding params) / (total params) for the two variants small
# enough that Meta chose to tie embeddings at all -- 8B+ don't, precisely
# because the share would otherwise exceed this band. Two reference points,
# not a threshold: they show what real small models actually tolerated,
# rather than asserting a cutoff.
LLAMA_REFERENCE_POINTS = [
    ("Llama-3.2-1B (tied)", 128_256 * 2_048 / 1_230_000_000),
    ("Llama-3.2-3B (tied)", 128_256 * 3_072 / 3_210_000_000),
]


def _fmt_vocab(v: int) -> str:
    k = v / 1000
    s = f"{k:.1f}".rstrip("0").rstrip(".")
    return f"{s}k"


def pick_chosen_vocab_size(sweep_rows: list[dict]) -> int:
    """Elbow of the overall compression-vs-vocab-size curve, found with
    the Kneedle algorithm (Satopaa et al., 2011; `kneed` package) rather
    than an asserted "gain < X%" threshold -- the curve is concave and
    increasing, so `curve="concave", direction="increasing"` finds the
    point of maximum distance from the chord between the first and last
    points, i.e. where returns bend hardest. Candidate indices (not raw
    vocab sizes) are used as x so Kneedle isn't biased by the swept sizes
    being unevenly spaced.

    Cost (embedding_param_share) is deliberately *not* part of this
    function -- it's reported alongside the elbow as context (see
    LLAMA_REFERENCE_POINTS / plot_sweep_tradeoff) for a human to weigh,
    not folded into an automated ceiling.
    """
    from kneed import KneeLocator

    vocab_sizes = sorted({r["vocab_size"] for r in sweep_rows})
    overall = {r["vocab_size"]: r["compression_ratio"] for r in sweep_rows if r["lang"] == "overall"}
    ys = [overall[v] for v in vocab_sizes]

    kl = KneeLocator(range(len(vocab_sizes)), ys, curve="concave", direction="increasing")
    if kl.knee is None:
        # No clear knee (e.g. a near-linear curve): fall back to the
        # middle candidate rather than silently picking an extreme.
        return vocab_sizes[len(vocab_sizes) // 2]
    return vocab_sizes[kl.knee]


def _style_axes(ax, grid_axis: str = "y", hide_left_spine: bool = False) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_AXIS)
    ax.spines["left"].set_visible(not hide_left_spine)
    if not hide_left_spine:
        ax.spines["left"].set_color(BASELINE_AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def plot_sweep_compression(sweep_rows: list[dict], chosen_vocab_size: int, out_path: Path) -> None:
    """Line chart: compression ratio vs. vocab size, one line per language.

    Job: tell distinct series apart -> categorical color, fixed order.
    Each line gets an end marker (identity carrier) + an ink-colored
    direct label (never colored text) beside it, plus a legend since
    there are >=2 series.
    """
    import matplotlib.pyplot as plt

    vocab_sizes = sorted({r["vocab_size"] for r in sweep_rows})
    x = list(range(len(vocab_sizes)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    for i, lang in enumerate(LANGUAGE_ORDER):
        color = CATEGORICAL[i]
        ys = [next(r["compression_ratio"] for r in sweep_rows if r["vocab_size"] == v and r["lang"] == lang) for v in vocab_sizes]
        ax.plot(x, ys, color=color, linewidth=2, solid_capstyle="round", zorder=3, label=lang)
        ax.scatter([x[-1]], [ys[-1]], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(
            lang, xy=(x[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", color=INK_PRIMARY, fontsize=10, fontweight="bold",
        )

    chosen_idx = vocab_sizes.index(chosen_vocab_size)
    ax.axvline(chosen_idx, color=BASELINE_AXIS, linewidth=1, zorder=1)
    _, ymax = ax.get_ylim()
    ax.annotate(
        f"chosen: {_fmt_vocab(chosen_vocab_size)}", xy=(chosen_idx, ymax), xytext=(4, -4),
        textcoords="offset points", va="top", ha="left", color=INK_MUTED, fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([_fmt_vocab(v) for v in vocab_sizes])
    ax.set_xlim(x[0] - 0.3, x[-1] + 0.6)
    ax.set_ylabel("compression ratio (bytes/token)", color=INK_SECONDARY)
    ax.set_title("Compression vs. vocab size, by language", loc="left", fontweight="bold", color=INK_PRIMARY, fontsize=12)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep_marginal_gain(sweep_rows: list[dict], chosen_vocab_size: int, out_path: Path) -> None:
    """Bar chart: marginal % compression gain per vocab-size doubling (overall).

    Job: one series is the point (the chosen size), rest are context ->
    emphasis form. The chosen candidate (the Kneedle elbow -- see
    pick_chosen_vocab_size) is in the accent hue; every other candidate
    is de-emphasis gray. No threshold line: the elbow is a curvature
    property of the whole curve, not a per-step cutoff, so there's no
    single number to draw here -- the bar heights themselves are the
    evidence.
    """
    import matplotlib.pyplot as plt

    from src.tokenizer.palette import DEEMPHASIS_GRAY

    vocab_sizes = sorted({r["vocab_size"] for r in sweep_rows})
    overall = {r["vocab_size"]: r["compression_ratio"] for r in sweep_rows if r["lang"] == "overall"}
    steps = [
        (cur, 100 * (overall[cur] - overall[prev]) / overall[prev]) for prev, cur in zip(vocab_sizes, vocab_sizes[1:])
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    xs = list(range(len(steps)))
    colors = [CATEGORICAL[0] if size == chosen_vocab_size else DEEMPHASIS_GRAY for size, _ in steps]
    ax.bar(xs, [g for _, g in steps], color=colors, width=0.6, zorder=3)

    for i, (size, gain) in enumerate(steps):
        is_chosen = size == chosen_vocab_size
        ax.annotate(
            f"{gain:.1f}%", xy=(i, gain), xytext=(0, 5), textcoords="offset points", ha="center",
            color=INK_PRIMARY, fontsize=9, fontweight="bold" if is_chosen else "normal",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"→{_fmt_vocab(size)}" for size, _ in steps])
    ax.set_ylabel("marginal compression gain (%)", color=INK_SECONDARY)
    ax.set_title(
        f"Marginal gain per vocab-size doubling — chosen: {_fmt_vocab(chosen_vocab_size)}",
        loc="left", fontweight="bold", color=INK_PRIMARY, fontsize=12,
    )
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep_tradeoff(sweep_rows: list[dict], chosen_vocab_size: int, out_path: Path) -> None:
    """Scatter: compression ratio vs. embedding-param cost, one point per
    candidate vocab size.

    The two costs/benefits from the sweep's own rationale in one view.
    Vocab size is an ordered quantity, encoded redundantly by position
    (x) and by a sequential single-hue ramp (light->dark = small->large)
    -- not a categorical rainbow, since these aren't distinct identities,
    they're steps of one ordered thing. Real small-model precedent
    (LLAMA_REFERENCE_POINTS) is drawn as labeled reference lines instead
    of an asserted cost ceiling, so the reader compares against what
    shipped models actually tolerated rather than a made-up cutoff.
    """
    import matplotlib.pyplot as plt

    vocab_sizes = sorted({r["vocab_size"] for r in sweep_rows})
    overall = {r["vocab_size"]: r for r in sweep_rows if r["lang"] == "overall"}

    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    n = len(vocab_sizes)
    for i, v in enumerate(vocab_sizes):
        ramp_idx = round(i * (len(SEQUENTIAL_BLUE) - 1) / max(n - 1, 1))
        color = SEQUENTIAL_BLUE[ramp_idx]
        row = overall[v]
        x, y = row["embedding_param_share"] * 100, row["compression_ratio"]
        is_chosen = v == chosen_vocab_size
        ax.scatter(
            [x], [y], s=170 if is_chosen else 90, color=color,
            edgecolor=INK_PRIMARY if is_chosen else SURFACE, linewidth=2, zorder=4,
        )
        ax.annotate(
            _fmt_vocab(v), xy=(x, y), xytext=(7, 7), textcoords="offset points",
            color=INK_PRIMARY, fontsize=9, fontweight="bold" if is_chosen else "normal",
        )

    for label, share in LLAMA_REFERENCE_POINTS:
        pct = share * 100
        ax.axvline(pct, color=INK_MUTED, linewidth=1, zorder=1)
        ax.annotate(
            label, xy=(pct, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
            color=INK_MUTED, fontsize=8, rotation=90, va="bottom",
        )
    ax.annotate(
        "darker = larger vocab", xy=(0.02, 0.96), xycoords="axes fraction",
        color=INK_MUTED, fontsize=9, va="top",
    )

    ax.set_xlabel("embedding+LM-head share of params, AT GPT-2-SMALL SCALE ONLY (%)", color=INK_SECONDARY)
    ax.set_ylabel("compression ratio (bytes/token, overall)", color=INK_SECONDARY)
    ax.set_title(
        "Compression vs. embedding-param cost — one fixed size anchor",
        loc="left", fontweight="bold", color=INK_PRIMARY, fontsize=12,
    )
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_cost_vs_model_scale(chosen_vocab_size: int, out_path: Path) -> None:
    """Bar chart: embedding_param_share of `chosen_vocab_size`, computed
    against each of REFERENCE_MODELS' real (hidden_dim, total_params).

    This project hasn't decided a target pre-training model size, so the
    cost axis on plot_sweep_tradeoff (computed against one fixed
    GPT-2-Small anchor) can't be the whole story -- this chart shows how
    much that single number actually depends on an assumption we don't
    control yet. Ordered tiers (small -> large model), so the ordinal
    ramp is used rather than categorical hues (these aren't distinct
    identities, they're steps of one ordered thing -- same reasoning as
    the vocab-size sequential ramp elsewhere in this module).
    """
    import matplotlib.pyplot as plt

    shares = [chosen_vocab_size * hidden_dim / total_params * 100 for _, hidden_dim, total_params in REFERENCE_MODELS]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    xs = list(range(len(REFERENCE_MODELS)))
    ax.bar(xs, shares, color=_ORDINAL_RAMP, width=0.6, zorder=3)
    for i, share in enumerate(shares):
        ax.annotate(
            f"{share:.1f}%", xy=(i, share), xytext=(0, 5), textcoords="offset points",
            ha="center", color=INK_PRIMARY, fontsize=9,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _ in REFERENCE_MODELS], rotation=15, ha="right")
    ax.set_ylabel("embedding+LM-head share of params (%)", color=INK_SECONDARY)
    ax.set_title(
        f"Cost of a {_fmt_vocab(chosen_vocab_size)} vocab across real model scales",
        loc="left", fontweight="bold", color=INK_PRIMARY, fontsize=12,
    )
    ax.annotate(
        "target model size not yet decided", xy=(0.98, 0.96), xycoords="axes fraction",
        ha="right", va="top", color=INK_MUTED, fontsize=9,
    )
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_baseline_comparison(baseline_rows: list[dict], out_path: Path) -> None:
    """Small multiples (one panel per language): horizontal bars ranked by
    compression ratio, one bar per tokenizer.

    Job: tell distinct series (tokenizers) apart, but "ours" is the
    point of the chart -> categorical color (so baseline-vs-baseline
    differences stay legible) with "ours" additionally emphasized via
    an ink outline + bold label, not desaturating the rest (that would
    destroy the "which baseline wins on hi" story this chart also
    tells). Bar-tip value labels double as the identity channel here,
    so no separate legend box is needed.
    """
    import matplotlib.pyplot as plt

    color_map = dict(zip(TOKENIZER_ORDER, CATEGORICAL))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, lang in zip(axes, LANGUAGE_ORDER):
        rows = [r for r in baseline_rows if r["lang"] == lang and r["tokenizer"] in TOKENIZER_ORDER]
        rows.sort(key=lambda r: r["compression_ratio"])
        names = [r["tokenizer"] for r in rows]
        values = [r["compression_ratio"] for r in rows]
        y = list(range(len(rows)))
        colors = [color_map[n] for n in names]
        edgecolors = [INK_PRIMARY if n == "ours" else "none" for n in names]
        linewidths = [2 if n == "ours" else 0 for n in names]

        ax.barh(y, values, color=colors, edgecolor=edgecolors, linewidth=linewidths, height=0.6, zorder=3)
        for yi, name, value in zip(y, names, values):
            ax.annotate(
                f"{value:.2f}", xy=(value, yi), xytext=(5, 0), textcoords="offset points", va="center",
                color=INK_PRIMARY, fontsize=9, fontweight="bold" if name == "ours" else "normal",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([TOKENIZER_LABELS[n] for n in names])
        for tick, name in zip(ax.get_yticklabels(), names):
            if name == "ours":
                tick.set_fontweight("bold")
                tick.set_color(INK_PRIMARY)
            else:
                tick.set_color(INK_MUTED)
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_title(lang, loc="left", fontweight="bold", color=INK_PRIMARY, fontsize=11)
        _style_axes(ax, grid_axis="x", hide_left_spine=True)

    fig.suptitle(
        "Compression ratio by tokenizer and language, held-out eval (higher = better)",
        x=0.01, ha="left", fontweight="bold", color=INK_PRIMARY, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
