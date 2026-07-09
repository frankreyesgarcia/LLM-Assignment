from pathlib import Path

from src.tokenizer.report import (
    REFERENCE_MODELS,
    pick_chosen_vocab_size,
    plot_baseline_comparison,
    plot_cost_vs_model_scale,
    plot_sweep_compression,
    plot_sweep_marginal_gain,
    plot_sweep_tradeoff,
)

# Mirrors the shape of (and exact values from) a real
# artifacts/tokenizer_sweep/results.csv run after parsing.
SWEEP_ROWS = [
    {"vocab_size": 8_000, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.0495}
    for lang, f, c in [("pt", 1.775, 3.582), ("es", 1.767, 3.600), ("hi", 3.445, 3.779), ("overall", 2.236, 3.670)]
] + [
    {"vocab_size": 16_000, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.0991}
    for lang, f, c in [("pt", 1.594, 3.989), ("es", 1.579, 4.028), ("hi", 3.387, 3.844), ("overall", 2.087, 3.932)]
] + [
    {"vocab_size": 32_000, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.1982}
    for lang, f, c in [("pt", 1.466, 4.338), ("es", 1.442, 4.411), ("hi", 3.351, 3.885), ("overall", 1.982, 4.140)]
] + [
    {"vocab_size": 50_000, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.3097}
    for lang, f, c in [("pt", 1.405, 4.526), ("es", 1.381, 4.605), ("hi", 3.331, 3.908), ("overall", 1.932, 4.246)]
] + [
    {"vocab_size": 65_536, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.4059}
    for lang, f, c in [("pt", 1.375, 4.626), ("es", 1.350, 4.710), ("hi", 3.323, 3.918), ("overall", 1.908, 4.300)]
] + [
    {"vocab_size": 100_000, "lang": lang, "fertility": f, "compression_ratio": c, "embedding_param_share": 0.6194}
    for lang, f, c in [("pt", 1.343, 4.734), ("es", 1.317, 4.831), ("hi", 3.314, 3.929), ("overall", 1.882, 4.360)]
]

# A near-linear curve: no real elbow, so pick_chosen_vocab_size should
# fall back to the middle candidate rather than picking an extreme.
LINEAR_SWEEP_ROWS = [
    {"vocab_size": v, "lang": "overall", "fertility": 2.0, "compression_ratio": c, "embedding_param_share": 0.1}
    for v, c in [(8_000, 3.0), (16_000, 3.5), (32_000, 4.0), (50_000, 4.5), (65_536, 5.0)]
]

BASELINE_ROWS = [
    {"tokenizer": name, "repo_id": "x", "vocab_size": vs, "lang": lang, "fertility": f, "compression_ratio": c}
    for name, vs, lang, f, c in [
        ("gpt2", 50257, "pt", 2.31, 2.75),
        ("gpt2", 50257, "es", 2.18, 2.92),
        ("gpt2", 50257, "hi", 7.66, 1.70),
        ("eurollm-1.7b", 128000, "pt", 1.57, 4.05),
        ("eurollm-1.7b", 128000, "es", 1.50, 4.25),
        ("eurollm-1.7b", 128000, "hi", 1.88, 6.91),
        ("sarvam-1", 68096, "pt", 2.78, 2.29),
        ("sarvam-1", 68096, "es", 2.66, 2.39),
        ("sarvam-1", 68096, "hi", 1.59, 8.20),
        ("laguna-m1", 100352, "pt", 2.05, 3.11),
        ("laguna-m1", 100352, "es", 1.98, 3.21),
        ("laguna-m1", 100352, "hi", 4.12, 3.16),
        ("ours", 16000, "pt", 1.59, 3.99),
        ("ours", 16000, "es", 1.58, 4.03),
        ("ours", 16000, "hi", 3.39, 3.84),
    ]
]


def test_pick_chosen_vocab_size_finds_kneedle_elbow():
    # Kneedle's point of maximum curvature on this exact curve is 32,000
    # (verified directly against the `kneed` package) -- not 16,000, which
    # was this project's earlier hand-picked-threshold answer. The two
    # disagreeing is expected: a curvature-based elbow and a "gain < X%"
    # threshold aren't the same rule, and this test pins the former.
    assert pick_chosen_vocab_size(SWEEP_ROWS) == 32_000


def test_pick_chosen_vocab_size_falls_back_on_no_elbow():
    # A straight line has no point of maximum curvature; Kneedle returns
    # no knee, and the function should fall back to the middle candidate
    # rather than silently picking an extreme.
    vocab_sizes = sorted({r["vocab_size"] for r in LINEAR_SWEEP_ROWS})
    assert pick_chosen_vocab_size(LINEAR_SWEEP_ROWS) == vocab_sizes[len(vocab_sizes) // 2]


def test_reference_models_share_shrinks_with_scale():
    # The whole point of plot_cost_vs_model_scale: embedding_param_share
    # for a fixed vocab size is far larger at the small end than the large
    # end of REFERENCE_MODELS (ordered small -> large), matching the real
    # Llama tied/untied-embedding pattern this project verified
    # independently. Not asserting strict pairwise monotonicity: GPT-2 and
    # Llama are separate model families with different depth/width
    # ratios, so e.g. GPT-2-Large and Llama-3.2-1B land within noise of
    # each other (2.65% vs 2.66%) even though the overall trend is a
    # sharp decrease -- real architectures aren't a smooth single curve.
    shares = [16_000 * hidden_dim / total_params for _, hidden_dim, total_params in REFERENCE_MODELS]
    assert shares[0] == max(shares)
    assert shares[-1] == min(shares)
    assert shares[0] > shares[-1] * 5


def test_plots_render_without_error(tmp_path: Path):
    plot_sweep_compression(SWEEP_ROWS, 16_000, tmp_path / "compression.png")
    plot_sweep_marginal_gain(SWEEP_ROWS, 16_000, tmp_path / "marginal_gain.png")
    plot_sweep_tradeoff(SWEEP_ROWS, 16_000, tmp_path / "tradeoff.png")
    plot_cost_vs_model_scale(16_000, tmp_path / "cost_vs_model_scale.png")
    plot_baseline_comparison(BASELINE_ROWS, tmp_path / "baseline.png")

    for name in ["compression.png", "marginal_gain.png", "tradeoff.png", "cost_vs_model_scale.png", "baseline.png"]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0
