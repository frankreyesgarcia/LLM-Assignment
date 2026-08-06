from __future__ import annotations

import numpy as np
import pytest

from src.model import GPT, GPTConfig
from src.model.scaling import (
    FLOPS_PER_PARAM_TOKEN,
    WIDTH_HEAD_DIM,
    describe_budget,
    fit_isoflop_parabola,
    fit_power_law,
    gpt_shape_for_width,
    max_iters_for_budget,
    nearest_width_for_params,
    optimal_params_and_tokens,
    parabola_vertex,
    tokens_for_budget,
)

VOCAB_SIZE = 32000
BLOCK_SIZE = 1024


def test_flops_per_param_token():
    # FLOPs(N, D) ~= 6*N*D -- the forward+backward FLOPs-per-parameter-per-token identity.
    assert FLOPS_PER_PARAM_TOKEN == 6


def test_tokens_for_budget_matches_flops_identity():
    # D = C / (6N) -- inverting FLOPs(N, D) = 6*N*D exactly (up to rounding).
    flops, n_params = 6e12, 1_000_000
    assert tokens_for_budget(flops, n_params) == round(flops / (FLOPS_PER_PARAM_TOKEN * n_params))


def test_tokens_for_budget_scales_inversely_with_params():
    flops = 1e18
    assert tokens_for_budget(flops, 2_000_000) == tokens_for_budget(flops, 1_000_000) // 2


def test_tokens_for_budget_grows_with_flops():
    n_params = 1_000_000
    assert tokens_for_budget(1e19, n_params) > tokens_for_budget(1e18, n_params)


def test_max_iters_for_budget_matches_tokens_over_step_size():
    flops, n_params, batch_size, block_size = 1e18, 5_000_000, 8, 1024
    tokens = tokens_for_budget(flops, n_params)
    assert max_iters_for_budget(flops, n_params, batch_size, block_size) == max(1, round(tokens / (batch_size * block_size)))


def test_max_iters_for_budget_is_never_zero():
    # Even a tiny budget should round up to at least one training step.
    assert max_iters_for_budget(1.0, 1_000_000_000, 32, 128) == 1


def test_max_iters_for_budget_uses_actual_model_param_count():
    # n_params should be the real GPT.num_params(non_embedding=False) for a
    # shape, not an approximation -- sanity-check against a real model.
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=8, n_head=8, n_embd=512)
    n_params = GPT(cfg).num_params(non_embedding=False)
    flops = 1e17
    max_iters = max_iters_for_budget(flops, n_params, batch_size=8, block_size=BLOCK_SIZE)
    tokens = max_iters * 8 * BLOCK_SIZE
    # Reconstructed compute should recover (approximately, given rounding) the original budget.
    assert abs(FLOPS_PER_PARAM_TOKEN * n_params * tokens - flops) / flops < 1e-2


def test_describe_budget_reports_consistent_numbers():
    flops, n_params = 1e17, 33_000_000
    text = describe_budget(flops, n_params, n_layer=7, n_head=7, n_embd=448, batch_size=8, block_size=1024)
    max_iters = max_iters_for_budget(flops, n_params, 8, 1024)
    assert f"max_iters={max_iters:,}" in text
    assert f"{n_params:,} total params" in text


# -- gpt_shape_for_width -------------------------------------------------


def test_gpt_shape_for_width_matches_real_run_shape():
    # scripts/slurm/07_pretrain.sh's real run is n_layer=8, n_head=8,
    # n_embd=512 -- exactly what width=512 should derive.
    assert gpt_shape_for_width(512) == (8, 8, 512)


def test_gpt_shape_for_width_keeps_head_dim_fixed_above_width_head_dim():
    for width in (64, 128, 192, 384, 640):
        n_layer, n_head, n_embd = gpt_shape_for_width(width)
        assert n_embd == width
        assert n_embd // n_head == WIDTH_HEAD_DIM


def test_gpt_shape_for_width_rejects_non_multiple_above_width_head_dim():
    with pytest.raises(ValueError):
        gpt_shape_for_width(100)


def test_gpt_shape_for_width_below_width_head_dim_uses_single_head():
    # Below WIDTH_HEAD_DIM=64 there isn't enough width for even one
    # WIDTH_HEAD_DIM-sized head, so n_head drops to 1 instead of the
    # "must be a multiple of WIDTH_HEAD_DIM" rule applying.
    for width in (16, 32, 48):
        n_layer, n_head, n_embd = gpt_shape_for_width(width)
        assert n_embd == width
        assert n_head == 1
        assert n_layer >= 1


def test_gpt_shape_for_width_depth_grows_slower_than_width():
    # DEPTH_WIDTH_EXPONENT < 1 -- quadrupling width should less-than-
    # quadruple n_layer (the "depth scales slower than width" property
    # this replaced the old n_layer = n_head = width/WIDTH_HEAD_DIM rule
    # for).
    n_layer_128, _, _ = gpt_shape_for_width(128)
    n_layer_512, _, _ = gpt_shape_for_width(512)
    assert n_layer_512 < 4 * n_layer_128


def test_gpt_shape_for_width_num_params_is_monotonic_in_width():
    # nearest_width_for_params's early-exit-on-first-non-improving-step
    # search relies on num_params(width) increasing monotonically with
    # width -- verify that still holds for this shape family, including
    # across the WIDTH_HEAD_DIM boundary where n_head's formula changes.
    widths = [16, 32, 48, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768]
    counts = []
    for width in widths:
        n_layer, n_head, n_embd = gpt_shape_for_width(width)
        cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        counts.append(GPT(cfg).num_params(non_embedding=False))
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


def test_nearest_width_for_params_recovers_exact_width():
    width = 256
    n_layer, n_head, n_embd = gpt_shape_for_width(width)
    cfg = GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    n_params = GPT(cfg).num_params(non_embedding=False)
    assert nearest_width_for_params(n_params, VOCAB_SIZE, BLOCK_SIZE) == width


# -- fit_isoflop_parabola / parabola_vertex ------------------------------


def test_fit_isoflop_parabola_recovers_known_coefficients():
    a, b, c = 2.0, -12.0, 20.0
    log_n = np.linspace(1.0, 5.0, 10)
    loss = a * log_n**2 + b * log_n + c
    fit_a, fit_b, fit_c = fit_isoflop_parabola(log_n, loss)
    assert fit_a == pytest.approx(a, abs=1e-8)
    assert fit_b == pytest.approx(b, abs=1e-8)
    assert fit_c == pytest.approx(c, abs=1e-8)


def test_parabola_vertex_matches_analytic_minimum():
    a, b, c = 3.0, -18.0, 10.0
    x_min, loss_min = parabola_vertex(a, b, c)
    assert x_min == pytest.approx(-b / (2 * a))
    assert loss_min == pytest.approx(a * x_min**2 + b * x_min + c)


def test_parabola_vertex_rejects_non_convex_fit():
    # a <= 0 means the sampled model sizes never bracketed a real minimum
    # (still-falling or straight-line loss) -- that budget's grid should
    # be widened, not silently reported with a bogus vertex.
    with pytest.raises(ValueError):
        parabola_vertex(a=0.0, b=-1.0, c=5.0)
    with pytest.raises(ValueError):
        parabola_vertex(a=-2.0, b=-1.0, c=5.0)


# -- fit_power_law / optimal_params_and_tokens ---------------------------


def test_fit_power_law_recovers_known_exponent():
    coefficient, exponent = 4.0, 0.5
    x = np.geomspace(1e12, 1e18, 8)
    y = coefficient * x**exponent
    fit_coeff, fit_exp = fit_power_law(x, y)
    assert fit_coeff == pytest.approx(coefficient, rel=1e-6)
    assert fit_exp == pytest.approx(exponent, rel=1e-6)


def test_optimal_params_and_tokens_applies_both_fits():
    n_fit = (2.0, 0.5)  # N_opt(C) = 2 * C**0.5
    d_fit = (0.1, 0.5)  # D_opt(C) = 0.1 * C**0.5
    flops = 1e16
    n_opt, d_opt = optimal_params_and_tokens(flops, n_fit, d_fit)
    assert n_opt == pytest.approx(2.0 * flops**0.5)
    assert d_opt == pytest.approx(0.1 * flops**0.5)
