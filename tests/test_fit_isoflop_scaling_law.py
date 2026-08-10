from __future__ import annotations

import math

import pytest

from scripts.fit_isoflop_scaling_law import EXPONENT_AGREEMENT_SIGMAS, compare_exponents, json_number


def test_compare_exponents_unknown_when_se_is_nan():
    # Too few budgets to jackknife an SE -- "unknown", not "differ". The
    # old code turned the nan into inf sigma and warned on every such run.
    sigmas, agree = compare_exponents(0.157, float("nan"))
    assert agree is None
    assert math.isnan(sigmas)


def test_compare_exponents_agrees_within_noise():
    # v2's real numbers: 0.55 (val) vs 0.40 (train), SEs 0.11 and 0.12.
    sigmas, agree = compare_exponents(0.157, math.hypot(0.11, 0.12))
    assert agree is True
    assert sigmas == pytest.approx(0.96, abs=0.01)


def test_compare_exponents_flags_a_gap_beyond_tolerance():
    sigmas, agree = compare_exponents(0.5, 0.1)
    assert agree is False
    assert sigmas > EXPONENT_AGREEMENT_SIGMAS


def test_compare_exponents_handles_zero_se():
    assert compare_exponents(0.0, 0.0) == (0.0, True)
    assert compare_exponents(0.1, 0.0) == (float("inf"), False)


def test_json_number_maps_non_finite_to_none():
    assert json_number(0.11) == 0.11
    assert json_number(float("nan")) is None
    assert json_number(float("inf")) is None
