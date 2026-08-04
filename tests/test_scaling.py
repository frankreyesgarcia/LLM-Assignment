from __future__ import annotations

from src.model import GPT, GPTConfig
from src.model.scaling import FLOPS_PER_PARAM_TOKEN, describe_budget, max_iters_for_budget, tokens_for_budget

VOCAB_SIZE = 32000
BLOCK_SIZE = 1024


def test_flops_per_param_token():
    # FLOPs(N, D) ~= 6*N*D -- Kaplan et al. 2020 Section 2.1; Chinchilla Section 3.3/Appendix F.
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
