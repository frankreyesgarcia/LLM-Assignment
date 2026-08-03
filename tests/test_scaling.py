from __future__ import annotations

from src.model import GPT, GPTConfig
from src.model.scaling import (
    FLOPS_PER_PF_DAY,
    architecture_for_params,
    kaplan_compute_optimal_run,
    kaplan_optimal_params,
    kaplan_optimal_tokens,
)


def test_flops_per_pf_day():
    # 1 PF-day = 1e15 FLOP/s * 86400 s -- Kaplan et al. 2020, Section 1.3.
    assert FLOPS_PER_PF_DAY == 1e15 * 86400


def test_kaplan_optimal_params_matches_eq_6_1_at_reference_point():
    # N(Cmin) = 1.3e9 * Cmin^0.73 (Eq 6.1) -- at Cmin = 1 PF-day, N = 1.3e9 exactly.
    assert kaplan_optimal_params(FLOPS_PER_PF_DAY) == round(1.3e9)


def test_kaplan_optimal_tokens_matches_eq_6_7_at_reference_point():
    # D(Cmin) ~= 4e10 * Cmin^0.26 (Eq 6.7) -- at Cmin = 1 PF-day, D = 4e10 exactly.
    assert kaplan_optimal_tokens(FLOPS_PER_PF_DAY) == round(4e10)


def test_optimal_params_and_tokens_grow_with_budget():
    small, big = 1e17, 1e21
    assert kaplan_optimal_params(big) > kaplan_optimal_params(small)
    assert kaplan_optimal_tokens(big) > kaplan_optimal_tokens(small)


def test_optimal_params_grows_faster_than_tokens():
    # Kaplan's central finding (Section 6): a 10x compute increase should go
    # mostly into model size (N ~ C^0.73), not data (D ~ C^0.26).
    small, big = 1e18, 1e19  # 10x compute increase
    n_ratio = kaplan_optimal_params(big) / kaplan_optimal_params(small)
    d_ratio = kaplan_optimal_tokens(big) / kaplan_optimal_tokens(small)
    assert n_ratio > d_ratio
    assert n_ratio > 4  # ~10^0.73 ~= 5.4
    assert d_ratio < 2  # ~10^0.26 ~= 1.8


def test_architecture_for_params_reproduces_repo_default_shape():
    # scripts/slurm/07_pretrain.sh hand-picked n_layer=12, n_head=12, n_embd=768
    # (head_dim=64) -- architecture_for_params should recover that shape when
    # given that config's own non-embedding param count as its target.
    n_target = 12 * 12 * 768**2  # Kaplan Eq (2.1): N ~= 12 * n_layer * n_embd^2
    n_layer, n_head, n_embd = architecture_for_params(n_target)
    assert (n_layer, n_head, n_embd) == (12, 12, 768)


def test_architecture_for_params_stays_close_to_target_and_valid():
    for n_target in (1e6, 1e7, 1e8, 1e9):
        n_layer, n_head, n_embd = architecture_for_params(int(n_target))
        assert n_embd % n_head == 0
        actual = GPT(GPTConfig(vocab_size=32000, block_size=1024, n_layer=n_layer, n_head=n_head, n_embd=n_embd)).num_params(
            non_embedding=True
        )
        # Kaplan's own formula is an approximation (ignores biases/layernorm);
        # the real model's non-embedding count should still land within 2x.
        assert 0.5 * n_target < actual < 2 * n_target


def test_kaplan_compute_optimal_run_is_self_consistent():
    flops = 1e19
    plan = kaplan_compute_optimal_run(flops)
    assert plan["n_opt"] == kaplan_optimal_params(flops)
    assert plan["d_opt_tokens"] == kaplan_optimal_tokens(flops)
    assert (plan["n_layer"], plan["n_head"], plan["n_embd"]) == architecture_for_params(plan["n_opt"])
    assert plan["pf_days"] == flops / FLOPS_PER_PF_DAY
