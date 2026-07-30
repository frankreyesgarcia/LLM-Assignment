from __future__ import annotations

import math

import torch

from src.model import GPT, GPTConfig


def _tiny_config(**overrides) -> GPTConfig:
    defaults = dict(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16)
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_forward_shapes():
    cfg = _tiny_config()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss is None


def test_loss_matches_uniform_prior_at_init():
    # A freshly-initialized model has no learned signal, so its predicted
    # next-token distribution should be close to uniform over the vocab --
    # cross-entropy against a uniform guess is ln(vocab_size).
    torch.manual_seed(0)
    cfg = _tiny_config(vocab_size=100)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    _, loss = model(x, y)
    assert math.isfinite(loss.item())
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_gradients_reduce_loss():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    _, first_loss = model(x, y)
    for _ in range(20):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, last_loss = model(x, y)

    assert math.isfinite(last_loss.item())
    assert last_loss.item() < first_loss.item()


def test_weight_tying():
    cfg = _tiny_config()
    model = GPT(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_num_params_excludes_embeddings():
    cfg = _tiny_config(vocab_size=1000, block_size=32, n_embd=64)
    model = GPT(cfg)
    total = model.num_params(non_embedding=False)
    non_embed = model.num_params(non_embedding=True)
    wte_params = cfg.vocab_size * cfg.n_embd
    wpe_params = cfg.block_size * cfg.n_embd
    assert total - non_embed == wte_params + wpe_params


def test_generate_produces_expected_length():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = GPT(cfg)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, top_k=10)
    assert out.shape == (1, 6)
    assert (out >= 0).all() and (out < cfg.vocab_size).all()
