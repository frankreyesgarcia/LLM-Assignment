from __future__ import annotations

import math

import torch

from torch.nn.attention.flex_attention import flex_attention

from src.model import GPT, GPTConfig
from src.model.gpt import (
    build_document_block_mask,
    build_document_dense_mask,
    build_document_mask,
    positions_within_document,
)
from src.model.train import document_ids


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


# --- intra-document attention masking -------------------------------------


def test_document_ids_keeps_eos_with_the_document_it_ends():
    x = torch.tensor([[5, 6, 1, 7, 8, 1, 9], [1, 4, 4, 4, 4, 4, 4]])
    ids = document_ids(x, eos_id=1)
    # EOS carries its own document's id; the next token opens the next one.
    assert ids.tolist() == [[0, 0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 1, 1, 1]]
    # Non-decreasing: the block-diagonal structure depends on it.
    assert (ids[:, 1:] >= ids[:, :-1]).all()


def test_positions_restart_at_each_document():
    doc_id = torch.tensor([[0, 0, 0, 1, 1, 2]])
    assert positions_within_document(doc_id).tolist() == [[0, 1, 2, 0, 1, 0]]


def test_block_mask_attention_matches_dense_masked_reference():
    # The fused kernel must agree with a naive dense-mask SDPA reference.
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 24, 8
    q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
    doc_id = torch.zeros(B, T, dtype=torch.long)
    doc_id[:, 9:] = 1
    doc_id[1, 17:] = 2

    with torch.no_grad():  # eager FlexAttention has no CPU backward kernel
        flex_out = flex_attention(q, k, v, block_mask=build_document_block_mask(doc_id))

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    same_doc = doc_id[:, :, None] == doc_id[:, None, :]
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=(causal & same_doc).unsqueeze(1)
    )
    assert torch.allclose(flex_out, ref, atol=1e-5)


def test_packed_documents_match_separate_forward_passes():
    # The property the feature is for: masking plus positional reset means
    # packing two documents into one window must be indistinguishable from
    # running each alone. Covers the mask, the reset, and the plumbing
    # through every layer -- derived from the spec, not from re-implementing
    # attention.
    torch.manual_seed(0)
    cfg = _tiny_config(block_size=12)
    model = GPT(cfg).eval()

    split = 5
    x = torch.randint(2, cfg.vocab_size, (2, 12))
    x[:, split - 1] = 1  # EOS ends the first document
    doc_id = document_ids(x, eos_id=1)

    packed, _ = model(x, doc_id=doc_id)
    first, _ = model(x[:, :split])
    second, _ = model(x[:, split:])

    assert torch.allclose(packed[:, :split], first, atol=1e-5)
    assert torch.allclose(packed[:, split:], second, atol=1e-5)


def test_earlier_documents_cannot_leak_into_later_ones():
    torch.manual_seed(0)
    cfg = _tiny_config(block_size=12)
    model = GPT(cfg).eval()

    x = torch.randint(2, cfg.vocab_size, (1, 12))
    x[:, 4] = 1
    doc_id = document_ids(x, eos_id=1)
    perturbed = x.clone()
    perturbed[:, 0] = (x[:, 0] + 7) % cfg.vocab_size  # rewrite the first document

    masked_a, _ = model(x, doc_id=doc_id)
    masked_b, _ = model(perturbed, doc_id=doc_id)
    assert torch.allclose(masked_a[:, 5:], masked_b[:, 5:], atol=1e-6)

    # Guard against passing vacuously: without doc_id the edit does reach it.
    plain_a, _ = model(x)
    plain_b, _ = model(perturbed)
    assert not torch.allclose(plain_a[:, 5:], plain_b[:, 5:], atol=1e-6)


def test_doc_masking_leaves_generation_path_untouched():
    # generate() passes no doc_id: a continuation is not a packed window.
    torch.manual_seed(0)
    model = GPT(_tiny_config()).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=4)
    assert out.shape == (1, 5)


def test_before_and_after_on_one_concrete_window():
    """Spells out exactly what changed, on a hand-written packed window.

    Two 3-token documents packed into one 6-token window.

        x = [10, 11, EOS, 20, 21, EOS]
             \_____ doc 0 _____/  \_____ doc 1 _____/
    """
    x = torch.tensor([[10, 11, 1, 20, 21, 1]])
    doc_id = document_ids(x, eos_id=1)
    assert doc_id.tolist() == [[0, 0, 0, 1, 1, 1]]

    T = x.shape[1]
    before = torch.tril(torch.ones(T, T, dtype=torch.bool))
    after = build_document_dense_mask(doc_id)[0, 0]

    # BEFORE -- plain causal: token 20 (index 3) sees all of document 0.
    assert before.int().tolist() == [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],  # <- row 3 reaches back into document 0
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    # AFTER -- block diagonal: document 1 starts from an empty context.
    assert after.int().tolist() == [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],  # <- row 3 sees only itself
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]
    # 9 of the 21 legal causal pairs were cross-document and are now gone
    # (each document keeps its own 1+2+3 = 6 within-document pairs).
    assert before.sum().item() == 21 and after.sum().item() == 12

    # BEFORE -- positions run straight through the window.
    assert torch.arange(T).tolist() == [0, 1, 2, 3, 4, 5]
    # AFTER -- each document counts from 0 again.
    assert positions_within_document(doc_id).tolist() == [[0, 1, 2, 0, 1, 2]]
