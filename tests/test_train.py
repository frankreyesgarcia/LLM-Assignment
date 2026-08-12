from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.model.train import (
    DEFAULT_VAL_BIN,
    TrainConfig,
    autocast_for,
    resolve_amp_dtype,
    train_model,
)


def _make_data_dir(tmp_path, vocab_size=50, n_train=2000, n_val=500):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, vocab_size, n_train, dtype=np.uint16).tofile(data_dir / "train.bin")
    rng.integers(0, vocab_size, n_val, dtype=np.uint16).tofile(data_dir / DEFAULT_VAL_BIN)
    (data_dir / "meta.json").write_text(json.dumps({"vocab_size": vocab_size, "eos_token_id": 0}))
    return data_dir


def _tiny_cfg(data_dir, out_dir, **overrides) -> TrainConfig:
    defaults = dict(
        data_dir=data_dir,
        out_dir=out_dir,
        block_size=8,
        batch_size=4,
        n_layer=2,
        n_head=2,
        n_embd=16,
        max_iters=6,
        warmup_iters=1,
        eval_interval=2,
        eval_iters=2,
        log_every_eval=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_checkpoint_interval_writes_resumable_ckpt_last(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out_dir = tmp_path / "run"
    cfg = _tiny_cfg(data_dir, out_dir, checkpoint_interval=2)
    train_model(cfg)

    import torch

    ckpt = torch.load(out_dir / "ckpt_last.pt", weights_only=False)
    assert set(["model_state_dict", "optimizer_state_dict", "model_cfg", "iter_num", "best_val_loss"]) <= ckpt.keys()
    # Written at it=2 and it=4 (checkpoint_interval=2, it>0, max_iters=6) -- last write wins.
    assert ckpt["iter_num"] == 4


def test_resume_continues_from_last_checkpoint_iter(tmp_path, capsys):
    data_dir = _make_data_dir(tmp_path)
    out_dir = tmp_path / "run"

    cfg1 = _tiny_cfg(data_dir, out_dir, max_iters=4, checkpoint_interval=2)
    train_model(cfg1)

    import torch

    ckpt_before = torch.load(out_dir / "ckpt_last.pt", weights_only=False)
    assert ckpt_before["iter_num"] == 2  # only checkpoint boundary hit within 0..3

    cfg2 = _tiny_cfg(data_dir, out_dir, max_iters=6, checkpoint_interval=2)
    train_model(cfg2)
    out = capsys.readouterr().out
    assert "Resumed from" in out
    assert "iter 3" in out  # resumes right after the saved iter_num=2

    ckpt_after = torch.load(out_dir / "ckpt_last.pt", weights_only=False)
    assert ckpt_after["iter_num"] == 4  # next checkpoint boundary after resuming at it=3


def test_resume_false_starts_over_despite_existing_checkpoint(tmp_path, capsys):
    data_dir = _make_data_dir(tmp_path)
    out_dir = tmp_path / "run"

    cfg1 = _tiny_cfg(data_dir, out_dir, max_iters=4, checkpoint_interval=2)
    train_model(cfg1)

    cfg2 = _tiny_cfg(data_dir, out_dir, max_iters=4, checkpoint_interval=2, resume=False)
    train_model(cfg2)
    out = capsys.readouterr().out
    assert "Resumed from" not in out


def test_resume_rejects_mismatched_model_shape(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out_dir = tmp_path / "run"

    cfg1 = _tiny_cfg(data_dir, out_dir, max_iters=4, checkpoint_interval=2)
    train_model(cfg1)

    cfg2 = _tiny_cfg(data_dir, out_dir, max_iters=4, checkpoint_interval=2, n_embd=32)
    with pytest.raises(ValueError, match="Refusing to resume"):
        train_model(cfg2)


def test_resume_after_completed_run_is_a_no_op_retrain(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    out_dir = tmp_path / "run"

    # checkpoint_interval=5 lands the last write exactly on max_iters-1, so
    # resuming has start_iter == max_iters -- an empty training loop.
    cfg1 = _tiny_cfg(data_dir, out_dir, max_iters=6, checkpoint_interval=5)
    train_model(cfg1)

    assert torch.load(out_dir / "ckpt_last.pt", weights_only=False)["iter_num"] == 5

    # Simulate re-submitting the same already-finished job: should not
    # error, and should not retrain (the loop body never executes).
    cfg2 = _tiny_cfg(data_dir, out_dir, max_iters=6, checkpoint_interval=5)
    result = train_model(cfg2)
    assert result["final_val_loss"] is not None


def test_resolve_amp_dtype_is_fp32_off_cuda():
    # CPU never autocasts, whatever was asked for -- tests and smoke runs
    # want exact fp32, and CPU autocast wouldn't be a speedup anyway.
    cpu = torch.device("cpu")
    assert resolve_amp_dtype("auto", cpu) is None
    assert resolve_amp_dtype("fp32", cpu) is None
    assert resolve_amp_dtype("bf16", cpu) is None


def test_resolve_amp_dtype_rejects_unknown_spec():
    with pytest.raises(ValueError, match="amp_dtype must be one of"):
        resolve_amp_dtype("fp16", torch.device("cpu"))


def test_resolve_amp_dtype_picks_bf16_on_capable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp_dtype("auto", torch.device("cuda")) is torch.bfloat16
    assert resolve_amp_dtype("bf16", torch.device("cuda")) is torch.bfloat16
    # An explicit opt-out still wins over capable hardware.
    assert resolve_amp_dtype("fp32", torch.device("cuda")) is None


def test_resolve_amp_dtype_without_bf16_support(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Quadro RTX 6000")
    # "auto" degrades silently; "bf16" is a demand, so it fails loudly
    # rather than quietly running 4x slower than the user asked for.
    assert resolve_amp_dtype("auto", torch.device("cuda")) is None
    with pytest.raises(ValueError, match="no bfloat16 support"):
        resolve_amp_dtype("bf16", torch.device("cuda"))


def test_autocast_for_is_a_no_op_in_fp32():
    with autocast_for(torch.device("cpu"), None):
        assert not torch.is_autocast_enabled()


def test_train_model_runs_in_explicit_fp32(tmp_path):
    data_dir = _make_data_dir(tmp_path)
    cfg = _tiny_cfg(data_dir, tmp_path / "run", amp_dtype="fp32")
    assert train_model(cfg)["final_val_loss"] > 0
