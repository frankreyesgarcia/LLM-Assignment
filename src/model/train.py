"""Task 3 — the pretraining loop itself, as importable functions.

Kept separate from `scripts/train.py` (a thin CLI wrapper) so
`scripts/run_scaling_sweep.py` can call `train_model()` directly in-process
for each grid point, instead of shelling out to a subprocess per run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.model.gpt import GPT, GPTConfig


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path | None = None
    block_size: int = 128
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    max_iters: int = 1000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    device: str = "auto"
    seed: int = 1337
    log_every_eval: bool = True


def load_data(data_dir: Path) -> tuple[np.memmap, np.memmap, dict]:
    meta = json.loads((data_dir / "meta.json").read_text())
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    return train, val, meta


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Pick batch_size random starting offsets into the flat token stream
    # and read block_size tokens from each -- the standard way to turn one
    # long concatenated token stream into training batches without
    # pre-chunking it into fixed windows (which would waste the tokens
    # that fall between window boundaries across epochs).
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    # The target for position i is simply the next token -- so y is the
    # same window shifted one position to the right.
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_at(it: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to `min_lr`.

    Warmup avoids taking large steps on a randomly-initialized model
    (unstable); cosine decay anneals the LR smoothly instead of dropping
    it abruptly, which empirically trains better than a fixed LR
    throughout.
    """
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # 1 -> 0 over training
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.AdamW:
    # Weight decay only on 2D+ params (the actual weight matrices) --
    # LayerNorm/bias 1D params aren't supposed to shrink toward zero, and
    # decaying them tends to hurt training for no benefit. Standard GPT
    # training practice (see e.g. nanoGPT, GPT-3 paper appendix).
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, data: dict[str, np.memmap], cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    out = {}
    model.eval()
    for split, arr in data.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(arr, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(cfg: TrainConfig) -> dict:
    """Run pretraining for `cfg.max_iters` steps; return a result summary.

    Used both by `scripts/train.py` (one run, full logging/checkpointing)
    and `scripts/run_scaling_sweep.py` (many runs, summary only) -- the
    `log_every_eval`/`out_dir` knobs let the sweep skip the I/O it doesn't
    need without duplicating the training loop itself.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device))

    train_data, val_data, meta = load_data(cfg.data_dir)
    model_cfg = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr)

    out_dir = cfg.out_dir
    log_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.jsonl"

    best_val_loss = float("inf")
    history: list[dict] = []
    start = time.time()
    tokens_per_iter = cfg.batch_size * cfg.block_size

    for it in range(cfg.max_iters):
        lr = lr_at(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device)
            record = {
                "iter": it,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "tokens_seen": it * tokens_per_iter,
                "elapsed_s": time.time() - start,
            }
            history.append(record)
            if cfg.log_every_eval:
                print(
                    f"iter {it:5d} | train_loss {losses['train']:.4f} | "
                    f"val_loss {losses['val']:.4f} | lr {lr:.2e} | "
                    f"tokens {record['tokens_seen']:,}"
                )
            if log_path is not None:
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            if out_dir is not None and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {"model_state_dict": model.state_dict(), "model_cfg": model_cfg, "iter_num": it},
                    out_dir / "ckpt.pt",
                )

        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

    final = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device)
    return {
        "n_params_total": model.num_params(non_embedding=False),
        "n_params_non_embed": model.num_params(non_embedding=True),
        "tokens_seen": cfg.max_iters * tokens_per_iter,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "history": history,
        "elapsed_s": time.time() - start,
    }
