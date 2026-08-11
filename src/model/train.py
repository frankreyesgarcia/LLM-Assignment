"""Task 3 — the pretraining loop itself, as importable functions.

Kept separate from `scripts/train.py` (a thin CLI wrapper) so the training
loop is importable/testable on its own -- `scripts/train.py` layers CLI
parsing and FLOPs-budget sizing (src/model/scaling.py) on top of
`train_model()` without the loop itself needing to know about either.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
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
    # W&B (https://wandb.ai) run logging -- off by default so existing
    # callers (tests, isoflop_sweep.py's many short-lived cells) don't
    # need a wandb account/API key just to call train_model(). wandb_mode
    # is passed straight through to wandb.init(mode=...) -- set it to
    # "offline" on cluster compute nodes without outbound internet access
    # (`wandb sync` the run directory from a node that has it afterward).
    use_wandb: bool = False
    wandb_project: str = "llm-und-pretrain"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] | None = None
    wandb_mode: str | None = None
    # Mask attention across document boundaries and restart positional
    # indices per document. 43.7% of tokens in a 1024-token window of this
    # repo's train.bin have an EOS before them, so this is not a rare
    # corner. A flag so it can be ablated at matched FLOPs.
    doc_masking: bool = True
    # Save a checkpoint (out_dir/ckpt_iter{N}.pt) every this many iterations,
    # regardless of val loss -- distinct from the "best val loss so far"
    # ckpt.pt below, which only helps if you're willing to lose everything
    # since the last improvement.
    checkpoint_interval: int | None = None


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


def document_ids(x: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Label each token in (B, T) with the index of its document in the window.

    Documents are packed as `[...tokens, EOS]`, so an EOS terminates its own
    document. Subtracting the EOS indicator keeps it there rather than
    opening the next one -- which is what lets the model still learn to
    predict EOS from that document's content.

        x       = [a, b, EOS, c, d]
        cumsum  = [0, 0,   1, 1, 1]
        - is_eos= [0, 0,   0, 1, 1]   <- EOS stays in document 0

    The window's first document is usually a fragment (the random offset
    lands mid-document); it gets id 0, which is correct -- its true prefix
    is simply not present.
    """
    is_eos = x == eos_id
    return is_eos.cumsum(dim=1) - is_eos.long()


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
def estimate_loss(
    model: GPT, data: dict[str, np.memmap], cfg: TrainConfig, device: torch.device, eos_id: int | None = None
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, arr in data.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_batch(arr, cfg.block_size, cfg.batch_size, device)
            # Same masking as training, or val_loss measures a context
            # regime the model was never trained in.
            doc_id = document_ids(x, eos_id) if eos_id is not None else None
            _, loss = model(x, y, doc_id=doc_id)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(cfg: TrainConfig) -> dict:
    """Run pretraining for `cfg.max_iters` steps; return a result summary.

    `out_dir=None`/`log_every_eval=False` skip checkpointing/console
    logging respectively, so this is also usable from tests or short
    smoke runs without writing files or spamming stdout.
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
    n_params_total = model.num_params(non_embedding=False)
    n_params_non_embed = model.num_params(non_embedding=True)

    eos_id = meta["eos_token_id"] if cfg.doc_masking else None

    out_dir = cfg.out_dir
    log_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.jsonl"

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
        run_config.update(
            device=str(device),
            n_params_total=n_params_total,
            n_params_non_embed=n_params_non_embed,
            vocab_size=meta["vocab_size"],
        )
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            mode=cfg.wandb_mode,
            config=run_config,
            dir=str(out_dir) if out_dir is not None else None,
        )

    best_val_loss = float("inf")
    history: list[dict] = []
    start = time.time()
    tokens_per_iter = cfg.batch_size * cfg.block_size

    try:
        for it in range(cfg.max_iters):
            lr = lr_at(it, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
                losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device, eos_id)
                elapsed_s = time.time() - start
                tokens_seen = it * tokens_per_iter
                record = {
                    "iter": it,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "elapsed_s": elapsed_s,
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
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": losses["train"],
                            "train/perplexity": float(np.exp(losses["train"])),
                            "val/loss": losses["val"],
                            "val/perplexity": float(np.exp(losses["val"])),
                            "val/best_loss": min(best_val_loss, losses["val"]),
                            "lr": lr,
                            "tokens_seen": tokens_seen,
                            "tokens_per_sec": tokens_seen / elapsed_s if elapsed_s > 0 else 0.0,
                            "elapsed_s": elapsed_s,
                        },
                        step=it,
                    )
                if out_dir is not None and losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(
                        {"model_state_dict": model.state_dict(), "model_cfg": model_cfg, "iter_num": it},
                        out_dir / "ckpt.pt",
                    )

            # Periodic snapshot, independent of eval_interval/best_val_loss
            # above -- see TrainConfig.checkpoint_interval's docstring for why
            # this is a separate knob rather than piggybacking on those.
            if out_dir is not None and cfg.checkpoint_interval is not None and it > 0 and it % cfg.checkpoint_interval == 0:
                torch.save(
                    {"model_state_dict": model.state_dict(), "model_cfg": model_cfg, "iter_num": it},
                    out_dir / f"ckpt_iter{it:07d}.pt",
                )

            x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
            # The EOS position's target is the next document's first token,
            # which masking makes unpredictable. Left in anyway: ~1 token in
            # 767, and it keeps loss comparable to the doc_masking=False arm.
            doc_id = document_ids(x, eos_id) if eos_id is not None else None
            _, loss = model(x, y, doc_id=doc_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({"train/step_loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}, step=it)

        final = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device, eos_id)
        elapsed_s = time.time() - start
        if out_dir is not None:
            # Unconditional, unlike the mid-training "best val loss so far"
            # save above -- callers that only ever care about the fully
            # trained model need this exact final state even when a noisier
            # earlier eval happened to score a lower val_loss.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_cfg": model_cfg,
                    "iter_num": cfg.max_iters - 1,
                    "final_train_loss": final["train"],
                    "final_val_loss": final["val"],
                },
                out_dir / "ckpt_final.pt",
            )
        if wandb_run is not None:
            wandb_run.summary["final_train_loss"] = final["train"]
            wandb_run.summary["final_val_loss"] = final["val"]
            wandb_run.summary["best_val_loss"] = min(best_val_loss, final["val"])
            wandb_run.summary["tokens_seen"] = cfg.max_iters * tokens_per_iter
            wandb_run.summary["elapsed_s"] = elapsed_s
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return {
        "doc_masking": cfg.doc_masking,
        "n_params_total": n_params_total,
        "n_params_non_embed": n_params_non_embed,
        "tokens_seen": cfg.max_iters * tokens_per_iter,
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "history": history,
        "elapsed_s": elapsed_s,
    }
