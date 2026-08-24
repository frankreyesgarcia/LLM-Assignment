"""The SFT training loop, as an importable function -- paired with
scripts/train_sft.py the same way src/model/train.py pairs with
scripts/train.py.

Deliberately a separate module from src/model/train.py rather than a
branch inside `train_model`: SFT differs from pretraining in exactly the
two places that matter (loss is masked per-token via a precomputed labels
stream instead of "next token", and the model starts from a pretrained
checkpoint instead of random init) but reuses everything else --
`resolve_amp_dtype`, `autocast_for`, `enable_tf32`, `lr_at`,
`configure_optimizer`, and critically `document_ids`/doc_masking itself
(see scripts/prepare_sft_data.py's docstring for why packed conversations
still get correct per-conversation attention boundaries "for free" from
that same mechanism). Keeping this as new functions rather than adding an
`if cfg.sft:` branch to `train_model` means pretraining's tested code path
never changes shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.model.gpt import GPT, GPTConfig
from src.model.train import (
    autocast_for,
    configure_optimizer,
    document_ids,
    enable_tf32,
    lr_at,
    resolve_amp_dtype,
)


@dataclass
class SFTTrainConfig:
    data_dir: Path  # output of scripts/prepare_sft_data.py
    init_from: Path  # a pretrain checkpoint (ckpt.pt/ckpt_final.pt/...) -- SFT never starts from random init
    out_dir: Path | None = None
    block_size: int = 128
    batch_size: int = 32
    dropout: float = 0.0
    qk_norm: bool = False
    max_iters: int = 1000
    lr: float = 5e-5  # SFT LR is conventionally ~10x lower than pretraining's -- see scripts/train_sft.py --help
    min_lr: float = 5e-6
    warmup_iters: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    device: str = "auto"
    amp_dtype: str = "auto"
    seed: int = 1337
    log_every_eval: bool = True
    progress_every: int | None = None
    # Attention stays within one packed conversation, not one packed
    # example's worth of a continuous corpus -- see this module's
    # docstring. Off only for ablation; SFT correctness depends on it
    # given how scripts/prepare_sft_data.py packs conversations.
    doc_masking: bool = True
    checkpoint_interval: int | None = None
    resume: bool = True
    use_wandb: bool = False
    wandb_project: str = "llm-und-sft"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] | None = None
    wandb_mode: str | None = None
    wandb_dir: Path | None = None


def load_sft_data(data_dir: Path) -> tuple[dict[str, np.memmap], dict[str, np.memmap], dict]:
    """Memory-map the train/val token + label streams written by
    scripts/prepare_sft_data.py. Labels are int16 (signed, unlike the
    uint16 token stream) so -1 ("no loss here") is representable.
    """
    meta = json.loads((data_dir / "meta.json").read_text())
    tokens = {
        split: np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r") for split in ("train", "val")
    }
    labels = {
        split: np.memmap(data_dir / f"{split}_labels.bin", dtype=np.int16, mode="r") for split in ("train", "val")
    }
    return tokens, labels, meta


def get_sft_batch(
    tokens: np.memmap, labels: np.memmap, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in ix])
    # Target for position i is the label of the *next* token (i+1) -- same
    # shift convention as src/model/train.py::get_batch, just reading the
    # value from the labels stream (token id, or -1 to mask) instead of
    # deriving it from the token stream itself.
    y = torch.stack([torch.from_numpy(labels[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_sft_loss(
    model: GPT,
    tokens: dict[str, np.memmap],
    labels: dict[str, np.memmap],
    cfg: SFTTrainConfig,
    device: torch.device,
    eos_id: int | None,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in tokens:
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = get_sft_batch(tokens[split], labels[split], cfg.block_size, cfg.batch_size, device)
            doc_id = document_ids(x, eos_id) if eos_id is not None else None
            with autocast_for(device, amp_dtype):
                _, loss = model(x, y, doc_id=doc_id)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def load_base_checkpoint(init_from: Path, device: torch.device) -> tuple[GPT, GPTConfig]:
    ckpt = torch.load(init_from, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, model_cfg


def train_sft_model(cfg: SFTTrainConfig) -> dict:
    """Run SFT for `cfg.max_iters` steps starting from `cfg.init_from`;
    return a result summary. Same out_dir=None/log_every_eval=False
    testing convenience as train_model.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device))

    amp_dtype = resolve_amp_dtype(cfg.amp_dtype, device)
    if device.type == "cuda":
        enable_tf32()

    tokens, labels, meta = load_sft_data(cfg.data_dir)
    model, model_cfg = load_base_checkpoint(cfg.init_from, device)
    if model_cfg.vocab_size != meta["vocab_size"]:
        raise ValueError(
            f"{cfg.init_from} was trained with vocab_size={model_cfg.vocab_size}, but "
            f"{cfg.data_dir}/meta.json says {meta['vocab_size']} -- SFT data must use the same "
            "tokenizer the base checkpoint was pretrained with."
        )
    if cfg.block_size > model_cfg.block_size:
        raise ValueError(f"--block-size {cfg.block_size} exceeds the base checkpoint's block_size {model_cfg.block_size}")
    if amp_dtype is torch.bfloat16 and not model_cfg.qk_norm and cfg.log_every_eval:
        print("WARNING: bf16 without qk_norm in the base checkpoint -- attention logits are unbounded.", flush=True)

    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr)
    eos_id = meta["eos_token_id"] if cfg.doc_masking else None

    out_dir = cfg.out_dir
    log_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.jsonl"

    best_val_loss = float("inf")
    start_iter = 0
    resume_path = out_dir / "ckpt_last.pt" if out_dir is not None else None
    if cfg.resume and resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_iter = ckpt["iter_num"] + 1
        print(f"Resumed from {resume_path} at iter {start_iter} (best_val_loss={best_val_loss:.4f})")

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
        wandb_dir = cfg.wandb_dir or out_dir
        if wandb_dir is not None:
            wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            settings=wandb.Settings(console="wrap"),
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            mode=cfg.wandb_mode,
            config=run_config,
            dir=str(wandb_dir) if wandb_dir is not None else None,
        )

    history: list[dict] = []
    start = time.time()
    tokens_per_iter = cfg.batch_size * cfg.block_size

    try:
        for it in range(start_iter, cfg.max_iters):
            lr = lr_at(it, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
                losses = estimate_sft_loss(model, tokens, labels, cfg, device, eos_id, amp_dtype)
                # tokens_seen counts from iter 0 (so it stays comparable across a
                # resumed run), elapsed_s only this submission's wall clock -- same
                # split as src/model/train.py, minus its elapsed_offset_s since the
                # SFT checkpoints don't carry one.
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
                        f"val_loss {losses['val']:.4f} | lr {lr:.2e} | tokens {tokens_seen:,}"
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

            if out_dir is not None and cfg.checkpoint_interval is not None and it > 0 and it % cfg.checkpoint_interval == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_cfg": model_cfg,
                        "iter_num": it,
                        "best_val_loss": best_val_loss,
                    },
                    out_dir / "ckpt_last.pt",
                )

            x, y = get_sft_batch(tokens["train"], labels["train"], cfg.block_size, cfg.batch_size, device)
            doc_id = document_ids(x, eos_id) if eos_id is not None else None
            with autocast_for(device, amp_dtype):
                _, loss = model(x, y, doc_id=doc_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({"train/step_loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}, step=it)
            if cfg.progress_every and (it + 1) % cfg.progress_every == 0:
                print(f"iter {it + 1:6d}/{cfg.max_iters} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        final = estimate_sft_loss(model, tokens, labels, cfg, device, eos_id, amp_dtype)
        elapsed_s = time.time() - start
        if out_dir is not None:
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
            wandb_run.summary["final_val_perplexity"] = float(np.exp(final["val"]))
            wandb_run.summary["tokens_seen"] = cfg.max_iters * tokens_per_iter
            wandb_run.summary["elapsed_s"] = elapsed_s
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return {
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "history": history,
        "elapsed_s": elapsed_s,
    }
