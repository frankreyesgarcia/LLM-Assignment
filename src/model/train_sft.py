"""The SFT training loop, as an importable function -- paired with
scripts/train_sft.py the same way src/model/train.py pairs with
scripts/train.py.

Deliberately a separate module from src/model/train.py rather than a
branch inside `train_model`: SFT differs from pretraining in exactly the
two places that matter (loss is masked per-token via a precomputed labels
stream instead of "next token", and the model starts from a pretrained
checkpoint instead of random init) but reuses everything else --
`resolve_amp_dtype`, `autocast_for`, `enable_tf32`, `lr_at`,
`configure_optimizer`, and the per-document attention masking in
src/model/gpt.py, which here isolates a padded row's real tokens from its
padding (one conversation per row, right-padded to the longest in the
batch -- see get_sft_batch). Keeping this as new functions rather than adding an
`if cfg.sft:` branch to `train_model` means pretraining's tested code path
never changes shape.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.model.gpt import GPT, GPTConfig
from src.model.train import (
    autocast_for,
    configure_optimizer,
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
    # Stops a padded row's real tokens from attending to its padding --
    # what a padding/attention mask does in a pipeline that has one. Only
    # has any effect on batches that actually needed padding (under
    # length-grouped batching most do not). Off for ablation only.
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


def load_sft_data(data_dir: Path) -> tuple[dict[str, np.memmap], dict[str, np.memmap], dict[str, np.ndarray], dict]:
    """Memory-map the train/val token + label streams written by
    scripts/prepare_sft_data.py, plus the conversation index.

    Labels are int16 (signed, unlike the uint16 token stream) so -1 ("no
    loss here") is representable. `{split}_index.bin` is (N, 2) int64 --
    (start, length) per conversation -- which is what makes one row of a
    batch one conversation rather than an arbitrary slice of the stream.
    """
    meta = json.loads((data_dir / "meta.json").read_text())
    tokens = {
        split: np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r") for split in ("train", "val")
    }
    labels = {
        split: np.memmap(data_dir / f"{split}_labels.bin", dtype=np.int16, mode="r") for split in ("train", "val")
    }
    index = {
        split: np.fromfile(data_dir / f"{split}_index.bin", dtype=np.int64).reshape(-1, 2)
        for split in ("train", "val")
    }
    return tokens, labels, index, meta


def length_grouped_batches(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator, megabatch_factor: int = 50
) -> Iterator[np.ndarray]:
    """Yield batches of conversation indices, grouped by similar length,
    forever (reshuffling every pass).

    Padding is to the longest conversation *in the batch*, so batching
    similar lengths together is what keeps padding waste near zero:
    measured on this corpus, random batches fill 59% of the padded tensor
    while length-grouped batches fill 99.99%.

    Globally sorting by length would maximize that, but then batch
    composition is perfectly correlated with length -- every batch is all
    short or all long, which is a systematically biased gradient. The
    standard compromise (HF's LengthGroupedSampler) and the one used here:
    shuffle, cut into megabatches of batch_size * megabatch_factor, sort
    only within a megabatch, then shuffle the resulting batch order.
    """
    n = len(lengths)
    if n < batch_size:
        raise ValueError(f"split has {n} conversations, fewer than one batch of {batch_size}")
    megabatch_size = batch_size * megabatch_factor
    while True:
        order = rng.permutation(n)
        batches = []
        for i in range(0, n - batch_size + 1, megabatch_size):
            mega = order[i : i + megabatch_size]
            mega = mega[np.argsort(lengths[mega], kind="stable")]
            for j in range(0, len(mega) - batch_size + 1, batch_size):
                batches.append(mega[j : j + batch_size])
        rng.shuffle(batches)
        yield from batches


def get_sft_batch(
    tokens: np.memmap,
    labels: np.memmap,
    index: np.ndarray,
    batch_idx: np.ndarray,
    block_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One batch = `len(batch_idx)` whole conversations, right-padded to the
    longest one in the batch.

    Returns (x, y, doc_id). doc_id marks the padding region as a separate
    document so real tokens cannot attend to it (and is None when the batch
    needs no padding at all, which lets the model skip the mask entirely --
    the common case under length-grouped batching).
    """
    rows = index[batch_idx]
    lengths = np.minimum(rows[:, 1], block_size)  # truncate only if the data was prepared for a larger window
    T = int(lengths.max())
    B = len(batch_idx)

    x = np.full((B, T), pad_token_id, dtype=np.int64)
    # -1 everywhere means padding is never a training target, exactly like
    # a masked (non-assistant) position.
    y = np.full((B, T), -1, dtype=np.int64)
    for r, (start, length) in enumerate(zip(rows[:, 0], lengths)):
        x[r, :length] = tokens[start : start + length]
        # Target for position i is the label of the *next* token (i+1) --
        # same shift convention as src/model/train.py::get_batch, reading
        # the value from the labels stream rather than deriving it. The last
        # real position predicts the EOS that terminates the conversation,
        # whose label is -1, so it contributes no loss.
        y[r, :length] = labels[start + 1 : start + 1 + length]

    doc_id = None
    if not (lengths == T).all():
        # Document 0 is the conversation, document 1 its padding: the
        # per-document attention mask in src/model/gpt.py then makes real
        # tokens unable to see pad positions, which is what a padding mask
        # would do in a pipeline that had one.
        doc_np = (np.arange(T)[None, :] >= lengths[:, None]).astype(np.int64)
        doc_id = torch.from_numpy(doc_np)

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        if doc_id is not None:
            doc_id = doc_id.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        if doc_id is not None:
            doc_id = doc_id.to(device)
    return x, y, doc_id


def estimate_sft_loss(
    model: GPT,
    tokens: dict[str, np.memmap],
    labels: dict[str, np.memmap],
    index: dict[str, np.ndarray],
    cfg: SFTTrainConfig,
    device: torch.device,
    pad_token_id: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split in tokens:
        # A fresh generator per eval, seeded identically every time, so
        # successive evals measure the model on the same conversations
        # rather than on a new random draw -- otherwise the eval-to-eval
        # difference mixes real progress with sampling noise, and
        # best-val checkpoint selection rewards a lucky draw.
        rng = np.random.default_rng(cfg.seed)
        batches = length_grouped_batches(index[split][:, 1], cfg.batch_size, rng)
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y, doc_id = get_sft_batch(
                tokens[split], labels[split], index[split], next(batches), cfg.block_size, pad_token_id, device
            )
            with autocast_for(device, amp_dtype):
                _, loss = model(x, y, doc_id=doc_id if cfg.doc_masking else None)
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

    tokens, labels, index, meta = load_sft_data(cfg.data_dir)
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
    pad_token_id = meta["pad_token_id"]
    batches = length_grouped_batches(index["train"][:, 1], cfg.batch_size, np.random.default_rng(cfg.seed))

    out_dir = cfg.out_dir
    log_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "log.jsonl"

    best_val_loss = float("inf")
    start_iter = 0
    tokens_seen_at_resume = 0
    resume_path = out_dir / "ckpt_last.pt" if out_dir is not None else None
    if cfg.resume and resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_iter = ckpt["iter_num"] + 1
        tokens_seen_at_resume = ckpt.get("tokens_seen", 0)
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
    # Counted, not derived: rows are whole conversations padded to the
    # longest in their batch, so a step no longer processes a fixed
    # batch_size * block_size tokens. tokens_seen is real (non-pad) tokens;
    # padded_tokens_seen is what the GPU actually ran, and their ratio is
    # the fill rate that length-grouped batching is there to keep high.
    tokens_seen = tokens_seen_at_resume
    padded_tokens_seen = 0

    try:
        for it in range(start_iter, cfg.max_iters):
            lr = lr_at(it, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
                losses = estimate_sft_loss(model, tokens, labels, index, cfg, device, pad_token_id, amp_dtype)
                # tokens_seen counts from iter 0 (so it stays comparable across a
                # resumed run), elapsed_s only this submission's wall clock -- same
                # split as src/model/train.py, minus its elapsed_offset_s since the
                # SFT checkpoints don't carry one.
                elapsed_s = time.time() - start
                record = {
                    "iter": it,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "padded_tokens_seen": padded_tokens_seen,
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
                            "padded_tokens_seen": padded_tokens_seen,
                            # Real tokens / tokens the GPU ran. Length-grouped
                            # batching should keep this near 1.0; a drop means
                            # batches are mixing lengths and burning compute
                            # on padding.
                            "batch_fill_rate": tokens_seen / padded_tokens_seen if padded_tokens_seen else 0.0,
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
                        "tokens_seen": tokens_seen,
                    },
                    out_dir / "ckpt_last.pt",
                )

            batch_idx = next(batches)
            x, y, doc_id = get_sft_batch(
                tokens["train"], labels["train"], index["train"], batch_idx, cfg.block_size, pad_token_id, device
            )
            tokens_seen += int(np.minimum(index["train"][batch_idx, 1], cfg.block_size).sum())
            padded_tokens_seen += x.numel()
            with autocast_for(device, amp_dtype):
                _, loss = model(x, y, doc_id=doc_id if cfg.doc_masking else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({"train/step_loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}, step=it)
            if cfg.progress_every and (it + 1) % cfg.progress_every == 0:
                print(f"iter {it + 1:6d}/{cfg.max_iters} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        final = estimate_sft_loss(model, tokens, labels, index, cfg, device, pad_token_id, amp_dtype)
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
            wandb_run.summary["tokens_seen"] = tokens_seen
            wandb_run.summary["padded_tokens_seen"] = padded_tokens_seen
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
