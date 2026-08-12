"""Task 3 — the pretraining loop itself, as importable functions.

Kept separate from `scripts/train.py` (a thin CLI wrapper) so the training
loop is importable/testable on its own -- `scripts/train.py` layers CLI
parsing and FLOPs-budget sizing (src/model/scaling.py) on top of
`train_model()` without the loop itself needing to know about either.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.model.gpt import GPT, GPTConfig


# Validation split held out from the training corpus by shard. The
# default because it's the only one comparable to the training loss: this
# project's val.bin was replaced by an unrelated corpus (job 17179385
# renamed the original to this name), and meta.json's val_source records
# what a given data_dir's val.bin actually holds. Pass --val-bin val.bin
# for a data_dir straight out of prepare_pretrain_data_streaming.py,
# which writes only that.
DEFAULT_VAL_BIN = "val.bin.trainsplit"

# Accepted TrainConfig.amp_dtype values. fp16 is deliberately absent: it
# would need a GradScaler, and every GPU this runs on has bf16.
AMP_DTYPE_CHOICES = ("auto", "bf16", "fp32")


@dataclass
class TrainConfig:
    data_dir: Path
    out_dir: Path | None = None
    # Which file in data_dir holds the validation tokens -- see load_data
    # for why this is named rather than hardcoded, and why the
    # in-distribution split is the default.
    val_bin: str = DEFAULT_VAL_BIN
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
    # Autocast dtype for the forward/backward pass: "auto" (bf16 where the
    # GPU supports it, fp32 otherwise), "bf16", or "fp32". Weights,
    # gradients and optimizer state stay fp32, and bf16 has fp32's exponent
    # range, so no gradient scaler is involved.
    amp_dtype: str = "auto"
    seed: int = 1337
    log_every_eval: bool = True
    # Print a progress line every N iterations. Independent of
    # eval_interval, which costs an eval each time -- this is just a
    # print, so a long cell can report progress without paying for it.
    # None disables it. W&B captures these into a run's Logs tab.
    progress_every: int | None = None
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
    # Groups many runs under one name in the W&B UI -- a sweep's cells are
    # one experiment, not 98 unrelated runs.
    wandb_group: str | None = None
    # Merged into the W&B run config on top of this dataclass's own
    # fields. For values that identify a run without being training knobs
    # -- a sweep cell's FLOPs budget, say, which TrainConfig never sees
    # because the sweep has already turned it into max_iters.
    wandb_config_extra: dict[str, object] | None = None
    # Where wandb writes its run directories; defaults to out_dir. Set it
    # when out_dir is None (sweep cells keep no checkpoints) so offline
    # runs still land somewhere deliberate instead of ./wandb.
    wandb_dir: Path | None = None
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
    # Auto-resume from out_dir/ckpt_last.pt if one exists, instead of
    # training from iter 0 -- lets a SLURM resubmission with the same
    # --out-dir pick up where a failed/timed-out run left off. Only takes
    # effect if checkpoint_interval actually produced a ckpt_last.pt to
    # resume from.
    resume: bool = True


def resolve_amp_dtype(spec: str, device: torch.device) -> torch.dtype | None:
    """Turn a `TrainConfig.amp_dtype` spec into an autocast dtype, or None for fp32.

    Non-CUDA devices always get None: CPU autocast is a slowdown here, and
    the tests and smoke runs that use CPU want plain fp32 anyway.
    """
    if spec not in AMP_DTYPE_CHOICES:
        raise ValueError(f"amp_dtype must be one of {AMP_DTYPE_CHOICES}, got {spec!r}")
    if spec == "fp32" or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if spec == "bf16":
        raise ValueError(
            f"amp_dtype='bf16' but {torch.cuda.get_device_name(device)} has no bfloat16 "
            "support (needs Ampere or newer); use 'auto' to fall back to fp32."
        )
    return None


def autocast_for(device: torch.device, amp_dtype: torch.dtype | None):
    """Autocast context for `amp_dtype`, or a no-op when running in fp32."""
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def enable_tf32() -> None:
    """Allow TF32 tensor cores for whatever stays in fp32.

    PyTorch leaves this off for matmul by default, which costs ~8x the
    matmul throughput on Ampere in exchange for mantissa bits that
    pretraining does not need. Matters most when bf16 is unavailable and
    the whole run is fp32.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_data(data_dir: Path, val_bin: str = DEFAULT_VAL_BIN) -> tuple[np.memmap, np.memmap, dict]:
    """Memory-map the train/val token streams and read meta.json.

    Missing files raise rather than falling back to another split, so a
    typo can't silently change what every val number downstream means.
    """
    meta = json.loads((data_dir / "meta.json").read_text())
    val_path = data_dir / val_bin
    if not val_path.exists():
        available = sorted(p.name for p in data_dir.glob("val*.bin*"))
        raise FileNotFoundError(f"{val_path} not found; validation files present in {data_dir}: {available or 'none'}")
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
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
    model: GPT,
    data: dict[str, np.memmap],
    cfg: TrainConfig,
    device: torch.device,
    eos_id: int | None = None,
    amp_dtype: torch.dtype | None = None,
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
            # Same precision as the training step, or val_loss measures a
            # numeric regime the model was never trained in.
            with autocast_for(device, amp_dtype):
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

    amp_dtype = resolve_amp_dtype(cfg.amp_dtype, device)
    precision = "fp32" if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
    if device.type == "cuda":
        enable_tf32()
        precision += " autocast + tf32 matmul" if amp_dtype is not None else " + tf32 matmul"
    if cfg.log_every_eval:
        print(f"device={device} precision={precision}")

    train_data, val_data, meta = load_data(cfg.data_dir, cfg.val_bin)
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

    best_val_loss = float("inf")
    start_iter = 0
    elapsed_offset_s = 0.0
    resume_path = out_dir / "ckpt_last.pt" if out_dir is not None else None
    if cfg.resume and resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt["model_cfg"] != model_cfg:
            raise ValueError(
                f"Refusing to resume from {resume_path}: it was trained with model_cfg "
                f"{ckpt['model_cfg']}, which doesn't match this run's {model_cfg}. Point "
                "--out-dir at an empty directory for a differently-shaped run."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_iter = ckpt["iter_num"] + 1
        elapsed_offset_s = ckpt.get("elapsed_s", 0.0)
        print(f"Resumed from {resume_path} at iter {start_iter} (best_val_loss={best_val_loss:.4f})")

    wandb_run = None
    if cfg.use_wandb:
        import wandb

        run_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
        run_config.update(run_config.pop("wandb_config_extra", None) or {})
        run_config.update(
            device=str(device),
            # cfg.amp_dtype is the request ("auto"); this is what it resolved to.
            precision=precision,
            n_params_total=n_params_total,
            n_params_non_embed=n_params_non_embed,
            vocab_size=meta["vocab_size"],
        )
        wandb_dir = cfg.wandb_dir or out_dir
        if wandb_dir is not None:
            wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            # 'wrap' wraps sys.stdout/stderr in-process. The default
            # 'auto' picks fd-level redirection, which several pool
            # workers sharing one inherited stdout mostly lose.
            settings=wandb.Settings(console="wrap"),
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            group=cfg.wandb_group,
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
                losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device, eos_id, amp_dtype)
                elapsed_s = time.time() - start + elapsed_offset_s
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
                # Separate from the ckpt_iter{N}.pt snapshot above: this one
                # carries optimizer state and gets overwritten in place (not
                # numbered), so resuming never has to guess which file is
                # newest -- see TrainConfig.resume.
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_cfg": model_cfg,
                        "iter_num": it,
                        "best_val_loss": best_val_loss,
                        "elapsed_s": elapsed_s,
                    },
                    out_dir / "ckpt_last.pt",
                )

            x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
            # The EOS position's target is the next document's first token,
            # which masking makes unpredictable. Left in anyway: ~1 token in
            # 767, and it keeps loss comparable to the doc_masking=False arm.
            doc_id = document_ids(x, eos_id) if eos_id is not None else None
            # backward() stays outside: autocast records the dtype each op
            # ran in, so gradients follow without the context being active.
            with autocast_for(device, amp_dtype):
                _, loss = model(x, y, doc_id=doc_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if wandb_run is not None:
                wandb_run.log({"train/step_loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}, step=it)

            if cfg.progress_every and (it + 1) % cfg.progress_every == 0:
                # flush: SIGKILL (e.g. a SLURM time limit) skips buffers, and
                # a killed cell's progress is exactly what you want to read.
                print(f"iter {it + 1:6d}/{cfg.max_iters} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        final = estimate_loss(model, {"train": train_data, "val": val_data}, cfg, device, eos_id, amp_dtype)
        elapsed_s = time.time() - start + elapsed_offset_s
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
