"""Task 3 -- fit a training run's token count / step count to a stated FLOPs
budget for a *given* model shape.

This file does not choose a model shape (n_layer/n_head/n_embd) -- that's
still picked by hand. It only answers a narrower question: given a model
shape you already chose, how many tokens / training steps fit inside a
stated FLOPs budget `--flops-budget`.
"""

from __future__ import annotations

# FLOPs(N, D) ~= 6*N*D -- 2 for the forward pass, 4 for the backward pass
# (twice the forward, since gradients are needed w.r.t. both inputs and
# weights), per parameter per token. N here is the model's *total*
# parameter count, embeddings included, since a forward pass genuinely
# spends FLOPs on the embedding lookup and (mainly) the output logits
# matmul -- see GPT.num_params(non_embedding=False), src/model/gpt.py.
FLOPS_PER_PARAM_TOKEN = 6


def tokens_for_budget(flops: float, n_params: int) -> int:
    """D(C, N): how many training tokens fit in FLOPs budget `C` for a model
    with `n_params` (total) parameters, at FLOPs(N, D) = 6*N*D. Assumes a
    single epoch (no repeated tokens) -- the caller is responsible for
    having enough distinct training tokens available.

    This is pure FLOPs bookkeeping for a *given* N -- it does not choose N;
    see this module's docstring for why choosing N is out of scope here.
    """
    return round(flops / (FLOPS_PER_PARAM_TOKEN * n_params))


def max_iters_for_budget(flops: float, n_params: int, batch_size: int, block_size: int) -> int:
    """Convert `tokens_for_budget`'s token count into a training-loop
    `max_iters`, given how many tokens one step consumes
    (batch_size * block_size).
    """
    tokens = tokens_for_budget(flops, n_params)
    return max(1, round(tokens / (batch_size * block_size)))


def describe_budget(
    flops: float, n_params: int, n_layer: int, n_head: int, n_embd: int, batch_size: int, block_size: int
) -> str:
    """One-line human-readable summary printed by scripts/train.py's
    --flops-budget.
    """
    tokens = tokens_for_budget(flops, n_params)
    max_iters = max_iters_for_budget(flops, n_params, batch_size, block_size)
    return (
        f"{flops:.3e} FLOPs budget for n_layer={n_layer}, n_head={n_head}, n_embd={n_embd} "
        f"({n_params:,} total params): D={tokens:,} tokens -> max_iters={max_iters:,} "
        f"(batch_size={batch_size} x block_size={block_size})"
    )


def main() -> None:
    """CLI: print describe_budget() for one or more --flops-budget values,
    without training anything. Model shape is an input here, not an
    output -- see this module's docstring for why.

    Usage: uv run src/model/scaling.py --flops-budget 1e17 1e18 \
        --vocab-size 32000 --n-layer 8 --n-head 8 --n-embd 512 \
        --batch-size 8 --block-size 1024
    """
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.model.gpt import GPT, GPTConfig

    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flops-budget", type=float, nargs="+", required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    n_params = GPT(GPTConfig(args.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd)).num_params(
        non_embedding=False
    )
    for flops in args.flops_budget:
        print(describe_budget(flops, n_params, args.n_layer, args.n_head, args.n_embd, args.batch_size, args.block_size))


if __name__ == "__main__":
    main()
