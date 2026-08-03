"""Loader throughput over the general-pretrain corpus vaults.

Times the full path -- vault read, render, tokenize, varlen pack -- at several
worker counts. Batches are timed only after the prefetch queue has been drained,
because timing fewer batches than the queue holds measures how fast a queue
empties rather than how fast it refills.

    .venv/bin/python scripts/bench/data/corpus.py --sources zh-tw/ultra-fineweb-l3
    .venv/bin/python scripts/bench/data/corpus.py --workers 8 16 --batches 40
"""

import argparse
import time

from transformers import AutoTokenizer

from kohakuwullm.data import build_loader


def bench(sources, tokenizer, workers: int, batches: int, k: int, ctx_max: int):
    """``(tokens_per_s, batches_timed, mean_doc_tokens)`` at one worker count."""
    loader = build_loader(
        "iterative",
        [{"name": s, "repeat": 1} for s in sources],
        tokenizer,
        renderer="plain",
        k=k,
        ctx_max=ctx_max,
        num_workers=workers,
        prefetch_factor=4 if workers else None,
        batches_per_epoch=10**9,
    )
    it = iter(loader)
    warm = max(workers * 4, 8)
    docs = tokens = 0
    for _ in range(warm):
        next(it)
    started = time.time()
    for _ in range(batches):
        batch = next(it)
        tokens += int(batch.num_tokens)
        docs += int(batch.seq_info.cu_seqlens.numel() - 1)
    elapsed = time.time() - started
    del it, loader
    return tokens / elapsed, batches, tokens / max(docs, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["zh-tw/ultra-fineweb-l3", "zh-tw/finepdfs-zh-zhtw"],
    )
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--workers", nargs="+", type=int, default=[0, 4, 8, 16])
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--k", type=int, default=262144)
    ap.add_argument("--ctx-max", type=int, default=2048)
    ap.add_argument("--target", type=float, default=500_000)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"sources: {args.sources}")
    print(f"k={args.k} ctx_max={args.ctx_max} batches={args.batches}\n")
    print(f"{'workers':>8s} {'tok/s':>12s} {'vs target':>10s} {'mean doc tok':>13s}")
    print("-" * 48)
    for workers in args.workers:
        rate, _, mean_doc = bench(
            args.sources, tokenizer, workers, args.batches, args.k, args.ctx_max
        )
        print(
            f"{workers:8d} {rate:12,.0f} {rate / args.target:9.2f}x {mean_doc:13.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
