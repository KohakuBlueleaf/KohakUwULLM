"""Token count of every corpus, sampled through the real loader.

One epoch's token count is what a training plan is built from, and it cannot be derived:
the TIPO renderer is pure Python and a record's rendered length is not a function of
anything cheap to look up. So it gets measured -- but only on a sample, because the
record count is exact and sampling only estimates the *mean* length. At 200k records the
standard error on that mean is a fraction of a percent, which is far tighter than any
decision this feeds.

The sample goes through ``build_dataset`` and ``collate_packed`` -- the loader the
trainer uses, not a reimplementation -- so ``num_tokens`` is what the model would see,
after truncation to ``--max-length`` and with the same loss masking.

Usage::

    .venv/bin/python scripts/data/token_census.py --limit 200000 --workers 16
"""

import argparse
import json
import os
import time

import numpy as np
import torch.utils.data as data
from transformers import AutoTokenizer

from kohakuwullm.data import build_dataset, collate_packed
from kohakuwullm.data.sources.vault import PATH_KEYED

ALL_SOURCES = [
    "danbooru",
    "danbooru_tagger",
    *[name for name in PATH_KEYED if name != "danbooru_tagger"],
]


def census_one(name: str, tokenizer, args) -> dict:
    dataset = build_dataset(
        [{"name": name, "repeat": 1}],
        tokenizer,
        root=args.root,
        max_length=args.max_length,
    )
    records = len(dataset)
    subset = dataset
    if args.limit and args.limit < records:
        rng = np.random.default_rng(0)
        subset = data.Subset(
            dataset, rng.choice(records, args.limit, replace=False).tolist()
        )

    loader = data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_packed,
        **({"prefetch_factor": 4} if args.workers else {}),
    )

    tokens = trained = seen = longest = 0
    started = time.perf_counter()
    for batch in loader:
        tokens += batch.num_tokens
        trained += batch.num_trained
        longest = max(longest, batch.seq_info.max_seqlen)
        seen += args.batch_size
    elapsed = time.perf_counter() - started

    counted = min(seen, len(subset))
    mean = tokens / counted if counted else 0.0
    return {
        "name": name,
        "records": records,
        "counted_records": counted,
        "sampled": counted < records,
        "mean_length": mean,
        "max_doc_len": int(longest),
        "trained_fraction": trained / tokens if tokens else 0.0,
        "tokens_counted": int(tokens),
        "tokens_epoch": int(round(mean * records)),
        "seconds": elapsed,
        "tokens_per_s": tokens / elapsed if elapsed else 0.0,
    }


def write_markdown(path: str, results: list[dict], args) -> None:
    total_records = sum(r["records"] for r in results)
    total_tokens = sum(r["tokens_epoch"] for r in results)
    lines = [
        "# Corpus token census",
        "",
        f"Root: `{args.root}` · tokenizer: `{args.tokenizer}` · "
        f"max_length: {args.max_length} · sample: {args.limit:,} records/source",
        "",
        "Record counts are exact. Token counts are the sampled mean length times the "
        "exact record count -- an estimate, and labelled one. Measured through "
        "`build_dataset` + `collate_packed`, so truncation and loss masking are the "
        "trainer's, not an approximation of them.",
        "",
        "One epoch at `repeat: 1`. A source listed with `repeat: 3` in a training config "
        "contributes three times this -- as re-renders, not duplicates.",
        "",
        "| source | records | tokens/epoch | mean len | max len | trained |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['records']:,} | **{r['tokens_epoch'] / 1e9:.3f}B** | "
            f"{r['mean_length']:.1f} | {r['max_doc_len']:,} | "
            f"{r['trained_fraction']:.1%} |"
        )
    lines += [
        f"| **total** | **{total_records:,}** | **{total_tokens / 1e9:.3f}B** | | | |",
        "",
        f"At 262,144 tokens/step that is **{total_tokens / 262144:,.0f} steps** per epoch.",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/xg7/caption-datasets")
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--sources", nargs="+", default=ALL_SOURCES)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="out/bench/dataset/corpus")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in args.sources:
        try:
            result = census_one(name, tokenizer, args)
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(result)
        print(
            f"{name:16s} {result['records']:>12,d} records  "
            f"{result['tokens_epoch'] / 1e9:7.3f}B tokens  "
            f"mean {result['mean_length']:6.1f}  "
            f"{result['tokens_per_s'] / 1e6:5.2f}M tok/s",
            flush=True,
        )
        with open(os.path.join(args.out, "census.json"), "w") as handle:
            json.dump({"config": vars(args), "sources": results}, handle, indent=2)
        write_markdown(os.path.join(args.out, "CORPUS.md"), results, args)

    total = sum(r["tokens_epoch"] for r in results)
    print(f"\ntotal {total / 1e9:.3f}B tokens, one epoch = {total / 262144:,.0f} steps")


if __name__ == "__main__":
    main()
