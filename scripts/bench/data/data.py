"""Data pipeline throughput: is the loader ever going to be the bottleneck?

Three questions, because they have different answers and different fixes:

1. **Raw record reads.** KohakuVault random-access rate against DataLoader worker
   count, local NVMe vs NFS. Random reads over NFS are latency-bound, not
   bandwidth-bound, so this scales with workers far past the point where a
   bandwidth-bound source would flatten.
2. **Full pipeline.** Records through rendering, tokenization and packing --
   which is what actually feeds the GPU. Rendering is pure Python and usually
   costs more than the read.
3. **Packing efficiency.** Token counts per batch, and what fraction of a padded
   batch would have been padding. This is the number that justifies the varlen
   path.

The axis here is **worker count x data root**, which is what answers "is the
source or the rendering the limit". ``loader.py`` holds the collation axis --
token-budget vs fixed-count vs padded -- at a fixed source. The full-pipeline
panel below and ``loader.py``'s ``packed`` variant do overlap; run this one when
the question is I/O and that one when the question is layout.

Usage:
    .venv/bin/python scripts/bench/data/data.py --out out/bench/dataset/data
    .venv/bin/python scripts/bench/data/data.py --roots /xg7/caption-datasets /Yamiyoru/caption-datasets
"""

import argparse
import json
import os
import time

import numpy as np
import torch.utils.data as data

from kohakuwullm.bench import Palette, bar_labels, finish_axis, new_figure, save_figure
from kohakuwullm.data import build_dataset, collate_packed
from kohakuwullm.data.sources.vault import DanbooruRecords


class _RecordOnly(data.Dataset):
    """Reads records without rendering, to separate I/O cost from CPU cost."""

    def __init__(self, root: str, n: int, seed: int = 0) -> None:
        self.records = DanbooruRecords(root=root)
        rng = np.random.default_rng(seed)
        self.picks = rng.choice(len(self.records), n, replace=False)

    def __len__(self) -> int:
        return len(self.picks)

    def __getitem__(self, index: int) -> int:
        rec = self.records[int(self.picks[index])]
        return len(rec.get("general") or []) if rec else 0


def _short_root(root: str) -> str:
    """`/xg7/caption-datasets` -> `xg7/caption-datasets`.

    Both roots share a basename, so the mount point is what distinguishes them
    and it has to stay in the legend.
    """
    parts = os.path.abspath(root).strip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else root


def bench_reads(roots, worker_counts, samples):
    rows = []
    for root in roots:
        if not os.path.exists(os.path.join(root, "raw/danbooru_meta.db")):
            print(f"  skip {root}: no danbooru_meta.db")
            continue
        for workers in worker_counts:
            # A fresh seed per configuration: reusing one makes the second run
            # read a page-cache-warm working set and report a fictional rate.
            seed = 1000 + workers + hash(root) % 100
            dataset = _RecordOnly(root, samples, seed=seed)
            loader = data.DataLoader(
                dataset, batch_size=64, num_workers=workers, collate_fn=list
            )
            start = time.perf_counter()
            total = sum(len(b) for b in loader)
            elapsed = time.perf_counter() - start
            rows.append(
                dict(
                    stage="read",
                    root=_short_root(root),
                    workers=workers,
                    rec_per_s=total / elapsed,
                )
            )
            print(
                f"  read {root} w={workers}: {total / elapsed:,.0f} rec/s", flush=True
            )
    return rows


def bench_pipeline(root, tokenizer, worker_counts, batches, batch_size, max_length):
    rows = []
    dataset = build_dataset(
        [{"name": "danbooru"}], tokenizer, root=root, max_length=max_length
    )
    for workers in worker_counts:
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_packed,
            drop_last=True,
            **({"prefetch_factor": 4} if workers else {}),
        )
        iterator = iter(loader)
        # Drain the whole prefetch buffer before timing. With 32 workers and
        # prefetch_factor 4 the loader has 128 batches queued; timing fewer than
        # that measures how fast a queue empties, not how fast it refills.
        prefetch_depth = max(2, workers * 4)
        for _ in range(prefetch_depth):
            next(iterator)
        start = time.perf_counter()
        tokens = trained = 0
        longest = 0
        for _ in range(batches):
            batch = next(iterator)
            tokens += batch.num_tokens
            trained += batch.num_trained
            longest = max(longest, batch.seq_info.max_seqlen)
        elapsed = time.perf_counter() - start
        rows.append(
            dict(
                stage="pipeline",
                workers=workers,
                tokens_per_s=tokens / elapsed,
                samples_per_s=batches * batch_size / elapsed,
                trained_frac=trained / max(tokens, 1),
                mean_doc_len=tokens / (batches * batch_size),
                max_doc_len=longest,
                # What a padded batch would have cost: every document stretched to
                # the longest one in its batch.
                pad_frac=1 - (tokens / (batches * batch_size)) / max(longest, 1),
            )
        )
        print(f"  pipeline w={workers}: {tokens / elapsed:,.0f} tok/s", flush=True)
    return rows


def bench_lengths(root, tokenizer, n, max_length):
    """Token-length distribution of rendered samples."""
    dataset = build_dataset(
        [{"name": "danbooru"}], tokenizer, root=root, max_length=max_length
    )
    rng = np.random.default_rng(0)
    lengths = [
        len(dataset[int(i)]["input_ids"])
        for i in rng.choice(len(dataset), n, replace=False)
    ]
    return lengths


def plot(rows, lengths, out_dir, args):
    pal = Palette()
    fig, axes = new_figure(2, 2, figsize=(13, 9))

    reads = [r for r in rows if r["stage"] == "read"]
    workers = sorted({r["workers"] for r in reads})
    for root in sorted({r["root"] for r in reads}):
        sel = sorted(
            (r for r in reads if r["root"] == root), key=lambda r: r["workers"]
        )
        axes[0].plot(
            [r["workers"] for r in sel],
            [r["rec_per_s"] for r in sel],
            marker="o",
            color=pal.color(root),
            label=root,
        )
    finish_axis(
        axes[0],
        "dataloader workers",
        "records / s",
        "Raw KohakuVault reads",
        xticks=workers,
        si_x=False,
        legend=True,
    )

    pipe = sorted(
        (r for r in rows if r["stage"] == "pipeline"), key=lambda r: r["workers"]
    )
    if pipe:
        axes[1].plot(
            [r["workers"] for r in pipe],
            [r["tokens_per_s"] for r in pipe],
            marker="s",
            color=pal.color("pipeline"),
            label="tokens/s",
        )
        finish_axis(
            axes[1],
            "dataloader workers",
            "tokens / s",
            "Full pipeline (render + tokenize + pack)",
            xticks=[r["workers"] for r in pipe],
            si_x=False,
        )
        # A 500M model on 4x5090 consumes roughly this much; the loader has to
        # clear it or the GPUs idle.
        axes[1].axhline(
            args.target_tokens_per_s, color="#D55E00", linestyle="--", linewidth=1.2
        )
        axes[1].annotate(
            f"training demand ~{args.target_tokens_per_s / 1000:.0f}k tok/s",
            (pipe[0]["workers"], args.target_tokens_per_s),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="#D55E00",
        )

    if lengths:
        axes[2].hist(lengths, bins=60, color=pal.color("lengths"), edgecolor="none")
        axes[2].axvline(
            float(np.mean(lengths)),
            color="#D55E00",
            linestyle="--",
            linewidth=1.2,
            label=f"mean {np.mean(lengths):.0f}",
        )
        axes[2].axvline(
            float(np.percentile(lengths, 99)),
            color="#0072B2",
            linestyle=":",
            linewidth=1.2,
            label=f"p99 {np.percentile(lengths, 99):.0f}",
        )
        axes[2].set_xlabel("tokens per rendered sample")
        axes[2].set_ylabel("count")
        axes[2].set_title("Sample length distribution")
        axes[2].legend()

    if pipe:
        best = max(pipe, key=lambda r: r["tokens_per_s"])
        labels = ["trained\ntokens", "masked\n(context)", "would-be\npadding"]
        values = [
            100 * best["trained_frac"],
            100 * (1 - best["trained_frac"]),
            100 * best["pad_frac"],
        ]
        bars = axes[3].bar(
            labels,
            values,
            color=[pal.color("trained"), pal.color("masked"), pal.color("padding")],
            width=0.6,
        )
        bar_labels(axes[3], bars, "{:.1f}%")
        axes[3].set_ylabel("% of packed tokens")
        axes[3].set_title("Where the tokens go (packing avoids the third bar)")

    fig.suptitle(
        "Data pipeline -- KohakuVault caption/tag corpus\n"
        f"batch {args.batch_size} documents, max length {args.max_length}",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "data.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/dataset/data")
    ap.add_argument("--roots", nargs="+", default=["/xg7/caption-datasets"])
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    ap.add_argument("--read-samples", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--length-samples", type=int, default=4000)
    ap.add_argument("--target-tokens-per-s", type=float, default=400_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = bench_reads(args.roots, args.workers, args.read_samples)
    rows += bench_pipeline(
        args.roots[0],
        tokenizer,
        args.workers,
        args.batches,
        args.batch_size,
        args.max_length,
    )
    lengths = bench_lengths(
        args.roots[0], tokenizer, args.length_samples, args.max_length
    )

    with open(os.path.join(args.out, "data.json"), "w") as handle:
        json.dump({"rows": rows, "lengths": lengths}, handle)
    plot(rows, lengths, args.out, args)
    print(f"wrote {args.out}/data.png")


if __name__ == "__main__":
    main()
