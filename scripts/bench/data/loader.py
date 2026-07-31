"""Dataloader throughput and batch-size stability: token budget vs fixed count.

Three loaders, all emitting a model-ready
:class:`~kohakuwullm.data.packing.PackedBatch`, all measured with tokenization
*inside* the timed loop because that is where 73% of the per-sample cost lives:

1. ``iterative``  token-budget varlen packing
   (:mod:`kohakuwullm.data.loader.iterative`) -- documents are appended until the total
   would cross ``k``.
2. ``packed``     fixed document count + ``collate_packed`` -- the current
   training path.
3. ``padded``     fixed document count + ``collate_padded`` -- the eval path,
   included because its token count is what a non-varlen kernel would compute
   on.

The baselines get ``k / mean_document_length`` documents per batch, so all three
carry the same *nominal* budget and the per-batch spread is comparable.

Two numbers per configuration, and the difference between them is the point of
the packed layout:

* **model tokens/s** -- what the GPU would compute on, padding included. This is
  the figure the instruction sheet asks for, and it flatters ``padded``.
* **real tokens/s** -- non-padding tokens only. This is the apples-to-apples
  loader comparison; ``padded`` spends ~85% of its model tokens on nothing.

The benchmark is CPU/IO bound and never touches CUDA, so it can run while the
GPUs are busy. That also rules out :func:`kohakuwullm.bench.bench_ms`, which
times with CUDA events; timing here is ``perf_counter`` around a batch count
larger than the prefetch buffer.

The axis is the **collation layout** at a fixed source. ``data.py`` holds the
other one -- worker count against local NVMe vs NFS, and the read/render split --
which is where "is the source or the rendering the limit" is answered.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench/data/loader.py
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/bench/data/loader.py --k 65536 --workers 0 8
"""

import argparse
import functools
import json
import os
import statistics
import time

import numpy as np
import torch.utils.data as data

from kohakuwullm.bench import Palette, bar_labels, finish_axis, new_figure, save_figure
from kohakuwullm.data import RenderedDataset, TIPORenderer, collate_packed
from kohakuwullm.data.loader.iterative import build_iterative_loader
from kohakuwullm.data.loader.padded import collate_padded, padding_fraction
from kohakuwullm.data.sources.vault import DanbooruRecords

VARIANTS = ["iterative", "packed", "padded"]


def _baseline_loader(records, tokenizer, args, layout, workers, batch_size):
    """Fixed-document-count loader over the existing map-style path."""
    if layout == "padded":
        collate = functools.partial(
            collate_padded,
            pad_token_id=0,
            max_length=args.ctx_max,
            pad_to_multiple=8,
        )
    else:
        collate = functools.partial(collate_packed, pad_to_multiple=0)

    dataset = RenderedDataset(
        records, TIPORenderer(), tokenizer, max_length=args.ctx_max, seed=args.seed
    )
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        # pin_memory allocates through the CUDA host allocator, which this
        # benchmark has no device for. Off for every variant, so the comparison
        # is unaffected.
        pin_memory=False,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return data.DataLoader(dataset, **kwargs)


def _build_loader(records, tokenizer, args, variant, workers, batch_size):
    if variant == "iterative":
        return build_iterative_loader(
            records,
            TIPORenderer(),
            tokenizer,
            k=args.k,
            m=args.m,
            ctx_max=args.ctx_max,
            num_workers=workers,
            seed=args.seed,
            prefetch_factor=args.prefetch_factor,
            pin_memory=False,
        )
    return _baseline_loader(records, tokenizer, args, variant, workers, batch_size)


def _batch_stats(batch, variant) -> dict:
    """Per-batch record: model tokens, real tokens, shape."""
    waste = padding_fraction(batch) if variant == "padded" else 0.0
    return dict(
        model_tokens=int(batch.num_tokens),
        real_tokens=int(round(batch.num_tokens * (1 - waste))),
        trained=int(batch.num_trained),
        docs=int(batch.seq_info.num_seqs),
        max_seqlen=int(batch.seq_info.max_seqlen),
        waste=waste,
    )


def _window(iterator, variant, timed: int) -> tuple[dict, list[dict]]:
    """One timed window of ``timed`` batches off an already-running iterator."""
    batches, marks = [], []
    start = time.perf_counter()
    for _ in range(timed):
        batches.append(_batch_stats(next(iterator), variant))
        marks.append(time.perf_counter())
    elapsed = marks[-1] - start

    # Steady-state check: if the drain was too short the first half is still
    # emptying a queue and runs *faster* than the workers can refill it, so only
    # a ratio above 1 impeaches the number. Counted in documents rather than
    # tokens because a padded batch's token count swings 4x with its longest
    # document, and that composition noise would swamp the signal.
    half = len(batches) // 2
    halves = [
        sum(r["docs"] for r in batches[:half]) / (marks[half - 1] - start),
        sum(r["docs"] for r in batches[half:]) / (marks[-1] - marks[half - 1]),
    ]
    return (
        dict(
            seconds=elapsed,
            model_tokens_per_s=sum(r["model_tokens"] for r in batches) / elapsed,
            real_tokens_per_s=sum(r["real_tokens"] for r in batches) / elapsed,
            docs_per_s=sum(r["docs"] for r in batches) / elapsed,
            half_ratio=halves[0] / max(halves[1], 1e-9),
        ),
        batches,
    )


def measure(
    loader, variant, drain: int, timed: int, repeats: int = 3
) -> tuple[dict, list[dict]]:
    """Median of ``repeats`` timed windows, after draining the prefetch buffer.

    Draining matters more than it looks. With 32 workers and ``prefetch_factor``
    2 the loader has 64 batches queued before the first ``next()`` returns;
    timing fewer than that measures how fast a queue empties, not how fast the
    workers refill it, and reports a throughput the pipeline can never sustain.

    Median over repeats, for the same reason :func:`kohakuwullm.bench.bench_ms`
    takes one: this machine has other tenants, and a single window of a 32-worker
    loader swings 15% run to run -- enough on its own to reverse the ranking of
    two variants that are actually equal.
    """
    iterator = iter(loader)
    records = []
    for _ in range(drain):
        records.append(_batch_stats(next(iterator), variant))

    windows = []
    for _ in range(repeats):
        window, batches = _window(iterator, variant, timed)
        windows.append(window)
        records.extend(batches)
    del iterator

    rates = sorted(w["model_tokens_per_s"] for w in windows)
    return (
        dict(
            batches=timed,
            repeats=repeats,
            model_tokens_per_s=statistics.median(rates),
            real_tokens_per_s=statistics.median(
                w["real_tokens_per_s"] for w in windows
            ),
            docs_per_s=statistics.median(w["docs_per_s"] for w in windows),
            half_ratio=max(w["half_ratio"] for w in windows),
            repeat_spread=rates[-1] / max(rates[0], 1e-9),
            windows=windows,
        ),
        records,
    )


def _spread(records: list[dict]) -> dict:
    """Mean / std / coefficient of variation of the per-batch token totals."""
    model = [r["model_tokens"] for r in records]
    real = [r["real_tokens"] for r in records]
    mean = statistics.fmean(model)
    std = statistics.pstdev(model)
    return dict(
        mean_model_tokens=mean,
        std_model_tokens=std,
        cv_model_tokens=std / max(mean, 1),
        min_model_tokens=min(model),
        max_model_tokens=max(model),
        mean_real_tokens=statistics.fmean(real),
        std_real_tokens=statistics.pstdev(real),
        cv_real_tokens=statistics.pstdev(real) / max(statistics.fmean(real), 1),
        mean_docs=statistics.fmean([r["docs"] for r in records]),
        mean_waste=statistics.fmean([r["waste"] for r in records]),
    )


def ablate_m(records, tokenizer, args):
    """Deficit and throughput against ``m``, the retry rule's only knob.

    ``m >= ctx_max`` makes no document "long" and disables the retry, which is
    the honest control: it says how much of the batch-size stability comes from
    token-budget batching alone and how much the retry adds on top.
    """
    rows = []
    for m in args.ablate_m:
        loader = build_iterative_loader(
            records,
            TIPORenderer(),
            tokenizer,
            k=args.k,
            m=m,
            ctx_max=args.ctx_max,
            num_workers=args.ablate_workers,
            seed=args.seed,
            prefetch_factor=args.prefetch_factor,
            pin_memory=False,
        )
        drain = max(4, args.ablate_workers * args.prefetch_factor)
        timing, batch_records = measure(
            loader, "iterative", drain, args.ablate_batches, repeats=1
        )
        deficits = [args.k - r["model_tokens"] for r in batch_records]
        rows.append(
            dict(
                m=m,
                max_deficit=max(deficits),
                mean_deficit=statistics.fmean(deficits),
                cv_model_tokens=_spread(batch_records)["cv_model_tokens"],
                mean_docs=statistics.fmean([r["docs"] for r in batch_records]),
                model_tokens_per_s=timing["model_tokens_per_s"],
            )
        )
        print(
            f"  m={m:<5} max deficit {rows[-1]['max_deficit']:>5} "
            f"mean {rows[-1]['mean_deficit']:>7.1f} "
            f"CV={rows[-1]['cv_model_tokens'] * 100:6.3f}% "
            f"{rows[-1]['model_tokens_per_s']:>12,.0f} tok/s",
            flush=True,
        )
        del loader
    return rows


def mean_document_length(records, tokenizer, args, n: int = 2000) -> float:
    """Mean tokenized length, which sets the baselines' documents per batch."""
    dataset = RenderedDataset(
        records, TIPORenderer(), tokenizer, max_length=args.ctx_max, seed=args.seed
    )
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(dataset), n, replace=False)
    lengths = [len(dataset[int(i)]["input_ids"]) for i in picks]
    return statistics.fmean(lengths)


def run(records, tokenizer, args):
    mean_len = mean_document_length(records, tokenizer, args)
    batch_size = max(1, round(args.k / mean_len))
    print(f"mean document {mean_len:.1f} tokens -> baseline batch {batch_size} docs")

    rows, traces = [], {}
    for variant in args.variants:
        for workers in args.workers:
            # One drain of the whole prefetch buffer, then at least as many
            # timed batches again. The floor is what warms the single-process
            # case, which has no buffer but still spends its first batches
            # growing the allocator's arena for 18 MB padded tensors.
            drain = max(4, workers * args.prefetch_factor)
            timed = max(args.min_batches, 2 * workers * args.prefetch_factor)
            loader = _build_loader(
                records, tokenizer, args, variant, workers, batch_size
            )
            timing, batch_records = measure(
                loader, variant, drain, timed, repeats=args.repeats
            )
            row = dict(
                variant=variant,
                workers=workers,
                batch_size=batch_size if variant != "iterative" else None,
                **timing,
                **_spread(batch_records),
            )
            rows.append(row)
            if workers == args.trace_workers:
                traces[variant] = batch_records
            drift = (
                "  QUEUE DRAIN, NOT STEADY STATE" if row["half_ratio"] > 1.15 else ""
            )
            print(
                f"  {variant:<10} w={workers:<3} "
                f"{row['model_tokens_per_s']:>12,.0f} model tok/s "
                f"{row['real_tokens_per_s']:>12,.0f} real tok/s "
                f"CV={row['cv_model_tokens'] * 100:5.2f}% "
                f"halves={row['half_ratio']:.2f} "
                f"spread={row['repeat_spread']:.2f}{drift}",
                flush=True,
            )
            del loader
    return rows, traces, mean_len, batch_size


def plot(rows, traces, ablation, out_dir, args, mean_len, batch_size):
    pal = Palette()
    for variant in VARIANTS:
        pal.color(variant)
    fig, axes = new_figure(2, 3, figsize=(19, 9.5))

    for variant in args.variants:
        sel = sorted(
            (r for r in rows if r["variant"] == variant), key=lambda r: r["workers"]
        )
        axes[0].plot(
            [r["workers"] for r in sel],
            [r["model_tokens_per_s"] for r in sel],
            marker="o",
            color=pal.color(variant),
            label=f"{variant} (model tokens)",
        )
        axes[0].plot(
            [r["workers"] for r in sel],
            [r["real_tokens_per_s"] for r in sel],
            marker="",
            linestyle="--",
            linewidth=1.2,
            color=pal.color(variant),
            label=f"{variant} (real tokens)",
        )
        # Min-max band over the repeats. Without it a reader takes the gap
        # between two lines for a result; on this machine a single window swings
        # far enough to reverse two variants that are equal.
        axes[0].fill_between(
            [r["workers"] for r in sel],
            [min(w["model_tokens_per_s"] for w in r["windows"]) for r in sel],
            [max(w["model_tokens_per_s"] for w in r["windows"]) for r in sel],
            color=pal.color(variant),
            alpha=0.15,
            linewidth=0,
        )
    finish_axis(
        axes[0],
        "dataloader workers",
        "tokens / s",
        "Throughput (tokenization inside the timed loop)",
        xticks=args.workers,
        si_x=False,
        logy=True,
        legend=True,
    )

    # |deviation| on a log axis, not the signed value on a linear one: the three
    # variants are two orders of magnitude apart, and a single 2048-token
    # document triples a padded batch, so one linear axis shows only the
    # padded tail and squashes the other two into the zero bin.
    bins = np.logspace(-3, 2.5, 56)
    for variant in args.variants:
        totals = np.array([r["model_tokens"] for r in traces.get(variant, [])])
        if not totals.size:
            continue
        cv = totals.std() / max(totals.mean(), 1)
        axes[1].hist(
            np.abs(100 * (totals / totals.mean() - 1)).clip(1e-3, None),
            bins=bins,
            color=pal.color(variant),
            alpha=0.7,
            label=f"{variant}  CV={100 * cv:.2f}%",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("|per-batch total - variant mean| / mean  (%)")
    axes[1].set_ylabel("batches")
    axes[1].set_title("Per-batch token count spread")
    axes[1].legend()

    for variant in args.variants:
        real = [r["real_tokens"] for r in traces.get(variant, [])][: args.trace_len]
        if not real:
            continue
        axes[2].plot(range(len(real)), real, color=pal.color(variant), label=variant)
    axes[2].axhline(args.k, color="#333333", linestyle=":", linewidth=1.2)
    axes[2].annotate(
        f"k = {args.k:,}",
        (0, args.k),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
    )
    finish_axis(
        axes[2],
        "batch index",
        "real (non-padding) tokens",
        f"Trace at {args.trace_workers} workers",
        si_x=False,
        legend=True,
    )

    # Both bars come from the trace configuration, so every panel in the figure
    # describes the same run.
    shown = [v for v in args.variants if traces.get(v)]
    spreads = {v: _spread(traces[v]) for v in shown}
    bars = axes[3].bar(
        shown,
        [100 * spreads[v]["cv_model_tokens"] for v in shown],
        color=[pal.color(v) for v in shown],
        width=0.55,
    )
    bar_labels(axes[3], bars, "{:.2f}%")
    axes[3].set_yscale("log")
    axes[3].set_ylabel("CV of per-batch tokens (%)")
    axes[3].set_title("Batch-size stability (lower is better)")

    bars = axes[4].bar(
        shown,
        [100 * spreads[v]["mean_waste"] for v in shown],
        color=[pal.color(v) for v in shown],
        width=0.55,
    )
    bar_labels(axes[4], bars, "{:.1f}%")
    axes[4].set_ylabel("% of model tokens that are padding")
    axes[4].set_title("Wasted compute")

    if ablation:
        ms = [r["m"] for r in ablation]
        axes[5].plot(
            ms,
            [r["max_deficit"] for r in ablation],
            marker="o",
            color=pal.color("iterative"),
            label="max deficit",
        )
        axes[5].plot(
            ms,
            [r["mean_deficit"] for r in ablation],
            marker="s",
            linestyle="--",
            color=pal.color("packed"),
            label="mean deficit",
        )
        # A batch ends on a document of at most `m`, so it cannot be more than
        # `m` short -- but only while the retry can find such a document. Below
        # roughly m=128 here no rejected document is ever that short, the batch
        # ends on `max_retry` instead, and the deficit floors out above the line
        # while staying small in absolute terms.
        axes[5].plot(
            ms, ms, color="#333333", linestyle=":", label="deficit = m (the bound)"
        )
        axes[5].set_xscale("log", base=2)
        axes[5].set_yscale("log", base=2)
        finish_axis(
            axes[5],
            "m (long-content threshold)",
            "tokens short of the budget",
            f"Retry ablation at {args.ablate_workers} workers "
            f"(m >= ctx_max disables the retry)",
            si_x=False,
            legend=True,
        )
        axes[5].annotate(
            "max_retry ends\nthe batch here",
            (ms[0], max(r["max_deficit"] for r in ablation[:2])),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color="#333333",
        )

    fig.suptitle(
        "Dataloader -- token-budget iterative packing vs fixed document count\n"
        f"k={args.k:,}  m={args.m}  ctx_max={args.ctx_max}  "
        f"baseline batch {batch_size} docs (mean document {mean_len:.0f} tokens)",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "loader.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/dataset/loader")
    ap.add_argument("--root", default="/xg7/caption-datasets")
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    ap.add_argument("--k", type=int, default=262144)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--ctx-max", type=int, default=2048)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--min-batches", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trace-workers", type=int, default=32)
    ap.add_argument(
        "--ablate-m", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1024, 2048]
    )
    ap.add_argument("--ablate-workers", type=int, default=8)
    ap.add_argument("--ablate-batches", type=int, default=64)
    ap.add_argument("--trace-len", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    # Redraw from a previous run's JSON. A 35-minute measurement should not be
    # the price of moving a legend.
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.replot:
        with open(os.path.join(args.out, "loader.json")) as handle:
            saved = json.load(handle)
        plot(
            saved["rows"],
            saved["traces"],
            saved["ablation"],
            args.out,
            args,
            saved["mean_doc_len"],
            saved["baseline_batch_size"],
        )
        print(f"wrote {args.out}/loader.png")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = DanbooruRecords(root=args.root)
    print(f"{len(records):,} records from {args.root}")

    rows, traces, mean_len, batch_size = run(records, tokenizer, args)
    print("retry ablation over m:")
    ablation = ablate_m(records, tokenizer, args)
    with open(os.path.join(args.out, "loader.json"), "w") as handle:
        json.dump(
            {
                "rows": rows,
                "traces": traces,
                "ablation": ablation,
                "mean_doc_len": mean_len,
                "baseline_batch_size": batch_size,
                "config": vars(args),
            },
            handle,
        )
    plot(rows, traces, ablation, args.out, args, mean_len, batch_size)
    print(f"wrote {args.out}/loader.png")


if __name__ == "__main__":
    main()
