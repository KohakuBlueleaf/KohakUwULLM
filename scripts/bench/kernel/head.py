"""The LM head: is the chunked linear-cross-entropy worth it, and does compile help?

At vocab 65536 the head is the single largest memory consumer in the model, so
the choice between PyTorch 2.13's reference and chunked ``linear_cross_entropy``
paths decides the batch size. This script settles it with four panels:

1. **Time** -- fwd+bwd latency vs token count, eager and compiled.
2. **Memory** -- peak allocation, against the card's capacity.
3. **The trade** -- time cost plotted against memory saved, so the exchange rate
   is visible directly rather than inferred from two other charts.
4. **What compile buys** -- speedup from ``torch.compile`` per variant.

The distinction that matters and is easy to miss: ``options=None`` is the
*reference* path and still materializes the full logits. The chunked
implementation only engages when an explicit ``LinearCrossEntropyOptions()`` is
passed.

The axis here is **token count**. ``head_options.py`` sweeps the precision regime
instead, over every implementation we have, and only that axis can see the
``options=None`` trap above; ``chunked_ce.py`` sweeps our own kernel's tile
geometry. Three axes, not three copies.

Usage:
    .venv/bin/python scripts/bench/kernel/head.py --out out/bench/kernel/head
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    bench_ms,
    bench_peak_memory,
    device_name,
    finish_axis,
    new_figure,
    save_figure,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def build_variants(dim, vocab, dtype, device, weight):
    """The five ways to compute a large-vocab CE, as callables of (x, target)."""
    module = torch.nn.LinearCrossEntropyLoss(
        dim, vocab, bias=False, reduction="none"
    ).to(device=device, dtype=dtype)
    module_chunked = torch.nn.LinearCrossEntropyLoss(
        dim,
        vocab,
        bias=False,
        reduction="none",
        # allow_retain_graph is auto-promoted under compile anyway; setting it
        # explicitly keeps eager and compiled measuring the same thing.
        options=LinearCrossEntropyOptions(allow_retain_graph=True),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        module.linear.weight.copy_(weight)
        module_chunked.linear.weight.copy_(weight)

    return {
        "naive": lambda x, t: F.cross_entropy(
            F.linear(x, weight).float(), t, reduction="none"
        ).sum(),
        "lce-reference": lambda x, t: F.linear_cross_entropy(
            x, weight, t, reduction="none", options=None
        )
        .float()
        .sum(),
        "lce-chunked": lambda x, t: F.linear_cross_entropy(
            x,
            weight,
            t,
            reduction="none",
            options=LinearCrossEntropyOptions(allow_retain_graph=True),
        )
        .float()
        .sum(),
        "nn.LinearCE": lambda x, t: module(x, t).float().sum(),
        "nn.LinearCE-chunked": lambda x, t: module_chunked(x, t).float().sum(),
    }


def run(args, device):
    rows = []
    for name, dtype in DTYPES.items():
        for tokens in args.token_counts:
            x = (
                torch.randn(tokens, args.dim, device=device, dtype=dtype) * 0.5
            ).requires_grad_()
            weight = (
                torch.randn(args.vocab, args.dim, device=device, dtype=dtype) * 0.02
            ).requires_grad_()
            target = torch.randint(0, args.vocab, (tokens,), device=device)
            variants = build_variants(args.dim, args.vocab, dtype, device, weight)

            for impl, fn in variants.items():
                for compiled in (False, True):
                    call = torch.compile(fn, dynamic=False) if compiled else fn

                    def run_step(c=call):
                        loss = c(x, target)
                        loss.backward()
                        x.grad = None
                        weight.grad = None

                    try:
                        run_step()  # warm up / trigger compilation
                        ms = bench_ms(run_step, warmup=2, iters=8)
                        mem = bench_peak_memory(run_step)
                    except torch.OutOfMemoryError:
                        print(f"  OOM {impl} compiled={compiled} {name} @{tokens}")
                        torch.cuda.empty_cache()
                        continue
                    except Exception as err:  # noqa: BLE001
                        print(
                            f"  skip {impl} compiled={compiled} {name} @{tokens}: "
                            f"{type(err).__name__}: {str(err)[:90]}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    rows.append(
                        dict(
                            impl=impl,
                            compiled=compiled,
                            dtype=name,
                            tokens=tokens,
                            fwdbwd_ms=ms,
                            peak_gib=mem,
                        )
                    )
                torch.cuda.empty_cache()
            x = weight = None
            torch.cuda.empty_cache()
            print(f"  {name} {tokens} done", flush=True)
    return rows


def plot(rows, out_dir, args):
    pal = Palette()
    fig, axes = new_figure(2, 2, figsize=(14, 10))
    sel = [r for r in rows if r["dtype"] == args.plot_dtype]
    impls = sorted({r["impl"] for r in sel})
    xs = sorted({r["tokens"] for r in sel})

    for impl in impls:
        for compiled in (False, True):
            pts = sorted(
                (r for r in sel if r["impl"] == impl and r["compiled"] == compiled),
                key=lambda r: r["tokens"],
            )
            if not pts:
                continue
            style = "--" if compiled else "-"
            label = impl + (" +compile" if compiled else "")
            axes[0].plot(
                [r["tokens"] for r in pts],
                [r["fwdbwd_ms"] for r in pts],
                style,
                marker="o",
                color=pal.color(impl),
                label=label,
            )
            axes[1].plot(
                [r["tokens"] for r in pts],
                [r["peak_gib"] for r in pts],
                style,
                marker="s",
                color=pal.color(impl),
                label=label,
            )

    finish_axis(
        axes[0],
        "tokens",
        "fwd+bwd (ms)",
        f"Latency ({args.plot_dtype}, vocab {args.vocab})",
        xticks=xs,
        logx=True,
        legend=True,
    )
    finish_axis(
        axes[1],
        "tokens",
        "peak memory (GiB)",
        "Peak memory",
        xticks=xs,
        logx=True,
        legend=True,
    )
    axes[1].axhline(args.gpu_gib, color="#D55E00", linestyle=":", linewidth=1.2)
    axes[1].annotate(
        f"{args.gpu_gib:.0f} GiB card",
        (xs[0], args.gpu_gib),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color="#D55E00",
    )

    # Panel 3: the exchange rate, at the largest token count measured.
    biggest = max(xs)
    at_max = [r for r in sel if r["tokens"] == biggest]
    ref = next(
        (r for r in at_max if r["impl"] == "lce-reference" and not r["compiled"]), None
    )
    if ref:
        for r in at_max:
            marker = "D" if r["compiled"] else "o"
            axes[2].scatter(
                r["peak_gib"],
                r["fwdbwd_ms"],
                s=90,
                marker=marker,
                color=pal.color(r["impl"]),
                zorder=3,
            )
            axes[2].annotate(
                r["impl"] + ("+c" if r["compiled"] else ""),
                (r["peak_gib"], r["fwdbwd_ms"]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7,
            )
        axes[2].set_xscale("log")
        finish_axis(
            axes[2],
            "peak memory (GiB)",
            "fwd+bwd (ms)",
            f"The trade at {biggest} tokens (down-left is better)",
            si_x=False,
        )

    # Panel 4: what compile is worth, per variant.
    labels, gains, colors = [], [], []
    for impl in impls:
        eager = [r for r in sel if r["impl"] == impl and not r["compiled"]]
        comp = [r for r in sel if r["impl"] == impl and r["compiled"]]
        pairs = [
            (e["fwdbwd_ms"], c["fwdbwd_ms"])
            for e in eager
            for c in comp
            if e["tokens"] == c["tokens"]
        ]
        if not pairs:
            continue
        labels.append(impl)
        gains.append(sum(e / c for e, c in pairs) / len(pairs))
        colors.append(pal.color(impl))
    if gains:
        bars = axes[3].bar(labels, gains, color=colors, width=0.6)
        bar_labels(axes[3], bars, "{:.2f}x")
        axes[3].axhline(1.0, color="#999999", linestyle="--", linewidth=1)
        axes[3].set_ylabel("eager time / compiled time")
        axes[3].set_title("What torch.compile buys (>1 = compile is faster)")
        axes[3].tick_params(axis="x", labelsize=7, rotation=20)

    fig.suptitle(
        f"LM head: chunked vs reference linear-cross-entropy -- {device_name()}\n"
        f"dim {args.dim}, vocab {args.vocab}  |  "
        "options=None is the MATERIALIZING reference path",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "head.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/head")
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument(
        "--token-counts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768]
    )
    ap.add_argument("--plot-dtype", default="bfloat16")
    ap.add_argument("--gpu-gib", type=float, default=31.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"device: {device_name()}")

    rows = run(args, torch.device("cuda"))
    with open(os.path.join(args.out, "head.json"), "w") as handle:
        json.dump(rows, handle, indent=2)
    plot(rows, args.out, args)
    print(f"wrote {args.out}/head.png")


if __name__ == "__main__":
    main()
