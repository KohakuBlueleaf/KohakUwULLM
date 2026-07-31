"""What the LM head costs, across every implementation we have.

The head is the largest single GEMM in the model and the pipeline's bottleneck
stage, so its cost decides both. Three things move it:

* **which cores the GEMM lands on** -- bf16 inputs reach the tensor cores
  (400 TFLOP/s mma peak, ~270 achievable), fp32 inputs run on the vector cores.
  Autocast decides this in a real step, where parameters are fp32.
* **chunked vs materializing** in ``F.linear_cross_entropy``.
* **our own kernel**, :func:`~kohakuwullm.kernels.loss.chunked_ce.chunked_linear_cross_entropy`,
  whose ``vocab_block`` and ``retain`` knobs trade memory for recompute
  continuously rather than at the two endpoints torch offers.

Everything is measured fwd+bwd under the autocast regime training uses, and
reported as time against peak memory, because a head that is fast and does not
fit is not usable at 8B.

The axis here is the **precision regime**, and it is the only one of the three
head sweeps that can see the autocast point above: ``head.py`` sweeps token count
at a fixed regime, ``chunked_ce.py`` sweeps our kernel's tile geometry.

Usage:
    .venv/bin/python scripts/bench/kernel/head_options.py --out out/bench/kernel/head_options
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
    device_name,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import bench_ms
from kohakuwullm.kernels.loss.chunked_ce import chunked_linear_cross_entropy

IGNORE = -100
MMA_PEAK = 400.0
MAMF = 270.0
# fp32 inputs never reach the tensor cores; 21760 lanes x 2 flop x ~2.41 GHz.
VECTOR_PEAK = 105.0

REGIMES = [
    ("bf16 param", torch.bfloat16, False),
    ("fp32 param + autocast", torch.float32, True),
    ("fp32 param, no autocast", torch.float32, False),
]


def _time_step(step, args):
    try:
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ms = bench_ms(step, warmup=args.warmup, iters=args.iters)
        return ms, torch.cuda.max_memory_allocated() / 2**30, True
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(
            exc
        ):
            raise
        torch.cuda.empty_cache()
        return float("nan"), float("nan"), False


def measure(tokens, dim, vocab, dtype, autocast, fn, args, device):
    """One fwd+bwd of ``fn(x, w, target)``, timed and memory-profiled."""
    x = (torch.randn(tokens, dim, device=device, dtype=dtype) * 0.5).requires_grad_(
        True
    )
    w = (torch.randn(vocab, dim, device=device, dtype=dtype) * 0.02).requires_grad_(
        True
    )
    target = torch.randint(0, vocab, (tokens,), device=device)
    target[::17] = IGNORE

    def step(x=x, w=w, target=target):
        x.grad = None
        w.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            loss = fn(x, w, target)
        loss.float().sum().backward()

    ms, peak, ok = _time_step(step, args)
    del x, w
    torch.cuda.empty_cache()
    flop = 6 * tokens * dim * vocab / 1e12
    width = torch.finfo(dtype).bits // 8
    # x and w are each read and written back as a gradient; the logit tile is
    # implementation-dependent and deliberately not counted, so this is a floor.
    gbytes = 2 * (tokens * dim + vocab * dim) * width / 1e9
    return dict(
        ms=ms,
        peak_gib=peak,
        tflops=(flop / (ms / 1e3)) if ok else 0.0,
        gbps=(gbytes / (ms / 1e3)) if ok else 0.0,
        ceiling=MAMF if (autocast or dtype is torch.bfloat16) else VECTOR_PEAK,
        ok=ok,
    )


def torch_fn(options):
    def call(x, w, target):
        return F.linear_cross_entropy(
            x, w, target, ignore_index=IGNORE, reduction="none", options=options
        )

    return call


def ours_fn(chunk, vocab_block, retain, compute_dtype=torch.bfloat16):
    def call(x, w, target):
        return chunked_linear_cross_entropy(
            x,
            w,
            target,
            ignore_index=IGNORE,
            chunk=chunk,
            vocab_block=vocab_block,
            retain=retain,
            compute_dtype=compute_dtype,
        )

    return call


def candidates(args):
    """Every head implementation worth putting on the Pareto plot."""
    out = [
        ("torch chunked", torch_fn(LinearCrossEntropyOptions())),
        ("torch materialize", torch_fn(None)),
    ]
    for vb in args.vocab_blocks:
        out.append((f"ours vb{vb}", ours_fn(args.chunk, vb, 0.0)))
    for retain in args.retains:
        out.append(
            (f"ours retain{retain:g}", ours_fn(args.chunk, args.vocab_block, retain))
        )
    return out


def run_pareto(args, device):
    rows = []
    for name, fn in candidates(args):
        row = measure(
            args.tokens, args.dim, args.vocab, torch.float32, True, fn, args, device
        )
        row["name"] = name
        rows.append(row)
        status = (
            f"{row['ms']:8.2f} ms  {row['tflops']:7.1f} TF/s  "
            f"{row['gbps']:7.0f} GB/s  {row['peak_gib']:6.3f} GiB"
            if row["ok"]
            else "OOM"
        )
        print(f"  {name:>22}: {status}", flush=True)
    return rows


def run_regimes(args, device):
    rows = []
    for name, dtype, autocast in REGIMES:
        for label, fn in (
            ("torch chunked", torch_fn(LinearCrossEntropyOptions())),
            ("torch materialize", torch_fn(None)),
            ("ours", ours_fn(args.chunk, args.vocab_block, args.retains[-1])),
            (
                "ours (fp32 compute)",
                ours_fn(args.chunk, args.vocab_block, args.retains[-1], torch.float32),
            ),
        ):
            row = measure(
                args.tokens, args.dim, args.vocab, dtype, autocast, fn, args, device
            )
            row.update(regime=name, impl=label)
            rows.append(row)
            print(
                f"  {name:24} {label:18}: {row['ms']:8.2f} ms  "
                f"{row['tflops']:7.1f} TF/s ({100 * row['tflops'] / row['ceiling']:5.1f}% of "
                f"{row['ceiling']:.0f}T)  {row['gbps']:7.0f} GB/s",
                flush=True,
            )
    return rows


def run_scaling(args, device):
    rows = []
    impls = (
        ("torch materialize", torch_fn(None)),
        ("ours", ours_fn(args.chunk, args.vocab_block, args.retains[-1])),
    )
    for tokens in args.token_sweep:
        for label, fn in impls:
            row = measure(
                tokens, args.dim, args.vocab, torch.float32, True, fn, args, device
            )
            row.update(tokens=tokens, impl=label)
            rows.append(row)
            print(
                f"  T={tokens:6d} {label:18}: {row['ms']:8.2f} ms  "
                f"{row['peak_gib']:6.3f} GiB",
                flush=True,
            )
    return rows


def frontier(points):
    """Points not dominated on both axes, ordered by memory."""
    ordered = sorted(points, key=lambda r: (r["peak_gib"], r["ms"]))
    best, out = float("inf"), []
    for row in ordered:
        if row["ms"] < best:
            out.append(row)
            best = row["ms"]
    return out


def plot(pareto, regimes, scaling, out_dir, args):
    pal = Palette()
    fig, axes = new_figure(1, 3, figsize=(19, 5.6))

    good = [r for r in pareto if r["ok"]]
    ours = [r for r in good if r["name"].startswith("ours")]
    theirs = [r for r in good if not r["name"].startswith("ours")]
    axes[0].scatter(
        [r["peak_gib"] for r in theirs],
        [r["ms"] for r in theirs],
        s=90,
        marker="s",
        color=pal.color("fwd"),
        label="torch",
        zorder=3,
    )
    axes[0].scatter(
        [r["peak_gib"] for r in ours],
        [r["ms"] for r in ours],
        s=70,
        color=pal.color("bwd"),
        label="chunked_ce",
        zorder=3,
    )
    front = frontier(good)
    axes[0].plot(
        [r["peak_gib"] for r in front],
        [r["ms"] for r in front],
        color="#c0392b",
        linewidth=1.2,
        linestyle="--",
        label="Pareto frontier",
        zorder=2,
    )
    for row in good:
        axes[0].annotate(
            row["name"].replace("ours ", ""),
            (row["peak_gib"], row["ms"]),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
        )
    finish_axis(
        axes[0],
        "peak memory (GiB)",
        "fwd+bwd (ms)",
        f"Time vs memory Pareto (T={args.tokens}, fp32 param + autocast)",
        si_x=False,
    )
    axes[0].legend(frameon=False, fontsize=8)

    impls = ["torch chunked", "torch materialize", "ours"]
    names = [r[0] for r in REGIMES]
    xs = range(len(names))
    width = 0.26
    for i, impl in enumerate(impls):
        vals = [
            next(
                (
                    r["tflops"]
                    for r in regimes
                    if r["regime"] == n and r["impl"] == impl
                ),
                0.0,
            )
            for n in names
        ]
        bars = axes[1].bar(
            [x + (i - 1) * width for x in xs],
            vals,
            width,
            color=pal.color(impl),
            label=impl,
        )
        bar_labels(axes[1], bars, "{:.0f}")
    axes[1].axhline(
        MMA_PEAK, color="#c0392b", linestyle=":", label=f"mma {MMA_PEAK:.0f}T"
    )
    axes[1].axhline(MAMF, color="#999999", linestyle="--", label=f"MAMF {MAMF:.0f}T")
    axes[1].axhline(
        VECTOR_PEAK, color="#2980b9", linestyle="-.", label=f"vector {VECTOR_PEAK:.0f}T"
    )
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_ylabel("achieved TFLOP/s")
    axes[1].set_title("Tensor cores vs vector cores")
    axes[1].legend(frameon=False, fontsize=7)

    for label, color in (
        ("torch materialize", pal.color("fwd")),
        ("ours", pal.color("bwd")),
    ):
        pts = [r for r in scaling if r["impl"] == label and r["ok"]]
        axes[2].plot(
            [r["tokens"] for r in pts],
            [r["ms"] for r in pts],
            marker="o",
            color=color,
            label=label,
        )
    finish_axis(
        axes[2],
        "tokens per microbatch",
        "fwd+bwd (ms)",
        "Scaling under the training regime",
        si_x=True,
    )
    axes[2].legend(frameon=False)

    fig.suptitle(
        f"LM head -- {device_name()}  |  D={args.dim}, V={args.vocab}",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "head_options.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/head_options")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=1792)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--vocab-block", type=int, default=4096)
    ap.add_argument(
        "--vocab-blocks", type=int, nargs="+", default=[2048, 4096, 16384, 65536]
    )
    ap.add_argument("--retains", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument(
        "--token-sweep", type=int, nargs="+", default=[2048, 4096, 8192, 16384]
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    print(f"device: {device_name()}  |  D={args.dim} V={args.vocab} T={args.tokens}")
    print("pareto candidates:")
    pareto = run_pareto(args, device)
    print("precision regimes:")
    regimes = run_regimes(args, device)
    print("token scaling:")
    scaling = run_scaling(args, device)

    with open(os.path.join(args.out, "head_options.json"), "w") as handle:
        json.dump(
            dict(config=vars(args), pareto=pareto, regimes=regimes, scaling=scaling),
            handle,
            indent=2,
        )
    plot(pareto, regimes, scaling, args.out, args)

    front = frontier([r for r in pareto if r["ok"]])
    print("\nPareto frontier (memory -> time):")
    for row in front:
        print(f"  {row['name']:>22}: {row['ms']:8.2f} ms  {row['peak_gib']:6.3f} GiB")
    print(f"\nwrote {args.out}/head_options.png")


if __name__ == "__main__":
    main()
