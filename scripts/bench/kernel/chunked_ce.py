"""Where the LM head's loss sits on the time/memory curve, and how exactly.

The two obvious ways to compute ``cross_entropy(x @ w.T, target)`` are the two
ends of a curve: materialize the ``(T, V)`` logits and run at cuBLAS speed with
``T * V`` of memory, or fuse everything into one Triton kernel and give up
cuBLAS. ``kernels/chunked_ce.py`` parameterizes the middle -- cuBLAS GEMMs into
a reused ``chunk x vocab_block`` tile, consumed in place by a Triton epilogue --
so this script's job is to plot the curve rather than argue about the endpoints.
The axis is **our kernel's tile geometry**; ``head.py`` sweeps token count and
``head_options.py`` the precision regime, over the same subsystem.

Four panels:

1. **The Pareto front** -- fwd+bwd time against peak memory for every tile
   geometry, with the two baselines and the four-GEMM cuBLAS roofline marked.
   This is the figure; everything else supports it.
2. **Tile shape at fixed area** -- the same scratch budget spent on a tall tile
   versus a wide one. They are not equivalent and the difference is not small.
3. **Retention** -- what buying back the recompute GEMM with cached logit tiles
   costs in memory, swept continuously from 0 to 1.
4. **Accuracy** -- ULP against an fp64 reference for loss, dX and dW, in both
   fp16 and bf16, next to the same numbers for the two torch paths. A fast
   wrong kernel is not a result.

Usage:
    .venv/bin/python scripts/bench/kernel/chunked_ce.py --out out/bench/kernel/chunked_ce
"""

import argparse
import itertools
import json
import os

import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    bench_ms,
    device_name,
    finish_axis,
    new_figure,
    save_figure,
    ulp_error,
)
from kohakuwullm.kernels.loss.chunked_ce import chunked_linear_cross_entropy

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def marginal_peak_gib(fn) -> float:
    """Peak allocation *attributable to* ``fn``, in GiB.

    Not ``max_memory_allocated`` on its own: the timing helper keeps a 256 MiB
    L2-flush buffer alive, and the inputs are resident before the call. Both
    would be charged to whichever kernel happened to run first.
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return (peak - resident) / 2**30


def fp64_reference(x, w, target, dloss, ignore_index, rows=256):
    """Loss, dX and dW in fp64, accumulated over row slices.

    Slicing is not an optimization: an fp64 ``(8192, 65536)`` logit tensor is
    4.3 GiB and its softmax needs a second one, which does not fit next to the
    rest of the benchmark on a 32 GiB card.
    """
    T, D = x.shape
    V = w.shape[0]
    xd, wd = x.double(), w.double()
    loss = torch.empty(T, device=x.device, dtype=torch.float64)
    dx = torch.empty(T, D, device=x.device, dtype=torch.float64)
    dw = torch.zeros(V, D, device=x.device, dtype=torch.float64)
    for t0 in range(0, T, rows):
        n = min(rows, T - t0)
        tgt = target[t0 : t0 + n]
        logits = xd[t0 : t0 + n] @ wd.t()
        lse = torch.logsumexp(logits, dim=-1)
        picked = logits.gather(1, tgt.clamp_min(0).unsqueeze(1)).squeeze(1)
        keep = (tgt != ignore_index).double()
        loss[t0 : t0 + n] = (lse - picked) * keep
        dlogit = torch.softmax(logits, dim=-1)
        dlogit.scatter_add_(
            1, tgt.clamp_min(0).unsqueeze(1), -torch.ones_like(picked).unsqueeze(1)
        )
        dlogit *= (dloss[t0 : t0 + n].double() * keep).unsqueeze(1)
        dx[t0 : t0 + n] = dlogit @ wd
        dw += dlogit.t() @ xd[t0 : t0 + n]
        del logits, dlogit
    return loss, dx, dw


def gemm_roofline(args, dtype, device):
    """Time the constituent cuBLAS GEMMs on their own.

    A roofline derived from a headline TFLOP/s number is wrong here: the three
    GEMM shapes do not run at the same rate (263 / 245 / 237 TFLOP/s measured),
    so the only defensible floor is the sum of the three, timed.
    """
    T, D, V = args.tokens, args.dim, args.vocab
    x = torch.randn(T, D, device=device, dtype=dtype)
    w = torch.randn(V, D, device=device, dtype=dtype)
    dlogits = torch.randn(T, V, device=device, dtype=dtype)
    logits = torch.empty(T, V, device=device, dtype=dtype)
    dx = torch.empty(T, D, device=device, dtype=torch.float32)
    dw = torch.empty(V, D, device=device, dtype=torch.float32)
    ms = {
        "logits": bench_ms(lambda: torch.mm(x, w.t(), out=logits), warmup=3, iters=15),
        "dX": bench_ms(
            lambda: torch.mm(dlogits, w, out=dx, out_dtype=torch.float32),
            warmup=3,
            iters=15,
        ),
        "dW": bench_ms(
            lambda: torch.mm(dlogits.t(), x, out=dw, out_dtype=torch.float32),
            warmup=3,
            iters=15,
        ),
    }
    x = w = dlogits = logits = dx = dw = None
    torch.cuda.empty_cache()
    ms["retain0"] = 2 * ms["logits"] + ms["dX"] + ms["dW"]
    ms["retain1"] = ms["logits"] + ms["dX"] + ms["dW"]
    print(
        f"  roofline: logits {ms['logits']:.2f} + dX {ms['dX']:.2f} + dW {ms['dW']:.2f}"
        f"  ->  retain=0 {ms['retain0']:.2f} ms, retain=1 {ms['retain1']:.2f} ms"
    )
    return ms


def make_inputs(args, dtype, device):
    x = (
        torch.randn(args.tokens, args.dim, device=device, dtype=dtype) * 0.5
    ).requires_grad_()
    w = (
        torch.randn(args.vocab, args.dim, device=device, dtype=dtype) * 0.02
    ).requires_grad_()
    target = torch.randint(0, args.vocab, (args.tokens,), device=device)
    target[:: args.ignore_stride] = -100
    return x, w, target


def variants(x, w, target, args):
    """Every implementation under test, as ``name -> callable returning (T,) loss``."""
    opts = LinearCrossEntropyOptions(allow_retain_graph=True)
    out = {
        "torch-materialize": lambda: F.linear_cross_entropy(
            x, w, target, reduction="none", options=None
        ).float(),
        "torch-chunked": lambda: F.linear_cross_entropy(
            x, w, target, reduction="none", options=opts
        ).float(),
    }
    for chunk, block in itertools.product(args.chunks, args.vocab_blocks):
        out[f"chunk{chunk}/vb{block}"] = (
            lambda c=chunk, b=block: chunked_linear_cross_entropy(
                x, w, target, chunk=c, vocab_block=b
            )
        )
    for retain in args.retains:
        out[f"retain{retain}"] = lambda r=retain: chunked_linear_cross_entropy(
            x, w, target, chunk=args.best_chunk, vocab_block=args.best_block, retain=r
        )
    return out


def time_variants(x, w, target, args, dtype_name):
    rows = []
    for name, fn in variants(x, w, target, args).items():

        def fwd():
            with torch.no_grad():
                fn()

        def fwdbwd():
            fn().sum().backward()
            x.grad = None
            w.grad = None

        try:
            fwdbwd()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM {name}")
            continue
        row = dict(
            impl=name,
            dtype=dtype_name,
            fwd_ms=bench_ms(fwd, warmup=3, iters=args.iters),
            fwd_gib=marginal_peak_gib(fwd),
            fwdbwd_ms=bench_ms(fwdbwd, warmup=3, iters=args.iters),
            fwdbwd_gib=marginal_peak_gib(fwdbwd),
        )
        rows.append(row)
        print(
            f"  {dtype_name:9s} {name:22s} fwd {row['fwd_ms']:6.2f} ms / "
            f"{row['fwd_gib']:5.3f} GiB   fwd+bwd {row['fwdbwd_ms']:6.2f} ms / "
            f"{row['fwdbwd_gib']:5.3f} GiB",
            flush=True,
        )
    return rows


def accuracy_rows(args, dtype_name, dtype, device):
    x, w, target = make_inputs(args, dtype, device)
    dloss = torch.randn(args.tokens, device=device, dtype=torch.float32)
    ref_loss, ref_dx, ref_dw = fp64_reference(
        x.detach(), w.detach(), target, dloss, -100
    )
    rows = []
    opts = LinearCrossEntropyOptions(allow_retain_graph=True)
    impls = {
        "torch-materialize": lambda: F.linear_cross_entropy(
            x, w, target, reduction="none", options=None
        ).float(),
        "torch-chunked": lambda: F.linear_cross_entropy(
            x, w, target, reduction="none", options=opts
        ).float(),
        # retain is deliberately absent: caching a logit tile instead of
        # recomputing it is bit-identical, pinned by
        # tests/test_kernels.py::test_chunked_ce_retention_is_bit_identical.
        "chunked_ce": lambda: chunked_linear_cross_entropy(
            x, w, target, chunk=args.best_chunk, vocab_block=args.best_block
        ),
    }
    for name, fn in impls.items():
        x.grad = w.grad = None
        loss = fn()
        loss.backward(dloss)
        row = dict(
            impl=name,
            dtype=dtype_name,
            loss_ulp=ulp_error(loss, ref_loss, dtype, "rms"),
            dx_ulp=ulp_error(x.grad, ref_dx, dtype, "rms"),
            dw_ulp=ulp_error(w.grad, ref_dw, dtype, "rms"),
        )
        rows.append(row)
        print(
            f"  {dtype_name:9s} {name:22s} ulp  loss {row['loss_ulp']:7.3f}  "
            f"dx {row['dx_ulp']:7.3f}  dw {row['dw_ulp']:7.3f}",
            flush=True,
        )
    x = w = ref_dx = ref_dw = None
    torch.cuda.empty_cache()
    return rows


def plot(timing, accuracy, roofline, args, out_dir):
    pal = Palette()
    fig, axes = new_figure(2, 2, figsize=(14, 10))
    sel = [r for r in timing if r["dtype"] == args.plot_dtype]
    tile = [r for r in sel if r["impl"].startswith("chunk")]
    retain = sorted(
        (r for r in sel if r["impl"].startswith("retain")),
        key=lambda r: r["fwdbwd_gib"],
    )
    roof = roofline[args.plot_dtype]

    axes[0].scatter(
        [r["fwdbwd_gib"] for r in tile],
        [r["fwdbwd_ms"] for r in tile],
        s=55,
        color=pal.color("chunked_ce"),
        alpha=0.55,
        zorder=3,
        label="chunked_ce, tile geometries (retain=0)",
    )
    front, best_so_far = [], float("inf")
    for row in sorted(tile, key=lambda r: r["fwdbwd_gib"]):
        if row["fwdbwd_ms"] < best_so_far:
            best_so_far = row["fwdbwd_ms"]
            front.append(row)
    axes[0].plot(
        [r["fwdbwd_gib"] for r in front],
        [r["fwdbwd_ms"] for r in front],
        drawstyle="steps-post",
        color=pal.color("chunked_ce"),
        label="Pareto front",
    )
    axes[0].plot(
        [r["fwdbwd_gib"] for r in retain],
        [r["fwdbwd_ms"] for r in retain],
        "-o",
        color=pal.color("retain"),
        label=f"retain 0->1 at chunk{args.best_chunk}/vb{args.best_block}",
    )
    for name in ("torch-materialize", "torch-chunked"):
        row = next((r for r in sel if r["impl"] == name), None)
        if row:
            axes[0].scatter(
                row["fwdbwd_gib"],
                row["fwdbwd_ms"],
                s=130,
                marker="D",
                color=pal.color(name),
                label=name,
                zorder=4,
            )
    for level, tag in (
        (roof["retain0"], "4 measured cuBLAS GEMMs (recompute)"),
        (roof["retain1"], "3 measured cuBLAS GEMMs (retain)"),
    ):
        axes[0].axhline(level, color="#7F7F7F", linestyle=":", linewidth=1.2)
        axes[0].annotate(
            f"{tag}: {level:.1f} ms",
            (0.02, level),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, 3),
            fontsize=8,
            color="#7F7F7F",
        )
    axes[0].set_xscale("log", base=2)
    top = 1.35 * max(r["fwdbwd_ms"] for r in retain)
    axes[0].set_ylim(roof["retain1"] * 0.9, top)
    clipped = sum(1 for r in tile if r["fwdbwd_ms"] > top)
    if clipped:
        axes[0].annotate(
            f"{clipped} starved tiles off-scale above "
            f"(up to {max(r['fwdbwd_ms'] for r in tile):.0f} ms)",
            (0.98, 0.03),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#7F7F7F",
        )
    finish_axis(
        axes[0],
        "peak memory attributable to the op (GiB)",
        "fwd+bwd (ms)",
        f"The curve ({args.plot_dtype}, T={args.tokens} D={args.dim} V={args.vocab})",
        si_x=False,
        legend=True,
    )

    areas: dict[int, list] = {}
    for row in tile:
        chunk = int(row["impl"].split("/")[0][5:])
        block = int(row["impl"].split("/")[1][2:])
        areas.setdefault(chunk * block, []).append((chunk, row))
    for area, points in sorted(areas.items()):
        if len(points) < 2:
            continue
        points.sort()
        axes[1].plot(
            [c for c, _ in points],
            [r["fwdbwd_ms"] for _, r in points],
            "-o",
            color=pal.color(f"area{area}"),
            label=f"tile = {area * 2 / 2**20:.0f} MiB",
        )
    finish_axis(
        axes[1],
        "chunk (rows per tile)",
        "fwd+bwd (ms)",
        "Same scratch budget, different tile shape",
        logx=True,
        legend=True,
    )

    for row in sorted(retain, key=lambda r: float(r["impl"][6:])):
        axes[2].scatter(
            float(row["impl"][6:]),
            row["fwdbwd_ms"],
            s=70,
            color=pal.color("retain"),
            zorder=3,
        )
        axes[2].annotate(
            f"{row['fwdbwd_gib']:.2f} GiB",
            (float(row["impl"][6:]), row["fwdbwd_ms"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )
    axes[2].plot(
        [
            float(r["impl"][6:])
            for r in sorted(retain, key=lambda r: float(r["impl"][6:]))
        ],
        [r["fwdbwd_ms"] for r in sorted(retain, key=lambda r: float(r["impl"][6:]))],
        "-",
        color=pal.color("retain"),
    )
    finish_axis(
        axes[2],
        "fraction of logit tiles cached from the forward",
        "fwd+bwd (ms)",
        "Buying back the recompute GEMM",
        si_x=False,
    )

    impls = list(dict.fromkeys(r["impl"] for r in accuracy))
    groups = [
        (dtype, field, mark)
        for dtype in DTYPES
        for field, mark in (("loss_ulp", "loss"), ("dx_ulp", "dX"), ("dw_ulp", "dW"))
    ]
    width = 0.8 / len(impls)
    for i, impl in enumerate(impls):
        heights = [
            next(
                r[field] for r in accuracy if r["impl"] == impl and r["dtype"] == dtype
            )
            for dtype, field, _ in groups
        ]
        bars = axes[3].bar(
            [g + i * width for g in range(len(groups))],
            heights,
            width=width,
            color=pal.color(impl),
            label=impl,
        )
        bar_labels(axes[3], bars, "{:.2g}")
    axes[3].set_yscale("log")
    axes[3].set_xticks([g + 0.4 - width / 2 for g in range(len(groups))])
    axes[3].set_xticklabels([f"{d[:2]}{d[-2:]}\n{m}" for d, _, m in groups], fontsize=8)
    axes[3].legend(fontsize=8)
    axes[3].set_ylabel("ULP vs fp64 reference (rms-scaled)")
    axes[3].set_title("Accuracy: lower is better, at the same shape as the timings")

    fig.suptitle(
        f"Linear + cross-entropy on the time/memory curve -- {device_name()}\n"
        f"cuBLAS GEMMs into a reused chunk x vocab_block tile, Triton epilogue in place",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "chunked_ce.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/chunked_ce")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=1792)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument(
        "--chunks", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192]
    )
    ap.add_argument(
        "--vocab-blocks", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 65536]
    )
    ap.add_argument(
        "--retains", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument("--best-chunk", type=int, default=8192)
    ap.add_argument("--best-block", type=int, default=4096)
    ap.add_argument("--ignore-stride", type=int, default=7)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--plot-dtype", default="bfloat16")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Seeded: the ULP panel compares implementations against each other, and an
    # unseeded redraw moves every bar by ~20% for reasons that are not the code.
    torch.manual_seed(0)
    device = torch.device("cuda")
    print(f"device: {device_name()}")

    timing, accuracy, roofline = [], [], {}
    for name, dtype in DTYPES.items():
        roofline[name] = gemm_roofline(args, dtype, device)
        x, w, target = make_inputs(args, dtype, device)
        timing += time_variants(x, w, target, args, name)
        del x, w
        torch.cuda.empty_cache()
        accuracy += accuracy_rows(args, name, dtype, device)

    with open(os.path.join(args.out, "chunked_ce.json"), "w") as handle:
        json.dump(
            {"timing": timing, "accuracy": accuracy, "roofline": roofline},
            handle,
            indent=2,
        )
    plot(timing, accuracy, roofline, args, args.out)
    print(f"wrote {args.out}/chunked_ce.png")


if __name__ == "__main__":
    main()
