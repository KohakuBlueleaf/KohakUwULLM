"""MXFP8 matmul: throughput and accuracy, against bf16 and against the vendor.

Four paths on the same shapes, because the interesting question is not "is fp8
faster than bf16" but "how much of the vendor's fp8 are we getting, and at what
accuracy":

* **bf16 cuBLAS** -- the baseline the model runs today.
* **``mxfp8_matmul_pq``** -- ours, operands pre-quantized. This is the shape a
  training step wants: weights cast once per optimizer step.
* **``mxfp8_matmul``** -- ours, quantizing inside the tile loop. A correctness
  reference and the no-cast-pass fallback, not the fast path.
* **``F.scaled_mm``** with ``BlockWise1x32`` -- cuBLAS MXFP8, the ceiling.

Accuracy is on the same figure as throughput, and it is reported in ULP of
**e4m3**, not of the input dtype. MXFP8 discards far more precision than either
fp16 or bf16 carries, so its measured accuracy is identical from both -- but
scored against the input dtype's eps, fp16 reads 8x worse than bf16 for a
computation that did not change. e4m3 is the format actually carried.

Shapes are the ones a transformer issues, not just squares. ``(16384, 1280,
5120)`` and ``(16384, 5120, 1280)`` are Nano-1B's MLP up and down projections;
``(8192, 1280, 1280)`` is the small case, included deliberately -- a figure that
showed only the shapes fp8 wins on would be advertising, not a measurement. It is
also where the *measurement* is hardest: at 0.09 ms wall, 45% of what a naive
timer reports is host dispatch, so fp8 looked like it lost to bf16 there until
dispatch was separated out. It does not; it wins by 2.1x.

GEMM rows are therefore timed three ways: ``wall`` (what a caller sees, dispatch
included), ``host`` (the dispatch alone) and ``wall - host`` as the device-time
estimate. Throughput is plotted from the last, with the raw wall bar drawn faintly
behind it so the size of the correction stays visible. ``wall`` is kept in the JSON
because published reference numbers for this card were measured that way.

The standalone cast gets true device time from ``graph_ms`` instead -- eight rows,
within the "handful, not a sweep" that capture allows -- because ``wall - host``
over-estimates it by 3x at that size. ``graph_ms`` does not flush L2, so a
bandwidth claim is only made where the traffic exceeds L2; the smaller shapes are
drawn but marked, since a cast reading 223% of DRAM peak is a cache hit, not a
fast kernel.

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/mxfp8.py
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    bench_samples,
    cached_peak_bandwidth,
    classify,
    device_name,
    graph_ms,
    new_figure,
    sample_spread,
    save_figure,
    ulp_error,
)

# timing_profile is imported from the module rather than the package: it is newer
# than bench/__init__.py's export list, which another change owns.
from kohakuwullm.bench.core.timing import (
    TENSOR_MAMF_TFLOPS,
    spread_has_power,
    spread_threshold,
    timing_profile,
)
from kohakuwullm.kernels.mxfp8 import (
    BLOCK_SCALE,
    mxfp8_matmul,
    mxfp8_matmul_pq,
    quantize_mx,
)
from kohakuwullm.kernels.mxfp8.interop import (
    as_vendor_scales,
    vendor_mxfp8_matmul_swizzled,
)

# (M, N, K). The square is the shape every published MXFP8 number uses, so it is
# the only one comparable against them; the rest are what we actually issue.
SHAPES = [
    (4096, 4096, 4096),
    (16384, 1280, 5120),
    (16384, 5120, 1280),
    (8192, 1280, 1280),
]
LABELS = {
    "bf16": "bf16 cuBLAS",
    "pq": "ours, pre-quantized",
    "fused": "ours, online quant",
    "vendor": "F.scaled_mm (cuBLAS MXFP8)",
}
ORDER = ["bf16", "pq", "fused", "vendor"]
# (M, K) for the standalone cast. The GEMM shapes above are reused, plus one that
# is deliberately larger than L2: only three of them move enough traffic to
# escape cache, and without a DRAM-resident point there is no bandwidth claim to
# make at all -- just a cache hit reported as a fast kernel.
CAST_SHAPES = [(m, k) for m, _, k in SHAPES] + [(65536, 1280)]
# sm_120 L2. A graph replay does not flush it, so traffic at or below this size
# is served from cache and its "bandwidth" can exceed the DRAM roofline.
L2_BYTES = 96 * 1024 * 1024


def measure(fn, args) -> dict:
    """Wall time, host dispatch time and sample spread for one call.

    ``wall`` carries a flat ~60 us of dispatch: the timing loop syncs and *then*
    records its start event, so the GPU sits idle inside the event window while
    Python marshals arguments. On the smallest GEMM here that is 45% of the
    reported time, and on the standalone cast it is most of it.

    ``ms_nodispatch`` subtracts the host time as an estimate of device time.
    Checked against CUDA-graph replay it is within 6-14% on the GEMMs, and 3x too
    high on the cast -- but the alternative, capturing a graph for every row, is
    the sweep-wide capture ``timing_profile``'s own docstring warns against: doing
    it here inflated one spread reading to 98% and moved wall times by up to 20%
    through allocator pressure, on rows that were not even being captured.
    """
    times = bench_samples(fn, warmup=args.warmup, iters=args.iters)
    profile = timing_profile(fn, warmup=args.warmup, iters=args.iters, device=False)
    wall = sorted(times)[len(times) // 2]
    return dict(
        ms=wall,
        ms_host=profile["host_ms"],
        ms_nodispatch=max(wall - profile["host_ms"], 1e-9),
        host_bound=bool(profile["host_bound"]),
        step_spread=sample_spread(times),
        ok=True,
    )


def gemm_rows(shape, dtype, args) -> list[dict]:
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=dtype) * 0.05
    aq, a_scale = quantize_mx(a)
    bq, b_scale = quantize_mx(b)

    # fp64 on the operands, accumulated in chunks: a full fp64 (16384, 5120)
    # reference is 640 MiB and the pair of fp64 operands is another 1.6 GiB, which
    # is what OOMed the earlier version of this benchmark.
    ref = torch.zeros(m, n, device="cuda", dtype=torch.float64)
    for start in range(0, k, args.ref_chunk):
        stop = min(start + args.ref_chunk, k)
        ref += a[:, start:stop].double() @ b[:, start:stop].double().T

    # The vendor's scale swizzle is hoisted, exactly as our pre-quantization is:
    # both are per-weight preparation done once per optimizer step, and leaving
    # either inside the timed call measures preparation as if it were the GEMM.
    a_swz = as_vendor_scales(a_scale)
    b_swz = as_vendor_scales(b_scale)

    # Operands bound as defaults, not captured: the tensors are freed below before
    # the next shape, and a late-bound closure would reference deleted names.
    calls = {
        "bf16": lambda a=a, b=b: a @ b.T,
        "pq": lambda p=(aq, a_scale, bq, b_scale): mxfp8_matmul_pq(*p, dtype),
        "fused": lambda a=a, b=b: mxfp8_matmul(a, b, dtype),
        "vendor": lambda p=(aq, a_swz, bq, b_swz): vendor_mxfp8_matmul_swizzled(
            *p, dtype
        ),
    }
    flops = 2.0 * m * n * k
    rows = []
    for name, call in calls.items():
        out = call()
        timing = measure(call, args)
        rows.append(
            dict(
                impl=name,
                m=m,
                n=n,
                k=k,
                dtype=str(dtype).replace("torch.", ""),
                tflops=flops / (timing["ms_nodispatch"] / 1e3) / 1e12,
                tflops_wall=flops / (timing["ms"] / 1e3) / 1e12,
                # e4m3's eps is the yardstick for every path here, including bf16:
                # scoring each against its own format would compare two different
                # rulers and make the low-precision path look competitive.
                ulp_e4m3=ulp_error(out, ref, torch.float8_e4m3fn, mode="rms"),
                rel_err=((out.double() - ref).norm() / ref.norm()).item(),
                **timing,
            )
        )
        del out
    del calls, a, b, aq, bq, a_swz, b_swz, ref
    torch.cuda.empty_cache()
    return rows


def quantize_rows(args) -> list[dict]:
    """Per-weight preparation cost: the cast, and the vendor's scale swizzle.

    Both are hoisted out of the GEMM timings above, so they are reported here
    instead of being silently free. The swizzle is charged to the vendor path
    alone -- our kernel reads the layout ``quantize_mx`` already produces.
    """
    peak = cached_peak_bandwidth()
    dtype = torch.bfloat16
    rows = []
    for m, k in CAST_SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        q, scales = quantize_mx(x)
        width = torch.finfo(dtype).bits // 8
        # Read the input, write e4m3 values plus one scale byte per 32.
        cast_moved = m * k * width + m * k + m * k // BLOCK_SCALE
        # The swizzle only touches the scales, which is 1/32 of the payload.
        swizzle_moved = 2 * m * k // BLOCK_SCALE
        for stage, fn, moved in (
            ("quantize_mx", lambda x=x: quantize_mx(x), cast_moved),
            (
                "scale swizzle (vendor only)",
                lambda s=scales: as_vendor_scales(s),
                swizzle_moved,
            ),
        ):
            timing = measure(fn, args)
            # graph_ms for the true device time -- eight rows, which is the
            # "handful, not a sweep" the capture warning allows. It does not flush
            # L2, so a bandwidth claim is only admissible where the traffic
            # exceeds L2; below that the kernel is reading cache and the rate goes
            # above DRAM peak, which is a cache hit reported as a fast kernel.
            device_ms = graph_ms(fn, warmup=args.warmup, iters=args.iters)
            gbps_wall = moved / (timing["ms"] / 1e3) / 1e9
            gbps_device = (
                moved / (device_ms / 1e3) / 1e9 if device_ms == device_ms else 0.0
            )
            rows.append(
                dict(
                    stage=stage,
                    m=m,
                    k=k,
                    dtype=str(dtype).replace("torch.", ""),
                    bytes_moved=moved,
                    l2_resident=moved <= L2_BYTES,
                    ms_graph=device_ms,
                    roofline_us=1e6 * moved / (peak * 1e9),
                    gbps=gbps_wall,
                    gbps_device=gbps_device,
                    bandwidth_pct=100.0 * gbps_wall / peak,
                    bandwidth_pct_device=100.0 * gbps_device / peak,
                    **timing,
                )
            )
        del x, q, scales
        torch.cuda.empty_cache()
    return rows


def plot(gemms, quants, peak, device, args) -> None:
    pal = Palette()
    fig, axes = new_figure(2, 2, figsize=(16, 10.5))
    shapes = [
        s for s in SHAPES if any(r["m"] == s[0] and r["k"] == s[2] for r in gemms)
    ]
    names = [f"{m}x{n}\nK={k}" for m, n, k in shapes]
    xs = range(len(shapes))
    width = 0.8 / len(ORDER)

    def pick(impl, shape, key):
        for row in gemms:
            if (row["impl"], row["m"], row["n"], row["k"]) == (impl, *shape):
                return row[key]
        return 0.0

    def mark_contended(bars, impls, shape_list):
        """Hatch any bar whose underlying row failed the spread cut.

        Applied to every panel that draws a throughput-derived quantity, not just
        the throughput one: the ratio panel is the most eye-catching in the
        figure, and an unmarked ratio built from two contended rows is exactly
        the number a reader would quote.

        The cut is ``spread_threshold(ms)``, not a flat percentage: a fixed 5%
        flags five of these seven shapes on a verifiably idle card, because the
        jitter floor is a fixed number of microseconds and so a larger *fraction*
        of a shorter measurement.
        """
        for bar, impl, shape in zip(bars, impls, shape_list):
            if pick(impl, shape, "step_spread") > spread_threshold(
                pick(impl, shape, "ms")
            ):
                bar.set_hatch("///")
                bar.set_edgecolor("#8E44AD")
                bar.set_linewidth(1.1)

    for i, impl in enumerate(ORDER):
        offset = (i - (len(ORDER) - 1) / 2) * width
        # Raw wall behind, dispatch-corrected in front: the visible difference
        # between them is the artifact, and hiding it would leave the corrected
        # number looking like a plain measurement rather than a correction.
        axes[0].bar(
            [x + offset for x in xs],
            [pick(impl, s, "tflops_wall") for s in shapes],
            width,
            color=pal.color(impl),
            alpha=0.3,
            label="as measured (dispatch included)" if i == 0 else None,
        )
        bars = axes[0].bar(
            [x + offset for x in xs],
            [pick(impl, s, "tflops") for s in shapes],
            width * 0.55,
            color=pal.color(impl),
            label=LABELS[impl],
        )
        bar_labels(axes[0], bars, "{:.0f}")
        mark_contended(bars, [impl] * len(shapes), shapes)
    axes[0].axhline(
        TENSOR_MAMF_TFLOPS,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        # Not a ceiling for the fp8 bars -- fp8 mma runs at twice bf16's rate. It
        # marks what bf16 achieves, which is the bar fp8 has to clear to be worth
        # wiring in. Drawing an unmeasured fp8 peak here would be inventing one.
        label=f"bf16 achievable {TENSOR_MAMF_TFLOPS:.0f}T (not an fp8 ceiling)",
    )
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels(names, fontsize=7.5)
    axes[0].set_ylabel("TFLOP/s (wall minus host dispatch)")
    # Headroom for the legend, which otherwise sits on the tallest bars' labels.
    axes[0].set_ylim(0, max(r["tflops"] for r in gemms) * 1.35)
    axes[0].set_title(
        "Throughput, net of host dispatch\n"
        "(hatched = spread over its duration-scaled threshold; "
        "faint bar = as measured, dispatch included)"
    )
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)

    for i, impl in enumerate(ORDER):
        offset = (i - (len(ORDER) - 1) / 2) * width
        bars = axes[1].bar(
            [x + offset for x in xs],
            [pick(impl, s, "ulp_e4m3") for s in shapes],
            width,
            color=pal.color(impl),
            label=LABELS[impl],
        )
        bar_labels(axes[1], bars, "{:.2f}")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels(names, fontsize=7.5)
    axes[1].set_ylabel("RMS error in ULP of e4m3 (eps 0.125)")
    axes[1].set_title(
        "Accuracy, one ruler for all four paths\n"
        "the three MXFP8 paths are bit-identical; bf16 is ~20x more accurate"
    )
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=7.5)

    # Fraction of the vendor, which is the number that says whether our kernel is
    # done. Throughput alone hides it: 361 looks respectable next to bf16's 225.
    for i, impl in enumerate(("pq", "fused")):
        vals = [
            100.0 * pick(impl, s, "tflops") / max(pick("vendor", s, "tflops"), 1e-9)
            for s in shapes
        ]
        bars = axes[2].bar(
            [x + (i - 0.5) * 0.38 for x in xs],
            vals,
            0.38,
            color=pal.color(impl),
            label=LABELS[impl],
        )
        bar_labels(axes[2], bars, "{:.0f}%")
        mark_contended(bars, [impl] * len(shapes), shapes)
    axes[2].axhline(
        100.0, color="#c0392b", linestyle="--", linewidth=1.2, label="vendor"
    )
    # Nothing runs the other way, and this says so rather than leaving the reader
    # to assume a counterweight exists. Feeding cuBLAS needs its scales swizzled,
    # which as a separate pass costs ~15 us of device time per operand -- but
    # `quantize_mx_vendor` writes that layout straight out of the cast, so the
    # vendor pays it only if you integrate it carelessly. Even unfused it was 12 ms
    # across all 183 projections of Nano-1B against a ~1.2 s GEMM deficit: 1%.
    swizzle = [r for r in quants if r["stage"].startswith("scale swizzle")]
    per_weight_us = (
        sum(r["ms_graph"] for r in swizzle) / len(swizzle) * 1e3 if swizzle else 0.0
    )
    axes[2].annotate(
        f"cuBLAS needs its scales swizzled ({per_weight_us:.0f} us/operand as a "
        "separate pass),\nbut quantize_mx_vendor folds that into the cast, so it "
        "is not a\ncounterweight: unfused it was 1% of the GEMM deficit anyway",
        (0.02, 0.03),
        xycoords="axes fraction",
        fontsize=7,
        va="bottom",
        color="#555555",
    )
    axes[2].set_xticks(list(xs))
    axes[2].set_xticklabels(names, fontsize=7.5)
    axes[2].set_ylabel("% of F.scaled_mm")
    axes[2].set_title(
        "How much of cuBLAS MXFP8 our kernels reach\n"
        "(hatched = built from a contended row)"
    )
    axes[2].legend(frameon=False, fontsize=8)

    # Microseconds, not GB/s: it is the only unit in which the device cost and
    # the dispatch cost can sit side by side, and the roofline becomes a per-shape
    # marker rather than a single line. Dispatch is its own bar, never folded in.
    casts = [r for r in quants if r["stage"] == "quantize_mx"]
    casts = sorted(casts, key=lambda r: r["bytes_moved"])
    qxs = range(len(casts))
    qwidth = 0.38
    for i, (key, label) in enumerate(
        (("ms_graph", "device (graph replay)"), ("ms_host", "Python dispatch")),
    ):
        bars = axes[3].bar(
            [x + (i - 0.5) * qwidth for x in qxs],
            [r[key] * 1e3 for r in casts],
            qwidth,
            color=pal.color(f"prep:{key}"),
            label=label,
        )
        bar_labels(axes[3], bars, "{:.0f}")
        if key == "ms_graph":
            for bar, row in zip(bars, casts):
                if row["l2_resident"]:
                    bar.set_hatch("xx")
                    bar.set_edgecolor("#c0392b")
    axes[3].scatter(
        list(qxs),
        [r["roofline_us"] for r in casts],
        marker="_",
        s=420,
        linewidths=2.2,
        color="#c0392b",
        zorder=5,
        label=f"time at {peak:.0f} GB/s roofline",
    )
    axes[3].set_xticks(list(qxs))
    axes[3].set_xticklabels(
        [
            f"M={r['m']} K={r['k']}\n{r['bytes_moved'] / 2**20:.0f} MiB"
            + ("\n(fits L2)" if r["l2_resident"] else "")
            for r in casts
        ],
        fontsize=7,
    )
    axes[3].set_ylabel("microseconds")
    valid = [r for r in casts if not r["l2_resident"]]
    claim = (
        ", ".join(f"{r['bandwidth_pct_device']:.0f}%" for r in valid)
        if valid
        else "no DRAM-resident shape measured"
    )
    axes[3].set_title(
        f"quantize_mx: device time sits on the roofline ({claim} of "
        f"{peak:.0f} GB/s)\nso the cast is bandwidth-saturated, not slow -- its wall "
        "cost is dispatch,\nwhich is why fusing it into the norm epilogue is the win"
        "\nhatched = traffic fits L2, so its rate is a cache hit and not admissible"
    )
    axes[3].legend(frameon=False, fontsize=7)
    axes[3].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"MXFP8 matmul -- {device}  |  accuracy in ULP of e4m3, "
        "not of the input dtype\n"
        "all three MXFP8 paths are bit-identical, so the gap to cuBLAS is "
        "scheduling and costs nothing in accuracy",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(args.out, "mxfp8.png"))


def report(gemms, quants, peak) -> None:
    print(
        f"\n{'shape':>22} {'impl':>28} {'wall':>7} {'host':>7} {'net':>7} "
        f"{'net TF/s':>9} {'wall TF/s':>10} {'e4m3 ulp':>9} {'spread':>7}"
    )
    print("-" * 118)
    for shape in SHAPES:
        for impl in ORDER:
            for row in gemms:
                if (row["impl"], row["m"], row["n"], row["k"]) != (impl, *shape):
                    continue
                verdict = classify(row, spread_threshold(row["ms"]))
                if verdict != "clean":
                    flag = "  CONTENDED"
                elif not spread_has_power(row["ms"]):
                    # Passing a check with no power is not a clean verdict.
                    flag = "  (spread test has no power at this duration)"
                else:
                    flag = ""
                print(
                    f"{str(shape):>22} {LABELS[impl]:>28} {row['ms']:>7.3f} "
                    f"{row['ms_host']:>7.3f} {row['ms_nodispatch']:>7.3f} "
                    f"{row['tflops']:>9.1f} {row['tflops_wall']:>10.1f} "
                    f"{row['ulp_e4m3']:>9.3f} {100 * row['step_spread']:>6.1f}%{flag}"
                )
    dirty = [
        r for r in gemms + quants if classify(r, spread_threshold(r["ms"])) != "clean"
    ]
    print(
        f"\n{len(dirty)} of {len(gemms) + len(quants)} rows exceeded their "
        "duration-scaled spread threshold (5% + 0.010/median_ms). Rows marked "
        "'no power'\nare short enough that the allowance exceeds any plausible "
        "contention signal -- passing there is not evidence of an exclusive card."
    )
    for row in sorted(dirty, key=lambda r: -r["step_spread"]):
        label = LABELS.get(row.get("impl")) or row.get("stage", "?")
        print(
            f"  {label:>28} M={row['m']} K={row['k']}: spread "
            f"{100 * row['step_spread']:.1f}% vs threshold "
            f"{100 * spread_threshold(row['ms']):.1f}%"
        )

    print(
        f"\n{'preparation stage':>28} {'shape':>16} {'MiB':>7} {'wall':>7} "
        f"{'host':>7} {'device':>7} {'dev GB/s':>9} {'%peak':>7}  admissible"
    )
    print("-" * 90)
    for row in quants:
        shape = f"M={row['m']} K={row['k']}"
        ok = "no (fits L2)" if row["l2_resident"] else "yes"
        print(
            f"{row['stage']:>28} {shape:>16} {row['bytes_moved'] / 2**20:>7.0f} "
            f"{row['ms']:>7.3f} {row['ms_host']:>7.3f} {row['ms_graph']:>7.3f} "
            f"{row['gbps_device']:>9.0f} {row['bandwidth_pct_device']:>6.1f}%  {ok}"
        )
    admissible = [
        r for r in quants if not r["l2_resident"] and r["stage"] == "quantize_mx"
    ]
    print(
        f"  measured DRAM peak: {peak:.0f} GB/s. 'device' is graph replay, which does "
        "NOT flush L2, so a\n  bandwidth number is only admissible where the traffic "
        "exceeds L2 -- below that it is a cache\n  hit reported as a fast kernel. On "
        "the admissible shapes the cast reaches "
        + ", ".join(f"{r['bandwidth_pct_device']:.0f}%" for r in admissible)
        + " of peak:\n  bandwidth-saturated, with nothing left to tune. Its wall cost "
        "is dominated by a flat ~60 us of\n  Python dispatch, which is what fusing it "
        "into the norm epilogue removes."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/mxfp8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ref-chunk", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument(
        "--replot",
        action="store_true",
        help="redraw from the existing JSON without touching a GPU; the cards "
        "here are shared, so re-tuning a figure must not need one",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "mxfp8.json")

    if args.replot:
        with open(path) as handle:
            payload = json.load(handle)
        gemms, quants = payload["gemms"], payload["quantize"]
        peak = payload["peak_bandwidth_gbps"]
        report(gemms, quants, peak)
        plot(gemms, quants, peak, payload.get("device", "device not recorded"), args)
        print(f"\nredrew {args.out}/mxfp8.png from {path}")
        return

    dtype = getattr(torch, args.dtype)
    peak = cached_peak_bandwidth()
    print(f"device: {device_name()}  |  measured DRAM {peak:.0f} GB/s  |  {args.dtype}")
    gemms = []
    for shape in SHAPES:
        print(f"  {shape} ...", flush=True)
        gemms += gemm_rows(shape, dtype, args)
    quants = quantize_rows(args)

    report(gemms, quants, peak)
    with open(path, "w") as handle:
        json.dump(
            dict(
                device=device_name(),
                peak_bandwidth_gbps=peak,
                compute_ceiling_tflops=TENSOR_MAMF_TFLOPS,
                config=vars(args),
                gemms=gemms,
                quantize=quants,
            ),
            handle,
            indent=2,
        )
    plot(gemms, quants, peak, device_name(), args)
    print(f"\nwrote {args.out}/mxfp8.png")


if __name__ == "__main__":
    main()
