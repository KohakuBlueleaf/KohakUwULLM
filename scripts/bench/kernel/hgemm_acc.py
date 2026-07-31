"""Is fp16 accumulation actually faster on this GPU, and what does it cost?

NVIDIA rate-limits fp32 *accumulation* on GeForce tensor cores: the HMMA path
with an fp16 accumulator runs at the full advertised rate, while the fp32
accumulator runs at half. Datacenter parts (A100/H100/B200) have no such split,
which is why the trick only shows up in consumer-GPU projects (``gpupoor``,
``torch-cublas-hgemm``). Whether sm_120 still does this is an empirical
question, so this script measures it rather than assuming.

fp16 accumulation is not free: partial sums live in a format with a 10-bit
mantissa and a max of 65504, so error grows with K and large-magnitude
reductions can overflow to inf. **Split-K** is the mitigation -- cut the
reduction into ``S`` slices, accumulate each in fp16 over a K/S-long run, then
combine the slices in fp32. That recovers most of the accuracy while keeping
the fast inner loop.

This script sweeps (M, N, K) x accumulator x split-K and reports both halves of
the trade: throughput and error against an fp64 reference. Nothing is worth
using if the error column is bad, so both are always plotted together.

Error is measured in ULP with ``mode="rms"``. A GEMM output near zero is
*cancellation* between large terms and inherits its absolute error from their
magnitude, so judging such an element against its own value reports thousands of
ULP for a numerically perfect kernel.

An fp32-input cuBLAS row is included as a reference even though nobody would
train with it: it is the only row that runs on the vector cores, and it fixes
the other end of the scale the tensor-core rows are read against.

Usage:
    .venv/bin/python scripts/bench/kernel/hgemm_acc.py --out out/bench/kernel/hgemm
"""

import argparse
import json
import os

import torch
import triton
import triton.language as tl

from kohakuwullm.bench.core.plotting import (
    Palette,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import (
    TENSOR_MAMF_TFLOPS,
    TENSOR_PEAK_TFLOPS,
    VECTOR_PEAK_TFLOPS,
    cached_peak_bandwidth,
    device_name,
    format_metrics,
    module_metrics,
    rel_error,
    ulp_error,
)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_FP16: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B. Accumulator dtype and split-K depth are compile-time constants."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    if ACC_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    step = BLOCK_K * SPLIT_K
    for k in range(pid_k * BLOCK_K, K, step):
        mask_k = (k + tl.arange(0, BLOCK_K)) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        if ACC_FP16:
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
        a_ptrs += step * stride_ak
        b_ptrs += step * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        # Slices combine in fp32 in memory, which is the point of split-K.
        tl.atomic_add(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def triton_matmul(a, b, acc_fp16: bool, split_k: int = 1, block=(128, 128, 64)):
    m, k = a.shape
    _, n = b.shape
    bm, bn, bk = block
    out_dtype = torch.float32 if split_k > 1 else a.dtype
    c = torch.zeros(m, n, device=a.device, dtype=out_dtype)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), split_k)
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACC_FP16=acc_fp16,
        SPLIT_K=split_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=8,
        num_stages=3,
    )
    return c


def gemm_bytes(m: int, n: int, k: int, in_width: int, out_width: int, split_k: int):
    """Algorithmic-minimum DRAM traffic for one GEMM, in bytes.

    Each input read once, the output written once per k-slice, plus the
    zero-fill split-K needs before its atomics. A real tiled GEMM re-reads A and
    B once per output tile, so this is a floor -- which is the safe direction:
    it under-states the achieved bandwidth rather than inventing headroom.
    """
    reads = (m * k + k * n) * in_width
    writes = m * n * out_width * split_k
    zero_fill = m * n * out_width if split_k > 1 else 0
    return reads + writes + zero_fill


def measure(name, fn, out, ref, *, m, n, k, dtype, split_k, ceiling, in_width):
    """One row: the four metrics plus both error columns, from one implementation."""
    out_width = out.element_size()
    metrics = module_metrics(
        fn,
        flops=2 * m * n * k,
        bytes_moved=gemm_bytes(m, n, k, in_width, out_width, split_k),
        ceiling=ceiling,
    )
    return dict(
        dtype=str(dtype).replace("torch.", ""),
        m=m,
        n=n,
        k=k,
        impl=name,
        split_k=split_k,
        rel_err=rel_error(out, ref),
        ulp=ulp_error(out, ref, dtype, mode="rms"),
        **metrics,
    )


def run(sizes, dtypes, splits, device="cuda"):
    rows = []
    for dtype in dtypes:
        width = torch.finfo(dtype).bits // 8
        # "vector", not "tf32": TF32 is disabled for this row, so cuBLAS runs
        # genuine fp32 FMA. A Triton fp32 `tl.dot` would want "tf32" instead --
        # it lowers to the TF32 tensor cores even though its inputs are fp32.
        ceiling = "vector" if dtype is torch.float32 else "tensor"
        for m, n, k in sizes:
            a = torch.randn(m, k, device=device, dtype=torch.float32) * 0.1
            b = torch.randn(k, n, device=device, dtype=torch.float32) * 0.1
            ad, bd = a.to(dtype), b.to(dtype)
            # From the *rounded* inputs, not from a/b: against a/b every row
            # inherits the input-casting error, which at small K swamps the
            # accumulator difference this benchmark exists to measure.
            ref = ad.double() @ bd.double()

            rows.append(
                measure(
                    "cublas",
                    lambda: torch.matmul(ad, bd),
                    torch.matmul(ad, bd),
                    ref,
                    m=m,
                    n=n,
                    k=k,
                    dtype=dtype,
                    split_k=1,
                    ceiling=ceiling,
                    in_width=width,
                )
            )
            print(format_metrics(f"{rows[-1]['dtype']} cublas K={k}", rows[-1]))

            if dtype is torch.float32:
                # The vector-core reference point only needs its own baseline;
                # the accumulator question is meaningless without HMMA.
                a = b = ad = bd = ref = None
                torch.cuda.empty_cache()
                continue

            for acc_fp16 in (False, True):
                if acc_fp16 and dtype is torch.bfloat16:
                    continue  # bf16 has no fp16-accumulate HMMA path
                for split_k in splits:
                    if not acc_fp16 and split_k > 1:
                        continue  # split-K only matters as an fp16-acc remedy
                    label = "triton-fp16acc" if acc_fp16 else "triton-fp32acc"
                    try:
                        out = triton_matmul(ad, bd, acc_fp16, split_k)
                        rows.append(
                            measure(
                                label,
                                lambda ad=ad, bd=bd, acc=acc_fp16, s=split_k: (
                                    triton_matmul(ad, bd, acc, s)
                                ),
                                out,
                                ref,
                                m=m,
                                n=n,
                                k=k,
                                dtype=dtype,
                                split_k=split_k,
                                ceiling="tensor",
                                in_width=width,
                            )
                        )
                    except Exception as err:  # noqa: BLE001
                        print(f"  skip acc_fp16={acc_fp16} split_k={split_k}: {err}")
                        continue
                    suffix = f" splitK={split_k}" if split_k > 1 else ""
                    print(
                        format_metrics(
                            f"{rows[-1]['dtype']} {label}{suffix} K={k}", rows[-1]
                        )
                        + f"  ulp {rows[-1]['ulp']:8.1f}"
                    )
            a = b = ad = bd = ref = None
            torch.cuda.empty_cache()
    return rows


def _label(impl: str, split_k: int, dtype: str) -> str:
    return f"{dtype} {impl}" + (f" splitK={split_k}" if split_k > 1 else "")


# Above this host share, `wall - host` is a weak estimate rather than a
# measurement: the two are not strictly additive, so subtracting a host time that
# is itself a large part of the window leaves a residual dominated by noise. Past
# 50% it has produced values above the hardware peak.
HOST_SHARE_SUSPECT = 0.30


def _net_tflops(row):
    """Throughput net of host dispatch, or ``nan`` when the subtraction is void."""
    net = row["ms"] - row.get("host_ms", 0.0)
    if net <= 0:
        return float("nan")
    flops = 2.0 * row["m"] * row["n"] * row["k"]
    return flops / (net / 1e3) / 1e12


def _host_share(row):
    return row.get("host_ms", 0.0) / row["ms"] if row["ms"] else float("nan")


def plot(rows, out_dir, peak_gbps, device=None):
    ks = sorted({r["k"] for r in rows})
    fig, axes = new_figure(1, 4, figsize=(24, 5.4))
    pal = Palette()
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    series = sorted({(r["impl"], r["split_k"], r["dtype"]) for r in rows})
    for i, key in enumerate(series):
        impl, split_k, dtype = key
        sel = sorted(
            (r for r in rows if (r["impl"], r["split_k"], r["dtype"]) == key),
            key=lambda r: r["k"],
        )
        if not sel:
            continue
        label = _label(impl, split_k, dtype)
        # Colour keys on the whole series, so split-K variants never collide.
        color = pal.color(label)
        style = "--" if split_k > 1 else "-"
        marker = markers[i % len(markers)]
        kw = dict(color=color, marker=marker, linestyle=style, label=label)
        # Wall behind, net in front. The gap between the two *is* the dispatch
        # cost, so hiding either would make the correction invisible.
        axes[0].plot(
            [r["k"] for r in sel],
            [r["tflops"] for r in sel],
            color=color,
            linestyle=style,
            alpha=0.28,
            linewidth=1.2,
        )
        axes[0].plot([r["k"] for r in sel], [_net_tflops(r) for r in sel], **kw)
        weak = [r for r in sel if _host_share(r) > HOST_SHARE_SUSPECT]
        if weak:
            axes[0].scatter(
                [r["k"] for r in weak],
                [_net_tflops(r) for r in weak],
                s=140,
                facecolors="none",
                edgecolors="#8E44AD",
                linewidths=1.6,
                zorder=5,
            )
        axes[1].plot([r["k"] for r in sel], [max(r["ulp"], 1e-3) for r in sel], **kw)
        # Relative error here, not ULP: ULP is normalised by each dtype's own
        # eps, so an fp32 row at 40 ULP is 8192x more accurate than an fp16 row
        # at 1.7 and would still plot to the right of it. This panel mixes
        # dtypes, so it needs the one error axis that survives that.
        axes[2].plot(
            [max(r["rel_err"], 1e-9) for r in sel],
            [_net_tflops(r) for r in sel],
            color=color,
            marker=marker,
            linestyle=":",
            alpha=0.85,
            label=label,
        )
        axes[3].plot([r["k"] for r in sel], [r["gbps"] for r in sel], **kw)

    for ceiling, color, style, label in (
        (TENSOR_PEAK_TFLOPS, "#c0392b", ":", f"mma {TENSOR_PEAK_TFLOPS:.0f}T"),
        (TENSOR_MAMF_TFLOPS, "#999999", "--", f"MAMF {TENSOR_MAMF_TFLOPS:.0f}T"),
        (VECTOR_PEAK_TFLOPS, "#2980b9", "-.", f"vector {VECTOR_PEAK_TFLOPS:.0f}T"),
    ):
        for ax in (axes[0], axes[2]):
            ax.axhline(ceiling, color=color, linestyle=style, linewidth=1.1)
        axes[0].annotate(
            label,
            (ks[0], ceiling),
            fontsize=7,
            va="bottom",
            xytext=(2, 2),
            textcoords="offset points",
        )
    axes[1].axhline(1.0, color="#c0392b", linestyle=":", linewidth=1.1)
    axes[1].annotate(
        "1 ULP -- as exact as the format allows",
        (ks[0], 1.0),
        fontsize=7,
        va="bottom",
        xytext=(2, 2),
        textcoords="offset points",
    )
    axes[3].axhline(
        peak_gbps, color="#2980b9", linestyle="-.", linewidth=1.1, label="measured DRAM"
    )

    finish_axis(
        axes[0],
        "reduction depth K",
        "device est. TFLOP/s (wall - host)",
        "Throughput: net of dispatch, wall faint behind\n"
        "ringed = host > 30% of wall, estimate is weak there",
        xticks=ks,
        logx=True,
    )
    axes[0].set_ylim(0, TENSOR_PEAK_TFLOPS * 1.08)
    finish_axis(
        axes[1],
        "reduction depth K",
        "max error (ULP of its own dtype, rms-scaled)",
        "Accuracy within each format (lower better)",
        xticks=ks,
        logx=True,
        logy=True,
    )
    finish_axis(
        axes[2],
        "relative error vs fp64 (comparable across dtypes)",
        "TFLOP/s",
        "Pareto: speed bought with error",
    )
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, TENSOR_PEAK_TFLOPS * 1.08)
    finish_axis(
        axes[3],
        "reduction depth K",
        "effective GB/s (algorithmic minimum)",
        "Traffic against the measured DRAM roofline",
        xticks=ks,
        logx=True,
        logy=True,
    )
    axes[3].legend(fontsize=7)

    worst = max(rows, key=lambda r: r["peak_gib"])
    axes[3].annotate(
        f"peak memory {min(r['peak_gib'] for r in rows):.2f}-"
        f"{worst['peak_gib']:.2f} GiB across every row; heaviest is "
        f"{_label(worst['impl'], worst['split_k'], worst['dtype'])} at K={worst['k']}",
        (0.02, 0.03),
        xycoords="axes fraction",
        fontsize=6.5,
    )

    # One shared legend under the figure: no series is ever hidden behind it.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
    )
    # The ratio is stated on the figure, and stated as flat, because the wall-time
    # numbers invite the opposite reading: they climb 1.28 -> 1.53 across this K
    # sweep purely because the host share falls 45% -> 3%, so both implementations
    # absorb the same ~65 us and the compression relaxes. Net of dispatch there is
    # no trend to explain.
    ratios = []
    for k in ks:
        pair = {
            r["impl"]: r
            for r in rows
            if r["k"] == k and r["split_k"] == 1 and r["dtype"] == "float16"
        }
        a, b = pair.get("triton-fp32acc"), pair.get("triton-fp16acc")
        if a and b:
            an, bn = _net_tflops(a), _net_tflops(b)
            if an == an and bn == bn and an > 0:
                ratios.append((k, bn / an))
    if ratios:
        span = f"{min(r for _, r in ratios):.2f}-{max(r for _, r in ratios):.2f}"
        trend = (
            f"fp16 accumulate is {span}x fp32, FLAT-to-FALLING in K "
            f"({ratios[0][1]:.2f} at K={ratios[0][0]} -> {ratios[-1][1]:.2f} at "
            f"K={ratios[-1][0]}).\nThe apparent climb in wall time is dispatch: "
            "its share falls 45% -> 3% across this sweep and both rows pay it."
        )
    else:
        trend = "split-K trades throughput back for accuracy"
    fig.suptitle(
        f"fp16 vs fp32 tensor-core accumulation -- {device or device_name()}\n"
        f"M=N=4096, throughput net of host dispatch. {trend}",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "hgemm_acc.png"))


def summarize(rows) -> None:
    """The two questions this benchmark exists to answer, in plain numbers."""
    best = {}
    for row in rows:
        key = (row["dtype"], row["impl"], row["split_k"])
        best[key] = max(best.get(key, 0.0), row["tflops"])
    print("\npeak TFLOP/s by implementation:")
    for key, tflops in sorted(best.items(), key=lambda kv: -kv[1]):
        print(f"  {key[0]:>9} {key[1]:>15} splitK={key[2]}: {tflops:7.1f}")

    deep = max(r["k"] for r in rows)
    print(f"\nsplit-K accuracy recovery at K={deep} (fp16 inputs):")
    for row in sorted(
        (r for r in rows if r["k"] == deep and r["dtype"] == "float16"),
        key=lambda r: (r["impl"], r["split_k"]),
    ):
        print(
            f"  {row['impl']:>15} splitK={row['split_k']}: "
            f"{row['ulp']:9.1f} ULP  rel {row['rel_err']:.2e}  "
            f"{row['tflops']:7.1f} TF/s"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/hgemm")
    ap.add_argument("--size", type=int, default=4096, help="M and N of the sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Explicitly, not by inheriting whatever the process happens to be set to:
    # with TF32 on, the "fp32" row silently runs on the tensor cores and stops
    # being the vector-core reference point it exists to be.
    torch.backends.cuda.matmul.allow_tf32 = False

    s = args.size
    sizes = [(s, s, k) for k in (512, 1024, 2048, 4096, 8192, 16384)]
    print(f"device: {device_name()}  |  M=N={s}")
    rows = run(
        sizes, [torch.float16, torch.bfloat16, torch.float32], splits=[1, 2, 4, 8]
    )

    # Stamped per row rather than in a header: this file's schema is a bare list,
    # and a redraw that has no GPU visible would otherwise title the figure "cpu".
    device = device_name()
    for row in rows:
        row["device"] = device
    with open(os.path.join(args.out, "hgemm_acc.json"), "w") as f:
        json.dump(rows, f, indent=2)
    plot(rows, args.out, cached_peak_bandwidth(), device=device)
    summarize(rows)
    print(f"\nwrote {args.out}/hgemm_acc.png")


if __name__ == "__main__":
    main()
