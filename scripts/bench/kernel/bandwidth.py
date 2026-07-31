"""The DRAM roofline: what "100% of peak" actually means on this box.

Every other benchmark in the suite divides by this number, so getting it wrong
corrupts all of them. ``torch.cuda.get_device_properties().memory_clock_rate``
reports the **stock** clock; this card's memory is factory-overclocked, and a
512-bit RTX 5090 gives 1792 GB/s at the stock 28 Gbps against 2035 GB/s at
31.8 Gbps. A kernel scored against the stock figure looks 13.6% better than it
is, and one scored against the overclocked *theoretical* figure looks worse than
the machine can actually do -- neither is the ceiling a kernel can reach.

Three access patterns, because no single microbenchmark is *the* ceiling:

* ``copy``  -- ``dst.copy_(src)``, 1 read + 1 write. May dispatch to the DMA
  engine rather than a kernel, so it measures the memory system but not
  necessarily what a compute kernel can reach.
* ``read``  -- a large reduction (``x.sum()``). No write traffic; read-heavy
  kernels can legitimately beat the copy figure, so this is their reference.
* ``triad`` -- ``a = b + s * c``: 2 reads + 1 write, the classic STREAM kernel
  and the closest analogue to a fused elementwise op.

The size sweep is the part that matters for *other* benchmarks: below the L2
working set a microbenchmark never touches DRAM at all and reports a number the
real model will never see. The sweep shows where that stops being true.

Per-GPU because cards in one box need not be clocked identically, and because a
number measured on a busy GPU is worthless -- the script refuses to report for a
device that is not idle.

Usage:
    .venv/bin/python scripts/bench/kernel/bandwidth.py
    .venv/bin/python scripts/bench/kernel/bandwidth.py --gpus 1 2 3 --mem-gbps 31.8
"""

import argparse
import json
import os
import subprocess

import torch

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    device_name,
    finish_axis,
    module_metrics,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import (
    TENSOR_MAMF_TFLOPS,
    TENSOR_PEAK_TFLOPS,
    TF32_MAMF_TFLOPS,
    VECTOR_PEAK_TFLOPS,
    stream_bandwidths,
)

# (name, bytes per element touched, flops per element). `copy` does no
# arithmetic at all; reporting 0 TFLOP/s for it is the honest answer and the
# anchor the roofline's memory-bound slope is drawn from.
PATTERNS = [
    ("copy", 8, 0.0),
    ("read", 4, 1.0),
    ("triad", 12, 2.0),
]


def _busy(index: int) -> tuple[int, int]:
    """(utilization %, memory MiB) for one torch device, via nvidia-smi.

    Selected by UUID, not by ordinal: under ``CUDA_VISIBLE_DEVICES=3`` torch's
    device 0 is nvidia-smi's device 3, and querying ``-i 0`` reads a completely
    different card -- which is how this check once refused to measure an idle
    GPU because a *different* one was busy.
    """
    uuid = f"GPU-{torch.cuda.get_device_properties(index).uuid}"
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                uuid,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        util, mem = (int(x) for x in out.split(","))
        return util, mem
    except Exception:  # noqa: BLE001
        return -1, -1


def _ops(n: int, device) -> dict:
    """The three callables, keeping their buffers alive via the closures."""
    a = torch.empty(n, dtype=torch.float32, device=device).uniform_()
    b = torch.empty_like(a).uniform_()
    c = torch.empty_like(a)
    return {
        "copy": lambda: c.copy_(a),
        "read": lambda: a.sum(),
        "triad": lambda: torch.add(a, b, alpha=2.0, out=c),
    }


def reference_intensities(
    dim: int = 1280, vocab: int = 65536, tokens: int = 8192, seq_len: int = 2048
) -> list[tuple[str, float]]:
    """FLOP/byte of the shapes this repo actually runs, at the Nano-500M scale.

    Placed on the roofline so it answers the question a ceiling alone cannot:
    which of our kernels could ever be compute-bound. Bytes are counted in bf16
    and only for tensors that must cross DRAM -- flash attention never writes
    the score matrix, which is why attention sits so far right.
    """
    head_flops = 2 * tokens * dim * vocab
    head_bytes = 2 * (tokens * dim + vocab * dim + tokens * vocab)
    # 4 flops/element (square, accumulate, rsqrt, scale) against a read + write.
    norm_ai = 4.0 / 4.0
    # silu is ~4 flops, the gate multiply 1, over 2 reads + 1 write.
    swiglu_ai = 5.0 / 6.0
    # Q@K and A@V are each 2 flops per (query, key, head-dim); only Q/K/V/O
    # cross DRAM, so intensity grows linearly with sequence length.
    attn_ai = seq_len / 4.0
    return [
        ("rmsnorm", norm_ai),
        ("swiglu", swiglu_ai),
        ("attn s2048", attn_ai),
        ("lm head", head_flops / head_bytes),
    ]


def sweep(index: int, sizes_mib: list[float], peak: float) -> list[dict]:
    """Achieved bandwidth for each pattern across working-set sizes.

    Timed with the device made current throughout: ``torch.cuda.Event`` records
    against the current device's stream, so timing work enqueued on a different
    device yields garbage -- an earlier version of this script measured GPU1 at
    367,921 GB/s that way, i.e. 180x the physical limit.
    """
    rows = []
    with torch.cuda.device(index):
        device = torch.device("cuda", index)
        for mib in sizes_mib:
            n = int(mib * (1 << 20)) // 4
            ops = _ops(n, device)
            for name, bytes_per_elem, flops_per_elem in PATTERNS:
                metrics = module_metrics(
                    ops[name],
                    flops=flops_per_elem * n,
                    bytes_moved=bytes_per_elem * n,
                    warmup=5,
                    iters=20,
                    # Genuine FMA: an elementwise stream has no tensor-core path
                    # at all, TF32 included.
                    ceiling="vector",
                    peak_bandwidth_gbps=peak,
                )
                rows.append(
                    dict(gpu=index, pattern=name, mib=mib, elements=n, **metrics)
                )
            del ops
            torch.cuda.empty_cache()
    return rows


def scan(
    indices: list[int], gib: float, mem_gbps: float | None, max_util: int
) -> list[dict]:
    """One measurement per device, to confirm the cards are clocked alike."""
    rows = []
    for index in indices:
        util, mem = _busy(index)
        props = torch.cuda.get_device_properties(index)
        bus = props.memory_bus_width
        stock = 2 * props.memory_clock_rate * 1e3 * (bus / 8) / 1e9
        actual = (bus / 8 * mem_gbps) if mem_gbps else None

        if util > max_util:
            print(
                f"GPU{index}: BUSY ({util}% util, {mem} MiB) -- skipped, a "
                f"measurement here would be meaningless"
            )
            rows.append({"gpu": index, "busy": True, "util": util})
            continue

        with torch.cuda.device(index):
            res = stream_bandwidths(int(gib * (1 << 30)))
        ceiling = actual or stock
        # A result above the theoretical ceiling means the measurement is
        # broken, not that the hardware is fast. Fail loudly rather than plot it.
        if max(res.values()) > 1.5 * ceiling:
            raise RuntimeError(
                f"GPU{index}: measured {max(res.values()):.0f} GB/s against a "
                f"{ceiling:.0f} GB/s ceiling -- the measurement is wrong, not "
                f"the hardware. Check that the device was current during timing."
            )
        rows.append(
            {
                "gpu": index,
                "uuid": str(props.uuid),
                "busy": False,
                "name": props.name,
                "bus_bits": bus,
                "stock_gbps": stock,
                "theoretical_oc_gbps": actual,
                "copy_gbps": res["copy"],
                "read_gbps": res["read"],
                "triad_gbps": res["triad"],
                "measured_peak_gbps": max(res.values()),
            }
        )
        line = (
            f"GPU{index} {props.name}  bus {bus}-bit\n"
            f"   copy  {res['copy']:7.0f} GB/s   "
            f"read  {res['read']:7.0f} GB/s   "
            f"triad {res['triad']:7.0f} GB/s\n"
            f"   theoretical: {stock:.0f} at stock clock"
        )
        if actual:
            best = max(res.values())
            line += (
                f", {actual:.0f} at {mem_gbps:g} Gbps"
                f"  -> best measured is {100 * best / actual:.1f}% of it"
            )
        print(line, flush=True)
    return rows


def plot(sweep_rows, scan_rows, peak, out_path, mem_gbps):
    pal = Palette()
    fig, axes = new_figure(1, 3, figsize=(19, 5.6))
    stock = next((r["stock_gbps"] for r in scan_rows if not r.get("busy")), 0.0)
    oc = next(
        (r.get("theoretical_oc_gbps") for r in scan_rows if not r.get("busy")), None
    )

    for name, _, _ in PATTERNS:
        pts = sorted(
            (r for r in sweep_rows if r["pattern"] == name), key=lambda r: r["mib"]
        )
        axes[0].plot(
            [r["mib"] for r in pts],
            [r["gbps"] for r in pts],
            marker="o",
            color=pal.color(name),
            label=name,
        )
    if oc:
        axes[0].axhline(
            oc, color="#c0392b", linestyle=":", label=f"theoretical {mem_gbps:g} Gbps"
        )
    axes[0].axhline(stock, color="#999999", linestyle="--", label="theoretical stock")
    axes[0].axhline(
        peak, color="#2980b9", linestyle="-.", label=f"measured peak {peak:.0f}"
    )
    finish_axis(
        axes[0],
        "working set (MiB)",
        "achieved GB/s",
        "Bandwidth vs working set",
        logx=True,
        si_x=False,
    )
    axes[0].legend(fontsize=8)

    live = [r for r in scan_rows if not r.get("busy")]
    xs = range(len(live))
    width = 0.26
    for i, key in enumerate(("copy_gbps", "read_gbps", "triad_gbps")):
        bars = axes[1].bar(
            [x + (i - 1) * width for x in xs],
            [r[key] for r in live],
            width,
            color=pal.color(key.removesuffix("_gbps")),
            label=key.removesuffix("_gbps"),
        )
        bar_labels(axes[1], bars, "{:.0f}")
    axes[1].set_xlim(-0.5, len(live) - 0.5)
    if oc:
        axes[1].axhline(oc, color="#c0392b", linestyle=":", label="theoretical OC")
    axes[1].axhline(stock, color="#999999", linestyle="--", label="theoretical stock")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels([f"GPU{r['gpu']}" for r in live])
    axes[1].set_ylabel("achieved GB/s")
    axes[1].set_title("Per-device agreement")
    axes[1].legend(fontsize=7, ncol=2)

    # The roofline the whole suite is scored against. The ridge points say which
    # kernels can ever be compute-bound: below them, arithmetic is free and only
    # bytes cost anything.
    intensity = [2.0**e for e in range(-4, 13)]
    for ceiling, color, style, label in (
        (TENSOR_PEAK_TFLOPS, "#c0392b", ":", f"mma {TENSOR_PEAK_TFLOPS:.0f}T"),
        (TENSOR_MAMF_TFLOPS, "#999999", "--", f"MAMF {TENSOR_MAMF_TFLOPS:.0f}T"),
        (
            TF32_MAMF_TFLOPS,
            "#7F7F7F",
            (0, (3, 1, 1, 1)),
            f"TF32 {TF32_MAMF_TFLOPS:.0f}T",
        ),
        (VECTOR_PEAK_TFLOPS, "#2980b9", "-.", f"vector {VECTOR_PEAK_TFLOPS:.0f}T"),
    ):
        axes[2].plot(
            intensity,
            [min(ceiling, peak * i / 1e3) for i in intensity],
            color=color,
            linestyle=style,
            label=f"{label}  (ridge {ceiling * 1e3 / peak:.0f} F/B)",
        )
    biggest = max(r["mib"] for r in sweep_rows)
    for name, bytes_per_elem, flops_per_elem in PATTERNS:
        if flops_per_elem == 0:
            continue
        row = next(
            r for r in sweep_rows if r["pattern"] == name and r["mib"] == biggest
        )
        axes[2].scatter(
            [flops_per_elem / bytes_per_elem],
            [row["tflops"]],
            s=80,
            color=pal.color(name),
            label=name,
            zorder=3,
        )
    for label, ai in reference_intensities():
        attainable = min(TENSOR_MAMF_TFLOPS, peak * ai / 1e3)
        axes[2].axvline(ai, color="#bbbbbb", linewidth=0.7, zorder=1)
        axes[2].annotate(
            f"{label}\n<= {attainable:.0f}T",
            (ai, attainable),
            fontsize=6.5,
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )
    finish_axis(
        axes[2],
        "arithmetic intensity (FLOP / byte)",
        "attainable TFLOP/s",
        "Roofline from the measured peak",
        logx=True,
        logy=True,
        si_x=False,
    )
    axes[2].legend(fontsize=7)

    fig.suptitle(
        f"DRAM roofline -- {device_name()}  |  measured peak {peak:.0f} GB/s"
        + (f" vs {oc:.0f} theoretical" if oc else ""),
        fontweight="bold",
    )
    save_figure(fig, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, nargs="+", default=None)
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument(
        "--sizes-mib",
        type=float,
        nargs="+",
        default=[1, 4, 16, 64, 128, 256, 512, 1024, 2048],
    )
    ap.add_argument(
        "--mem-gbps",
        type=float,
        default=31.8,
        help="actual per-pin memory speed; 28 is the RTX 5090 stock clock",
    )
    ap.add_argument(
        "--max-util",
        type=int,
        default=5,
        help="refuse to measure a GPU busier than this (%%)",
    )
    ap.add_argument("--out", default="out/bench/kernel/bandwidth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    count = torch.cuda.device_count()
    targets = args.gpus if args.gpus is not None else list(range(count))
    print(f"device: {device_name()}")

    scan_rows = scan(targets, args.gib, args.mem_gbps, args.max_util)
    live = [r for r in scan_rows if not r.get("busy")]
    if not live:
        raise RuntimeError("every requested GPU was busy; nothing measured")
    home = live[0]["gpu"]
    peak = live[0]["measured_peak_gbps"]

    print(f"\nsize sweep on GPU{home} (peak denominator {peak:.0f} GB/s):")
    sweep_rows = sweep(home, args.sizes_mib, peak)
    for row in sweep_rows:
        print(
            f"  {row['pattern']:>5} {row['mib']:7.0f} MiB: "
            f"{row['ms']:8.3f} ms  {row['tflops']:8.2f} TF/s "
            f"({row['compute_pct']:5.1f}% of {row['compute_ceiling_tflops']:.0f}T)  "
            f"{row['gbps']:8.0f} GB/s ({row['bandwidth_pct']:5.1f}%)  "
            f"{row['peak_gib']:7.3f} GiB",
            flush=True,
        )

    payload = dict(
        config=vars(args),
        measured_peak_gbps=peak,
        scan=scan_rows,
        sweep=sweep_rows,
    )
    with open(os.path.join(args.out, "bandwidth.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    plot(
        sweep_rows,
        scan_rows,
        peak,
        os.path.join(args.out, "bandwidth.png"),
        args.mem_gbps,
    )

    best = max(r["gbps"] for r in sweep_rows)
    floor = min(
        (r["mib"] for r in sweep_rows if r["gbps"] >= 0.95 * best), default=float("nan")
    )
    print(
        f"\nmeasured peak {peak:.0f} GB/s -- use this as the denominator.\n"
        f"the sweep tops out at {best:.0f} GB/s; a working set below "
        f"{floor:g} MiB does not reach 95% of that, so a microbenchmark smaller "
        f"than this is measuring launch overhead, not the memory system."
    )
    print(f"wrote {args.out}/bandwidth.png")


if __name__ == "__main__":
    main()
