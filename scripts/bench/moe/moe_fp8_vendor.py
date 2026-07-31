"""The boring, obviously-correct MoE fp8 path: a per-expert loop of vendor GEMMs.

The custom grouped MXFP8 kernel is fast on paper and currently disagrees with its
own oracle. This benchmark answers the separate question that decides whether that
kernel is a *dependency* or an *optimization*: **what does a path whose numerics are
not in question cost?** Every fp8 arm here is `torch.nn.functional.scaled_mm`, the
same cuBLAS call that carries MXFP8 on dense models at 0.6 sigma against bf16.

The enabler, verified before anything was timed. ``quantize_mx_vendor`` writes swizzle
tile ``(row_tile, col_tile)`` at ``(row_tile*col_tiles + col_tile)*512``, so a span of
whole row-tiles is contiguous in the flat buffer. An expert whose rows start at a
**128-aligned offset** therefore takes its scale operand as a plain view -- no
re-swizzle, no copy, no extra launch. The count need not be aligned, only the start:
``expected_swizzled_numel(count, K)`` is exactly ``ceil(count/128)*col_tiles*512``. So
the whole loop costs **two** quantize launches (one activation buffer, one weight bank)
plus one ``scaled_mm`` per expert, and `tests/test_kernels.py` pins both the positive
case and the unaligned negative case.

Five arms, because three different questions are being asked at once and no single
comparison answers more than one:

* ``bf16_grouped`` -- the incumbent Triton grouped GEMM. The denominator that matters.
* ``bf16_loop`` -- a per-expert ``torch.mm`` loop. The **loop control**: same launch
  count as the vendor loop, no fp8. Without it, "the vendor loop is slower" cannot be
  told apart from "a loop is slower", and those have opposite implications. It runs at
  0.50-0.72x of the grouped kernel, so most of the sequential vendor loop's deficit is
  the looping, not the precision.
* ``fp8_vendor_loop`` -- the deliverable, read literally: one stream.
* ``fp8_vendor_streamed`` -- the same GEMMs across concurrent streams. This is the arm
  that wins, and the gap between it and the previous one is the whole finding.
* ``fp8_triton_grouped`` -- the custom kernel, for context and as a **correctness
  cross-check**: the vendor loop is a bit-exact oracle for it, so every row reports
  whether the two agree bitwise at that shape. The existing test covers two toy shapes;
  a divergence appearing only at a preset shape would be invisible to it. Measured:
  48/48 rows bit-identical, 0.0 ULP.

Rows per expert is ``tokens * top_k / experts``, which is ``tokens/8`` at every preset
here, and it is the axis that decides everything.

**The loop is not launch-bound, and that was the expected answer.** At the 1B preset with
1024 rows per expert the loop issues in 2.6 ms of host time and executes in 0.46 ms of
device time -- 98% host, so CUDA-graphing it removes all of the dispatch. It is still
slower than bf16 grouped afterwards (0.75x). One expert's GEMM1 there is 48 CTAs against
170 SMs, so 64 of them in one stream leave most of the card idle however cheaply they are
issued. **The binding constraint is wave quantization**, the fix is concurrency, and it
saturates at 8 streams while staying bit-identical to the sequential loop.

**``wall - host`` is not usable here and is not the primary timer.** At a 96-99% host
share the subtraction is a difference of two nearly-equal numbers: the first run of this
benchmark reported 1401 TF/s for a *bf16* arm against a 227 TF/s ceiling. The primary
timer is ``device_time`` -- one graph capture, replayed through the ordinary **flushed**
sample loop. That is device time with no dispatch and a cold cache, which is what a real
MoE layer sees: plain ``graph_ms`` has no flush, and the fp8 weight bank fits in this
card's 96 MiB L2 at every preset here while the bf16 bank does not, so an unflushed
replay would hand the fp8 arm a cache advantage the model never gets. Both banks' L2
residency is in every row so the reader can check that rather than take it on trust.

Graph capture is only available because the shapes are static, which the 128-alignment
already implies: a capacity-padded dispatch has fixed counts and needs no device->host
sync to build the slices. The sync a *dynamic* loop would pay is measured rather than
assumed -- see :func:`sync_cost_us`.

Usage:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/bench/moe/moe_fp8_vendor.py \
        --out out/bench/moe/moe_fp8_vendor
"""

import argparse
import json
import os
import time

import torch

from kohakuwullm.bench.analysis.admissibility import (
    contention_notes,
    median_and_contention,
    visible_devices,
)
from kohakuwullm.bench.core.timing import (
    cached_peak_bandwidth,
    cpu_enqueue_ms,
    device_name,
    stream_bandwidths,
    ulp_error,
)
from kohakuwullm.bench.vendor.vendor_moe import (
    StreamedLoop,
    aligned_offsets,
    dense_ceilings,
    device_time,
    measure_replay_floor,
    scale_views,
)
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, quantize_mx, quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.grouped import grouped_mxfp8_gemm
from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul_swizzled
from kohakuwullm.models.presets import get_preset

# Both ceilings are **measured in this process on this card**, by `dense_ceilings`,
# rather than carried as constants. The stored 227 TF/s bf16 figure disqualified a
# grouped bf16 row at 228 TF/s on its first run here: it is a dense-GEMM measurement
# from another day at another sustained clock, not a hardware bound, and a threshold
# that fires on 0.4% is measuring the constant's staleness. The theoretical fp8 peak
# stays as a constant because it *is* a bound -- nothing can exceed it.
FP8_PEAK_TFLOPS = 838.0
CEILINGS: dict[str, float] = {}

# The aten dispatch floor measured on this box, kept as the yardstick for the per-expert
# host cost each row reports. Measured here at ~41 us per expert, i.e. this loop's Python
# side costs more than twice the bare dispatch -- which is moot under a graph and fatal
# without one.
ATEN_DISPATCH_US = 18.0

PRESETS = ("Kohaku-MoE-1B", "Kohaku-MoE-2B", "Kohaku-MoE-3B", "MoE-1B-A280M")

# Filled in by `measure_replay_floor` before any row runs: a replay's host enqueue lands
# between the two GPU timestamps, so it inflates every device time additively -- which
# biases every *ratio* toward 1.0 and makes the speedups reported here conservative.
GRAPH_REPLAY_FLOOR_MS = float("nan")


def expert_shapes(preset: str) -> list[tuple[str, int, int]]:
    """``(name, N, K)`` for the two expert GEMMs of one preset.

    GEMM1 is the fused gate+up projection, so its N is ``2*hidden``. Both are FPROP,
    contracting K, which is the axis the MX block scale must lie along.
    """
    cfg = get_preset(preset)
    return [
        ("gemm1", 2 * cfg.moe_hidden, cfg.dim),
        ("gemm2", cfg.dim, cfg.moe_hidden),
    ]


def build(preset: str, name: str, n: int, k: int, rows_per_expert: int, dtype):
    """Every operand both paths need, quantized once, sliced into per-expert views."""
    experts = get_preset(preset).moe_num_experts
    starts, counts, total, offsets = aligned_offsets(rows_per_expert, experts)
    torch.manual_seed(0)

    x = torch.randn(total, k, device="cuda", dtype=dtype)
    w = torch.randn(experts, n, k, device="cuda", dtype=dtype) * k**-0.5

    # One quantize for the whole activation buffer and one for the whole weight bank.
    # The weight bank's per-expert start is `e*n`, and every preset's n is a multiple
    # of 128 (gemm1 n = 2*hidden, gemm2 n = dim), so those starts are aligned too.
    xq, xs_sw = quantize_mx_vendor(x)
    wq_flat, ws_sw = quantize_mx_vendor(w.reshape(experts * n, k))
    wq = wq_flat.view(experts, n, k)
    x_views = scale_views(xs_sw, starts, counts, k)
    w_views = scale_views(ws_sw, [e * n for e in range(experts)], [n] * experts, k)
    xq_rows = [xq[s : s + c] for s, c in zip(starts, counts)]

    # Natural-layout scales for the Triton kernel. Same `_quantize_block`, so the e4m3
    # values and ue8m0 exponents are identical to the vendor pair -- which is what
    # makes the vendor loop a bit-exact oracle for it rather than merely a close one.
    xq_nat, xs_nat = quantize_mx(x)
    wq_nat, ws_nat = quantize_mx(w.reshape(experts * n, k))
    return dict(
        preset=preset,
        name=name,
        experts=experts,
        n=n,
        k=k,
        rows_per_expert=rows_per_expert,
        total=total,
        padding_frac=1.0 - (rows_per_expert * experts) / total,
        dtype=dtype,
        x=x,
        w=w,
        offsets=offsets,
        xq_rows=xq_rows,
        x_views=x_views,
        wq=wq,
        w_views=w_views,
        xq_nat=xq_nat,
        xs_nat=xs_nat,
        wq_nat=wq_nat.view(experts, n, k),
        ws_nat=ws_nat.view(experts, n, k // BLOCK_SCALE),
        starts=starts,
        counts=counts,
    )


def arms(b: dict, streams: int = 8):
    """The callables, sharing operands so no arm is charged for another's setup."""
    xq_rows, x_views, wq, w_views = b["xq_rows"], b["x_views"], b["wq"], b["w_views"]
    x, w, offsets, dtype = b["x"], b["w"], b["offsets"], b["dtype"]
    experts = b["experts"]
    x_rows = [x[s : s + c] for s, c in zip(b["starts"], b["counts"])]

    # Bound per expert rather than closed over the loop variable: a `lambda: f(e)` in a
    # comprehension captures the *name*, so every job would run the last expert -- and
    # the result would still be the right shape and the right dtype.
    fp8_jobs = [
        (
            lambda a=xq_rows[e], sa=x_views[e], bq=wq[e], sb=w_views[e]: (
                vendor_mxfp8_matmul_swizzled(a, sa, bq, sb, dtype)
            )
        )
        for e in range(experts)
    ]
    bf16_jobs = [(lambda a=x_rows[e], bw=w[e]: a @ bw.T) for e in range(experts)]

    def fp8_vendor_loop():
        return [job() for job in fp8_jobs]

    def bf16_loop():
        return [job() for job in bf16_jobs]

    def bf16_grouped():
        return grouped_gemm(x, w, offsets)

    def fp8_triton_grouped():
        return grouped_mxfp8_gemm(
            b["xq_nat"], b["xs_nat"], b["wq_nat"], b["ws_nat"], offsets, out_dtype=dtype
        )

    return {
        "bf16_grouped": bf16_grouped,
        "bf16_loop": bf16_loop,
        "fp8_vendor_loop": fp8_vendor_loop,
        "fp8_vendor_streamed": StreamedLoop(fp8_jobs, streams),
        "fp8_triton_grouped": fp8_triton_grouped,
    }


def x_slice(b: dict, e: int) -> torch.Tensor:
    return b["x"][b["starts"][e] : b["starts"][e] + b["counts"][e]]


def concat_rows(parts: list[torch.Tensor]) -> torch.Tensor:
    """The loop's per-expert outputs assembled into the buffer a combine step wants.

    Costs one launch, not E: `torch.cat` is a single kernel. It is not free -- see
    `moe_fp8_vendor_path.py`, where it is 120 us of unoverlapped copy at the 1B preset,
    forced by `F.scaled_mm` having no `out=` overload.
    """
    return torch.cat(parts, dim=0)


def unpadded(b: dict, out: torch.Tensor) -> torch.Tensor:
    """The grouped kernels' output restricted to real rows, dropping alignment padding."""
    return torch.cat([out[s : s + c] for s, c in zip(b["starts"], b["counts"])], dim=0)


def row(b: dict, graphs: bool = True, streams: int = 8) -> dict:
    """One shape, every arm, timed and cross-checked."""
    fns = arms(b, streams=streams)
    experts, n, k = b["experts"], b["n"], b["k"]
    real_rows = b["rows_per_expert"] * experts
    flops = 2.0 * real_rows * n * k

    with torch.no_grad():
        vendor_out = concat_rows(fns["fp8_vendor_loop"]())
        triton_out = unpadded(b, fns["fp8_triton_grouped"]())
        bf16_out = unpadded(b, fns["bf16_grouped"]())
    # Bit-equal, not close. Both consume the same e4m3 values and the same ue8m0
    # exponents; anything other than equality is a layout bug, and that is exactly the
    # failure an fp64 reference passes.
    bitwise = bool(torch.equal(vendor_out, triton_out))
    triton_ulp = ulp_error(triton_out, vendor_out, torch.float8_e4m3fn, "rms")

    out: dict = dict(
        preset=b["preset"],
        gemm=b["name"],
        experts=experts,
        N=n,
        K=k,
        rows_per_expert=b["rows_per_expert"],
        padded_rows=b["total"],
        padding_frac=b["padding_frac"],
        tokens_equivalent=b["rows_per_expert"] * 8,
        gflop=flops / 1e9,
        # Both banks, because their L2 residency differs and that is exactly what an
        # unflushed graph replay would turn into a fake fp8 win.
        fp8_bank_mib=experts * n * k * (1 + 1 / BLOCK_SCALE) / 2**20,
        bf16_bank_mib=experts * n * k * 2 / 2**20,
        fp8_bank_fits_l2=experts * n * k * (1 + 1 / BLOCK_SCALE) < _l2_bytes(),
        bf16_bank_fits_l2=experts * n * k * 2 < _l2_bytes(),
        vendor_vs_triton_bitwise=bitwise,
        vendor_vs_triton_ulp=triton_ulp,
        vendor_ulp_vs_bf16=ulp_error(
            vendor_out.float(), bf16_out.float(), torch.float8_e4m3fn, "rms"
        ),
    )

    for label, fn in fns.items():
        wall, contended, power = median_and_contention(fn)
        host = cpu_enqueue_ms(fn)
        dev, dev_contended, dev_power = (
            device_time(fn) if graphs else (float("nan"),) * 2 + (False,)
        )
        out[f"{label}_wall_ms"] = wall
        out[f"{label}_host_ms"] = host
        out[f"{label}_device_ms"] = dev
        out[f"{label}_host_share"] = host / wall if wall else float("nan")
        out[f"{label}_contended"] = contended
        out[f"{label}_device_contended"] = dev_contended
        out[f"{label}_power"] = power and dev_power
        out[f"{label}_capturable"] = dev == dev
        # The replay's host enqueue lands between the two GPU timestamps, so every
        # device_ms is inflated by roughly one floor. Reported, not silently subtracted:
        # the raw number stays the headline because the correction biases every *ratio*
        # toward 1.0, which makes the fp8 speedups quoted here conservative rather than
        # flattering. The fraction says how conservative.
        out[f"{label}_floor_frac"] = (
            GRAPH_REPLAY_FLOOR_MS / dev if dev and dev == dev else float("nan")
        )
        out[f"{label}_device_less_floor_ms"] = dev - GRAPH_REPLAY_FLOOR_MS
        for stem in ("wall", "device"):
            ms = out[f"{label}_{stem}_ms"]
            out[f"{label}_{stem}_tflops"] = flops / (ms / 1e3) / 1e12 if ms > 0 else 0.0
        torch.cuda.empty_cache()

    # Per-expert issue cost against the aten dispatch floor: a loop whose host time per
    # expert sits at the floor is issuing, not computing.
    out["vendor_us_per_expert_host"] = out["fp8_vendor_loop_host_ms"] * 1e3 / experts
    out["vendor_us_per_expert_device"] = (
        out["fp8_vendor_loop_device_ms"] * 1e3 / experts
    )
    out["launch_bound"] = bool(
        out["fp8_vendor_loop_host_ms"] > out["fp8_vendor_loop_device_ms"]
    )
    for stem in ("wall", "device"):
        base = out[f"bf16_grouped_{stem}_ms"]
        for arm in ("fp8_vendor_loop", "fp8_vendor_streamed", "fp8_triton_grouped"):
            got = out[f"{arm}_{stem}_ms"]
            out[f"{arm}_speedup_{stem}"] = base / got if got > 0 else float("nan")
        out[f"loop_penalty_{stem}"] = (
            out[f"bf16_loop_{stem}_ms"] / base if base > 0 else float("nan")
        )
    out["graph_gain"] = (
        out["fp8_vendor_loop_wall_ms"] / out["fp8_vendor_loop_device_ms"]
        if out["fp8_vendor_loop_device_ms"] > 0
        else float("nan")
    )
    out["stream_gain"] = (
        out["fp8_vendor_loop_device_ms"] / out["fp8_vendor_streamed_device_ms"]
        if out["fp8_vendor_streamed_device_ms"] > 0
        else float("nan")
    )
    out["suspect"] = _suspect(out, fns)
    return out


def _suspect(r: dict, fns: dict) -> str:
    """Every impossibility and disqualification this row can state, before it prints.

    Both ceilings come from ``dense_ceilings`` in this same process, with 10% of slack:
    a grouped kernel is not obliged to be slower than a dense square GEMM, so an arm a
    few percent above it is a shape difference, not a broken timer. What the check is
    for is the failure that produced the first run of this benchmark -- a ``wall - host``
    bf16 arm reporting 1401 TF/s, which no amount of slack excuses.
    """
    notes = []
    for label in fns:
        if not r[f"{label}_capturable"]:
            notes.append(f"{label} did not capture: no device time for it")
            continue
        tf = r[f"{label}_device_tflops"]
        kind = "fp8" if "fp8" in label else "bf16"
        # The hardware bound first, then this card's own measured dense rate. Only the
        # first is an impossibility; the second is a strong prior a grouped kernel may
        # legitimately edge past, so it says "above" rather than disqualifying the row.
        if kind == "fp8" and tf > FP8_PEAK_TFLOPS:
            notes.append(f"{label}_device={tf:.0f} TF/s above the fp8 hardware peak")
        elif tf > 1.1 * CEILINGS.get(kind, float("inf")):
            notes.append(
                f"{label}_device={tf:.0f} TF/s is >10% above this card's measured "
                f"dense {kind} rate ({CEILINGS[kind]:.0f})"
            )
        if r[f"{label}_floor_frac"] > 0.2:
            notes.append(
                f"{label} is {r[f'{label}_floor_frac']:.0%} replay floor "
                f"({r[f'{label}_device_ms'] * 1e3:.0f} us): its rate is understated"
            )
    certifiable = all(r[f"{label}_power"] for label in fns)
    notes += contention_notes(
        r, tuple(f"{label}_device_contended" for label in fns), certifiable
    )
    if not r["vendor_vs_triton_bitwise"]:
        notes.append(
            "triton grouped differs from the vendor oracle "
            f"({r['vendor_vs_triton_ulp']:.1f} ULP rms)"
        )
    return "; ".join(notes)


def _l2_bytes() -> int:
    return torch.cuda.get_device_properties(0).L2_cache_size


def stream_sweep(b: dict, counts=(1, 2, 4, 8, 16, 32, 64)) -> list[dict]:
    """Device time against concurrent stream count, with bit-equality against 1 stream.

    The equality check is not decoration. Concurrency changes *when* each GEMM runs and
    must not change *what* it computes -- every stream reads disjoint rows and writes a
    disjoint output, so any difference would mean the fan-out aliased something. It is
    checked at every stream count rather than once, because an aliasing bug that only
    appears past some degree of overlap is exactly the kind this would otherwise miss.
    """
    fns = arms(b)
    flops = 2.0 * b["rows_per_expert"] * b["experts"] * b["n"] * b["k"]
    tiles = -(-b["rows_per_expert"] // 128) * -(-b["n"] // 128)
    base_ms = device_time(fns["bf16_grouped"])[0]
    with torch.no_grad():
        reference = torch.cat(fns["fp8_vendor_loop"](), 0)
    out = []
    for n_streams in counts:
        loop = arms(b, streams=n_streams)["fp8_vendor_streamed"]
        with torch.no_grad():
            got = torch.cat(loop(), 0)
        torch.cuda.synchronize()
        ms, contended, power = device_time(loop)
        out.append(
            dict(
                preset=b["preset"],
                gemm=b["name"],
                rows_per_expert=b["rows_per_expert"],
                tiles_per_expert=tiles,
                streams=n_streams,
                device_ms=ms,
                device_tflops=flops / (ms / 1e3) / 1e12 if ms > 0 else 0.0,
                speedup_vs_bf16_grouped=base_ms / ms if ms > 0 else float("nan"),
                bitwise_vs_sequential=bool(torch.equal(got, reference)),
                contended=contended,
                power=power,
            )
        )
        del loop, got
        torch.cuda.empty_cache()
    return out


def sync_cost_us(experts: int, iters: int = 200) -> float:
    """What a *dynamic* per-expert loop pays that a padded one does not.

    A loop needs its expert counts on the host to build slices. With capacity padding
    they are static and this cost is zero; without it, every MoE layer pays one
    device->host sync. Measured rather than assumed, because it is the one structural
    cost the vendor path has and the grouped kernel does not -- `grouped_mxfp8_gemm`
    bounds its grid from `rows` and `E` alone and never reads offsets on the host.
    """
    offsets = torch.arange(experts + 1, dtype=torch.int32, device="cuda") * 128
    torch.cuda.synchronize()
    for _ in range(10):
        offsets.tolist()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        offsets.tolist()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iters


def main():
    global GRAPH_REPLAY_FLOOR_MS
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--skip-stream-sweep", action="store_true")
    args = parser.parse_args()

    bandwidth = stream_bandwidths()
    print(f"{device_name()}  CUDA_VISIBLE_DEVICES={visible_devices()}")
    print("STREAM GB/s " + "  ".join(f"{k} {v:.0f}" for k, v in bandwidth.items()))
    if max(bandwidth.values()) < 1700:
        print("!! below 1700 GB/s: the card is shared and every row below is worthless")
    GRAPH_REPLAY_FLOOR_MS = measure_replay_floor()
    CEILINGS.update(dense_ceilings())
    print(f"flushed graph-replay floor: {GRAPH_REPLAY_FLOOR_MS * 1e3:.1f} us")
    print(
        "measured dense 4096^3 ceilings on THIS card: "
        + "  ".join(f"{k} {v:.0f} TF/s" for k, v in CEILINGS.items())
    )

    sweeps = []
    if not args.skip_stream_sweep:
        configs = dict.fromkeys(
            [(args.presets[0], 256), (args.presets[0], 1024), (args.presets[-1], 1024)]
        )
        for preset, rpe in configs:
            name, n, k = expert_shapes(preset)[0]
            b = build(preset, name, n, k, rpe, torch.bfloat16)
            sweeps += stream_sweep(b)
            del b
            torch.cuda.empty_cache()
        print(
            f"\n{'preset':>13}{'rows/E':>7}{'tiles/E':>8}{'streams':>8}{'us':>8}"
            f"{'TF/s':>7}{'vs bf16':>9}{'bitwise':>9}"
        )
        for s in sweeps:
            print(
                f"{s['preset'].replace('Kohaku-', ''):>13}{s['rows_per_expert']:>7}"
                f"{s['tiles_per_expert']:>8}{s['streams']:>8}"
                f"{s['device_ms'] * 1e3:>8.1f}{s['device_tflops']:>7.0f}"
                f"{s['speedup_vs_bf16_grouped']:>8.2f}x"
                f"{'ok' if s['bitwise_vs_sequential'] else 'DIFF':>9}"
            )

    print(
        f"\n{'preset':>13}{'gemm':>6}{'rows/E':>7}{'bf16grp':>8}{'bf16lp':>8}"
        f"{'vendor':>8}{'strmed':>8}{'triton':>8}{'vend x':>8}{'strm x':>8}"
        f"{'us/E':>7}{'host%':>7}{'bits':>7}"
        "     device TF/s from a flushed graph replay; x is against bf16_grouped"
    )
    rows = []
    for preset in args.presets:
        for name, n, k in expert_shapes(preset):
            for rpe in args.rows:
                b = build(preset, name, n, k, rpe, torch.bfloat16)
                rows.append(row(b, streams=args.streams))
                del b
                torch.cuda.empty_cache()
                r = rows[-1]
                print(
                    f"{r['preset'].replace('Kohaku-', ''):>13}{r['gemm']:>6}"
                    f"{r['rows_per_expert']:>7}"
                    f"{r['bf16_grouped_device_tflops']:>8.0f}"
                    f"{r['bf16_loop_device_tflops']:>8.0f}"
                    f"{r['fp8_vendor_loop_device_tflops']:>8.0f}"
                    f"{r['fp8_vendor_streamed_device_tflops']:>8.0f}"
                    f"{r['fp8_triton_grouped_device_tflops']:>8.0f}"
                    f"{r['fp8_vendor_loop_speedup_device']:>7.2f}x"
                    f"{r['fp8_vendor_streamed_speedup_device']:>7.2f}x"
                    f"{r['vendor_us_per_expert_device']:>7.1f}"
                    f"{100 * r['fp8_vendor_loop_host_share']:>7.0f}"
                    f"{'BITEQ' if r['vendor_vs_triton_bitwise'] else 'DIFF':>7}"
                )
                if r["suspect"]:
                    print(f"{'':>13}  SUSPECT: {r['suspect']}")

    sync_us = {e: sync_cost_us(e) for e in sorted({r["experts"] for r in rows})}
    print(
        "\ndevice->host sync per layer (dynamic counts): "
        + ", ".join(f"E={e} {v:.1f} us" for e, v in sync_us.items())
    )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "moe_fp8_vendor.json"), "w") as fh:
            json.dump(
                {
                    "device": device_name(),
                    "visible_devices": visible_devices(),
                    "bandwidth": bandwidth,
                    "peak_bandwidth_gbps": cached_peak_bandwidth(),
                    "aten_dispatch_us": ATEN_DISPATCH_US,
                    "graph_replay_floor_ms": GRAPH_REPLAY_FLOOR_MS,
                    "dense_ceilings_tflops": CEILINGS,
                    "streams": args.streams,
                    "sync_us_by_experts": {str(k): v for k, v in sync_us.items()},
                    "stream_sweep": sweeps,
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {os.path.join(args.out, 'moe_fp8_vendor.json')}")


if __name__ == "__main__":
    main()
