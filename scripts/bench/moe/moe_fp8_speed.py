"""Where the fused MXFP8 expert path's device time goes, kernel by kernel, per preset.

``mxfp8_moe.py`` answers whether the fused path is faster than the bf16 one. This
answers *which kernel* the time is in, at the shapes the presets actually train at --
which is the question a 2.53x whole-step regression against a 2.30x microbenchmark
forces, because a whole-path number cannot distinguish "the GEMM is slow here" from
"something beside the GEMM is".

**Every time here is profiler device time**, summed per kernel over ``ITERS`` calls.
Not ``wall - host``: that estimate has produced three physically impossible figures in
this repo, and a per-kernel split is exactly the case where it cannot work, since the
dispatch it subtracts belongs to the whole launch sequence rather than to any one
kernel. The profiler attributes to the kernel that ran.

Each row carries the fused path's relative error against the bf16 path in the same
row, per the repo convention: a kernel that is fast and wrong is not a result.

The **weight refresh is charged here too**, and it has to be: it is per optimizer step
rather than per microbatch, so a per-call benchmark that omits it flatters a path whose
weights are quantized four ways. It is reported as its own line rather than folded in,
because its cost divides by ``grad_accum`` and the GEMMs' does not.

Usage:
    .venv/bin/python scripts/bench/moe/moe_fp8_speed.py --out out/bench/moe/moe_fp8_speed
"""

import argparse
import collections
import json
import os
import statistics

import torch
import torch.nn.functional as F

from kohakuwullm.bench.core.timing import (
    bench_samples,
    cached_peak_bandwidth,
    cpu_enqueue_ms,
    device_name,
    rel_error,
    stream_bandwidths,
)
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed
from kohakuwullm.kernels.mxfp8 import quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul_swizzled
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts
from kohakuwullm.models.presets import get_preset

# The token budget the MoE A/B trains at, so rows/expert here is the ratio a step sees.
TOKENS = 8104
# The square shape both dense ceilings are quoted at. Large enough that host dispatch is
# ~2% of the call, so wall is device without a graph capture standing in for one.
CEILING_N = 4096
ITERS = 20
WARMUP = 8
PRESETS = (
    "MoE-1B-A280M",
    "Kohaku-MoE-1B",
    "Kohaku-MoE-2B",
    "Kohaku-MoE-3B",
    "Kohaku-MoE-5B",
    "MoE-8B-A1B",
)
# The triad this box peaks at. Checked against the absolute number, not only against
# the run's own ratio, because a uniformly slowed card passes a self-consistency check.
TRIAD_PEAK_GBPS = 1791.0
TRIAD_FLOOR = 0.97

# Kernels the fused path owns, in issue order, so a table row reads as a pipeline.
FUSED_ORDER = (
    "_quantize_mx_kernel",
    "_experts_gemm1",
    "_experts_gemm2",
    "_experts_gemm2_dgrad",
    "_experts_gemm1_dgrad",
    "grouped_mxfp8_wgrad_kernel",
)
BF16_ORDER = (
    "_grouped_gemm_fwd",
    "_grouped_gemm_dx",
    "_grouped_gemm_dw",
    "_combine_fwd",
    "_combine_bwd",
)


def dense_ceilings_profiled(size: int = CEILING_N) -> dict:
    """Dense bf16 and dense MXFP8 at ``size^3``, measured **in this process**.

    Carried constants go stale and then reject good work: a stored 227 TF/s bf16
    figure disqualified a legitimate 228 TF/s row on this box, which was measuring
    the constant's age rather than the kernel. So both ceilings are re-measured on
    whichever card the run is on, in the same process as the result they bound.

    Named apart from :func:`kohakuwullm.bench.vendor.vendor_moe.dense_ceilings`, which
    measures the same two rates by CUDA-graph replay. Neither is the right one in
    general: each matches the timer its own table's rows use, and that is the whole
    requirement -- a ceiling on one clock over rows on another is not a percentage.

    **Profiler device time, the same timer every row of the table uses**, and that
    consistency is the point rather than a detail. Timing the ceiling by wall and the
    rows by device time divides one clock by another: measured here, the vendor MXFP8
    call carries a 15% host share, so a wall ceiling reads 513 TF/s where the device
    one reads ~600, and every "percent of ceiling" in the table would be inflated by
    that much. ``wall_tflops`` and ``host_share`` are kept beside it so the size of
    that correction stays visible instead of being silently applied.
    """
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    flops = 2.0 * size**3

    aq, a_scale = quantize_mx_vendor(a)
    bq, b_scale = quantize_mx_vendor(b)

    def bf16_call():
        return a @ b.T

    # Scales swizzled outside the timed call: swizzling inside charges the GEMM for two
    # layout copies and understated cuBLAS by 27% the first time this repo timed it.
    def fp8_call():
        return vendor_mxfp8_matmul_swizzled(aq, a_scale, bq, b_scale)

    out = {}
    for name, call in (("bf16", bf16_call), ("mxfp8", fp8_call)):
        device_ms = sum(kernel_times(call).values())
        wall_ms = statistics.median(bench_samples(call, warmup=10, iters=30))
        host_ms = cpu_enqueue_ms(call, warmup=10)
        out[name] = {
            "tflops": flops / (device_ms / 1e3) / 1e12,
            "ms": device_ms,
            "wall_tflops": flops / (wall_ms / 1e3) / 1e12,
            "host_share": host_ms / wall_ms,
        }
    return out


def routing(tokens: int, top_k: int, experts: int, device="cuda", seed=0):
    """A balanced routing assignment plus the sorted-order indices it implies."""
    torch.manual_seed(seed)
    pairs = tokens * top_k
    expert_of = torch.arange(pairs, device=device) % experts
    counts = torch.bincount(expert_of, minlength=experts)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    order = expert_of.argsort(stable=True).to(torch.int32)
    token_of = torch.div(order, top_k, rounding_mode="floor").to(torch.int32)
    gate = torch.rand(pairs, device=device) * 0.5 + 0.5
    return offsets, order, token_of, gate


def bf16_expert_path(x, w_in, w_out, gate, token_of, order, offsets, num_tokens):
    """What ``MoEMLP`` does today: gather, two grouped GEMMs, SwiGLU, combine."""
    rows = x.index_select(0, token_of.long()).contiguous()
    h = grouped_gemm(rows, w_in, offsets)
    gate_h, value = h.chunk(2, dim=-1)
    out_sorted = grouped_gemm(F.silu(gate_h) * value, w_out, offsets)
    return combine_routed(out_sorted, gate, order, token_of, num_tokens)


def kernel_times(fn, iters: int = ITERS, warmup: int = WARMUP) -> dict[str, float]:
    """Per-kernel device ms for one call of ``fn``, keyed by short kernel name.

    Warmed at the *same* shape it is measured at, since a partial warmup over a
    varlen stream charges Triton compilation as throughput.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    out: dict[str, float] = collections.defaultdict(float)
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        out[event.key] += (event.self_device_time_total or 0.0) / 1e3 / iters
    return dict(out)


def bucket(times: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    """Named kernels of interest, plus everything else as one ``other`` line."""
    row = {n: times.get(n, 0.0) for n in names}
    row["other"] = sum(v for k, v in times.items() if k not in names)
    row["total"] = sum(times.values())
    return row


def preset_row(preset: str, tokens: int = TOKENS, dtype=torch.bfloat16) -> dict:
    """One preset: fused and bf16 per-kernel device time, refresh cost, and error."""
    cfg = get_preset(preset)
    dim, experts, top_k, hidden = (
        cfg.dim,
        cfg.moe_num_experts,
        cfg.moe_top_k,
        cfg.moe_hidden,
    )
    offsets, order, token_of, gate_f32 = routing(tokens, top_k, experts)
    gate = gate_f32.to(dtype).requires_grad_()
    m = token_of.numel()

    torch.manual_seed(0)
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype, requires_grad=True)
    w_in = (
        torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    ).requires_grad_()
    w_out = (
        torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    ).requires_grad_()
    packed = MXFP8ExpertWeights(w_in.detach(), w_out.detach())

    fused = lambda: mxfp8_moe_experts(  # noqa: E731
        x, w_in, w_out, gate, token_of, order, offsets, packed
    )
    eager = lambda: bf16_expert_path(  # noqa: E731
        x, w_in, w_out, gate, token_of, order, offsets, tokens
    )
    with torch.no_grad():
        fwd_err = rel_error(fused().float(), eager().float())

    dout = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    inputs = [x, w_in, w_out, gate]
    step_fused = lambda: torch.autograd.grad(fused(), inputs, dout)  # noqa: E731
    step_eager = lambda: torch.autograd.grad(eager(), inputs, dout)  # noqa: E731
    grad_err = {
        name: rel_error(got.float(), ref.float())
        for name, got, ref in zip(
            ("dx", "dw_in", "dw_out", "dgate"), step_fused(), step_eager()
        )
    }

    fused_times = bucket(kernel_times(step_fused), FUSED_ORDER)
    bf16_times = bucket(kernel_times(step_eager), BF16_ORDER)
    refresh = kernel_times(lambda: packed.refresh(w_in.detach(), w_out.detach()))

    # 2 GEMMs x (fprop + dgrad + wgrad), 2 MAC per FLOP.
    flops = 3 * 6.0 * m * dim * hidden
    return {
        "preset": preset,
        "tokens": tokens,
        "dim": dim,
        "experts": experts,
        "top_k": top_k,
        "hidden": hidden,
        "rows_per_expert": m / experts,
        "fwd_rel_error": fwd_err,
        "grad_rel_error": grad_err,
        "fused_ms": fused_times,
        "bf16_ms": bf16_times,
        "refresh_ms": sum(refresh.values()),
        "speedup": bf16_times["total"] / fused_times["total"],
        "fused_tflops": flops / (fused_times["total"] / 1e3) / 1e12,
        "bf16_tflops": flops / (bf16_times["total"] / 1e3) / 1e12,
    }


def format_row(row: dict) -> str:
    lines = [
        f"\n=== {row['preset']}  dim={row['dim']} hidden={row['hidden']} "
        f"E={row['experts']} top_k={row['top_k']} "
        f"{row['rows_per_expert']:.0f} rows/expert ==="
    ]
    lines.append(f"{'kernel':32s}{'fused ms':>12s}{'bf16 ms':>12s}")
    for name in FUSED_ORDER:
        lines.append(f"  {name:30s}{row['fused_ms'][name]:12.3f}{'':>12s}")
    for name in BF16_ORDER:
        lines.append(f"  {name:30s}{'':>12s}{row['bf16_ms'][name]:12.3f}")
    for name in ("other", "total"):
        lines.append(
            f"  {name:30s}{row['fused_ms'][name]:12.3f}{row['bf16_ms'][name]:12.3f}"
        )
    ceiling = row.get("pct_of_composite_ceiling")
    lines.append(
        f"  {'speedup':30s}{row['speedup']:11.2f}x  "
        f"({row['fused_tflops']:.0f} vs {row['bf16_tflops']:.0f} TF/s"
        + (f", {ceiling:.0f}% of composite ceiling)" if ceiling else ")")
    )
    lines.append(
        f"  {'refresh (per optim step)':30s}{row['refresh_ms']:12.3f}"
        f"   {100 * row['refresh_ms'] / row['fused_ms']['total']:.1f}% of a microbatch"
    )
    errs = " ".join(f"{k} {v:.2e}" for k, v in row["grad_rel_error"].items())
    lines.append(f"  rel error vs bf16: out {row['fwd_rel_error']:.2e}  {errs}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/bench/moe/moe_fp8_speed")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    args = parser.parse_args()

    triad = stream_bandwidths()["triad"]
    if triad < TRIAD_FLOOR * TRIAD_PEAK_GBPS:
        raise RuntimeError(
            f"triad {triad:.0f} GB/s is {100 * triad / TRIAD_PEAK_GBPS:.1f}% of the "
            f"{TRIAD_PEAK_GBPS:.0f} GB/s this box peaks at: the card is contended, so "
            "nothing measured here would be a measurement"
        )
    print(
        f"{device_name()} | CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} | triad {triad:.0f} GB/s",
        flush=True,
    )

    ceilings = dense_ceilings_profiled()
    # The composite the fused path is actually bounded by: FPROP and DGRAD run in fp8
    # and WGRAD does not, so neither dense ceiling alone is the right denominator. Three
    # equal-FLOP products, two at the fp8 rate and one at the 16-bit rate.
    composite = 3 / (2 / ceilings["mxfp8"]["tflops"] + 1 / ceilings["bf16"]["tflops"])
    ceilings["composite_2fp8_1bf16_tflops"] = composite
    print(
        f"in-process dense ceilings: bf16 {ceilings['bf16']['tflops']:.0f} TF/s "
        f"(host {100 * ceilings['bf16']['host_share']:.0f}%), "
        f"MXFP8 {ceilings['mxfp8']['tflops']:.0f} TF/s "
        f"(host {100 * ceilings['mxfp8']['host_share']:.0f}%) "
        f"-> composite {composite:.0f} TF/s",
        flush=True,
    )

    rows = []
    for preset in PRESETS:
        row = preset_row(preset, tokens=args.tokens)
        row["pct_of_composite_ceiling"] = 100 * row["fused_tflops"] / composite
        rows.append(row)
        print(format_row(row), flush=True)
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "moe_fp8_speed.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "device": device_name(),
                # `device_name()` reads visible index 0 and cannot identify a physical
                # card, so the visible-device string is what makes the row provenanced.
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                "triad_gbps": triad,
                "peak_bandwidth_gbps": cached_peak_bandwidth(),
                "tokens": args.tokens,
                "iters": ITERS,
                "ceilings": ceilings,
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
