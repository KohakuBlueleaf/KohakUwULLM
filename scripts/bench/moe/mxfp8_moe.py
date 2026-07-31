"""Fused MXFP8 expert path against the bf16 path it replaces, speed and accuracy.

Two things are being measured and they need different denominators, so the table
carries both:

* the **grouped MXFP8 GEMM** on its own, against whichever roofline binds. Which
  one that is flips with rows per expert -- at 384 rows the weight bank dominates
  the working set and bandwidth binds; by ~700 rows the arithmetic per weight byte
  has doubled and compute does. The compute ceiling quoted is the 692 TF/s this
  card *demonstrates* on dense vendor MXFP8, not the 838 TF/s theoretical peak
  nothing on it has reached.
* the **whole expert path**, against the bf16 path in ``MoEMLP._expert_compute``
  plus its gather and combine. That comparison is the deliverable: fp8 that is
  fast and wrong is not a result, so every row carries the relative error against
  the bf16 path in the same row it reports TF/s.
* the **launch count** of both paths, forward and forward+backward. It is reported
  beside the times because it is the claimed *mechanism*: the full model is ~93%
  host time, so an op that disappears is worth more than an op that gets faster,
  and a speedup with an unchanged launch count would mean the explanation is wrong
  even if the number is right. Measured with the profiler rather than counted from
  the source, since a `torch.zeros` beside a fused epilogue is still a dispatch.

Rows per expert is ``tokens * top_k / experts`` and it is the axis that decides
everything here, so each preset is measured at its real ratio and at a
gradient-accumulation-sized microbatch, where the same kernel is
weight-bandwidth-bound and has nothing left to give.

Every row carries the contended fraction of the samples its own median came from,
and a row above the limit is flagged ``suspect``. The rule for a shared card is
that the row is discarded and re-measured, so the run has to be able to state
which rows those are -- and ``CUDA_VISIBLE_DEVICES`` goes in the JSON because
``device_name()`` is byte-identical across this box's four cards, whose sustained
clocks are not.

**``wall - host`` is the primary timer, not ``graph_ms``.** A graph replay charges no
dispatch, which is exactly what a kernel benchmark wants, but it also cannot have an
L2 flush spliced into it -- and GEMM2's whole weight bank is 81 MiB against this
card's 96 MiB L2, so a replay reads it out of cache and reports 11% more than the
same kernel reaches on a cold bank. A real MoE layer touches each expert matrix once
per layer, cold. So the L2-flushed measurement is the honest one and ``graph_ms``
runs beside it as a cross-check; a row where they disagree by more than a quarter is
flagged rather than averaged.

**The fwd+bwd rows predate the WGRAD operand change and are stale.** The backward now
contracts the 16-bit activation instead of its fp8 copy -- a loss decision, made because
the fp8 operand cost 0.21 nats at MoE-1B-A280M -- which widens the WGRAD's B reads from
1 byte per element to 2 and adds one elementwise pass to rebuild ``h``. Whole-path rows
(the 2.30x at MoE-8B-A1B, 2.36x at MoE-3B-A500M) therefore describe a backward this repo
no longer runs and want re-measuring. The ``_launch_wgrad`` rows below are unaffected:
they compare **dw buffer dtypes** and still pass a scale plane, so they still time the
fp8-operand path on purpose.

Usage:
    .venv/bin/python scripts/bench/moe/mxfp8_moe.py --out out/bench/moe/mxfp8_moe
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from kohakuwullm.bench.analysis.admissibility import (
    contention_notes,
    median_and_contention,
    net_and_graph,
    suspect_speedup,
    visible_devices,
)
from kohakuwullm.bench.core.timing import (
    bench_peak_memory,
    cached_peak_bandwidth,
    cpu_enqueue_ms,
    device_name,
    device_op_counts,
    graph_ms,
    rel_error,
    stream_bandwidths,
)
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed
from kohakuwullm.kernels.mxfp8 import quantize_mx
from kohakuwullm.kernels.mxfp8.grouped import BLOCK_SCALE, grouped_mxfp8_gemm
from kohakuwullm.kernels.mxfp8.moe import (
    MXFP8ExpertWeights,
    _launch_wgrad,
    mxfp8_moe_experts,
)
from kohakuwullm.models.presets import get_preset

# The rate this card reaches on *dense* vendor MXFP8 at 4096^3, measured. Used as
# the compute ceiling in preference to the 838 TF/s theoretical fp8 peak: a
# fraction of a rate nothing achieves is not a useful number, and the theoretical
# one is kept only as an impossibility bound.
FP8_DEMONSTRATED_TFLOPS = 692.0
FP8_PEAK_TFLOPS = 838.0

PRESETS = ("MoE-1B-A280M", "MoE-2B-A370M", "MoE-3B-A500M", "MoE-8B-A1B")


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


def cast_row(preset, tokens, dtype=torch.bfloat16):
    """What folding the weight-gradient cast into the WGRAD epilogue saved.

    Both arms in **one process on one card**, because this is a 10-20% effect and the
    card-to-card offset on this box is ~1% -- comparing against a stored pre-fold run
    would put a systematic bias inside the thing being measured. Isolated to the two
    WGRAD launches rather than read off the whole step, since that is exactly what
    changed and the step carries four other kernels' noise.

    The L2 boundary is the reason this is a per-preset row and not one number. A
    saved round trip is only DRAM traffic if the buffer misses L2: ``dw_out`` at
    MoE-1B-A280M is 64 MiB of fp32 against a 96 MiB L2, so its cast is served from
    cache and the saving should compress there while ``dw_in`` (128 MiB) still pays.
    """
    cfg = get_preset(preset)
    dim, experts, top_k = cfg.dim, cfg.moe_num_experts, cfg.moe_top_k
    hidden = cfg.moe_hidden
    offsets, order, token_of, gate_f32 = routing(tokens, top_k, experts)
    gate = gate_f32.to(dtype)
    m = token_of.numel()
    torch.manual_seed(0)

    dout = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    hq, hs = quantize_mx(torch.randn(m, hidden, device="cuda", dtype=dtype))
    dpre = torch.randn(m, 2 * hidden, device="cuda", dtype=dtype)
    xq, xs = quantize_mx(torch.randn(tokens, dim, device="cuda", dtype=dtype))

    def run(buffer_dtype, cast: bool):
        dw_out = torch.empty(experts, dim, hidden, device="cuda", dtype=buffer_dtype)
        _launch_wgrad(
            dout,
            hq,
            hs,
            token_of,
            gate,
            order,
            dw_out,
            offsets,
            dim,
            hidden,
            True,
            False,
        )
        dw_in = torch.empty(experts, 2 * hidden, dim, device="cuda", dtype=buffer_dtype)
        _launch_wgrad(
            dpre,
            xq,
            xs,
            token_of,
            gate,
            order,
            dw_in,
            offsets,
            2 * hidden,
            dim,
            False,
            True,
        )
        if cast:
            return dw_in.to(dtype), dw_out.to(dtype)
        return dw_in, dw_out

    folded = lambda: run(dtype, cast=False)  # noqa: E731
    legacy = lambda: run(torch.float32, cast=True)  # noqa: E731
    # Bit-equality is asserted here too, not only in the test suite: a benchmark that
    # reports a saving without checking the two arms still agree is how a "win" that
    # is really a dropped rounding step gets published.
    with torch.no_grad():
        got, ref = folded(), legacy()
    identical = all(torch.equal(a, b) for a, b in zip(got, ref))

    folded_ms, folded_dirty, folded_power = median_and_contention(folded)
    legacy_ms, legacy_dirty, legacy_power = median_and_contention(legacy)
    saved_mib = (experts * (2 * hidden * dim + dim * hidden) * 2 * 4) / 2**20
    row = {
        "preset": preset,
        "tokens": tokens,
        "dw_in_fp32_mib": experts * 2 * hidden * dim * 4 / 2**20,
        "dw_out_fp32_mib": experts * dim * hidden * 4 / 2**20,
        "dw_out_fits_l2": experts * dim * hidden * 4 < _l2_bytes(),
        "traffic_saved_mib": saved_mib,
        "folded_ms": folded_ms,
        "legacy_ms": legacy_ms,
        "speedup": legacy_ms / folded_ms if folded_ms else float("nan"),
        "saved_ms": legacy_ms - folded_ms,
        # What the saving would be if the removed traffic streamed at the measured
        # peak. A measured saving far below this means the cast was served by L2.
        "saved_ms_at_peak_bw": saved_mib
        * 2**20
        / (cached_peak_bandwidth() * 1e9)
        * 1e3,
        "bit_identical": identical,
        "contended_folded": folded_dirty,
        "contended_legacy": legacy_dirty,
        "contention_certifiable": bool(folded_power and legacy_power),
    }
    notes = []
    if not identical:
        notes.append("arms disagree bitwise: the saving is a precision change")
    notes += contention_notes(
        row, ("contended_folded", "contended_legacy"), row["contention_certifiable"]
    )
    row["suspect"] = "; ".join(notes)
    return row


def gemm_row(name, rows_per_expert, experts, n, k, dtype=torch.bfloat16):
    """One bare grouped-GEMM measurement with both rooflines resolved."""
    torch.manual_seed(0)
    total = rows_per_expert * experts
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = (
        torch.full((experts,), rows_per_expert, device="cuda").cumsum(0).to(torch.int32)
    )
    x = torch.randn(total, k, device="cuda", dtype=dtype)
    w = torch.randn(experts, n, k, device="cuda", dtype=dtype) * k**-0.5
    xq, xs = quantize_mx(x)
    wq, ws = quantize_mx(w.reshape(experts * n, k))
    wq, ws = wq.view(experts, n, k), ws.view(experts, n, k // BLOCK_SCALE)

    fp8 = lambda: grouped_mxfp8_gemm(xq, xs, wq, ws, offsets)  # noqa: E731
    bf16 = lambda: grouped_gemm(x, w, offsets)  # noqa: E731
    flops = 2.0 * total * n * k
    weight_bank = experts * n * k * (1 + 1 / BLOCK_SCALE)
    footprint = weight_bank + total * k * (1 + 1 / BLOCK_SCALE) + total * n * 2
    bw_ceiling = flops / (footprint / (cached_peak_bandwidth() * 1e9)) / 1e12
    ceiling = min(FP8_DEMONSTRATED_TFLOPS, bw_ceiling)
    fp8_ms, fp8_graph, fp8_dirty, fp8_power = net_and_graph(fp8)
    bf16_ms, _, bf16_dirty, bf16_power = net_and_graph(bf16)
    tf = flops / (fp8_ms / 1e3) / 1e12
    row = {
        "name": name,
        "rows_per_expert": rows_per_expert,
        "experts": experts,
        "N": n,
        "K": k,
        "footprint_mib": footprint / 2**20,
        "weight_bank_mib": weight_bank / 2**20,
        "fits_l2": weight_bank < _l2_bytes(),
        "fp8_ms": fp8_ms,
        "fp8_graph_ms": fp8_graph,
        "bf16_ms": bf16_ms,
        "fp8_tflops": tf,
        "fp8_graph_tflops": flops / (fp8_graph / 1e3) / 1e12,
        "bf16_tflops": flops / (bf16_ms / 1e3) / 1e12,
        "ceiling_tflops": ceiling,
        "ceiling_kind": (
            "bandwidth" if bw_ceiling < FP8_DEMONSTRATED_TFLOPS else "compute"
        ),
        # Both rooflines, not only the binding one: which one binds flips inside this
        # sweep, and a table that reports only the minimum makes the crossover look
        # like a discontinuity in the kernel rather than a change of denominator.
        "bandwidth_ceiling_tflops": bw_ceiling,
        "compute_ceiling_tflops": FP8_DEMONSTRATED_TFLOPS,
        "pct_of_ceiling": 100.0 * tf / ceiling,
        "pct_of_bandwidth_ceiling": 100.0 * tf / bw_ceiling,
        "speedup": bf16_ms / fp8_ms,
        "contended_fp8": fp8_dirty,
        "contended_bf16": bf16_dirty,
        "contention_certifiable": bool(fp8_power and bf16_power),
        "rel_err": rel_error(fp8().float(), bf16().float()),
    }
    row["suspect"] = _suspect_gemm(row)
    row["suspect_speedup"] = suspect_speedup(row, ("contended_bf16",))
    return row


def path_row(preset, tokens, dtype=torch.bfloat16):
    """One whole-expert-path measurement: fused MXFP8 against the bf16 path."""
    cfg = get_preset(preset)
    dim, experts, top_k = cfg.dim, cfg.moe_num_experts, cfg.moe_top_k
    hidden = cfg.moe_hidden
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
        err = rel_error(fused().float(), eager().float())

    flops_fwd = 6.0 * m * dim * hidden
    # Grad stays enabled for the forward rows: the fused kernel skips the
    # pre-activation store when nothing needs it, so a no_grad forward measures a
    # different kernel than the one a training step runs.
    fwd_wall_fp8, fwd_dirty_fp8, fwd_power_fp8 = median_and_contention(fused)
    fwd_host_fp8 = cpu_enqueue_ms(fused)
    fwd_wall_bf16, fwd_dirty_bf16, fwd_power_bf16 = median_and_contention(eager)
    fwd_host_bf16 = cpu_enqueue_ms(eager)
    fwd_fp8 = fwd_wall_fp8 - fwd_host_fp8
    fwd_bf16 = fwd_wall_bf16 - fwd_host_bf16

    dout = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    inputs = [x, w_in, w_out, gate]

    def step(fn):
        return torch.autograd.grad(fn(), inputs, dout)

    full_fp8, dirty_fp8, power_fp8 = median_and_contention(lambda: step(fused))
    full_bf16, dirty_bf16, power_bf16 = median_and_contention(lambda: step(eager))
    host_fp8 = cpu_enqueue_ms(lambda: step(fused))
    host_bf16 = cpu_enqueue_ms(lambda: step(eager))
    net_fp8, net_bf16 = full_fp8 - host_fp8, full_bf16 - host_bf16

    # The launch count is the mechanism, so it is measured rather than counted by
    # reading the source: an epilogue that fuses on paper still costs a dispatch if
    # a `torch.zeros` or a stride-fixing copy slipped in beside it.
    launch_fwd_fp8 = device_op_counts(fused)
    launch_fwd_bf16 = device_op_counts(eager)
    launch_step_fp8 = device_op_counts(lambda: step(fused))
    launch_step_bf16 = device_op_counts(lambda: step(eager))

    # A graph replay beside the flushed measurement, because at small token counts
    # dispatch is most of the wall and `wall - host` stops being evidence -- the
    # first clean run of this had four of ten path rows disqualified that way. The
    # replay's warm L2 would invalidate a *bandwidth* claim, but both arms read the
    # same weight bank through the same cache, so the ratio survives it.
    fwd_graph_fp8 = graph_ms(fused, iters=20)
    fwd_graph_bf16 = graph_ms(eager, iters=20)

    # Peak memory carries the part of the story the timers cannot. The weight-gradient
    # buffers dominate the backward's transient footprint, and a row disqualified for
    # host share still reports this honestly -- an allocation is not a measurement of
    # elapsed time, so nothing about dispatch noise touches it.
    peak_fp8 = bench_peak_memory(lambda: step(fused))
    peak_bf16 = bench_peak_memory(lambda: step(eager))

    row = {
        "preset": preset,
        "tokens": tokens,
        "dim": dim,
        "experts": experts,
        "top_k": top_k,
        "hidden": hidden,
        "rows_per_expert": m / experts,
        "fwd_fp8_ms": fwd_fp8,
        "fwd_bf16_ms": fwd_bf16,
        "fwd_speedup": fwd_bf16 / fwd_fp8,
        "fwd_graph_fp8_ms": fwd_graph_fp8,
        "fwd_graph_bf16_ms": fwd_graph_bf16,
        "fwd_graph_speedup": fwd_graph_bf16 / fwd_graph_fp8,
        "fwd_graph_fp8_tflops": flops_fwd / (fwd_graph_fp8 / 1e3) / 1e12,
        "fwd_fp8_tflops": flops_fwd / (fwd_fp8 / 1e3) / 1e12,
        "fwd_bf16_tflops": flops_fwd / (fwd_bf16 / 1e3) / 1e12,
        "step_fp8_net_ms": net_fp8,
        "step_bf16_net_ms": net_bf16,
        "step_speedup": net_bf16 / net_fp8,
        "step_fp8_tflops": 3.0 * flops_fwd / (net_fp8 / 1e3) / 1e12,
        "host_share_fp8": host_fp8 / full_fp8,
        "host_share_bf16": host_bf16 / full_bf16,
        "host_share_fwd_fp8": fwd_host_fp8 / fwd_wall_fp8,
        "host_share_fwd_bf16": fwd_host_bf16 / fwd_wall_bf16,
        "step_peak_fp8_gib": peak_fp8,
        "step_peak_bf16_gib": peak_bf16,
        "step_peak_ratio": peak_bf16 / peak_fp8 if peak_fp8 else float("nan"),
        "launch_fwd_fp8": launch_fwd_fp8,
        "launch_fwd_bf16": launch_fwd_bf16,
        "launch_step_fp8": launch_step_fp8,
        "launch_step_bf16": launch_step_bf16,
        "contended_fwd_fp8": fwd_dirty_fp8,
        "contended_fwd_bf16": fwd_dirty_bf16,
        "contended_step_fp8": dirty_fp8,
        "contended_step_bf16": dirty_bf16,
        "contention_certifiable": bool(
            fwd_power_fp8 and fwd_power_bf16 and power_fp8 and power_bf16
        ),
        "rel_err_vs_bf16": err,
    }
    row["suspect"] = _suspect(row)
    row["suspect_speedup"] = suspect_speedup(
        row,
        (
            "contended_fwd_bf16",
            "contended_step_bf16",
            "host_share_bf16",
            "host_share_fwd_bf16",
        ),
    )
    return row


def _l2_bytes() -> int:
    return torch.cuda.get_device_properties(0).L2_cache_size


def launch_label(counts: dict[str, int]) -> str:
    """``kernel+memory`` -- the memset behind a scatter-add is a dispatch too."""
    return f"{counts['kernel']}+{counts['memory']}"


def _suspect_gemm(row: dict) -> str:
    """Impossibilities and cache artifacts in one bare-GEMM row.

    A **percentage of a ceiling above 100 is checked here, not left to the reader**:
    the first run of this benchmark printed 184% of a bandwidth roofline and it was
    caught by eye. The cause was a `cached_peak_bandwidth` measured while a stale
    background job held the card, which halved the denominator without touching the
    numerator -- so the check has to be on the ratio, not on either half.
    """
    notes = []
    if row["fp8_tflops"] > FP8_PEAK_TFLOPS:
        notes.append(
            f"{row['fp8_tflops']:.0f} TF/s above the {FP8_PEAK_TFLOPS} fp8 peak"
        )
    if row["pct_of_ceiling"] > 100.0:
        notes.append(
            f"{row['pct_of_ceiling']:.0f}% of the {row['ceiling_kind']} ceiling"
        )
    gap = abs(row["fp8_graph_tflops"] - row["fp8_tflops"]) / max(
        row["fp8_tflops"], 1e-9
    )
    if gap > 0.25:
        notes.append(f"timers disagree {gap:.0%} (graph {row['fp8_graph_tflops']:.0f})")
    notes += contention_notes(row, ("contended_fp8",), row["contention_certifiable"])
    return "; ".join(notes)


def _suspect(row: dict) -> str:
    """Every impossibility a path row can state, checked before it is printed.

    Three separate results in this repo's history were physically impossible and
    were caught only by a person noticing. A ``wall - host`` estimate past a 50% host
    share is the specific failure that produced two of them.

    A contended sample set is flagged here rather than left to the run log, because
    the rule for one is that it is **discarded and re-measured**, not caveated -- and
    a verdict nobody reads cannot enforce that.
    """
    notes = []
    for key in ("fwd_fp8_tflops", "step_fp8_tflops", "fwd_graph_fp8_tflops"):
        if row[key] > FP8_PEAK_TFLOPS:
            notes.append(f"{key}={row[key]:.0f} above the {FP8_PEAK_TFLOPS} fp8 peak")
    for key in ("host_share_fp8", "host_share_fwd_fp8"):
        if row[key] > 0.5:
            notes.append(f"{key}={row[key]:.0%}: net_ms is not evidence")
    notes += contention_notes(
        row,
        ("contended_fwd_fp8", "contended_step_fp8"),
        row["contention_certifiable"],
    )
    return "; ".join(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--tokens", type=int, default=8192)
    args = parser.parse_args()

    bandwidth = stream_bandwidths()
    print(f"{device_name()}  CUDA_VISIBLE_DEVICES={visible_devices()}")
    print("STREAM GB/s " + "  ".join(f"{k} {v:.0f}" for k, v in bandwidth.items()))
    if max(bandwidth.values()) < 1700:
        print("!! below 1700 GB/s: the card is shared and every row below is worthless")

    gemms = []
    cfg = get_preset("MoE-8B-A1B")
    # 512 and 768 are the interior of the regime a real microbatch lands in, so the
    # sweep has to contain them rather than bracket them: the binding roofline flips
    # from bandwidth to compute inside this range, and an interpolation across the
    # crossover is not a measurement of either side of it.
    for rows in (192, 384, 512, 682, 768, 1364):
        for name, n, k in (
            ("gemm1", 2 * cfg.moe_hidden, cfg.dim),
            ("gemm2", cfg.dim, cfg.moe_hidden),
        ):
            gemms.append(gemm_row(name, rows, cfg.moe_num_experts, n, k))
            torch.cuda.empty_cache()
    print(
        f"\n{'grouped MXFP8 GEMM':>22}{'rows/E':>8}{'MiB':>6}{'TF/s':>8}{'graph':>7}"
        f"{'ceil':>7}{'%':>5}{'bind':>11}{'bwceil':>8}{'bw%':>5}"
        f"{'vs bf16':>9}{'relerr':>9}"
    )
    for r in gemms:
        label = f"{r['name']} {r['N']}x{r['K']}"
        print(
            f"{label:>22}{r['rows_per_expert']:>8}"
            f"{r['footprint_mib']:>6.0f}{r['fp8_tflops']:>8.0f}"
            f"{r['fp8_graph_tflops']:>7.0f}"
            f"{r['ceiling_tflops']:>7.0f}{r['pct_of_ceiling']:>5.0f}"
            f"{r['ceiling_kind']:>11}{r['bandwidth_ceiling_tflops']:>8.0f}"
            f"{r['pct_of_bandwidth_ceiling']:>5.0f}"
            f"{r['speedup']:>8.2f}x{r['rel_err']:>9.4f}"
        )
        for field, tag in (("suspect", "SUSPECT"), ("suspect_speedup", "RATIO ONLY")):
            if r[field]:
                print(f"{'':>22}  {tag}: {r[field]}")

    casts = []
    for preset in PRESETS:
        casts.append(cast_row(preset, args.tokens))
        torch.cuda.empty_cache()
    print(
        f"\n{'WGRAD dw cast':>14}{'dw_in':>8}{'dw_out':>8}{'L2?':>5}{'saved':>8}"
        f"{'folded':>9}{'legacy':>9}{'x':>7}{'saved':>9}{'at peak':>9}{'bits':>6}"
    )
    for r in casts:
        print(
            f"{r['preset'].replace('MoE-', ''):>14}{r['dw_in_fp32_mib']:>7.0f}M"
            f"{r['dw_out_fp32_mib']:>7.0f}M{'yes' if r['dw_out_fits_l2'] else 'no':>5}"
            f"{r['traffic_saved_mib']:>7.0f}M{r['folded_ms']:>9.3f}{r['legacy_ms']:>9.3f}"
            f"{r['speedup']:>6.2f}x{r['saved_ms']:>9.3f}{r['saved_ms_at_peak_bw']:>9.3f}"
            f"{'ok' if r['bit_identical'] else 'DIFF':>6}"
        )
        if r["suspect"]:
            print(f"{'':>14}  SUSPECT: {r['suspect']}")

    paths = []
    for preset, tokens in [(p, args.tokens) for p in PRESETS] + [
        ("MoE-8B-A1B", args.tokens // 8)
    ]:
        paths.append(path_row(preset, tokens))
        torch.cuda.empty_cache()
    print(
        f"\n{'expert path':>14}{'tok':>6}{'rows/E':>8}{'fwd TF/s':>10}{'fwd x':>7}"
        f"{'gph TF/s':>10}{'gph x':>7}{'step TF/s':>11}{'step x':>8}"
        f"{'host%':>7}{'relerr':>9}"
    )
    for r in paths:
        print(
            f"{r['preset'].replace('MoE-', ''):>14}{r['tokens']:>6}"
            f"{r['rows_per_expert']:>8.0f}{r['fwd_fp8_tflops']:>10.0f}"
            f"{r['fwd_speedup']:>6.2f}x{r['fwd_graph_fp8_tflops']:>10.0f}"
            f"{r['fwd_graph_speedup']:>6.2f}x{r['step_fp8_tflops']:>11.0f}"
            f"{r['step_speedup']:>7.2f}x{100 * r['host_share_fp8']:>7.0f}"
            f"{r['rel_err_vs_bf16']:>9.4f}"
        )
        for field, tag in (("suspect", "SUSPECT"), ("suspect_speedup", "RATIO ONLY")):
            if r[field]:
                print(f"{'':>14}  {tag}: {r[field]}")

    print(
        f"\n{'launches / layer':>18}{'tok':>6}{'fwd fp8':>10}{'fwd bf16':>10}"
        f"{'step fp8':>10}{'step bf16':>11}{'peak fp8':>10}{'peak bf16':>11}{'x':>7}"
    )
    for r in paths:
        print(
            f"{r['preset'].replace('MoE-', ''):>18}{r['tokens']:>6}"
            + "".join(
                f"{launch_label(r[key]):>{width}}"
                for key, width in (
                    ("launch_fwd_fp8", 10),
                    ("launch_fwd_bf16", 10),
                    ("launch_step_fp8", 10),
                    ("launch_step_bf16", 11),
                )
            )
            + f"{r['step_peak_fp8_gib']:>10.2f}{r['step_peak_bf16_gib']:>11.2f}"
            + f"{r['step_peak_ratio']:>6.2f}x"
        )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "mxfp8_moe.json"), "w") as fh:
            json.dump(
                {
                    "device": device_name(),
                    "visible_devices": visible_devices(),
                    "bandwidth": bandwidth,
                    "peak_bandwidth_gbps": cached_peak_bandwidth(),
                    "gemms": gemms,
                    "casts": casts,
                    "paths": paths,
                },
                fh,
                indent=2,
            )


if __name__ == "__main__":
    main()
