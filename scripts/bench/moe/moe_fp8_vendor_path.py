"""The whole MoE expert path in vendor MXFP8, against the bf16 path it would replace.

``moe_fp8_vendor.py`` answers the GEMM question. This answers the shipping question,
which is not the same one: a path that wins on both GEMMs can still lose overall,
because the fp8 arm has to **quantize the SwiGLU output** between them -- a full extra
pass over the ``(rows, hidden)`` intermediate that the bf16 arm never makes. That cost
scales with rows and the GEMM saving scales with rows*hidden, so which way it goes is a
question about the preset, not a matter of principle.

Both arms run the identical gather, SwiGLU and combine, using the same ops. The only
difference is the two GEMMs and the quantize between them, which is exactly the change
under evaluation; matching everything else is what makes the difference attributable.

Three fp8 arms, because the answer differs by construction and the middle one is the
finding:

* ``fp8_seq`` -- per-expert chain, one stream. What "the boring implementation" means
  literally, and it loses to bf16.
* ``fp8_streamed`` -- the same chain with one expert per stream, round-robin over 8.
  One expert's GEMM is far fewer CTAs than this card has SMs, so the sequential
  version leaves most of the machine idle regardless of dispatch cost.
* ``fp8_fused`` -- the repo's existing fused MXFP8 expert path, for context. It is the
  fast option and the one currently under repair; the point of the vendor arms is what
  is available *without* trusting it.

The bf16 arm is ``MoEMLP._expert_compute`` written out: gather, grouped GEMM, SwiGLU,
grouped GEMM, fused combine.

Primary timer is device time from a flushed graph replay -- see
``kohakuwullm.bench.vendor.vendor_moe`` for why ``wall - host`` cannot be used on a per-expert
loop. Eager wall time is reported beside it because "with and without CUDA graphs" is a
real question here and the answers are far apart.

Usage:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/bench/moe/moe_fp8_vendor_path.py \
        --out out/bench/moe/moe_fp8_vendor
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from kohakuwullm.bench.analysis.admissibility import (
    contention_notes,
    median_and_contention,
    visible_devices,
)
from kohakuwullm.bench.core.timing import (
    cpu_enqueue_ms,
    device_name,
    device_op_counts,
    rel_error,
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
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed
from kohakuwullm.kernels.mxfp8 import quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul_swizzled
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts
from kohakuwullm.models.presets import get_preset

FP8_PEAK_TFLOPS = 838.0
PRESETS = ("Kohaku-MoE-1B", "Kohaku-MoE-2B", "Kohaku-MoE-3B", "MoE-1B-A280M")


def routing(tokens: int, top_k: int, experts: int, device="cuda", seed=0):
    """A balanced assignment, plus the padded gather index the aligned layout needs.

    Balanced because that is the capacity-padded case the vendor loop is built for and
    the one that graph-captures. ``padded_token_of`` points alignment padding at token
    0: those rows are computed and discarded, which is cheaper than masking and does
    not change any real row's result.
    """
    torch.manual_seed(seed)
    pairs = tokens * top_k
    per_expert = pairs // experts
    expert_of = torch.arange(pairs, device=device) % experts
    order = expert_of.argsort(stable=True).to(torch.int32)
    token_of = torch.div(order, top_k, rounding_mode="floor").to(torch.int32)
    gate = torch.rand(pairs, device=device) * 0.5 + 0.5

    starts, counts, total, offsets = aligned_offsets(per_expert, experts, device)
    padded_token_of = torch.zeros(total, dtype=torch.int64, device=device)
    for e, (start, count) in enumerate(zip(starts, counts)):
        padded_token_of[start : start + count] = token_of[
            e * per_expert : e * per_expert + count
        ]
    return dict(
        pairs=pairs,
        per_expert=per_expert,
        order=order,
        token_of=token_of,
        gate=gate,
        starts=starts,
        counts=counts,
        total=total,
        offsets=offsets,
        padded_token_of=padded_token_of,
        unpadded_offsets=_plain_offsets(per_expert, experts, device),
    )


def _plain_offsets(per_expert: int, experts: int, device) -> torch.Tensor:
    """Unpadded expert boundaries -- what the bf16 grouped path actually runs on."""
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.arange(1, experts + 1, device=device, dtype=torch.int32) * (
        per_expert
    )
    return offsets


def build(preset: str, tokens: int, dtype=torch.bfloat16):
    """Weights, activations and routing for one preset at one token count."""
    cfg = get_preset(preset)
    dim, experts, top_k, hidden = (
        cfg.dim,
        cfg.moe_num_experts,
        cfg.moe_top_k,
        cfg.moe_hidden,
    )
    r = routing(tokens, top_k, experts)
    # The gate weight rides the same dtype as the rows it scales: `combine_routed`
    # multiplies them elementwise, and an fp32 gate would silently promote the whole
    # combine to fp32 and charge the comparison for it.
    r["gate"] = r["gate"].to(dtype)
    torch.manual_seed(0)
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5

    # The weight bank is quantized once per optimizer step in a real run, not per
    # layer call, so it is hoisted out of every timed region here. Charging the GEMM
    # for a refresh it does not do would understate the vendor path by more than the
    # effect being measured.
    w_in_q, w_in_s = quantize_mx_vendor(w_in.reshape(experts * 2 * hidden, dim))
    w_out_q, w_out_s = quantize_mx_vendor(w_out.reshape(experts * dim, hidden))
    return dict(
        preset=preset,
        cfg=cfg,
        dim=dim,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        tokens=tokens,
        dtype=dtype,
        x=x,
        w_in=w_in,
        w_out=w_out,
        w_in_q=w_in_q.view(experts, 2 * hidden, dim),
        w_in_views=scale_views(
            w_in_s,
            [e * 2 * hidden for e in range(experts)],
            [2 * hidden] * experts,
            dim,
        ),
        w_out_q=w_out_q.view(experts, dim, hidden),
        w_out_views=scale_views(
            w_out_s, [e * dim for e in range(experts)], [dim] * experts, hidden
        ),
        **r,
    )


def bf16_path(b: dict):
    """``MoEMLP._expert_compute`` plus its gather and combine, written out."""
    rows = b["x"].index_select(0, b["token_of"].long())
    h = grouped_gemm(rows, b["w_in"], b["unpadded_offsets"])
    gate_h, value = h.chunk(2, dim=-1)
    out_sorted = grouped_gemm(F.silu(gate_h) * value, b["w_out"], b["unpadded_offsets"])
    return combine_routed(out_sorted, b["gate"], b["order"], b["token_of"], b["tokens"])


def expert_jobs(b: dict, sink: torch.Tensor | None = None):
    """One closure per expert: GEMM1 -> SwiGLU -> quantize -> GEMM2, all in fp8.

    Bound per expert rather than closed over the loop variable. A ``lambda: f(e)`` in a
    comprehension captures the *name*, so every job would run the last expert -- and
    the result would still have the right shape and dtype, which is why this is written
    out rather than left to look obvious.

    With ``sink``, each job copies its result into that buffer's rows and returns
    nothing. This exists because ``F.scaled_mm`` has **no ``out=`` overload**, so a
    per-expert loop cannot write its GEMM2 result where the combine needs it and has to
    assemble one afterwards. ``torch.cat`` does that in a single kernel, but a single
    kernel at the end of the loop is 120 us of unoverlapped copy at the 1B preset; the
    same bytes moved as 64 small copies *inside* the streams hide behind other experts'
    GEMMs. Same traffic, different place on the timeline.
    """
    dtype = b["dtype"]
    jobs = []
    for e in range(b["experts"]):
        start, count = b["starts"][e], b["counts"][e]

        def job(
            e=e,
            rows=slice(start, start + count),
            w_in_q=b["w_in_q"][e],
            w_in_s=b["w_in_views"][e],
            w_out_q=b["w_out_q"][e],
            w_out_s=b["w_out_views"][e],
            dest=None if sink is None else sink[e * b["per_expert"] :][:count],
        ):
            h = vendor_mxfp8_matmul_swizzled(
                b["_xq"][rows], b["_x_views"][e], w_in_q, w_in_s, dtype
            )
            gate_h, value = h.chunk(2, dim=-1)
            swig = F.silu(gate_h) * value
            hq, hs = quantize_mx_vendor(swig)
            out = vendor_mxfp8_matmul_swizzled(hq, hs, w_out_q, w_out_s, dtype)
            if dest is None:
                return out
            dest.copy_(out)
            return None

        jobs.append(job)
    return jobs


def gather_and_quantize(b: dict):
    """Gather routed rows into the aligned buffer and quantize it -- two launches.

    Written as its own step because it is what the per-expert loop shares with the
    grouped kernel, and because it is the answer to "how many quantizes does this
    cost": one, for the whole batch, not one per expert. The alignment rule is what
    buys that.
    """
    rows = b["x"].index_select(0, b["padded_token_of"])
    return quantize_mx_vendor(rows)


def expert_outputs(b: dict, runner) -> torch.Tensor:
    """Gather, quantize, and run every expert through ``runner`` -- everything but the
    combine.

    Split from :func:`fp8_path` because this is the tensor a sequential-vs-streamed
    equality check has to be taken on. ``combine_routed`` is a scatter-add and its
    float atomics do not commute, so it is **nondeterministic**: measured here, the
    bf16 path disagrees with *itself* by 0.0625 absolute across two calls. Checking the
    combined output would therefore report an aliasing bug on every run and prove
    nothing on any of them.
    """
    b["_xq"], b["_xs"] = gather_and_quantize(b)
    b["_x_views"] = scale_views(b["_xs"], b["starts"], b["counts"], b["dim"])
    parts = runner()
    # A sink-writing runner returns Nones and has already assembled the buffer.
    return b["_sink"] if parts[0] is None else torch.cat(parts, dim=0)


def fp8_path(b: dict, runner):
    """Gather, quantize, run every expert through ``runner``, combine."""
    return combine_routed(
        expert_outputs(b, runner), b["gate"], b["order"], b["token_of"], b["tokens"]
    )


def arms(b: dict, streams: int = 8):
    """Every path callable, sharing weights so no arm pays another's setup."""
    # The views the jobs close over are rebuilt by `fp8_path` on each call, so the
    # closures must read them through `b` rather than capture them here.
    b["_xq"], b["_xs"] = gather_and_quantize(b)
    b["_x_views"] = scale_views(b["_xs"], b["starts"], b["counts"], b["dim"])
    jobs = expert_jobs(b)
    streamed = StreamedLoop(jobs, streams)
    # The sink is allocated once, outside every timed region, exactly as a real layer
    # would hold it: allocating it per call would charge this arm for an allocation the
    # `torch.cat` arm makes too, and hide the only difference being measured.
    b["_sink"] = torch.empty(b["pairs"], b["dim"], device="cuda", dtype=b["dtype"])
    sink_streamed = StreamedLoop(expert_jobs(b, sink=b["_sink"]), streams)
    packed = MXFP8ExpertWeights(b["w_in"].detach(), b["w_out"].detach())

    def fp8_fused():
        return mxfp8_moe_experts(
            b["x"],
            b["w_in"],
            b["w_out"],
            b["gate"].to(b["dtype"]),
            b["token_of"],
            b["order"],
            b["unpadded_offsets"],
            packed,
        )

    def sequential():
        return [job() for job in jobs]

    return {
        "bf16": lambda: bf16_path(b),
        "fp8_seq": lambda: fp8_path(b, sequential),
        "fp8_streamed": lambda: fp8_path(b, streamed),
        "fp8_sink": lambda: fp8_path(b, sink_streamed),
        "fp8_fused": fp8_fused,
        # Handed back so the caller can take its equality check before the combine,
        # rather than re-deriving the runners and risking a different closure.
        "_runners": {"seq": sequential, "streamed": streamed, "sink": sink_streamed},
    }


def row(b: dict, streams: int, floor_ms: float, ceilings: dict) -> dict:
    """One preset at one token count, every arm, timed and checked."""
    fns = arms(b, streams=streams)
    m = b["pairs"]
    # 6*m*dim*hidden: two GEMMs, one (dim x 2*hidden) and one (hidden x dim), so
    # 2*m*dim*2*hidden + 2*m*hidden*dim.
    flops = 6.0 * m * b["dim"] * b["hidden"]

    runners = fns.pop("_runners")
    with torch.no_grad():
        reference = fns["bf16"]()
        # The combine's own run-to-run spread, from two calls of the *same* arm. It is
        # the floor any fp8-vs-bf16 error has to be read against: without it, the fp8
        # column looks like a precision result when part of it is scatter-add noise.
        reference_twice = fns["bf16"]()
        seq_out = fns["fp8_seq"]()
        seq_parts = expert_outputs(b, runners["seq"])
        streamed_parts = expert_outputs(b, runners["streamed"])
        sink_parts = expert_outputs(b, runners["sink"]).clone()
    torch.cuda.synchronize()

    out: dict = dict(
        preset=b["preset"],
        tokens=b["tokens"],
        dim=b["dim"],
        hidden=b["hidden"],
        experts=b["experts"],
        top_k=b["top_k"],
        rows_per_expert=b["per_expert"],
        padding_frac=1.0 - m / b["total"],
        gflop=flops / 1e9,
        streams=streams,
        # Taken on the expert outputs, before the combine. Concurrency must not change
        # arithmetic -- every stream reads disjoint rows and writes a disjoint output --
        # but `combine_routed` is a nondeterministic scatter-add, so the same check on
        # the combined result fires every run and means nothing.
        streamed_matches_sequential=bool(torch.equal(streamed_parts, seq_parts)),
        # The sink arm reorders *where* results are written, not what they are. Checked
        # separately because a slice-offset error there would land every expert's rows
        # in the wrong place and still produce a full, plausible buffer.
        sink_matches_sequential=bool(torch.equal(sink_parts, seq_parts)),
        bf16_self_rel_err=rel_error(reference_twice.float(), reference.float()),
        fp8_rel_err_vs_bf16=rel_error(seq_out.float(), reference.float()),
        fp8_ulp_vs_bf16=ulp_error(
            seq_out.float(), reference.float(), torch.float8_e4m3fn, "rms"
        ),
    )

    for label, fn in fns.items():
        wall, contended, power = median_and_contention(fn)
        host = cpu_enqueue_ms(fn)
        dev, dev_contended, dev_power = device_time(fn)
        out[f"{label}_wall_ms"] = wall
        out[f"{label}_host_ms"] = host
        out[f"{label}_device_ms"] = dev
        out[f"{label}_host_share"] = host / wall if wall else float("nan")
        out[f"{label}_contended"] = contended
        out[f"{label}_device_contended"] = dev_contended
        out[f"{label}_power"] = power and dev_power
        out[f"{label}_capturable"] = dev == dev
        out[f"{label}_floor_frac"] = (
            floor_ms / dev if dev and dev == dev else float("nan")
        )
        out[f"{label}_launches"] = device_op_counts(fn)
        for stem in ("wall", "device"):
            ms = out[f"{label}_{stem}_ms"]
            out[f"{label}_{stem}_tflops"] = flops / (ms / 1e3) / 1e12 if ms > 0 else 0.0
        torch.cuda.empty_cache()

    for stem in ("wall", "device"):
        base = out[f"bf16_{stem}_ms"]
        for arm in ("fp8_seq", "fp8_streamed", "fp8_sink", "fp8_fused"):
            got = out[f"{arm}_{stem}_ms"]
            out[f"{arm}_speedup_{stem}"] = base / got if got > 0 else float("nan")
    out["suspect"] = _suspect(out, fns, ceilings)
    return out


def _suspect(r: dict, fns: dict, ceilings: dict) -> str:
    """What would make this row inadmissible, checked before it is printed."""
    notes = []
    for label in fns:
        if not r[f"{label}_capturable"]:
            notes.append(f"{label} did not capture: no device time for it")
            continue
        tf = r[f"{label}_device_tflops"]
        kind = "fp8" if "fp8" in label else "bf16"
        if kind == "fp8" and tf > FP8_PEAK_TFLOPS:
            notes.append(f"{label}_device={tf:.0f} TF/s above the fp8 hardware peak")
        elif tf > 1.1 * ceilings.get(kind, float("inf")):
            notes.append(
                f"{label}_device={tf:.0f} TF/s >10% above this card's dense {kind} "
                f"rate ({ceilings[kind]:.0f})"
            )
        if r[f"{label}_floor_frac"] > 0.2:
            notes.append(
                f"{label} is {r[f'{label}_floor_frac']:.0%} replay floor: understated"
            )
    certifiable = all(r[f"{label}_power"] for label in fns)
    notes += contention_notes(
        r, tuple(f"{label}_device_contended" for label in fns), certifiable
    )
    if not r["streamed_matches_sequential"]:
        notes.append("streamed output differs from sequential: the fan-out aliased")
    if not r["sink_matches_sequential"]:
        notes.append("sink output differs from sequential: its row offsets are wrong")
    return "; ".join(notes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--presets", nargs="+", default=list(PRESETS))
    parser.add_argument("--streams", type=int, default=8)
    args = parser.parse_args()

    bandwidth = stream_bandwidths()
    print(f"{device_name()}  CUDA_VISIBLE_DEVICES={visible_devices()}")
    print("STREAM GB/s " + "  ".join(f"{k} {v:.0f}" for k, v in bandwidth.items()))
    if max(bandwidth.values()) < 1700:
        print("!! below 1700 GB/s: the card is shared and every row below is worthless")
    floor = measure_replay_floor()
    ceilings = dense_ceilings()
    print(f"flushed graph-replay floor: {floor * 1e3:.1f} us")
    print(
        "measured dense 4096^3 ceilings on THIS card: "
        + "  ".join(f"{k} {v:.0f} TF/s" for k, v in ceilings.items())
    )

    print(
        f"\n{'preset':>13}{'tok':>7}{'rows/E':>7}{'bf16':>7}{'seq':>7}{'strm':>7}"
        f"{'sink':>7}{'fused':>7}{'seq x':>8}{'strm x':>8}{'sink x':>8}{'fusd x':>8}"
        f"{'eager x':>9}{'relerr':>9}{'noise':>8}{'bits':>6}"
        "     device TF/s; x is against bf16"
    )
    rows = []
    for preset in args.presets:
        for tokens in args.tokens:
            b = build(preset, tokens)
            rows.append(row(b, args.streams, floor, ceilings))
            del b
            torch.cuda.empty_cache()
            r = rows[-1]
            print(
                f"{r['preset'].replace('Kohaku-', ''):>13}{r['tokens']:>7}"
                f"{r['rows_per_expert']:>7}{r['bf16_device_tflops']:>7.0f}"
                f"{r['fp8_seq_device_tflops']:>7.0f}"
                f"{r['fp8_streamed_device_tflops']:>7.0f}"
                f"{r['fp8_sink_device_tflops']:>7.0f}"
                f"{r['fp8_fused_device_tflops']:>7.0f}"
                f"{r['fp8_seq_speedup_device']:>7.2f}x"
                f"{r['fp8_streamed_speedup_device']:>7.2f}x"
                f"{r['fp8_sink_speedup_device']:>7.2f}x"
                f"{r['fp8_fused_speedup_device']:>7.2f}x"
                f"{r['fp8_sink_speedup_wall']:>8.2f}x"
                f"{r['fp8_rel_err_vs_bf16']:>9.4f}"
                f"{r['bf16_self_rel_err']:>8.4f}"
                f"{'ok' if r['streamed_matches_sequential'] and r['sink_matches_sequential'] else 'DIFF':>6}"
            )
            if r["suspect"]:
                print(f"{'':>13}  SUSPECT: {r['suspect']}")

    print(
        f"\n{'preset':>13}{'tok':>7}"
        + "".join(
            f"{k:>16}"
            for k in ("bf16", "fp8_seq", "fp8_streamed", "fp8_sink", "fp8_fused")
        )
        + "   launches kernel+memory"
    )
    for r in rows:
        print(
            f"{r['preset'].replace('Kohaku-', ''):>13}{r['tokens']:>7}"
            + "".join(
                f"{r[f'{k}_launches']['kernel']}+{r[f'{k}_launches']['memory']:<10}".rjust(
                    16
                )
                for k in ("bf16", "fp8_seq", "fp8_streamed", "fp8_sink", "fp8_fused")
            )
        )

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "moe_fp8_vendor_path.json")
        with open(path, "w") as fh:
            json.dump(
                {
                    "device": device_name(),
                    "visible_devices": visible_devices(),
                    "bandwidth": bandwidth,
                    "graph_replay_floor_ms": floor,
                    "dense_ceilings_tflops": ceilings,
                    "streams": args.streams,
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
