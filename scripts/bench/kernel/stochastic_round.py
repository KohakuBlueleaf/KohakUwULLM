"""What stochastic rounding costs on the writeback, and what it buys in bias.

The writeback into a 16-bit master weight is pure streaming: read the fp32
update, read the parameter, write the parameter. So the questions are how close
each spelling gets to DRAM peak and how many launches it takes to cover a whole
model -- never FLOPs.

Three writeback spellings, and the point is that the fastest one is wrong:

* ``nearest`` -- ``dst.copy_(src)``. The floor, and biased by construction: it is
  what silently freezes a 16-bit weight once the update drops below half an ulp.
* ``torch-sr`` -- ``training.lowbit.stochastic_round_``. Correct, and pays for a
  full-size ``torch.randint`` tensor plus several more passes over the data. bf16
  only; it refuses fp16 rather than producing garbage, so that row is absent.
* ``triton-sr`` -- ``kernels.stochastic_round.stochastic_round_``. Same bits, one
  pass, randomness generated in registers.

**Every bandwidth row is timed twice, because neither timer alone is right.**

:func:`bench_ms` flushes L2 -- necessary, because this device has a **96 MiB** L2
and an earlier draft of this script reported **4612 GB/s**, 2.4x DRAM peak, from a
working set that fit inside it. But its window also contains the Python that
issues the launch, and at ~27 us against a 435 us kernel that is 6%: under
``bench_ms`` alone this kernel reads 9% slower than a plain cast, and *all* of the
gap is host time. The un-synchronized loop keeps the queue full so the host
overlaps, and its device span puts the two within 0.4% of each other, both at
~99.5% of peak. ``gbps`` is computed from the loop, and the ``bench_ms`` figure is
kept beside it so the discrepancy is visible instead of load-bearing.

Rows are still sized to at least 1.5x L2 so the loop's warm cache cannot flatter
it: at 732 MiB against 96 MiB at most 13% could be resident, and the two timers
agreeing to 0.4% on ``nearest`` is the check that it is not.

The launch arm uses only the loop, because pipelining *is* the question there. It
reports host issue time next to the device span, since a per-parameter arm starves
the device and the event span then measures the host -- the finding, not an error,
but it has to be labelled.

Accuracy is the mean deviation over many roundings, in ULP, reported beside its
own standard error so "unbiased" is a comparison rather than an assertion.

The other two thirds of this question: ``lost_update.py`` measures the bias SR
exists to remove, and ``sr_whole_step.py`` whether the grouping speedup measured
here survives a pipelined training step -- at Kohaku-1.5B it does not.

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/stochastic_round.py
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/stochastic_round.py \
        --preset Nano-200M --out out/bench/kernel/sr
"""

import argparse
import json
import math
import os
import statistics
import time

import torch

from kohakuwullm.bench.core.plotting import (
    bar_labels,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import (
    bench_ms,
    cached_peak_bandwidth,
    device_name,
    sample_spread,
    ulp_error,
)
from kohakuwullm.kernels.optim.stochastic_round import (
    stochastic_round_,
    stochastic_round_update_,
)
from kohakuwullm.kernels.optim.stochastic_round_grouped import GroupedWriteback
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training.optim import lowbit

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
# Bytes the cast arms move per element: read fp32, write 16-bit. The update arms
# read the parameter as well, so they move 2 more.
CAST_BYTES = 6
UPDATE_BYTES = 8
# Multiple of L2 a bandwidth row's working set must exceed. Bounds how much of a
# row the un-flushed loop can keep resident between iterations. 1.5x was not
# enough and did not look wrong: 183 MiB against 96 MiB reported 105.7% and 107.9%
# of DRAM peak. `implausible` below is the durable guard -- this number alone is
# the kind that rots the next time the sweep is edited.
L2_MARGIN = 2.5
# Fraction of a quarter-ulp offset the accuracy probe sits above the grid.
PROBE_FRACTION = 0.25


def loop_ms(fn, iters: int = 50, warmup: int = 10) -> dict:
    """Per-call wall, host-issue and device time from one pipelined loop.

    Nothing synchronizes inside the loop, so the launch queue stays full the way
    it does in a training step. ``issue_ms`` is the host cost, taken before the
    final synchronize; when it matches ``wall_ms`` the arm is host-bound and
    ``device_ms`` is measuring host issue through a starved queue rather than
    kernel time. No L2 flush -- use :func:`bench_ms` for anything bandwidth.
    """
    marks = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for begin, end in marks:
        begin.record()
        fn()
        end.record()
    issue = (time.perf_counter() - start) * 1e3 / iters
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1e3 / iters

    spans = [a.elapsed_time(b) for a, b in marks]
    return {
        "wall_ms": wall,
        "issue_ms": issue,
        "device_ms": statistics.median(spans),
        "spread": sample_spread(spans),
        "host_bound": issue >= 0.9 * wall,
    }


def cast_arms(dtype: torch.dtype) -> dict:
    """``name -> (dst, src, seed) -> None`` for every writeback spelling."""
    arms = {
        "nearest": lambda dst, src, seed: dst.copy_(src),
        "triton-sr": lambda dst, src, seed: stochastic_round_(dst, src, seed),
    }
    if dtype is torch.bfloat16:
        arms["torch-sr"] = lambda dst, src, seed: lowbit.stochastic_round_(dst, src)
    return arms


def torch_update(param: torch.Tensor, update: torch.Tensor, keep: float) -> None:
    """The writeback ``StochasticAdamW`` performs today, unfused."""
    working = param.to(torch.float32, copy=True)
    working.mul_(keep)
    working.add_(update)
    lowbit.stochastic_round_(param, working)


def bandwidth_sweep(dtype: torch.dtype, peak: float, l2: int) -> list[dict]:
    """Achieved bandwidth for every arm, at sizes that cannot sit in L2.

    One allocation for the whole sweep, sliced: reallocating per row churns the
    caching allocator into the timing loop, and a leading slice of a contiguous
    tensor is contiguous, which is all these kernels require.
    """
    floor = int(L2_MARGIN * l2 / CAST_BYTES)
    sizes = [m * 1_000_000 for m in (24, 32, 48, 64, 96, 128)]
    sizes = [n for n in sizes if n >= floor]
    src_full = torch.randn(max(sizes), device="cuda") * 0.02
    dst_full = torch.empty(max(sizes), dtype=dtype, device="cuda")
    rows = []
    for n in sizes:
        src, dst = src_full[:n], dst_full[:n]
        for name, arm in cast_arms(dtype).items():
            call = lambda a=arm, d=dst, s=src: a(d, s, 7)  # noqa: E731
            flushed = bench_ms(call, iters=25)
            timing = loop_ms(call, iters=30)
            gbps = n * CAST_BYTES / (timing["device_ms"] * 1e-3) / 1e9
            rows.append(
                {
                    "arm": name,
                    "dtype": str(dtype).replace("torch.", ""),
                    "numel": n,
                    "working_set_mib": n * CAST_BYTES / 2**20,
                    "flushed_ms": flushed,
                    "flushed_gbps": n * CAST_BYTES / (flushed * 1e-3) / 1e9,
                    "gbps": gbps,
                    "pct_peak": 100.0 * gbps / peak if peak else 0.0,
                    "implausible": bool(peak and gbps > peak),
                    **timing,
                }
            )
            if rows[-1]["implausible"]:
                print(
                    f"    IMPLAUSIBLE: {gbps:.0f} GB/s exceeds DRAM peak {peak:.0f} "
                    f"at {n * CAST_BYTES / 2**20:.0f} MiB -- L2-resident, raise L2_MARGIN"
                )
            print(
                f"  {name:11s} {dtype} n={n:>11,} "
                f"{n * CAST_BYTES / 2**20:6.0f} MiB  dev {timing['device_ms'] * 1e3:8.1f} us  "
                f"{gbps:7.1f} GB/s  {rows[-1]['pct_peak']:5.1f}% peak  "
                f"(flushed {rows[-1]['flushed_gbps']:7.1f} GB/s, "
                f"issue {timing['issue_ms'] * 1e3:5.1f} us)"
            )
    return rows


def launch_floor(dtype: torch.dtype, n: int = 4096) -> list[dict]:
    """Per-call cost at a size too small to hide it, which is the launch cost.

    Deliberately not reported as bandwidth: at 4096 elements the kernel moves
    24 KB and every microsecond here is dispatch. These figures also include the
    loop's two ``Event.record`` calls, which cost host time of their own, so they
    compare the spellings against each other rather than give an absolute -- the
    clean per-launch number is ``per_launch_us`` from :func:`launch_sweep`, where
    162 launches amortise one pair of records.
    """
    src = torch.randn(n, device="cuda") * 0.02
    dst = torch.empty(n, dtype=dtype, device="cuda")
    rows = []
    for name, arm in cast_arms(dtype).items():
        timing = loop_ms(lambda a=arm: a(dst, src, 7))
        rows.append(
            {
                "arm": name,
                "dtype": str(dtype).replace("torch.", ""),
                "numel": n,
                **timing,
            }
        )
        print(
            f"  {name:11s} {dtype} wall {timing['wall_ms'] * 1e3:7.1f} us  "
            f"issue {timing['issue_ms'] * 1e3:7.1f} us  host_bound={timing['host_bound']}"
        )
    return rows


def accuracy(dtype: torch.dtype, n: int = 1 << 14, reps: int = 2048) -> list[dict]:
    """Bias per arm on values a quarter of an ulp above a representable one.

    Not a random tensor: random values average out near the midpoint, where
    round-to-nearest is *also* nearly unbiased across many elements. A fixed
    quarter-ulp offset is the case the freeze actually presents.

    ``bias_ulp`` is the mean over every element and every draw, which is the only
    statistic that isolates bias; ``sem_ulp`` is its standard error, so a reader
    can see whether zero is inside the error bar. ``max_ulp`` is the per-element
    worst case from :func:`ulp_error`, which at 2048 draws is dominated by
    sampling noise -- an earlier draft reported *that* as the bias and made both
    correct arms look 0.04 ulp off.
    """
    grid = torch.full((n,), 1.0, device="cuda").to(dtype).float()
    ulp = ((grid + torch.finfo(dtype).eps).to(dtype).float() - grid)[0].item()
    src = grid + PROBE_FRACTION * ulp
    sem = (PROBE_FRACTION * (1 - PROBE_FRACTION) / (reps * n)) ** 0.5
    rows = []
    for name, arm in cast_arms(dtype).items():
        total = torch.zeros_like(src, dtype=torch.float64)
        dst = torch.empty_like(src, dtype=dtype)
        for step in range(reps):
            arm(dst, src, step + 1)
            total += dst.double()
        mean = total / reps
        rows.append(
            {
                "arm": name,
                "dtype": str(dtype).replace("torch.", ""),
                "bias_ulp": ((mean - src.double()).mean() / ulp).item(),
                "sem_ulp": sem,
                "max_ulp": ulp_error(mean, src.double(), dtype, mode="rms"),
            }
        )
        print(
            f"  {name:11s} {dtype} bias {rows[-1]['bias_ulp']:+.5f} +- {sem:.5f} ulp  "
            f"(max per element {rows[-1]['max_ulp']:.4f})"
        )
    return rows


def launch_sweep(preset: str, dtype: torch.dtype) -> dict:
    """Per-parameter launches against one launch over the same element count.

    ``triton-flat`` is not something the current code can reach -- it rounds one
    contiguous buffer. It is the floor a grouped kernel would approach, so the
    ratio to ``triton-update`` is the grouping headroom and nothing else.
    """
    config = get_preset(preset)
    model = LMBackbone(config).to("cuda").to(dtype)
    params = [p.detach() for p in model.parameters() if p.dtype is dtype]
    updates = [torch.randn_like(p, dtype=torch.float32) for p in params]
    total = sum(p.numel() for p in params)
    flat_param = torch.empty(total, dtype=dtype, device="cuda")
    flat_update = torch.empty(total, dtype=torch.float32, device="cuda")

    def triton_per_param():
        offset = 0
        for param, update in zip(params, updates):
            stochastic_round_update_(param, update, 11, decay=1e-4, rng_offset=offset)
            offset += param.numel()

    def torch_per_param():
        for param, update in zip(params, updates):
            torch_update(param, update, 1.0 - 1e-4)

    def triton_flat():
        stochastic_round_update_(flat_param, flat_update, 11, decay=1e-4)

    grouped = GroupedWriteback(params, updates)

    arms = {
        "triton-update": triton_per_param,
        "triton-grouped": lambda: grouped(11, decay=1e-4),
        "triton-flat": triton_flat,
    }
    if dtype is torch.bfloat16:
        arms["torch-update"] = torch_per_param
    timings = {name: loop_ms(fn, iters=30) for name, fn in arms.items()}

    per, one = timings["triton-update"], timings["triton-flat"]
    group_timing = timings["triton-grouped"]
    out = {
        "preset": preset,
        "dtype": str(dtype).replace("torch.", ""),
        "tensors": len(params),
        "numel": total,
        # The launch arm is the only one that reads the model definition, and this
        # repo's presets move: two runs of this script an hour apart straddled a
        # tie_embeddings flip and reported 162/192.7M against 163/251.4M for the
        # same preset name. A launch row is only comparable to another with an
        # identical fingerprint.
        "tie_embeddings": bool(config.tie_embeddings),
        "moved_gb": total * UPDATE_BYTES / 1e9,
        "arms": timings,
        "per_launch_us": (per["issue_ms"] - one["issue_ms"])
        * 1e3
        / max(len(params) - 1, 1),
        "grouping_headroom": per["wall_ms"] / one["wall_ms"],
        "grouped_speedup": per["wall_ms"] / group_timing["wall_ms"],
        "grouped_chunks": grouped.n_chunks,
    }
    print(
        f"  {preset} {dtype}: {len(params)} tensors, {total / 1e6:.1f}M params, "
        f"tie_embeddings={config.tie_embeddings}"
    )
    for name, timing in timings.items():
        print(
            f"    {name:14s} wall {timing['wall_ms']:8.3f}  issue {timing['issue_ms']:8.3f}"
            f"  device {timing['device_ms']:8.3f} ms  host_bound={timing['host_bound']}"
        )
    print(
        f"    per-launch host cost {out['per_launch_us']:.1f} us, "
        f"headroom {out['grouping_headroom']:.2f}x, "
        f"grouped {out['grouped_speedup']:.2f}x over per-parameter "
        f"({out['grouped_chunks']} chunks)"
    )
    return out


# Bytes per parameter the ladder sweep allocates: a 16-bit parameter plus its
# fp32 update. The model itself is built on `meta`, so it costs nothing -- only
# the shapes are wanted, and the fp32 build would otherwise double the peak.
LADDER_BYTES = 6
# Leaves headroom on a 32 GiB card for the chunk table, the RNG and fragmentation.
LADDER_BUDGET_GIB = 24.0


def _meta_shapes(preset: str) -> tuple[list[tuple[int, ...]], bool]:
    config = get_preset(preset)
    with torch.device("meta"):
        model = LMBackbone(config)
    shapes = [tuple(p.shape) for p in model.parameters() if p.is_floating_point()]
    return shapes, bool(config.tie_embeddings)


def _time_ladder_rung(shapes, dtype: torch.dtype, numel: int) -> dict:
    """Allocate one rung, time both dispatch strategies, and let the frame free it."""
    params = [
        torch.empty(s, dtype=dtype, device="cuda").normal_(0, 0.02) for s in shapes
    ]
    updates = [
        torch.empty(s, dtype=torch.float32, device="cuda").normal_(0, 1e-6)
        for s in shapes
    ]
    grouped = GroupedWriteback(params, updates)

    def per_param():
        offset = 0
        for param, update in zip(params, updates):
            stochastic_round_update_(param, update, 11, decay=1e-4, rng_offset=offset)
            offset += param.numel()

    per = loop_ms(per_param, iters=20)
    one = loop_ms(lambda: grouped(11, decay=1e-4), iters=20)
    return {
        "measured": True,
        "per_param": per,
        "grouped": one,
        "speedup": per["wall_ms"] / one["wall_ms"],
        "per_launch_us": (per["issue_ms"] - one["issue_ms"]) * 1e3 / len(shapes),
        "chunks": grouped.n_chunks,
        "grouped_gbps": numel * UPDATE_BYTES / (one["device_ms"] * 1e-3) / 1e9,
    }


def ladder_sweep(presets: list[str], dtype: torch.dtype) -> list[dict]:
    """Per-parameter against grouped across the preset ladder.

    Launch cost scales with the *tensor count* and device time with the byte
    count, so the interesting axis is how the ratio moves along a ladder whose
    rungs differ in both. Rungs too large for the card are recorded with their
    tensor count and an explicitly-labelled projection rather than silently
    dropped -- the projection is ``n_tensors * per_launch_us``, arithmetic, not a
    measurement.
    """
    rows = []
    for preset in presets:
        shapes, tied = _meta_shapes(preset)
        numel = sum(math.prod(s) for s in shapes)
        need = numel * LADDER_BYTES / 2**30
        row = {
            "preset": preset,
            "dtype": str(dtype).replace("torch.", ""),
            "tensors": len(shapes),
            "numel": numel,
            "tie_embeddings": tied,
            "needs_gib": need,
        }
        if need > LADDER_BUDGET_GIB:
            row["measured"] = False
            print(
                f"  {preset:16s} {len(shapes):4d} tensors {numel / 1e9:6.3f}B "
                f"-- needs {need:.1f} GiB, over the {LADDER_BUDGET_GIB:.0f} GiB "
                f"budget; tensor count recorded, timing NOT measured"
            )
            rows.append(row)
            continue

        row.update(_time_ladder_rung(shapes, dtype, numel))
        print(
            f"  {preset:16s} {len(shapes):4d} tensors {numel / 1e9:6.3f}B  "
            f"per-param {row['per_param']['wall_ms']:8.3f} ms  "
            f"grouped {row['grouped']['wall_ms']:7.3f} ms  "
            f"{row['speedup']:5.2f}x  {row['per_launch_us']:5.1f} us/launch  "
            f"{row['grouped_gbps']:6.0f} GB/s"
        )
        rows.append(row)
        # Freed by the helper's frame exiting, not by `del`: a `del` here would
        # unbind names its closures capture, which ruff flags as F821 and which
        # has now bitten three files in this repo.
        torch.cuda.empty_cache()
    return rows


def figure(rows, acc, floors, launches, path: str) -> None:
    fig, axes = new_figure(1, 3, figsize=(15.5, 4.4))
    peak = cached_peak_bandwidth()

    ax = axes[0]
    for name in ("nearest", "triton-sr", "torch-sr"):
        series = [r for r in rows if r["arm"] == name and r["dtype"] == "bfloat16"]
        if series:
            ax.plot(
                [r["numel"] for r in series],
                [r["gbps"] for r in series],
                marker="o",
                label=name,
            )
    if peak:
        ax.axhline(peak, ls="--", lw=1.0, color="#7F7F7F")
        ax.text(rows[0]["numel"], peak * 1.02, f"DRAM peak {peak:.0f} GB/s", fontsize=8)
    ax.set_ylim(0, (peak or 2000) * 1.15)
    finish_axis(
        ax,
        f"elements (all above {L2_MARGIN}x L2)",
        "GB/s",
        "bf16 writeback bandwidth",
        logx=True,
        legend=True,
    )

    ax = axes[1]
    bf16 = [a for a in acc if a["dtype"] == "bfloat16"]
    vals = [a["bias_ulp"] for a in bf16]
    bars = ax.bar(
        [a["arm"] for a in bf16],
        vals,
        color=["#D55E00" if abs(v) > 0.01 else "#009E73" for v in vals],
    )
    bar_labels(ax, bars, "{:+.4f}")
    ax.axhline(0.0, lw=0.8, color="#333333")
    finish_axis(ax, "", "bias (ulp)", f"bias at {PROBE_FRACTION} ulp, bf16")

    ax = axes[2]
    order = ["torch-update", "triton-update", "triton-grouped", "triton-flat"]
    colors = {
        "torch-update": "#E69F00",
        "triton-update": "#D55E00",
        "triton-grouped": "#009E73",
        "triton-flat": "#0072B2",
    }
    width = 0.2
    labels = [f"{row['dtype']}\n{row['tensors']} tensors" for row in launches]
    for slot, name in enumerate(order):
        xs = [
            i + (slot - 1) * width for i, r in enumerate(launches) if name in r["arms"]
        ]
        ys = [r["arms"][name]["wall_ms"] for r in launches if name in r["arms"]]
        bars = ax.bar(xs, ys, width=width, label=name, color=colors[name])
        bar_labels(ax, bars, "{:.2f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    finish_axis(
        ax,
        "",
        "ms per whole-model writeback",
        f"launch cost ({launches[0]['preset']})",
        legend=True,
        si_x=False,
    )
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="Nano-200M")
    parser.add_argument("--out", default="out/bench/kernel/sr")
    parser.add_argument(
        "--ladder",
        default="",
        help="comma-separated presets; sweeps per-parameter vs grouped over them",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    peak = cached_peak_bandwidth()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"{device_name()} CUDA_VISIBLE_DEVICES={visible} "
        f"peak {peak:.0f} GB/s  L2 {l2 / 2**20:.0f} MiB"
    )

    rows, acc, floors, launches = [], [], [], []
    for name, dtype in DTYPES.items():
        print(f"-- bandwidth, {name}")
        rows += bandwidth_sweep(dtype, peak, l2)
        torch.cuda.empty_cache()
        print(f"-- launch floor, {name}")
        floors += launch_floor(dtype)
        print(f"-- accuracy, {name}")
        acc += accuracy(dtype)
        torch.cuda.empty_cache()
    print("-- launch count")
    for dtype in DTYPES.values():
        launches.append(launch_sweep(args.preset, dtype))
        torch.cuda.empty_cache()

    ladder = []
    if args.ladder:
        print("-- ladder")
        ladder = ladder_sweep(args.ladder.split(","), torch.bfloat16)

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "device": device_name(),
        "cuda_visible_devices": visible,
        "peak_gbps": peak,
        "l2_bytes": l2,
        "bandwidth": rows,
        "launch_floor": floors,
        "accuracy": acc,
        "launches": launches,
        "ladder": ladder,
    }
    with open(os.path.join(args.out, "stochastic_round.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    figure(rows, acc, floors, launches, os.path.join(args.out, "stochastic_round.png"))
    print(f"wrote {args.out}/stochastic_round.json and .png")


if __name__ == "__main__":
    main()
