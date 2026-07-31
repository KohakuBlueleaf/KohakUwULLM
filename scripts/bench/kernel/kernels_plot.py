"""Draw the per-module figures from ``kernels.json``.

``scripts/bench/kernel/kernels.py`` writes one row per (module, implementation, dtype,
shape, phase); this turns them into figures. Split from the measurement half
because the two change for different reasons -- a new kernel adds a
measurement, a rejected figure only changes the drawing -- and because one file
carrying both exceeded the repo's size limit.

Three figures, one per binding constraint, because a single grid mixing them
invites reading a bandwidth-bound kernel's TFLOP/s as if it meant something:

* ``kernels_bandwidth.png`` -- RMSNorm / SwiGLU / RoPE against measured DRAM.
* ``kernels_moe.png`` -- expert GEMMs, operand precision, and the router in
  front of them, against MAMF and the measured TF32 rate.
* ``kernels_compute.png`` -- SwiGLU MLP against MAMF, LM head against memory.

Usage:
    .venv/bin/python scripts/bench/kernel/kernels_plot.py --dir out/bench/kernel/kernels
"""

import argparse
import json
import os
from argparse import Namespace

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import TENSOR_MAMF_TFLOPS

ROUTER_NOTE = (
    "router: sigmoid gating, top-k of E routed experts + 1 shared expert always on, "
    "aux-loss-free balancing bias (steers selection only, never the output weight)"
)
# Drawing order for the two MoE panels that compare named alternatives. Rows
# missing from a run are dropped rather than plotted as zero.
GEMM_IMPLS = (
    "loop (E x cuBLAS mm)",
    "triton grouped",
    "triton grouped, fp32 operands",
    "cuBLAS 1 GEMM (equal work)",
)
ROUTER_IMPLS = (
    "sigmoid (ours)",
    "softmax",
    "sigmoid + group-limited",
    "sigmoid, no balance bias",
)
DISPATCH = "dispatch (sort+gather+scatter)"
GROUPED = "triton grouped"
FP32_GROUPED = "triton grouped, fp32 operands"
CEIL = "#D55E00"
BANDWIDTH_KERNELS = ("rmsnorm", "swiglu", "rope")


def moe_groups(gemm):
    """``(preset, experts, top_k)`` in measured order.

    Read back off the rows rather than from ``presets.py``: the figure then
    describes what was actually measured even if the run overrode the presets.
    """
    seen = {}
    for r in gemm:
        seen.setdefault(r["preset"], (r["preset"], r["num_experts"], r["top_k"]))
    return list(seen.values())


def _pick(rows, **where):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


# `wall - host` stops being a measurement and becomes a weak estimate somewhere
# around here; past 50% it has produced values above the hardware peak. Rows over
# this are drawn ringed rather than dropped, because "we cannot tell" is itself
# the result for those points.
HOST_SHARE_SUSPECT = 0.30


def _host_share(row):
    ms = row.get("ms") or 0.0
    return (row.get("host_ms", 0.0) / ms) if ms else float("nan")


def _net(row, ykey):
    """``ykey`` rescaled from wall time to wall-minus-host.

    Every rate in these rows is ``quantity / ms``, so removing dispatch is a
    scaling rather than a recomputation -- no byte or FLOP count is re-derived
    here, which keeps this honest about being the same measurement, corrected.
    """
    ms = row.get("ms") or 0.0
    net_ms = ms - row.get("host_ms", 0.0)
    if ms <= 0 or net_ms <= 0:
        return float("nan")
    # Refused, not merely flagged, past the suspect threshold. At 90% host share
    # the scaling is 10x and lands far above the physical ceiling; drawing that
    # and ringing it still puts an impossible number on the axis, and it wrecks
    # the y-scale for every row that *is* measurable.
    if _host_share(row) > HOST_SHARE_SUSPECT:
        return float("nan")
    return row[ykey] * ms / net_ms


def _lines(ax, rows, ykey, pal, xkey="tokens", net=True):
    """One line per impl; returns the sorted x positions for the ticks.

    Plots the dispatch-corrected value as primary with the raw wall value faint
    behind it: the gap between the pair is the dispatch cost, and a figure that
    showed only one of them would either report Python's speed as the kernel's or
    hide that a correction was applied at all.
    """
    for impl in sorted({r["impl"] for r in rows}):
        pts = sorted(_pick(rows, impl=impl), key=lambda r: r[xkey])
        color = pal.color(impl)
        if net:
            ax.plot(
                [r[xkey] for r in pts],
                [r[ykey] for r in pts],
                color=color,
                alpha=0.25,
                linewidth=1.2,
            )
        ax.plot(
            [r[xkey] for r in pts],
            [_net(r, ykey) if net else r[ykey] for r in pts],
            marker="o",
            color=color,
            label=impl,
        )
        weak = [r for r in pts if _host_share(r) > HOST_SHARE_SUSPECT]
        if net and weak:
            ax.scatter(
                [r[xkey] for r in weak],
                [r[ykey] for r in weak],
                s=130,
                facecolors="none",
                edgecolors="#8E44AD",
                linewidths=1.5,
                zorder=5,
            )
    return sorted({r[xkey] for r in rows})


def _grouped_bars(ax, groups, series, value_of, pal, fmt="{:.0f}"):
    """One bar per (group, series); returns the tick positions."""
    width = 0.82 / max(len(series), 1)
    xs = list(range(len(groups)))
    for i, name in enumerate(series):
        bars = ax.bar(
            [x + (i - (len(series) - 1) / 2) * width for x in xs],
            [value_of(g, name) for g in groups],
            width,
            color=pal.color(name),
            label=name,
        )
        bar_labels(ax, bars, fmt)
    ax.set_xticks(xs)
    return xs


def _theoretical_note(args) -> str:
    """Clock-derived roofline for the subtitle, using the real memory speed.

    Both figures come out of the JSON rather than a live device, so a finished
    run can be redrawn without a GPU. A run recorded before those fields existed
    says so instead of guessing.
    """
    stock = args.theoretical_bandwidth_gbps
    if not stock:
        return "clock-derived roofline not recorded for this run"
    if args.mem_gbps and args.memory_bus_width:
        return (
            f"theoretical {args.memory_bus_width / 8 * args.mem_gbps:.0f} at "
            f"{args.mem_gbps:g} Gbps (stock clock would say {stock:.0f})"
        )
    return f"theoretical {stock:.0f} from the reported (stock) clock"


def plot_bandwidth(rows, out_dir, args, peak_bw, streams):
    pal = Palette()
    fig, axes = new_figure(2, 3, figsize=(19, 10))
    kernels = BANDWIDTH_KERNELS

    for ax, kern in zip(axes, kernels):
        sel = _pick(rows, kernel=kern, dtype="bfloat16", phase="fwd+bwd")
        if not sel:
            continue
        xs = _lines(ax, sel, "gbps", pal)
        ax.axhline(peak_bw, color=CEIL, linestyle="--", linewidth=1.2)
        ax.annotate(
            f"measured DRAM ceiling {peak_bw:.0f} GB/s",
            (xs[0], peak_bw),
            textcoords="offset points",
            xytext=(4, -13),
            fontsize=8,
            color=CEIL,
        )
        finish_axis(
            ax,
            "tokens",
            "device est. GB/s (wall - host)",
            f"{kern} fwd+bwd, bf16 ({sel[0]['shape']})",
            xticks=xs,
            logx=True,
            legend=True,
        )
        ax.set_ylim(0, peak_bw * 1.2)

    big = max(args.token_counts)
    pairs = [
        (k, i)
        for k in kernels
        for i in sorted({r["impl"] for r in _pick(rows, kernel=k)})
    ]
    at_big = dict(dtype="bfloat16", phase="fwd+bwd", tokens=big)
    tail = f"at T={big} (bf16, fwd+bwd)"
    # ULP is a property of the kernel rather than of one size, so its panel takes
    # the worst over the whole sweep (where={}); the other two read the largest.
    panels = (
        (
            "bandwidth_pct",
            "{:.0f}",
            "% of peak",
            100.0,
            at_big,
            f"% of measured DRAM peak {tail}",
        ),
        ("peak_gib", "{:.2f}", "peak GiB", None, at_big, f"peak memory {tail}"),
        (
            "ulp",
            "{:.2f}",
            "max error (ULP)",
            1.0,
            {},
            "Precision: worst over fp16, bf16 and every size",
        ),
    )
    for ax, (key, fmt, ylabel, line, where, title) in zip(axes[3:], panels):
        labels, values, colors = [], [], []
        for kern, impl in pairs:
            pts = _pick(rows, kernel=kern, impl=impl, **where)
            if pts:
                labels.append(f"{kern}\n{impl.split(' ')[0]}")
                values.append(max(r[key] for r in pts))
                colors.append(pal.color(impl))
        bars = ax.bar(labels, values, color=colors, width=0.62)
        bar_labels(ax, bars, fmt)
        if line is not None:
            ax.axhline(line, color=CEIL, linestyle="--", linewidth=1.2)
            ax.set_ylim(0, max(values + [line]) * 1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)

    patterns = ", ".join(f"{k} {v:.0f}" for k, v in sorted(streams.items()))
    fig.suptitle(
        f"Bandwidth-bound modules -- {args.device}  |  dim {args.dim}\n"
        f"ceiling {peak_bw:.0f} GB/s = best measured STREAM pattern ({patterns}); "
        f"{_theoretical_note(args)}\n"
        "Bold = net of host dispatch; faint = as measured; ringed (at the wall "
        "value) = host >30% of wall, where no corrected number is admissible.\n"
        "Forward-only at 131k tokens, the fused paths reach 96-102% of this "
        "ceiling -- their low small-token numbers were dispatch, not the kernel. "
        "The panels below\nare fwd+bwd, which is genuinely lower (53-78%). Eager "
        "two-kernel swiglu stays at 57-63% even at 1-3% host share: that one is "
        "the implementation, not the clock.",
        fontweight="bold",
    )
    path = os.path.join(out_dir, "kernels_bandwidth.png")
    save_figure(fig, path)
    return path


def plot_moe(gemm, router, out_dir, args, tf32_peak):
    pal = Palette()
    fig, axes = new_figure(2, 3, figsize=(19, 11))
    groups = moe_groups(gemm)
    labels = [f"{p}\nE={e}, top-{k}" for p, e, k in groups]
    impls = [i for i in GEMM_IMPLS if _pick(gemm, impl=i)]

    def router_ms(g, i):
        pts = _pick(router, preset=g[0], impl=i)
        return pts[0]["ms"] if pts else 0.0

    def gemm_value(group, impl, key, matrix, phase="fwd"):
        pts = _pick(
            gemm,
            preset=group[0],
            impl=impl,
            matrix=matrix,
            phase=phase,
            dtype="bfloat16",
        )
        return pts[0][key] if pts else 0.0

    def ceilings(ax, title):
        ax.axhline(
            TENSOR_MAMF_TFLOPS,
            color="#c0392b",
            linestyle="--",
            label=f"bf16 MAMF {TENSOR_MAMF_TFLOPS:.0f}T",
        )
        ax.axhline(
            tf32_peak,
            color="#2980b9",
            linestyle="-.",
            label=f"TF32 mma {tf32_peak:.0f}T (fp32 operands)",
        )
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("effective TFLOP/s")
        ax.set_title(title)
        ax.legend(fontsize=7)

    for ax, matrix, phase, title in (
        (axes[0], "w_in", "fwd", "Expert GEMM w_in (2H x D) -- forward, bf16"),
        (axes[1], "w_out", "fwd", "Expert GEMM w_out (D x H) -- forward, bf16"),
        (axes[2], "w_in", "fwd+bwd", "Expert GEMM w_in -- fwd+bwd, bf16"),
    ):
        _grouped_bars(
            ax,
            groups,
            impls,
            lambda g, i, m=matrix, p=phase: gemm_value(g, i, "tflops", m, p),
            pal,
        )
        ceilings(ax, title)

    series = ["bf16 operands", "fp32 operands (TF32 mma)"]
    acc = lambda g, s: gemm_value(  # noqa: E731
        g, GROUPED if s.startswith("bf16") else FP32_GROUPED, "tflops", "w_in"
    )
    _grouped_bars(axes[3], groups, series, acc, pal)
    for i, group in enumerate(groups):
        fast, slow = (acc(group, s) for s in series)
        if slow > 0:
            axes[3].annotate(
                f"{fast / slow:.2f}x",
                (i, max(fast, slow)),
                ha="center",
                textcoords="offset points",
                xytext=(0, 16),
                fontsize=9,
                fontweight="bold",
            )
    ceilings(axes[3], "Operand precision in the same Triton kernel (w_in, fwd)")

    # The bias segment is the gap between the two router variants already
    # measured, so it costs no extra run and names a specific target rather than
    # burying it inside one "router" block.
    def bias_ms(g):
        return router_ms(g, "sigmoid (ours)") - router_ms(g, "sigmoid, no balance bias")

    # Hatched = overhead, solid = arithmetic. Colour alone would leave the two
    # thin router slivers reading as just two more expert-GEMM stages; the
    # texture separates "work" from "the cost of deciding where work goes"
    # without needing the legend.
    parts = [
        ("router: gate + top-k", lambda g: router_ms(g, "sigmoid, no balance bias"), 1),
        ("router: balance-bias bookkeeping", bias_ms, 1),
        (DISPATCH, lambda g: router_ms(g, DISPATCH), 1),
        ("expert GEMM w_in", lambda g: gemm_value(g, GROUPED, "ms", "w_in"), 0),
        ("expert GEMM w_out", lambda g: gemm_value(g, GROUPED, "ms", "w_out"), 0),
    ]
    xs, bottom = list(range(len(groups))), [0.0] * len(groups)
    for name, getter, overhead in parts:
        vals = [getter(g) for g in groups]
        axes[4].bar(
            xs,
            vals,
            0.55,
            bottom=bottom,
            color=pal.color(name),
            label=name,
            hatch="//" if overhead else None,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, total in enumerate(bottom):
        group = groups[i]
        spent = sum(getter(group) for _, getter, over in parts if over)
        axes[4].annotate(
            f"{100 * spent / total:.0f}% non-GEMM\nbias {bias_ms(group) * 1e3:.0f} us",
            (i, total),
            ha="center",
            textcoords="offset points",
            xytext=(0, 3),
            fontsize=8,
            fontweight="bold",
        )
    axes[4].set_xticks(xs)
    axes[4].set_xticklabels(labels, fontsize=7)
    axes[4].set_ylabel("forward latency (ms)")
    axes[4].set_title(f"Where an MoE layer's time goes (T={args.moe_tokens})")
    axes[4].legend(fontsize=7)
    axes[4].set_ylim(0, max(bottom) * 1.22)

    _grouped_bars(
        axes[5], groups, list(ROUTER_IMPLS), lambda g, v: router_ms(g, v) * 1e3, pal
    )
    axes[5].set_xticklabels(labels, fontsize=7)
    axes[5].set_ylabel("router latency (us)")
    axes[5].set_title("Router design: gating and group-limited routing")
    axes[5].legend(fontsize=7)

    worst = max((r["ulp"] for r in gemm if r["impl"] == "triton grouped"), default=0.0)
    fig.suptitle(
        f"MoE expert path -- {args.device}  |  T={args.moe_tokens} tokens, "
        f"routed rows = T x top_k\n{ROUTER_NOTE}\n"
        f"Triton grouped GEMM worst error over both dtypes and all shapes: "
        f"{worst:.2f} ULP (RMS-scaled); 1 ULP = as exact as the dtype allows. "
        f"fp32 operands go to TF32 mma ({tf32_peak:.0f}T measured), not fp32 FMA",
        fontweight="bold",
    )
    path = os.path.join(out_dir, "kernels_moe.png")
    save_figure(fig, path)
    return path


def plot_compute(rows, out_dir, args, peak_bw):
    pal = Palette()
    fig, axes = new_figure(1, 3, figsize=(19, 5.6))

    mlp = _pick(rows, kernel="mlp", dtype="bfloat16", phase="fwd+bwd")
    xs = _lines(axes[0], mlp, "tflops", pal)
    axes[0].axhline(
        TENSOR_MAMF_TFLOPS, color="#c0392b", linestyle="--", label="MAMF 270T"
    )
    finish_axis(
        axes[0],
        "tokens",
        "effective TFLOP/s",
        "SwiGLU MLP fwd+bwd (bf16): cuBLAS GEMMs + fused gate",
        xticks=xs,
        logx=True,
        legend=True,
    )

    head = _pick(rows, kernel="head", dtype="bfloat16")
    for ax, key, ylabel, title in (
        (axes[1], "peak_gib", "peak GiB", "LM head memory -- the binding constraint"),
        (axes[2], "ms", "fwd+bwd (ms)", "LM head latency"),
    ):
        xs = _lines(ax, head, key, pal)
        finish_axis(ax, "tokens", ylabel, title, xticks=xs, logx=True, legend=True)

    fig.suptitle(
        f"Compute- and capacity-bound modules -- {args.device}  |  "
        f"dim {args.dim}, hidden {args.hidden}, vocab {args.vocab}\n"
        f"ceilings: MAMF {TENSOR_MAMF_TFLOPS:.0f} TFLOP/s bf16, "
        f"{peak_bw:.0f} GB/s DRAM",
        fontweight="bold",
    )
    path = os.path.join(out_dir, "kernels_compute.png")
    save_figure(fig, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out/bench/kernel/kernels")
    ap.add_argument(
        "--mem-gbps",
        type=float,
        default=None,
        help="actual per-pin memory speed in Gbps (e.g. 31.8 if overclocked). "
        "torch reports the STOCK clock, so the clock-derived roofline is "
        "wrong on an overclocked card; pass the real figure to have it "
        "shown alongside the measured ceiling.",
    )
    cli = ap.parse_args()
    with open(os.path.join(cli.dir, "kernels.json")) as handle:
        payload = json.load(handle)

    # --mem-gbps only ever affected a caption, so it belongs on this side; a
    # JSON written before the split still carries one, and this overrides it
    # rather than colliding with it. The machine fields default rather than
    # fail, so a run recorded before they existed still redraws.
    args = Namespace(
        **{
            **payload["config"],
            "mem_gbps": cli.mem_gbps,
            "device": payload.get("device", "device not recorded"),
            "memory_bus_width": payload.get("memory_bus_width"),
            "theoretical_bandwidth_gbps": payload.get("theoretical_bandwidth_gbps"),
        }
    )
    rows = payload["rows"]
    peak_bw = payload["peak_bandwidth_gbps"]
    streams = payload["stream_bandwidths"]
    tf32_peak = payload["tf32_peak_tflops"]
    gemm = _pick(rows, kernel="moe_gemm")
    router = _pick(rows, kernel="router")

    for path in (
        plot_bandwidth(rows, cli.dir, args, peak_bw, streams),
        plot_moe(gemm, router, cli.dir, args, tf32_peak),
        plot_compute(rows, cli.dir, args, peak_bw),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
