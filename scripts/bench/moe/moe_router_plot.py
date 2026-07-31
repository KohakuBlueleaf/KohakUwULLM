"""Draw the router + dispatch figures from ``moe_router.json``.

Two figures, split by the question they answer:

* ``moe_router_layer.png`` -- did the layer get faster, and did the non-GEMM
  share move? A stacked bar per shape, eager beside fused, annotated with the
  share. This is the figure to read first; everything else explains it.
* ``moe_router_stages.png`` -- which stage won, by how much, and against what.
  Includes the ``torch.compile`` and MSLK baselines, both of which lost, and the
  precision panel that makes the speedups admissible.

Latency is drawn from ``graph_us`` (CUDA-graph replay, pure device time) with
``cpu_us`` shown beside it, because on this path the two disagree and the
disagreement *is* the finding: the eager router's cost is Python issuing eight
ops, not the GPU running them.

Usage:
    .venv/bin/python scripts/bench/moe/moe_router_plot.py --dir out/bench/moe/moe_router
"""

import argparse
import json
import math
import os

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    new_figure,
    save_figure,
)

CEIL = "#D55E00"
# Seeded in a fixed order so a given implementation keeps one colour in every
# panel. Left to first-use order the palette runs out by the fifth panel and
# hands the expert-sort comparison two near-indistinguishable late steps.
IMPL_ORDER = (
    "eager",
    "torch.compile",
    "fused triton",
    "mslk scatter_add",
    "eager (scale + index_add)",
    "fused (scale+scatter)",
    "eager (argsort+cumsum+div)",
    "fused (counting sort)",
)
# Drawing order. A stage missing from a run is dropped rather than plotted zero.
ROUTER_IMPLS = ("eager", "torch.compile", "fused triton")
COMBINE_IMPLS = (
    "eager (scale + index_add)",
    "mslk scatter_add",
    "fused (scale+scatter)",
)
SORT_IMPLS = ("eager (argsort+cumsum+div)", "fused (counting sort)")
# (stage, eager impl, fused impl, counts as non-GEMM overhead)
LAYER_PARTS = (
    ("router", "eager", "fused triton", True),
    ("expert-sort", "eager (argsort+cumsum+div)", "fused (counting sort)", True),
    ("gather", "index_select", "index_select", True),
    ("combine", "eager (scale + index_add)", "fused (scale+scatter)", True),
    ("expert_gemm", "triton grouped", "triton grouped", False),
)


def _pick(rows, **where):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def _value(rows, key="graph_us", **where):
    hits = _pick(rows, **where)
    if not hits:
        return 0.0
    value = hits[0].get(key, 0.0)
    return 0.0 if value is None or math.isnan(value) else value


def shape_groups(rows):
    """``(experts, top_k, dim)`` in measured order, read off the rows."""
    seen = {}
    for row in rows:
        key = (row["num_experts"], row["top_k"], row["dim"])
        seen.setdefault(key, key)
    return list(seen.values())


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
            edgecolor="white",
            linewidth=0.5,
        )
        bar_labels(ax, bars, fmt)
    ax.set_xticks(xs)
    return xs


def _mslk_note(rows, shape_count: int) -> str:
    """One line on whether MSLK's fused routing op could run here at all.

    The raw assertion text is a full C++ sentence; quoted verbatim it runs off
    the canvas, so the *constraint* is recovered from it and the prose dropped.
    """
    tried = _pick(rows, kernel="index_shuffling")
    refused = [r for r in tried if not r.get("supported")]
    if not tried or not refused:
        return ""
    allowed = sorted(
        {
            int(token)
            for row in refused
            for token in row.get("note", "").replace("==", " ").split()
            if token.isdigit()
        }
    )
    limit = f" (accepts num_experts in {allowed})" if allowed else ""
    return (
        f"MSLK index_shuffling refused {len(refused)}/{shape_count} of these "
        f"shapes{limit}, so its fused top-k+sort+histogram was never available"
    )


def plot_layer(rows, out_dir, payload):
    """Stacked layer cost, eager vs fused, with the non-GEMM share annotated."""
    pal = Palette()
    fig, axes = new_figure(1, 2, figsize=(15, 6.4))
    groups = shape_groups(rows)
    labels = [f"E={e}, top-{k}\ndim {d}" for e, k, d in groups]

    def stage_us(group, stage, impl):
        experts, top_k, dim = group
        return _value(
            rows,
            kernel=stage,
            impl=impl,
            num_experts=experts,
            top_k=top_k,
            dim=dim,
        )

    ax = axes[0]
    width = 0.36
    xs = list(range(len(groups)))
    totals = {}
    for slot, (variant, which) in enumerate((("eager", 1), ("fused", 2))):
        offset = (slot - 0.5) * width
        bottom = [0.0] * len(groups)
        for stage, eager_impl, fused_impl, overhead in LAYER_PARTS:
            impl = eager_impl if which == 1 else fused_impl
            vals = [stage_us(g, stage, impl) for g in groups]
            ax.bar(
                [x + offset for x in xs],
                vals,
                width,
                bottom=bottom,
                color=pal.color(stage),
                label=stage if slot == 0 else None,
                # Hatched = the cost of deciding where work goes; solid = the
                # arithmetic. Colour alone would let the thin router sliver read
                # as just another GEMM stage.
                hatch="//" if overhead else None,
                edgecolor="white",
                linewidth=0.4,
            )
            bottom = [b + v for b, v in zip(bottom, vals)]
        totals[variant] = bottom
        overhead_us = [
            sum(
                stage_us(g, stage, eager_impl if which == 1 else fused_impl)
                for stage, eager_impl, fused_impl, over in LAYER_PARTS
                if over
            )
            for g in groups
        ]
        for i, total in enumerate(bottom):
            if total <= 0:
                continue
            ax.annotate(
                f"{variant}\n{total:.0f} us\n{100 * overhead_us[i] / total:.0f}% non-GEMM",
                (xs[i] + offset, total),
                ha="center",
                textcoords="offset points",
                xytext=(0, 4),
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("forward device time (us)")
    ax.set_title("Where an MoE layer's forward goes (CUDA-graph replay)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, max(max(v) for v in totals.values()) * 1.34)

    ax = axes[1]
    speedup = []
    for i, group in enumerate(groups):
        eager_total = totals["eager"][i]
        fused_total = totals["fused"][i]
        speedup.append(eager_total / fused_total if fused_total else 0.0)
    bars = ax.bar(xs, speedup, 0.5, color=pal.color("router"))
    bar_labels(ax, bars, "{:.2f}x")
    ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2)
    ax.annotate(
        "parity",
        (xs[0] - 0.35, 1.0),
        textcoords="offset points",
        xytext=(0, -13),
        fontsize=8,
        color=CEIL,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("whole-layer speedup")
    ax.set_title("Layer forward: fused vs eager")
    ax.set_ylim(0, max(speedup + [1.0]) * 1.25)

    hidden = next((r.get("hidden") for r in rows if r.get("hidden")), None)
    fig.suptitle(
        f"MoE layer forward -- {payload['device']}  |  "
        f"T={payload['config']['tokens']} tokens, one expert hidden={hidden}\n"
        "hatched = non-GEMM overhead (router, expert-sort, gather, combine); "
        "solid = the expert GEMMs\n"
        "narrow experts make the GEMMs small, so overhead dominates at this "
        "scale in a way it does not at frontier width",
        fontweight="bold",
    )
    path = os.path.join(out_dir, "moe_router_layer.png")
    save_figure(fig, path)
    return path


def plot_stages(rows, out_dir, payload):
    """Per-stage winners and losers, plus the precision that admits them."""
    pal = Palette()
    for name in IMPL_ORDER:
        pal.color(name)
    fig, axes = new_figure(2, 3, figsize=(19, 10.5))
    groups = shape_groups(rows)
    labels = [f"E={e}, top-{k}\ndim {d}" for e, k, d in groups]

    def getter(stage, key="graph_us"):
        def value_of(group, impl):
            experts, top_k, dim = group
            return _value(
                rows,
                key=key,
                kernel=stage,
                impl=impl,
                num_experts=experts,
                top_k=top_k,
                dim=dim,
            )

        return value_of

    panels = (
        (
            axes[0],
            "router",
            ROUTER_IMPLS,
            "graph_us",
            "Router: device time (CUDA-graph replay)",
            "device time (us)",
        ),
        (
            axes[1],
            "router",
            ROUTER_IMPLS,
            "cpu_us",
            "Router: host time to ISSUE the work",
            "host enqueue (us)",
        ),
        (
            axes[2],
            "combine",
            COMBINE_IMPLS,
            "graph_us",
            "Combine (scale + sum back onto token)",
            "device time (us)",
        ),
        (
            axes[3],
            "expert-sort",
            SORT_IMPLS,
            "graph_us",
            "Expert-sort (group pairs by expert)",
            "device time (us)",
        ),
        (
            axes[4],
            "non_gemm_path",
            ("eager", "fused triton"),
            "graph_us",
            "Whole non-GEMM path",
            "device time (us)",
        ),
    )
    for ax, stage, impls, key, title, ylabel in panels:
        present = [i for i in impls if _pick(rows, kernel=stage, impl=i)]
        _grouped_bars(ax, groups, present, getter(stage, key), pal)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7.5)

    axes[1].annotate(
        "eager: host time == device time, so the GPU waits on Python",
        (0.02, 0.94),
        xycoords="axes fraction",
        fontsize=8,
        color=CEIL,
        fontweight="bold",
    )

    ax = axes[5]
    names, values, colors = [], [], []
    for stage, impl in (
        ("router", "fused triton"),
        ("combine", "fused (scale+scatter)"),
    ):
        # `"ulp" in r`, not truthiness: an exact kernel reports 0.0, and dropping
        # it would hide the best result in the figure.
        hits = [
            r for r in rows if r["kernel"] == stage and r["impl"] == impl and "ulp" in r
        ]
        if hits:
            names.append(f"{stage}\n{impl.split(' ')[0]}")
            values.append(max(r["ulp"] for r in hits))
            colors.append(pal.color(impl))
    if values:
        xs = list(range(len(values)))
        bars = ax.bar(xs, values, 0.4, color=colors)
        bar_labels(ax, bars, "{:.2f}")
        ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2)
        ax.annotate(
            "1 ULP = as exact as bf16 allows",
            (xs[0] - 0.2, 1.0),
            textcoords="offset points",
            xytext=(0, 5),
            fontsize=8,
            color=CEIL,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_ylim(0, max(values + [1.0]) * 1.45)
    exact = all(r.get("idx_exact", True) for r in _pick(rows, kernel="router"))
    ax.set_ylabel("max divergence from eager (ULP, RMS-scaled)")
    # "divergence", not "error": for the combine, eager is not the truth either.
    # Both sum top_k bf16 terms with atomics, so they differ by summation order,
    # and against fp64 the fused one is the more accurate of the two (its
    # grad_weight dot product accumulates in fp32). Labelling this "error" would
    # read as the kernel being the worse of the pair.
    ax.set_title(
        "Precision -- worst over every shape\n"
        f"top-k index agreement with eager: {'EXACT' if exact else 'MISMATCH'}"
    )

    peak = payload["peak_bandwidth_gbps"]
    fig.suptitle(
        f"MoE router + dispatch -- {payload['device']}  |  "
        f"T={payload['config']['tokens']} tokens, measured DRAM ceiling "
        f"{peak:.0f} GB/s\n"
        "the router is launch- and bandwidth-bound, never compute-bound: its gate "
        "GEMM is under 1% of tensor peak\n" + _mslk_note(rows, len(groups)),
        fontweight="bold",
    )
    path = os.path.join(out_dir, "moe_router_stages.png")
    save_figure(fig, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/moe/moe_router")
    cli = parser.parse_args()
    with open(os.path.join(cli.dir, "moe_router.json")) as handle:
        payload = json.load(handle)
    rows = payload["rows"]

    for path in (
        plot_layer(rows, cli.dir, payload),
        plot_stages(rows, cli.dir, payload),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
