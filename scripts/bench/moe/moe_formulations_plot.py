"""Draw the six-formulation figures from ``moe_formulations.json``.

``scripts/bench/moe/moe_formulations.py`` writes one row per (formulation, shape,
stage); this turns them into figures. Split from the measurement half for the
same reason as ``kernels.py`` / ``kernels_plot.py``: a new formulation adds a
measurement, a rejected figure only changes the drawing.

Two figures, because the question "which formulation is fastest" is not
answerable from latency alone:

* ``moe_formulations.png`` -- the four metrics at every shape, ceilings drawn.
  Enough to see that all six sit at 43-55% of MAMF, i.e. none of them is losing
  to a kernel problem.
* ``moe_formulations_choice.png`` -- what actually decides a choice: how much of
  each formulation's cost is its unfused router (fixable) rather than its own
  arithmetic (not), and its cost per unit of *capacity*, without which a
  formulation that simply runs a smaller model looks like a speed win.

Usage:
    .venv/bin/python scripts/bench/moe/moe_formulations_plot.py \\
        --dir out/bench/moe/moe_formulations
"""

import argparse
import json
import os

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.core.timing import TENSOR_MAMF_TFLOPS, TF32_MAMF_TFLOPS

# Drawing order. Fixed here rather than read off the rows so the colours stay
# stable when a run is missing a formulation.
ORDER = [
    "DeepSeek-V3",
    "DeepSeek-V4",
    "MiMo-V2.5",
    "OLMoE/Switch",
    "ReMoE",
    "Kimi-K3 latent",
]
CEIL = "#D55E00"
FLAGSHIP = 1536


def _pick(rows, **where):
    return [r for r in rows if all(r.get(k) == v for k, v in where.items())]


def shape_labels(rows) -> list[tuple[str, int]]:
    """``(label, dim)`` per measured shape, in measured order."""
    seen = {}
    for r in rows:
        seen.setdefault(
            r["dim"],
            f"E={r['num_experts']} k={r['top_k']}\nd={r['dim']} h={r['hidden']}",
        )
    return [(label, dim) for dim, label in seen.items()]


def _grouped_bars(ax, shapes, formulations, value_of, pal, fmt="{:.0f}", hatch_of=None):
    width = 0.84 / max(len(formulations), 1)
    xs = list(range(len(shapes)))
    for i, name in enumerate(formulations):
        offset = (i - (len(formulations) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in xs],
            [value_of(dim, name) for _, dim in shapes],
            width,
            color=pal.color(name),
            label=name,
            hatch="//" if hatch_of and hatch_of(name) else None,
        )
        bar_labels(ax, bars, fmt)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in shapes], fontsize=8)
    return xs


def _present(rows) -> list[str]:
    have = {r["formulation"] for r in rows}
    return [name for name in ORDER if name in have]


def plot_metrics(rows, out_dir, meta, stage="layer_fwd_bwd"):
    """All four metrics at every shape, with both ceilings drawn."""
    data = _pick(rows, stage=stage)
    pal = Palette()
    shapes = shape_labels(data)
    names = _present(data)
    peak_bw = meta.get("peak_bandwidth_gbps") or 0.0

    def value(key):
        def get(dim, name):
            hit = _pick(data, dim=dim, formulation=name)
            return hit[0][key] if hit else 0.0

        return get

    def device_ms(dim, name):
        """Device time, falling back to wall clock if a capture failed."""
        hit = _pick(data, dim=dim, formulation=name)
        if not hit:
            return 0.0
        graph = hit[0].get("graph_us", float("nan"))
        return graph / 1e3 if graph == graph else hit[0]["ms"]

    fig, axes = new_figure(2, 2, figsize=(15.5, 10.4))

    captured = any(r.get("graph_us") == r.get("graph_us") for r in data)
    _grouped_bars(axes[0], shapes, names, device_ms, pal, "{:.2f}")
    finish_axis(
        axes[0],
        "",
        "ms of device time" if captured else "ms of wall time",
        f"latency, T={meta['config']['tokens']} tokens (lower is better)",
        legend=True,
    )

    _grouped_bars(axes[1], shapes, names, value("tflops"), pal)
    axes[1].axhline(
        TENSOR_MAMF_TFLOPS, color=CEIL, ls="--", lw=1.2, label="bf16 MAMF 270"
    )
    finish_axis(
        axes[1], "", "effective TFLOP/s", "compute rate against MAMF", legend=True
    )

    _grouped_bars(axes[2], shapes, names, value("gbps"), pal)
    if peak_bw:
        axes[2].axhline(
            peak_bw, color=CEIL, ls="--", lw=1.2, label=f"STREAM {peak_bw:.0f}"
        )
    finish_axis(
        axes[2],
        "",
        "effective GB/s",
        "traffic against measured DRAM (bytes are a floor)",
        legend=True,
    )

    _grouped_bars(axes[3], shapes, names, value("peak_gib"), pal, "{:.2f}")
    finish_axis(axes[3], "", "peak GiB", "peak allocated during the step", legend=True)

    fig.suptitle(
        f"MoE formulations, {stage.replace('_', ' ')} -- {meta['device']}\n"
        "same E / k / expert geometry and the same gate matrix per shape; "
        "ReMoE dispatches 4k padded slots, Kimi-K3 latent runs a half-size bank",
        fontsize=11,
    )
    save_figure(fig, os.path.join(out_dir, f"moe_{stage}.png"))


def plot_choice(rows, out_dir, meta):
    """What decides the choice: fixable overhead, and cost per unit capacity."""
    layer = _pick(rows, stage="layer_fwd_bwd")
    forward = _pick(rows, stage="layer_fwd")
    router = _pick(rows, stage="router")
    pal = Palette()
    for name in ORDER:
        pal.color(name)
    shapes = shape_labels(layer)
    names = _present(layer)

    fig, axes = new_figure(1, 3, figsize=(19.5, 5.8))

    # Router stage alone: this is the panel that separates "slow because our
    # kernel does not cover this gate" from "slow because of the formulation".
    #
    # `graph_us`, not `ms`. A router is one 15-34 us kernel behind ~155 us of
    # Python, and the wall-clock timer reports the Python, bimodally -- see the
    # note in moe_formulations.py. Device time is the only figure that says
    # anything about the gate itself.
    def router_us(dim, name):
        hit = _pick(router, dim=dim, formulation=name)
        return hit[0]["graph_us"] if hit else 0.0

    def is_eager(name):
        hit = _pick(router, formulation=name)
        return bool(hit) and hit[0]["router_path"] != "fused"

    _grouped_bars(axes[0], shapes, names, router_us, pal, hatch_of=is_eager)
    finish_axis(
        axes[0],
        "",
        "router stage, us of device time",
        "router alone -- hatched bars are on the eager path (fixable, not intrinsic)",
        legend=True,
    )

    # Cost per unit of capacity. Without this panel a formulation that simply
    # runs a smaller expert bank reads as a speed win.
    # One shape, not four: six labelled bars per group collide, and the flagship
    # is the only shape where the layer is device-bound at every formulation --
    # at d=640 host issue exceeds device time and the ranking is Python's, not
    # the formulation's.
    def per_active(name):
        hit = _pick(forward, dim=FLAGSHIP, formulation=name)
        if not hit or not hit[0]["active_params"]:
            return 0.0
        return hit[0]["graph_us"] / (hit[0]["active_params"] / 1e6)

    xs = list(range(len(names)))
    bars = axes[1].bar(
        xs, [per_active(n) for n in names], 0.62, color=[pal.color(n) for n in names]
    )
    bar_labels(axes[1], bars, "{:.1f}")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    finish_axis(
        axes[1],
        "",
        "us of forward device time per M active parameter",
        f"capacity-adjusted cost at d={FLAGSHIP} (lower is better)",
    )

    # Absolute capacity, so the panel above is readable as a trade rather than a
    # correction: a bar that is short in both is genuinely cheaper.
    flagship = _pick(layer, dim=FLAGSHIP)
    xs = list(range(len(names)))
    total = [
        (
            _pick(flagship, formulation=n)[0]["params"] / 1e6
            if _pick(flagship, formulation=n)
            else 0.0
        )
        for n in names
    ]
    active = [
        (
            _pick(flagship, formulation=n)[0]["active_params"] / 1e6
            if _pick(flagship, formulation=n)
            else 0.0
        )
        for n in names
    ]
    bars = axes[2].bar(
        [x - 0.2 for x in xs], total, 0.4, color=[pal.color(n) for n in names]
    )
    bar_labels(axes[2], bars, "{:.0f}")
    bars = axes[2].bar(
        [x + 0.2 for x in xs],
        active,
        0.4,
        color=[pal.color(n) for n in names],
        alpha=0.45,
        hatch="//",
    )
    bar_labels(axes[2], bars, "{:.1f}")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    finish_axis(
        axes[2],
        "",
        "M parameters",
        f"capacity at d={FLAGSHIP} (solid: total, hatched: active per token)",
    )

    fig.suptitle(
        f"Choosing a MoE formulation -- {meta['device']}, "
        f"T={meta['config']['tokens']}\n"
        f"router ceiling for reference: TF32 MAMF {TF32_MAMF_TFLOPS:.0f} TFLOP/s; "
        "an eager router costs ~0.25 ms of per-op dispatch regardless of shape",
        fontsize=11,
    )
    save_figure(fig, os.path.join(out_dir, "moe_formulations_choice.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/moe/moe_formulations")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "moe_formulations.json")) as handle:
        payload = json.load(handle)
    rows = payload["rows"]

    # Two metric figures. The forward one is device time from a graph replay and
    # is the comparable number; the fwd+bwd one is wall clock, which is what a
    # training step pays but which at the smaller shapes includes host issue time
    # exceeding the device work.
    plot_metrics(rows, args.dir, payload, stage="layer_fwd")
    plot_metrics(rows, args.dir, payload, stage="layer_fwd_bwd")
    plot_choice(rows, args.dir, payload)
    print(f"wrote figures to {args.dir}")


if __name__ == "__main__":
    main()
