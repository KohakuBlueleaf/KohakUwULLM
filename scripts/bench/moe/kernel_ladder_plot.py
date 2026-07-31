"""Draw the kernel ladder: what each rewrite bought, step by step.

The figure exists to stop a single number standing in for four decisions. A bar labelled
"14x faster" invites the reading that MXFP8 did it; the ladder shows that most of that
factor was banked earlier by writing a grouped kernel, and that the part attributable to
MXFP8 is the last two rungs.

Left panel is the MoE expert path, annotated with the **increment** between adjacent
rungs rather than only the cumulative height, because the increment is the thing a
decision was made about. The shipped bf16 baseline is marked, since every end-to-end
ratio in `kohaku_e2e` is measured from there and not from the eager floor.

Right panel is the quantizer, which is bandwidth-bound, so it is scored in GB/s against
this box's measured 1791 GB/s DRAM peak rather than in absolute time.

Usage::

    .venv/bin/python scripts/bench/moe/kernel_ladder_plot.py --dir out/bench/moe/kernel_ladder
"""

import argparse
import json
import os

from kohakuwullm.bench.core.plotting import (
    Palette,
    finish_axis,
    new_figure,
    save_figure,
)

DRAM_PEAK_GBPS = 1791.0
# The rung the end-to-end sweep compares against. Everything below it was already in
# place before the MXFP8 work started.
SHIPPED_BASELINE = "2-grouped-bf16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/bench/moe/kernel_ladder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.dir, "kernel_ladder.json")) as handle:
        data = json.load(handle)

    moe = [r for r in data["moe"] if "error" not in r]
    quant = [r for r in data["quantize"] if "error" not in r]
    if not moe:
        print("no usable MoE rows")
        return

    palette = Palette()
    fig, axes = new_figure(1, 2, figsize=(14, 5.5))
    ax_moe, ax_quant = axes

    presets = sorted({r["preset"] for r in moe})
    stages = sorted({r["stage"] for r in moe})
    width = 0.8 / max(len(presets), 1)

    for i, preset in enumerate(presets):
        rows = {r["stage"]: r for r in moe if r["preset"] == preset}
        heights = [rows[s]["ms"] if s in rows else 0.0 for s in stages]
        offs = [j + (i - (len(presets) - 1) / 2) * width for j in range(len(stages))]
        ax_moe.bar(offs, heights, width, label=preset, color=palette.color(preset))
        # The increment, not the cumulative factor: each rung was a separate decision
        # and the question is what that decision was worth.
        for j in range(1, len(stages)):
            if heights[j] <= 0 or heights[j - 1] <= 0:
                continue
            ax_moe.annotate(
                f"{heights[j - 1] / heights[j]:.2f}x",
                (offs[j], heights[j]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
            )

    if SHIPPED_BASELINE in stages:
        ax_moe.axvline(stages.index(SHIPPED_BASELINE), color="crimson", ls="--", lw=1)
        ax_moe.annotate(
            "end-to-end ratios are measured from here",
            (stages.index(SHIPPED_BASELINE), max(r["ms"] for r in moe) * 0.6),
            textcoords="offset points",
            xytext=(8, 0),
            color="crimson",
            fontsize=8,
        )

    finish_axis(
        ax_moe,
        "",
        "ms per forward",
        title="MoE expert path: what each rewrite bought",
        logy=True,
        legend=True,
    )
    ax_moe.set_xticks(range(len(stages)))
    ax_moe.set_xticklabels(stages, rotation=20, ha="right", fontsize=8)

    shapes = sorted({tuple(r["shape"]) for r in quant})
    qstages = sorted({r["stage"] for r in quant})
    qwidth = 0.8 / max(len(shapes), 1)
    for i, shape in enumerate(shapes):
        rows = {r["stage"]: r for r in quant if tuple(r["shape"]) == shape}
        heights = [rows[s]["gbps"] if s in rows else 0.0 for s in qstages]
        offs = [j + (i - (len(shapes) - 1) / 2) * qwidth for j in range(len(qstages))]
        ax_quant.bar(
            offs,
            heights,
            qwidth,
            label=f"{shape[0]}x{shape[1]}",
            color=palette.color(str(shape)),
        )

    ax_quant.axhline(DRAM_PEAK_GBPS, color="crimson", ls="--", lw=1)
    ax_quant.annotate(
        "1791 GB/s measured DRAM peak",
        (0, DRAM_PEAK_GBPS),
        textcoords="offset points",
        xytext=(4, 4),
        color="crimson",
        fontsize=8,
    )
    finish_axis(
        ax_quant,
        "",
        "effective GB/s",
        title="MX quantizer: bandwidth-bound, so scored against DRAM",
        legend=True,
    )
    ax_quant.set_xticks(range(len(qstages)))
    ax_quant.set_xticklabels(qstages, rotation=20, ha="right", fontsize=8)

    fig.suptitle(
        "Kernel ladder -- the increment between rungs is what each decision was worth",
        fontsize=10,
    )
    path = args.out or os.path.join(args.dir, "kernel_ladder.png")
    save_figure(fig, path)
    print(f"wrote {path}")

    for preset in presets:
        rows = {r["stage"]: r for r in moe if r["preset"] == preset}
        if SHIPPED_BASELINE in rows and "4-fused-fp8" in rows:
            shipped = rows[SHIPPED_BASELINE]["ms"] / rows["4-fused-fp8"]["ms"]
            total = rows["1-eager"]["ms"] / rows["4-fused-fp8"]["ms"]
            print(
                f"{preset:15s} total {total:5.1f}x from eager, "
                f"{shipped:5.2f}x from the shipped bf16 baseline"
            )


if __name__ == "__main__":
    main()
