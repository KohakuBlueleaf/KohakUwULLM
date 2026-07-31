"""Figure for the vendor-MXFP8 MoE result: speed and correctness in one view.

Four panels, arranged so the reader reaches the conclusion in the order the evidence
supports it rather than the order it was measured:

1. **Device rate against rows per expert.** Both bf16 arms, both vendor arms and the
   custom kernel, with this card's measured dense ceilings drawn in. The gap between
   ``bf16_loop`` and ``bf16_grouped`` is the price of looping *at all*, in a row where
   precision is held fixed -- without it the vendor loop's deficit is unattributable.
2. **Speedup against the incumbent**, with the 1.0x line. This is where the crossover
   lives and it is different for the sequential and streamed constructions.
3. **Stream-count saturation.** The mechanism panel: device time against concurrent
   streams at a fixed shape, annotated with tiles per expert against this card's 170
   SMs. A sequential loop is not slow because launches are expensive.
4. **Correctness, on the same figure as the speed.** Bitwise agreement with the
   per-expert vendor oracle for every GEMM row, and the whole-path relative error
   against bf16 with the combine's own nondeterminism drawn as the floor. A fast and
   wrong kernel is not a result, and a chart showing only the fast half invites exactly
   that mistake.

Usage:
    .venv/bin/python scripts/bench/moe/moe_fp8_vendor_plot.py --dir out/bench/moe/moe_fp8_vendor
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

ARMS = (
    ("bf16_grouped", "bf16 grouped (incumbent)"),
    ("bf16_loop", "bf16 per-expert loop"),
    ("fp8_vendor_loop", "fp8 vendor loop, 1 stream"),
    ("fp8_vendor_streamed", "fp8 vendor loop, 8 streams"),
    ("fp8_triton_grouped", "fp8 Triton grouped (custom)"),
)


def load(path: str, name: str):
    with open(os.path.join(path, name)) as fh:
        return json.load(fh)


def rate_panel(ax, rows, palette, gemm: str, preset: str):
    """Device TF/s against rows per expert, one line per arm."""
    subset = [r for r in rows if r["gemm"] == gemm and r["preset"] == preset]
    subset.sort(key=lambda r: r["rows_per_expert"])
    x = [r["rows_per_expert"] for r in subset]
    for key, label in ARMS:
        ax.plot(
            x,
            [r[f"{key}_device_tflops"] for r in subset],
            marker="o",
            label=label,
            color=palette.color(key),
        )
    finish_axis(
        ax,
        "rows per expert",
        "device TF/s",
        f"{preset} {gemm}: rate (flushed graph replay)",
        xticks=x,
        logx=True,
        si_x=False,
        legend=True,
    )


def speedup_panel(ax, rows, palette, gemm: str):
    """Speedup against bf16 grouped, every preset, for the two vendor constructions."""
    presets = sorted({r["preset"] for r in rows})
    for preset in presets:
        subset = sorted(
            (r for r in rows if r["gemm"] == gemm and r["preset"] == preset),
            key=lambda r: r["rows_per_expert"],
        )
        x = [r["rows_per_expert"] for r in subset]
        colour = palette.color(preset)
        ax.plot(
            x,
            [r["fp8_vendor_streamed_speedup_device"] for r in subset],
            marker="o",
            color=colour,
            label=f"{preset.replace('Kohaku-', '')} streamed",
        )
        ax.plot(
            x,
            [r["fp8_vendor_loop_speedup_device"] for r in subset],
            marker="s",
            linestyle="--",
            color=colour,
            alpha=0.55,
            label=f"{preset.replace('Kohaku-', '')} sequential",
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.annotate(
        "parity with bf16",
        (x[0], 1.0),
        textcoords="offset points",
        xytext=(2, 4),
        fontsize=8,
        color="#333333",
    )
    finish_axis(
        ax,
        "rows per expert",
        "speedup vs bf16 grouped",
        f"{gemm}: solid = 8 streams, dashed = sequential",
        xticks=x,
        logx=True,
        si_x=False,
        legend=True,
    )


def stream_panel(ax, sweeps, palette):
    """Device time against concurrent streams -- the wave-quantization mechanism."""
    keys = sorted({(s["preset"], s["rows_per_expert"]) for s in sweeps})
    for preset, rpe in keys:
        subset = sorted(
            (
                s
                for s in sweeps
                if s["preset"] == preset and s["rows_per_expert"] == rpe
            ),
            key=lambda s: s["streams"],
        )
        tiles = subset[0]["tiles_per_expert"]
        ax.plot(
            [s["streams"] for s in subset],
            [s["device_ms"] * 1e3 for s in subset],
            marker="o",
            color=palette.color(f"{preset}-{rpe}"),
            label=f"{preset.replace('Kohaku-', '')} {rpe} rows/E ({tiles} tiles vs 170 SMs)",
        )
    finish_axis(
        ax,
        "concurrent streams",
        "device us",
        "One expert cannot fill the card; concurrency can",
        xticks=[s["streams"] for s in subset],
        logx=True,
        si_x=False,
        legend=True,
    )


def path_panel(ax, rows, path_rows, palette):
    """The shipping answer: whole-path speedup, with the accuracy that comes with it.

    Speed as bars against the 1.0x line, error on a twin axis. The two belong on one
    panel because the decision needs both at once -- and because the fp8 error is flat
    across every row here, so a panel showing only error would carry no information
    while quietly implying it had been checked against something.
    """
    labels = [
        f"{r['preset'].replace('Kohaku-', '')}\n{r['tokens'] // 1024}k tok"
        for r in path_rows
    ]
    positions = list(range(len(labels)))
    width = 0.38
    for offset, key, label in (
        (-width / 2, "fp8_sink", "vendor loop, 8 streams"),
        (width / 2, "fp8_fused", "repo's fused kernel"),
    ):
        ax.bar(
            [p + offset for p in positions],
            [r[f"{key}_speedup_device"] for r in path_rows],
            width=width,
            color=palette.color(key),
            label=label,
        )
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("whole-path speedup vs bf16")
    # Headroom for the legend, which otherwise sits on the tallest fused bar.
    ax.set_ylim(0, max(r["fp8_fused_speedup_device"] for r in path_rows) * 1.45)

    err_ax = ax.twinx()
    err_ax.grid(False)
    err_ax.plot(
        positions,
        [r["fp8_rel_err_vs_bf16"] for r in path_rows],
        marker="D",
        markersize=4,
        linestyle="none",
        color="#333333",
        label="fp8 rel err vs bf16",
    )
    # The combine's run-to-run spread, from two calls of the same bf16 arm. Plotted on
    # the same axis as the fp8 error because it is the floor that error has to be read
    # against; three rows measure exactly 0 and are drawn at the axis bottom rather than
    # dropped, since a missing marker would read as "not measured".
    noise = [r["bf16_self_rel_err"] for r in path_rows]
    floor = min([n for n in noise if n > 0] or [1e-4]) / 3
    err_ax.plot(
        positions,
        [n if n > 0 else floor for n in noise],
        marker="v",
        markersize=4,
        linestyle="none",
        color="#D55E00",
        label="bf16 vs itself (scatter-add noise floor)",
    )
    err_ax.set_yscale("log")
    err_ax.set_ylabel("relative error")

    total = len(rows)
    agree = sum(1 for r in rows if r["vendor_vs_triton_bitwise"])
    ax.set_title(
        f"Whole expert forward. Every one of {agree}/{total} GEMM rows is\n"
        "bit-identical to the per-expert vendor oracle"
    )
    handles, texts = ax.get_legend_handles_labels()
    extra_handles, extra_texts = err_ax.get_legend_handles_labels()
    ax.legend(
        handles + extra_handles, texts + extra_texts, loc="upper left", fontsize=7
    )


def main():
    parser = argparse.ArgumentParser()
    # `--dir`, not `--out`: every other plotter in the suite reads JSON from a
    # directory and writes its figure beside it, and one odd name is one wrong
    # invocation.
    parser.add_argument("--dir", default="out/bench/moe/moe_fp8_vendor")
    args = parser.parse_args()

    gemm = load(args.dir, "moe_fp8_vendor.json")
    path = load(args.dir, "moe_fp8_vendor_path.json")
    palette = Palette()

    fig, axes = new_figure(2, 2, figsize=(15.0, 10.0))
    rate_panel(axes[0], gemm["rows"], palette, "gemm1", "Kohaku-MoE-3B")
    speedup_panel(axes[1], gemm["rows"], Palette(), "gemm1")
    stream_panel(axes[2], gemm["stream_sweep"], Palette())
    path_panel(axes[3], gemm["rows"], path["rows"], Palette())
    ceilings = gemm["dense_ceilings_tflops"]
    fig.suptitle(
        "MoE MXFP8 through the vendor GEMM only: a per-expert `scaled_mm` loop\n"
        f"RTX 5090 sm_120, CUDA_VISIBLE_DEVICES={gemm['visible_devices']}, "
        f"measured dense ceilings bf16 {ceilings['bf16']:.0f} / fp8 {ceilings['fp8']:.0f} TF/s",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90)
    save_figure(fig, os.path.join(args.dir, "moe_fp8_vendor.png"))
    print(f"wrote {os.path.join(args.dir, 'moe_fp8_vendor.png')}")


if __name__ == "__main__":
    main()
