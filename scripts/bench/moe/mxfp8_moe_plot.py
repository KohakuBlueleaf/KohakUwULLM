"""Draw the fused-MXFP8-MoE figure from ``mxfp8_moe.json``.

Split from the measurement half for the same reason as ``kernels.py`` /
``kernels_plot.py``: a re-measurement should not touch the drawing, and a rejected
layout should not force a re-run of a 15-minute sweep on a contended box.

Six panels, because the claim has that many parts and no one of them stands alone:

* **throughput** of the fused path against the bf16 path it replaces, forward and
  forward+backward. The backward panel is the one that decides anything -- a
  forward-only win is not a training speedup.
* **launch counts**, split into the GEMMs and everything else. This is the
  mechanism, and the split is the whole point: the two paths issue the *same*
  number of matmuls, so every launch the fusion removes is a launch that was not
  arithmetic. A speedup panel without this one invites "the fp8 GEMM is faster",
  which at these shapes is only about half of it.
* the **bare grouped GEMM against its binding roofline** across rows per expert,
  with the realistic microbatch band marked. Which roofline binds flips inside that
  band -- bandwidth below ~450 rows, compute above -- so the dotted line is
  ``min(bandwidth, compute)`` per shape rather than either one alone.
* **host share**, which is why three presets appear in no timing panel. Above a 50%
  host share ``wall - host`` is a difference of two comparable numbers and stops
  being evidence, and the rows that fail it are exactly the rows this panel is for.
* **accuracy in the same figure as the speed**, per the house rule. A fused path
  that is 2x faster and wrong is not a result, and a figure that reports only the
  fast half is how that ships.

Three filters, one per claim, rather than one global one: a high host share
disqualifies a device-time estimate, a contended *baseline* arm disqualifies only a
ratio, and neither touches a relative error or a launch count.

Usage:
    .venv/bin/python scripts/bench/moe/mxfp8_moe_plot.py --dir out/bench/moe/mxfp8_moe
"""

import argparse
import json
import os

from kohakuwullm.bench import (
    bar_labels,
    finish_axis,
    new_figure,
    save_figure,
)

# Rows per expert a real microbatch lands in. Below it is the gradient-accumulation
# case, above it is a batch this card has no memory for.
REGIME = (384, 768)
# Reference lines are recessive and neutral. A roofline drawn in a categorical hue
# competes with the series it is a reference *for*, and when that hue is also a
# series' colour the two read as the same thing.
CEIL = "#333333"
BAND = "#F0E442"
# Assigned explicitly rather than drawn from `Palette` in call order, because the
# order the panels run in is not a meaning. **Hue is the precision, lightness is
# forward vs forward+backward** -- one thing to learn, and it holds in every panel, so
# a reader who learns "orange is MXFP8" in the first panel still knows it in the last.
# Left to assignment order, `bf16` came out blue in one panel and yellow in another,
# and `matmul` landed on Okabe-Ito yellow under white text.
BF16_FWD = "#56B4E9"
BF16_STEP = "#0072B2"
FP8_FWD = "#E69F00"
FP8_STEP = "#D55E00"
MATMUL = "#0072B2"
OTHER = "#7F7F7F"
SHAPE_COLORS = {"gemm1": "#0072B2", "gemm2": "#009E73"}


def _is_matmul(name: str) -> bool:
    """Whether a kernel name is one of the path's matrix products.

    Name-matched rather than counted from a hand-written list per path, so a future
    kernel lands in the right bucket without an edit here. ``wgrad`` is included and
    ``combine`` is not: the combine is a scatter-add whose cost the fusion removes,
    while both weight gradients are irreducible arithmetic that only moves.
    """
    return "gemm" in name.lower() or "wgrad" in name.lower()


def _split_launches(counts: dict) -> tuple[int, int]:
    """``(matmuls, everything else)`` for one measured launch listing."""
    matmul = sum(_is_matmul(name) for name in counts["names"])
    return matmul, counts["total"] - matmul


def _label(preset: str, tokens: int) -> str:
    return f"{preset.replace('MoE-', '')}\n{tokens // 1024}k tok"


def _clean(rows: list[dict]) -> list[dict]:
    """Rows whose fp8 measurement is admissible.

    A ``suspect`` row is not evidence, so it is not drawn. Dropping it silently
    would be worse than plotting it, so the caller prints what was dropped.
    """
    return [row for row in rows if not row["suspect"]]


def _clean_ratio(rows: list[dict]) -> list[dict]:
    """Rows whose *ratio* is admissible -- both arms clean, not just the fp8 one.

    Stricter than :func:`_clean`, and what both panels that *draw the bf16 arm* need
    -- the throughput panel as much as the speedup one, since a bar is as much a
    claim as a ratio. A contended bf16 baseline leaves the fp8 rate untouched, which
    is why :func:`_clean` exists for the GEMM panel; it does not make the baseline
    bar drawable.
    """
    return [row for row in rows if not row["suspect"] and not row["suspect_speedup"]]


def throughput_panel(ax, paths):
    """Fused against eager, forward and forward+backward, in TF/s."""
    labels = [_label(row["preset"], row["tokens"]) for row in paths]
    positions = range(len(paths))
    width = 0.2
    series = (
        ("bf16 fwd", "fwd_bf16_tflops", -1.5, BF16_FWD),
        ("MXFP8 fwd", "fwd_fp8_tflops", -0.5, FP8_FWD),
        ("bf16 fwd+bwd", "step_bf16_tflops", 0.5, BF16_STEP),
        ("MXFP8 fwd+bwd", "step_fp8_tflops", 1.5, FP8_STEP),
    )
    for name, key, offset, color in series:
        bars = ax.bar(
            [p + offset * width for p in positions],
            [row[key] for row in paths],
            width,
            label=name,
            color=color,
        )
        bar_labels(ax, bars)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    finish_axis(
        ax,
        "",
        "TFLOP/s",
        "Fused MXFP8 expert path vs the bf16 path it replaces",
        legend=True,
    )


def speedup_panel(ax, paths):
    """The ratio the panel above implies, which is what a training step feels."""
    labels = [_label(row["preset"], row["tokens"]) for row in paths]
    positions = range(len(paths))
    width = 0.35
    for name, key, offset, color in (
        ("forward", "fwd_speedup", -0.5, FP8_FWD),
        ("fwd+bwd", "step_speedup", 0.5, FP8_STEP),
    ):
        bars = ax.bar(
            [p + offset * width for p in positions],
            [row[key] for row in paths],
            width,
            label=name,
            color=color,
        )
        bar_labels(ax, bars, fmt="{:.2f}x")
    ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2, label="parity")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    finish_axis(ax, "", "speedup over bf16", "Speedup, fused vs eager", legend=True)


def launch_panel(ax, paths):
    """Launches per MoE layer, split into matmuls and everything else.

    Stacked rather than grouped: the reader's question is what the *total* is and
    how much of it is arithmetic, and a grouped pair makes the total something to
    add up by eye.
    """
    row = paths[0]
    arms = (
        ("bf16\nfwd", "launch_fwd_bf16"),
        ("MXFP8\nfwd", "launch_fwd_fp8"),
        ("bf16\nfwd+bwd", "launch_step_bf16"),
        ("MXFP8\nfwd+bwd", "launch_step_fp8"),
    )
    # A launch count is per layer and structural, so it must not vary with the preset.
    # Asserted rather than assumed, because if it did vary the single row drawn here
    # would be a claim about one preset wearing a title that says "every".
    varying = [
        key
        for _, key in arms
        if len({tuple(other[key]["names"]) for other in paths}) != 1
    ]
    if varying:
        raise SystemExit(f"launch count varies across presets for {varying}")
    positions = range(len(arms))
    matmuls = [_split_launches(row[key])[0] for _, key in arms]
    others = [_split_launches(row[key])[1] for _, key in arms]
    lower = ax.bar(list(positions), matmuls, 0.6, label="matmul", color=MATMUL)
    upper = ax.bar(
        list(positions),
        others,
        0.6,
        bottom=matmuls,
        label="everything else",
        color=OTHER,
    )
    for index, (bar, total) in enumerate(
        zip(upper, [m + o for m, o in zip(matmuls, others)])
    ):
        ax.annotate(
            f"{total}",
            (bar.get_x() + bar.get_width() / 2, total),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        del index
    for bar, value in zip(lower, matmuls):
        ax.annotate(
            f"{value}",
            (bar.get_x() + bar.get_width() / 2, value / 2),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels([name for name, _ in arms])
    finish_axis(
        ax,
        "",
        "device launches per MoE layer",
        "Launch count (identical at every preset) -- same matmuls, less around them",
        legend=True,
    )


def gemm_panel(ax, gemms):
    """Bare grouped MXFP8 GEMM against both rooflines, over rows per expert."""
    ax.axvspan(*REGIME, color=BAND, alpha=0.25, zorder=0, label="real microbatch")
    for name in ("gemm1", "gemm2"):
        rows = sorted(
            [row for row in gemms if row["name"] == name],
            key=lambda row: row["rows_per_expert"],
        )
        ax.plot(
            [row["rows_per_expert"] for row in rows],
            [row["fp8_tflops"] for row in rows],
            marker="o",
            label=f"{name} ({rows[0]['N']}x{rows[0]['K']})",
            color=SHAPE_COLORS[name],
        )
        # The **binding** roofline, i.e. min(bandwidth, compute), not the bandwidth
        # curve alone. Above ~700 rows the bandwidth number runs to 1200 TF/s on a card
        # that has never exceeded 692, so plotting it squeezed the measurements this
        # panel is about into the bottom third of the axis while implying headroom that
        # does not exist.
        ax.plot(
            [row["rows_per_expert"] for row in rows],
            [row["ceiling_tflops"] for row in rows],
            linestyle=":",
            linewidth=1.4,
            color=SHAPE_COLORS[name],
        )
    compute = gemms[0]["compute_ceiling_tflops"]
    ax.axhline(
        compute,
        color=CEIL,
        linestyle="--",
        linewidth=1.2,
        label=f"{compute:.0f} demonstrated dense MXFP8",
    )
    finish_axis(
        ax,
        "rows per expert",
        "TFLOP/s",
        "Grouped MXFP8 GEMM; dotted = each shape's own binding roofline",
        # Ticks from every shape present, not from gemm1's rows: a row dropped for
        # contention is dropped per shape, so gemm2 can carry a point that gemm1's
        # tick list does not have.
        xticks=sorted({row["rows_per_expert"] for row in gemms}),
        si_x=False,
        legend=True,
    )


def cast_saving_panel(ax, casts):
    """Measured saving from folding the dw cast, against its bandwidth upper bound.

    The pair is the point. A saved round trip only costs DRAM traffic if the buffer
    misses L2, so the at-peak bar is what the saving *would* be if every removed byte
    came from memory. Where the measured bar falls well short, the cast was being
    served from cache -- which is the same effect, with the opposite sign, as a norm
    epilogue whose fusion loses below the L2 boundary.
    """
    labels = [_label(row["preset"], row["tokens"]) for row in casts]
    positions = range(len(casts))
    width = 0.35
    for name, key, offset, color in (
        ("measured", "saved_ms", -0.5, FP8_STEP),
        ("if all removed bytes were DRAM", "saved_ms_at_peak_bw", 0.5, OTHER),
    ):
        bars = ax.bar(
            [p + offset * width for p in positions],
            [row[key] for row in casts],
            width,
            label=name,
            color=color,
        )
        bar_labels(ax, bars, fmt="{:.2f}")
    ax.set_xticks(list(positions))
    # A rung whose dw_out fits in L2 is named on the axis, since it is the one place
    # the two bars are expected to disagree for a reason that is not a defect.
    ax.set_xticklabels(
        [
            f"{label}\n(dw_out in L2)" if row["dw_out_fits_l2"] else label
            for label, row in zip(labels, casts)
        ]
    )
    finish_axis(
        ax,
        "",
        "ms saved per MoE layer (2 WGRAD launches)",
        "Folding the fp32 weight-gradient cast into the WGRAD epilogue",
        legend=True,
    )


def cast_speedup_panel(ax, casts):
    """The same saving as a ratio on the two WGRAD launches it applies to."""
    labels = [_label(row["preset"], row["tokens"]) for row in casts]
    positions = range(len(casts))
    bars = ax.bar(
        list(positions),
        [row["speedup"] for row in casts],
        0.55,
        label="folded vs fp32-buffer-then-cast",
        color=FP8_FWD,
    )
    bar_labels(ax, bars, fmt="{:.2f}x")
    ax.axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2, label="parity")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    finish_axis(
        ax,
        "",
        "speedup on the WGRAD pair",
        "Bit-identical output, so this is traffic removed rather than work skipped",
        legend=True,
    )


def host_share_panel(ax, paths):
    """Dispatch as a fraction of wall, per preset -- why some rows cannot be timed.

    Drawn from **every** row, including the ones the throughput panels drop. Host
    share is not a timing estimate, it is the reason an estimate is or is not
    admissible, so the rows disqualified by it are exactly the rows this panel is
    about. Above the 50% line ``wall - host`` is a difference of two comparable
    numbers and stops being evidence of device time; the launch-count panel is what
    speaks for those presets.
    """
    labels = [_label(row["preset"], row["tokens"]) for row in paths]
    positions = range(len(paths))
    width = 0.35
    for name, key, offset, color in (
        ("bf16", "host_share_bf16", -0.5, BF16_STEP),
        ("MXFP8", "host_share_fp8", 0.5, FP8_STEP),
    ):
        bars = ax.bar(
            [p + offset * width for p in positions],
            [100.0 * row[key] for row in paths],
            width,
            label=name,
            color=color,
        )
        bar_labels(ax, bars, fmt="{:.0f}%")
    ax.axhline(
        50.0,
        color=CEIL,
        linestyle="--",
        linewidth=1.2,
        label="above: wall-host is not evidence",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    finish_axis(
        ax,
        "",
        "host dispatch as % of wall (fwd+bwd)",
        "What the host pays -- and which rows can be timed at all",
        legend=True,
    )
    # The house default is a frameless legend, overridden only here: the 50% rule line
    # runs straight through the label text at every axis limit this panel can take.
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")


def accuracy_panel(ax, paths, gemms):
    """Relative error against the bf16 path, beside the speed that cost it.

    Drawn from every row regardless of ``suspect``: this is a correctness
    measurement, and nothing about a contended card or a high host share changes
    what the two paths computed.
    """
    labels = [_label(row["preset"], row["tokens"]) for row in paths]
    positions = range(len(paths))
    bars = ax.bar(
        list(positions),
        [row["rel_err_vs_bf16"] for row in paths],
        0.55,
        label="fused path vs bf16 path",
        color=FP8_STEP,
    )
    bar_labels(ax, bars, fmt="{:.4f}")
    gemm_err = max(row["rel_err"] for row in gemms)
    ax.axhline(
        gemm_err,
        color=CEIL,
        linestyle="--",
        linewidth=1.2,
        label=f"bare GEMM vs bf16 ({gemm_err:.4f})",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    finish_axis(
        ax,
        "",
        "relative L2 error",
        "Accuracy -- the cost side of every number above",
        legend=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="out/bench/moe/mxfp8_moe")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "mxfp8_moe.json")) as fh:
        data = json.load(fh)

    paths, gemms = data["paths"], data["gemms"]
    for label, rows in (("path", paths), ("gemm", gemms)):
        for row in rows:
            if row["suspect"]:
                print(f"throughput excludes this {label} row: {row['suspect']}")
            elif row["suspect_speedup"]:
                print(f"ratio only excludes this {label} row: {row['suspect_speedup']}")
    clean_paths, clean_gemms = _clean(paths), _clean(gemms)
    ratio_paths = _clean_ratio(paths)
    if not clean_paths or not clean_gemms or not ratio_paths:
        raise SystemExit("no rows survived the suspect filter; re-measure first")

    # `step_bf16_tflops` is not in the JSON -- the measurement half reports the fp8
    # rate and the ratio, which is what its own table needs. Derived here rather than
    # added there so a re-run is not required to draw a panel.
    for row in paths:
        row["step_bf16_tflops"] = row["step_fp8_tflops"] / row["step_speedup"]

    fig, axes = new_figure(2, 3, figsize=(21.0, 10.5))
    # Three filters, one per claim. The throughput and speedup panels both draw the
    # bf16 arm, so both need *both* arms admissible; the GEMM panel draws only the fp8
    # rate, so a contended baseline does not disqualify it -- which is where the split
    # pays, at 11 of 12 GEMM rows instead of 9. Neither filter touches a relative error
    # or a launch count, which are not timings at all.
    throughput_panel(axes[0], ratio_paths)
    speedup_panel(axes[1], ratio_paths)
    launch_panel(axes[2], paths)
    gemm_panel(axes[3], clean_gemms)
    host_share_panel(axes[4], paths)
    accuracy_panel(axes[5], paths, gemms)
    fig.suptitle(
        f"Fused MXFP8 MoE expert path -- {data['device']}, "
        f"CUDA_VISIBLE_DEVICES={data.get('visible_devices', '?')}; "
        f"timing panels show only rows that passed every admissibility check",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, os.path.join(args.dir, "mxfp8_moe.png"))

    # Its own figure rather than a seventh panel: "what did folding the cast buy" is a
    # different question from "is the fused path faster than bf16", and the presets
    # admissible for it are different too -- a bit-identical traffic saving needs no
    # host-share cut, so all four rungs appear here.
    casts = [row for row in data.get("casts", []) if not row["suspect"]]
    if casts:
        fig, axes = new_figure(1, 2, figsize=(15.0, 5.2))
        cast_saving_panel(axes[0], casts)
        cast_speedup_panel(axes[1], casts)
        fig.suptitle(
            f"WGRAD fp32 cast removal -- {data['device']}, "
            f"CUDA_VISIBLE_DEVICES={data.get('visible_devices', '?')}",
            fontsize=12,
            fontweight="bold",
        )
        save_figure(fig, os.path.join(args.dir, "mxfp8_moe_casts.png"))
        print(f"wrote {args.dir}/mxfp8_moe_casts.png")
    print(f"wrote {args.dir}/mxfp8_moe.png")


if __name__ == "__main__":
    main()
