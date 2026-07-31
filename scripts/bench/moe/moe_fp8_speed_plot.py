"""Draw the MoE fp8 speed figure from ``moe_fp8_speed.json`` and the A/B arm JSONs.

Split from the measurement half for the same reason as ``mxfp8_moe_plot.py``: a
rejected layout should not force a re-run on a shared box.

Four panels, because the claim has four parts:

* **the real step**, which is the only panel that settles anything. A per-kernel win
  that does not survive a training loop is what this whole investigation was about,
  so the arms measured over 400 real steps go first and the microbenchmarks come after.
* **where the fused path's device time goes**, stacked per kernel, per preset. The
  point of the stack is that ``grouped_mxfp8_wgrad_kernel`` is 38-43% of it at every
  shape -- the fused GEMMs are not the constraint, and a reader who sees only a
  speedup number cannot tell that.
* **speedup against the bf16 path** across the preset ladder, which is what kills the
  narrow-expert hypothesis: the narrowest preset is not the slowest.
* **accuracy in the same figure as the speed**, per the house rule. A kernel that is
  fast and wrong is not a result, and a figure reporting only the fast half is how
  that ships.

Usage:
    .venv/bin/python scripts/bench/moe/moe_fp8_speed_plot.py --dir out/bench/moe/moe_fp8_speed
"""

import argparse
import glob
import json
import os

from kohakuwullm.bench import bar_labels, finish_axis, new_figure, save_figure
from kohakuwullm.bench.core.plotting import Palette

ARMS = ("bf16", "fp8_linears_only", "fp8_up")
ARM_LABELS = {
    "bf16": "bf16",
    "fp8_linears_only": "fp8 linears only",
    "fp8_up": "fp8 linears + MoE",
}
# The published pre-fix figure, kept as its own bar rather than as an annotation: the
# regression is the reason this benchmark exists and a reader should see its size.
BEFORE_MS = {"bf16": 238.84, "fp8_linears_only": 252.18, "fp8_up": 603.97}
KERNELS = (
    "_experts_gemm1",
    "_experts_gemm2",
    "_experts_gemm2_dgrad",
    "_experts_gemm1_dgrad",
    "grouped_mxfp8_wgrad_kernel",
    "_quantize_mx_kernel",
    "other",
)
SHORT = {
    "_experts_gemm1": "GEMM1 (fprop)",
    "_experts_gemm2": "GEMM2 (fprop)",
    "_experts_gemm2_dgrad": "GEMM2 dgrad",
    "_experts_gemm1_dgrad": "GEMM1 dgrad",
    "grouped_mxfp8_wgrad_kernel": "WGRAD (16-bit operands)",
    "_quantize_mx_kernel": "quantize",
    "other": "other",
}


def step_panel(ax, arms: dict, palette: Palette) -> None:
    """Wall ms/step of the real 400-step arms, before and after the fix."""
    labels = [ARM_LABELS[a] for a in ARMS]
    positions = range(len(ARMS))
    width = 0.38
    before = ax.bar(
        [p - width / 2 for p in positions],
        [BEFORE_MS[a] for a in ARMS],
        width,
        color=palette.color("before"),
        label="before (published)",
    )
    after = ax.bar(
        [p + width / 2 for p in positions],
        [arms[a]["ms_per_step"] for a in ARMS],
        width,
        color=palette.color("after"),
        label="after",
    )
    bar_labels(ax, before, "{:.0f}")
    bar_labels(ax, after, "{:.0f}")
    ax.axhline(arms["bf16"]["ms_per_step"], color="0.4", lw=1, ls="--")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=8)
    finish_axis(
        ax,
        "",
        "ms / step (wall)",
        "Real 400-step training, MoE-1B-A280M\n3.24M tokens, one card, dashed = bf16",
        legend=True,
    )


def stack_panel(ax, rows: list, palette: Palette) -> None:
    """Fused-path device time per kernel, one stacked bar per preset."""
    names = [r["preset"].replace("Kohaku-", "K-") for r in rows]
    bottom = [0.0] * len(rows)
    for kernel in KERNELS:
        values = [r["fused_ms"][kernel] for r in rows]
        ax.bar(
            names,
            values,
            bottom=bottom,
            color=palette.color(kernel),
            label=SHORT[kernel],
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    finish_axis(
        ax,
        "",
        "device ms per MoE layer (fwd+bwd)",
        "Where the fused path's time goes\n8104 tokens, 1013 rows/expert",
        legend=True,
    )


def speedup_panel(ax, rows: list, palette: Palette) -> None:
    """Fused against the bf16 expert path, per preset, annotated with expert width."""
    names = [r["preset"].replace("Kohaku-", "K-") for r in rows]
    bars = ax.bar(names, [r["speedup"] for r in rows], color=palette.color("speedup"))
    bar_labels(ax, bars, "{:.2f}x")
    ax.axhline(1.0, color="0.4", lw=1, ls="--")
    for index, row in enumerate(rows):
        ax.annotate(
            f"h={row['hidden']}",
            (index, 0.12),
            ha="center",
            fontsize=7,
            color="0.25",
        )
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    finish_axis(
        ax,
        "",
        "fused MXFP8 / bf16 expert path",
        "Speedup is flat across expert width\n(the narrow-expert hypothesis, falsified)",
    )


def error_panel(ax, rows: list, palette: Palette) -> None:
    """The accuracy half, in the same figure as the speed."""
    names = [r["preset"].replace("Kohaku-", "K-") for r in rows]
    series = [("out", lambda r: r["fwd_rel_error"])] + [
        (k, lambda r, k=k: r["grad_rel_error"][k])
        for k in ("dx", "dw_in", "dw_out", "dgate")
    ]
    width = 0.16
    for index, (name, getter) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(
            [i + offset for i in range(len(rows))],
            [getter(r) for r in rows],
            width,
            color=palette.color(f"err/{name}"),
            label=name,
        )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=30, fontsize=8)
    finish_axis(
        ax,
        "",
        "relative L2 error vs the bf16 path",
        "Accuracy, unchanged by anything here\n(the tile change is bit-identical)",
        legend=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/moe/moe_fp8_speed")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "moe_fp8_speed.json")) as handle:
        bench = json.load(handle)
    # A missing arm is usually a *disqualified* arm: a run whose exclusivity could not
    # be established gets its summary renamed rather than deleted, so the file is
    # sitting right there under another name. A bare FileNotFoundError sends the reader
    # to re-run 400 GPU steps that already happened, so the suffixed files are named.
    arms = {}
    for arm in ARMS:
        path = os.path.join(args.dir, f"{arm}.json")
        if not os.path.exists(path):
            aside = sorted(glob.glob(os.path.join(args.dir, f"{arm}.*.json")))
            hint = f" -- set aside as {', '.join(os.path.basename(a) for a in aside)}"
            parser.error(f"missing arm {arm}.json in {args.dir}{hint if aside else ''}")
        with open(path) as handle:
            arms[arm] = json.load(handle)

    palette = Palette()
    fig, axes = new_figure(2, 2)
    step_panel(axes[0], arms, palette)
    stack_panel(axes[1], bench["rows"], palette)
    speedup_panel(axes[2], bench["rows"], palette)
    error_panel(axes[3], bench["rows"], palette)
    fig.suptitle(
        "Fused MXFP8 MoE expert path: the regression was a per-step autotune, "
        "not the kernels",
        fontsize=12,
    )
    path = os.path.join(args.dir, "moe_fp8_speed.png")
    save_figure(fig, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
