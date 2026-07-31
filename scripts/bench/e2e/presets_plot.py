"""Draw the four Kohaku-ladder figures from ``presets.json``.

Split from the census half for the same reason as ``kernels.py`` /
``kernels_plot.py``: a rejected layout should not force a re-count, and the JSON is
then the one table every figure is reachable from.

Each figure answers one question about the ladder:

1. ``presets_ladder`` -- do the nine capacities form one smooth line through the
   target sequence, dense and sparse alternating? This is the headline.
2. ``presets_total_vs_active`` -- how far the sparse rungs sit off the diagonal.
3. ``presets_shape`` -- is the width/depth trajectory coherent or scattered?
4. ``presets_composition`` -- how much of each rung is the untied embedding and
   head, which is what decides whether a rung is named by total or by body.

Two conventions the captions state and this docstring will not repeat, because
getting either wrong misreads the ladder by more than any architecture choice in
it: ``total`` includes the untied embedding and head, and ``active`` on a sparse
rung is the *body* count (no embedding, head or router) -- the convention the
design targets were solved in. ``count_active_parameters`` is the other one, and it
is 41-69% higher on a sparse rung.

Usage:
    .venv/bin/python scripts/bench/e2e/presets_plot.py --dir out/bench/model/presets
"""

import argparse
import json
import os
import textwrap

from matplotlib.ticker import FuncFormatter

from kohakuwullm.bench import finish_axis, new_figure, save_figure

# Okabe-Ito, the repo-wide ramp in `bench.plotting`, assigned explicitly rather
# than by `Palette`'s rotation: slots 3 (purple) and 6 (yellow) are skipped
# because the three composition segments have to separate from each other under
# deuteranopia, and green/purple measures 7.6 dE where green/orange/sky measures
# 11.4. Validated light-surface, all pairs (the whole bench suite renders on
# white); the two sub-3:1 contrast slots carry direct labels and a JSON table.
COLOR = {
    "dense": "#0072B2",
    "sparse": "#D55E00",
    "embed": "#009E73",
    "active": "#E69F00",
    "inactive": "#56B4E9",
}
# A design target and a trajectory guide are not categories, so neither takes a
# categorical slot from the rungs.
REFERENCE = "#7F7F7F"

_MARKER = {"dense": "o", "sparse": "s"}
_GRANULARITY_NOTE = (
    "* Kohaku-MoE-8B routes 16 of 128 experts, not 8 of 64, so its expert width is "
    "0.25 x dim: 17 experts run per token against the ladder's 9, and holding "
    "0.5 x dim there would starve attention. Sparsity (0.125) is unchanged. It is the "
    "intended design, not an outlier."
)


def _class_of(row: dict) -> str:
    return "sparse" if row["sparse"] else "dense"


def _caption(fig, text: str, width: int = 145) -> None:
    """Put the caption below the axes, hard-wrapped.

    ``fig.text`` at negative y rather than a second ``suptitle`` line, because a
    caption wants a lighter face than a bold title and ``save_figure``'s
    ``bbox="tight"`` expands the crop to include it. That expansion is why the wrap
    is not optional: an unwrapped caption widens the *figure*, squeezing the axes it
    describes into a third of the image.
    """
    wrapped = "\n".join(
        textwrap.fill(paragraph, width=width) for paragraph in text.split("\n")
    )
    fig.text(
        0.5,
        -0.03,
        wrapped,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
        linespacing=1.5,
    )


def _log_ticks(ax, values, axis: str = "y") -> None:
    """Plain numbers on a log axis; ``10^3`` is not a parameter count."""
    getattr(ax, f"set_{axis}ticks")(list(values))
    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    getattr(ax, f"{axis}axis").set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _rung_xaxis(ax, rows: list[dict]) -> None:
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right")


def _series_scatter(ax, rows, xs, ys, size: int = 60) -> None:
    """One scatter call per class, so hue and marker shape stay paired."""
    for cls in ("dense", "sparse"):
        idx = [i for i, r in enumerate(rows) if _class_of(r) == cls]
        if not idx:
            continue
        ax.scatter(
            [xs[i] for i in idx],
            [ys[i] for i in idx],
            s=size,
            color=COLOR[cls],
            marker=_MARKER[cls],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
            label=f"{cls} rung",
        )


def _class_paths(ax, rows: list[dict], xs, ys) -> None:
    """One polyline per class, sorted along x.

    Not one line through all nine rungs: a dense rung and the sparse rung beside it
    reach the same capacity by different means, so a single polyline zigzags at
    every pair and implies an ordering the shapes do not have.
    """
    for cls in ("dense", "sparse"):
        pairs = sorted((x, y) for r, x, y in zip(rows, xs, ys) if _class_of(r) == cls)
        ax.plot(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            color=COLOR[cls],
            linewidth=1.2,
            alpha=0.55,
            zorder=2,
        )


def _bars(ax, rows: list[dict], values, fmt: str) -> None:
    """Bars coloured by class, each labelled -- an unlabelled bar is a shape."""
    for i, (row, value) in enumerate(zip(rows, values)):
        ax.bar(i, value, width=0.62, color=COLOR[_class_of(row)])
        ax.annotate(
            fmt.format(value),
            (i, value),
            textcoords="offset points",
            xytext=(0, 3 if value >= 0 else -11),
            ha="center",
            fontsize=8,
        )
    _rung_xaxis(ax, rows)


def plot_ladder(rows: list[dict], out_dir: str) -> None:
    """The headline: nine capacities against the sequence they were solved to."""
    fig, axes = new_figure(1, 2, figsize=(13.0, 4.8))
    xs = list(range(len(rows)))
    capacity = [r["capacity_m"] for r in rows]
    targets = [r["target_capacity_m"] for r in rows]

    # This connecting line *is* the claim -- that the nine rungs are one ladder,
    # not two -- so unlike every other line in the suite it crosses both classes,
    # and it is drawn in the reference ink because it belongs to neither.
    axes[0].plot(xs, capacity, color=REFERENCE, linewidth=1.2, zorder=2)
    axes[0].plot(
        xs,
        targets,
        color=REFERENCE,
        linewidth=1.4,
        linestyle=(0, (4, 2)),
        marker="x",
        markersize=6,
        zorder=1,
        label="design target",
    )
    _series_scatter(axes[0], rows, xs, capacity)
    _rung_xaxis(axes[0], rows)
    finish_axis(
        axes[0],
        "",
        "effective capacity  sqrt(active x total), M",
        "One ladder, dense and sparse alternating",
        logy=True,
    )
    _log_ticks(axes[0], (200, 300, 500, 700, 1000, 1500, 2000, 3000))
    axes[0].legend(loc="upper left")
    # A corner note rather than an arrow to the point: the ladder's lower-right is
    # the only empty region, and an arrow reaching it from the top-right marker
    # crosses the curve it is annotating.
    axes[0].text(
        0.98,
        0.05,
        "* MoE-8B: E=128, top_k=16, expert 0.25 x dim (kappa unchanged)",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=8,
        color="#444444",
    )

    deviation = [r["capacity_vs_target_pct"] for r in rows]
    _bars(axes[1], rows, deviation, "{:+.1f}")
    axes[1].axhline(0, color="#333333", linewidth=0.9)
    finish_axis(axes[1], "", "capacity - target (%)", "Fit to the target sequence")
    axes[1].set_ylim(min(deviation) - 4, max(deviation) + 4)

    fig.suptitle(
        "Kohaku ladder -- effective capacity per rung "
        "(parameters counted on meta device)",
        fontweight="bold",
    )
    _caption(
        fig,
        "Left: sqrt(active x total) per rung on a log axis, with the sequence the "
        "ladder was solved to overlaid. Right: each rung's distance from that "
        "target, signed; the worst fit is "
        f"{max(deviation, key=abs):+.1f}%. Dense rungs are fully active, so their "
        "active is their total; a sparse rung's active is its body count "
        "(embedding, head and router excluded) -- the convention the targets were "
        "solved in.\n" + _GRANULARITY_NOTE,
    )
    save_figure(fig, os.path.join(out_dir, "presets_ladder.png"))


def plot_total_vs_active(rows: list[dict], out_dir: str) -> None:
    """Where the sparse rungs sit relative to the dense diagonal."""
    fig, axes = new_figure(1, 2, figsize=(13.0, 4.8))
    total = [r["total"] / 1e6 for r in rows]
    active = [r["active_design"] / 1e6 for r in rows]
    ratio = [r["total_over_active"] for r in rows]
    sparse_ratio = [v for v, r in zip(ratio, rows) if r["sparse"]]
    span_text = (
        f"{min(sparse_ratio):.1f}-{max(sparse_ratio):.1f}x" if sparse_ratio else "n/a"
    )

    span = (min(total + active) * 0.75, max(total) * 1.4)
    axes[0].plot(
        span,
        span,
        color=REFERENCE,
        linewidth=1.2,
        linestyle=(0, (4, 2)),
        zorder=1,
        label="active = total (dense)",
    )
    _series_scatter(axes[0], rows, total, active)
    for row, x, y in zip(rows, total, active):
        # Only the sparse rungs get the ratio: on a dense one it is 1.00x by
        # construction, and nine labels where five carry information is chaos.
        suffix = f"  {row['total_over_active']:.1f}x" if row["sparse"] else ""
        axes[0].annotate(
            row["label"] + suffix,
            (x, y),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=8,
            color="#333333",
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(*span)
    axes[0].set_ylim(*span)
    _log_ticks(axes[0], (200, 500, 1000, 2000, 5000, 10000), axis="x")
    _log_ticks(axes[0], (200, 500, 1000, 2000, 5000, 10000))
    finish_axis(
        axes[0],
        "total parameters (M)",
        "active parameters (M)",
        f"Sparse rungs sit {span_text} below the diagonal",
        si_x=False,
    )
    axes[0].legend(loc="upper left")

    _bars(axes[1], rows, ratio, "{:.2f}x")
    axes[1].axhline(1.0, color="#333333", linewidth=0.9)
    finish_axis(axes[1], "", "total / active", "Sparsity multiplier per rung")
    axes[1].set_ylim(0, max(ratio) * 1.18)

    fig.suptitle("Kohaku ladder -- total against active parameters", fontweight="bold")
    _caption(
        fig,
        "Left: total vs active on log-log; the dashed line is active = total, where "
        "every dense rung sits by construction. Right: the same information as a "
        f"ratio. The sparse rungs span {span_text}, which is what holding "
        "kappa = top_k/num_experts = 0.125 across the ladder buys -- a fixed "
        "sparsity is what makes one hyperparameter set walk the whole ladder.\n"
        + _GRANULARITY_NOTE,
    )
    save_figure(fig, os.path.join(out_dir, "presets_total_vs_active.png"))


def plot_shape(rows: list[dict], out_dir: str) -> None:
    """Width against depth: the aspect-ratio trajectory."""
    fig, axes = new_figure(1, 2, figsize=(13.0, 4.8))
    depth = [r["depth"] for r in rows]
    dim = [r["dim"] for r in rows]
    aspect = [r["dim"] / r["depth"] for r in rows]
    gqa = [r["gqa_ratio"] for r in rows]

    _class_paths(axes[0], rows, depth, dim)
    _series_scatter(axes[0], rows, depth, dim)
    for row, x, y in zip(rows, depth, dim):
        # Dense labels above the marker, sparse below: Kohaku-200M (17, 768) and
        # Kohaku-MoE-1B (16, 768) are one marker-width apart, and a single offset
        # renders the two names on top of each other.
        axes[0].annotate(
            row["label"],
            (x, y),
            textcoords="offset points",
            xytext=(9, -11) if row["sparse"] else (9, 4),
            fontsize=8,
            color="#333333",
        )
    finish_axis(axes[0], "depth (layers)", "dim", "Width against depth", si_x=False)
    axes[0].set_xlim(min(depth) - 3, max(depth) + 7)
    axes[0].set_ylim(min(dim) - 120, max(dim) + 160)
    axes[0].legend(loc="upper left")

    _bars(axes[1], rows, aspect, "{:.0f}")
    finish_axis(axes[1], "", "dim / depth", "Aspect ratio per rung")
    axes[1].set_ylim(0, max(aspect) * 1.18)

    widest = max(rows, key=lambda r: r["dim"] / r["depth"])
    narrowest = min(rows, key=lambda r: r["dim"] / r["depth"])
    fig.suptitle(
        f"Kohaku ladder -- shape trajectory (head_dim 64 throughout, "
        f"GQA {min(gqa):.0f}-{max(gqa):.0f}x)",
        fontweight="bold",
    )
    _caption(
        fig,
        "Left: each class walks its own path through (depth, dim), the sparse rungs "
        "staying narrower at equal capacity -- which is what keeps a pipeline seam "
        "cheap. Both paths rise in dim and depth together at every step. "
        "Right: dim/depth per rung, "
        f"{min(aspect):.0f}-{max(aspect):.0f} across the ladder "
        f"({narrowest['label'].strip(' *')} narrowest, "
        f"{widest['label'].strip(' *')} widest).\n"
        "* Kohaku-MoE-8B is that turn: with expert_hidden = 0.5 x dim and "
        "kappa = 0.125 both fixed, 8B total was reachable only through depth, so it "
        f"is {narrowest['dim']} wide over {narrowest['depth']} layers.",
    )
    save_figure(fig, os.path.join(out_dir, "presets_shape.png"))


def plot_composition(rows: list[dict], out_dir: str) -> None:
    """What fraction of a rung is the untied embedding and head."""
    fig, axes = new_figure(1, 2, figsize=(13.4, 4.8))
    segments = (
        ("embed", "embedding + head", lambda r: r["embedding_and_head"]),
        ("active", "active body", lambda r: r["active_full"] - r["embedding_and_head"]),
        (
            "inactive",
            "routed experts not taken",
            lambda r: r["total"] - r["active_full"],
        ),
    )

    # `label if i == 0` would drop "routed experts not taken" from the legend
    # entirely: the first rung is dense, so its third segment is zero width and is
    # never drawn. A label has to attach to the first *drawn* segment of its kind.
    labelled: set[str] = set()
    for i, row in enumerate(rows):
        left = 0.0
        for key, label, getter in segments:
            width = 100 * getter(row) / row["total"]
            if width <= 0:
                continue
            axes[0].barh(
                i,
                # A 2px-equivalent surface gap between segments rather than an edge
                # colour: a border around a fill reads as another mark.
                width - 0.35,
                left=left,
                height=0.62,
                color=COLOR[key],
                label=None if key in labelled else label,
            )
            labelled.add(key)
            if width >= 8:
                axes[0].annotate(
                    f"{width:.0f}%",
                    (left + width / 2, i),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
            left += width
        axes[0].annotate(
            f"{row['total'] / 1e6:,.0f}M",
            (101.5, i),
            va="center",
            fontsize=8,
            color="#333333",
        )
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r["label"] for r in rows])
    # One empty row below the last bar to seat the legend in: at "lower right" it
    # lands on top of the MoE-8B bar, and outside the axes it collides with either
    # the title or the x label.
    axes[0].set_ylim(len(rows) + 0.5, -0.7)
    axes[0].set_xlim(0, 118)
    finish_axis(
        axes[0], "share of total parameters (%)", "", "Composition per rung", si_x=False
    )
    axes[0].legend(loc="lower center", ncol=3)

    share = [100 * r["embed_share"] for r in rows]
    total = [r["total"] / 1e6 for r in rows]
    _class_paths(axes[1], rows, total, share)
    _series_scatter(axes[1], rows, total, share)
    for row, x, y in zip(rows, total, share):
        axes[1].annotate(
            row["label"],
            (x, y),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
            color="#333333",
        )
    axes[1].set_xscale("log")
    _log_ticks(axes[1], (200, 500, 1000, 2000, 5000, 10000), axis="x")
    finish_axis(
        axes[1],
        "total parameters (M)",
        "embedding + head, share of total (%)",
        "Untying costs 131072 x dim",
        si_x=False,
    )
    axes[1].set_ylim(0, max(share) * 1.2)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "Kohaku ladder -- parameter composition (vocab 65536, untied)",
        fontweight="bold",
    )
    _caption(
        fig,
        "Left: each rung's total split three ways. 'Active body' is what one "
        "token's arithmetic touches outside the embedding and head, the router "
        "matrix included (under 0.1% of any rung, and excluded from 'active' in the "
        "other three figures); 'routed experts not taken' is the rest of the expert "
        "bank -- memory and optimizer state, but not FLOPs. Right: the "
        "embedding+head share against total, one curve per class. Untied, embed and "
        f"head are 2 x 65536 x dim together: {max(share):.0f}% of the smallest rung "
        f"and {min(share):.1f}% of the largest, which is why the rungs are named by "
        "total rather than by body.\n" + _GRANULARITY_NOTE,
    )
    save_figure(fig, os.path.join(out_dir, "presets_composition.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out/bench/model/presets")
    args = ap.parse_args()

    with open(os.path.join(args.dir, "presets.json")) as handle:
        rows = json.load(handle)["rungs"]

    plot_ladder(rows, args.dir)
    plot_total_vs_active(rows, args.dir)
    plot_shape(rows, args.dir)
    plot_composition(rows, args.dir)
    print(f"wrote 4 figures (PNG + SVG) to {args.dir}")


if __name__ == "__main__":
    main()
