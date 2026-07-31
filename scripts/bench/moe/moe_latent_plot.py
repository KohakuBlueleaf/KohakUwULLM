"""Draw the latent-MoE figure from ``moe_latent.json``.

The result this figure has to carry is negative: **compressing the expert input
width does not pay for itself as a dispatch optimization.** Dispatch is linear in
width, so halving the width can save at most half of a gather and a scatter --
tens of microseconds. The ``down``/``up`` projections that buy that compression
are two dense ``T x dim x (dim/alpha)`` GEMMs, i.e. quadratic in ``dim``. A
linear saving cannot fund a quadratic price, and the panels show it not funding
it: net is negative on every shape at alpha=2.

Four panels, because four separate things have to be true before "latent MoE is
faster" would be an honest sentence, and only the first is usually checked:

1. the dispatch budget has to come out ahead (it does not until alpha=4, and then
   only at dim >= 1280),
2. the dispatch saving has to actually materialize (it floors -- below ~600 wide
   gather stops responding to width at all),
3. the expert GEMMs have to keep their efficiency (they lose 24-31% at alpha=4),
4. the comparison has to be capacity-matched (it is not -- alpha=2 is half the
   expert bank, alpha=4 a quarter).

``whole_layer`` is deliberately not drawn as a speedup. It is wall time including
host dispatch, and ``cpu_us`` shows 95%+ of it is Python issuing ops: all three
shapes land within 10 us of each other at alpha=4 despite 3x different GEMM work,
which is a launch-rate floor, not a GPU result. Panel 4 pairs it with the
parameter counts so the reader sees what the "win" is actually made of.

Usage:
    .venv/bin/python scripts/bench/moe/moe_latent_plot.py --dir out/bench/moe/moe_latent
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

CEIL = "#D55E00"
ZERO = "#333333"
NET = "#111111"
# bf16 achievable dense GEMM on this card, the same figure head_options.py uses.
MAMF = 270.0
# Seeded in a fixed order so alpha keeps one colour across panels 3 and 4. Left
# to first-use order the shape series in panel 2 consume the strong entries and
# alpha=2 lands on the yellow, which carries no annotation legibly.
SERIES_ORDER = ("alpha1", "alpha2", "alpha4", "saved", "price")


def index(rows: list[dict]) -> dict[tuple[int, int, str], dict]:
    return {(r["dim"], r["alpha"], r["kernel"]): r for r in rows}


def shapes(rows: list[dict]) -> list[tuple[int, int, int]]:
    """``(dim, num_experts, top_k)`` per measured shape, narrowest first."""
    seen = {r["dim"]: (r["dim"], r["num_experts"], r["top_k"]) for r in rows}
    return [seen[d] for d in sorted(seen)]


def label(shape: tuple[int, int, int]) -> str:
    dim, experts, top_k = shape
    return f"d{dim}\nE{experts} k{top_k}"


def dispatch_us(by_key: dict, dim: int, alpha: int) -> float:
    return by_key[(dim, alpha, "gather")]["graph_us"] + (
        by_key[(dim, alpha, "combine")]["graph_us"]
    )


def budget(rows: list[dict], alphas: list[int]) -> list[dict]:
    """Per (shape, alpha>1): what dispatch saves, what the projections cost."""
    by_key = index(rows)
    out = []
    for dim, experts, top_k in shapes(rows):
        base = dispatch_us(by_key, dim, 1)
        for alpha in alphas:
            if alpha == 1:
                continue
            saved = base - dispatch_us(by_key, dim, alpha)
            price = by_key[(dim, alpha, "down_up_proj")]["graph_us"]
            out.append(
                dict(
                    dim=dim,
                    experts=experts,
                    top_k=top_k,
                    alpha=alpha,
                    saved=saved,
                    price=price,
                    net=saved - price,
                )
            )
    return out


def plot_budget(entries: list[dict], axis, pal: Palette) -> None:
    names = [f"d{e['dim']}\na{e['alpha']}" for e in entries]
    xs = range(len(entries))
    width = 0.38
    bars = axis.bar(
        [x - width / 2 for x in xs],
        [e["saved"] for e in entries],
        width,
        color=pal.color("saved"),
        label="dispatch saved",
    )
    bar_labels(axis, bars, "{:+.0f}")
    bars = axis.bar(
        [x + width / 2 for x in xs],
        [-e["price"] for e in entries],
        width,
        color=pal.color("price"),
        label="down/up projections",
    )
    bar_labels(axis, bars, "{:+.0f}")
    axis.scatter(
        list(xs),
        [e["net"] for e in entries],
        marker="D",
        s=70,
        color=NET,
        edgecolors="#FFFFFF",
        linewidths=1.1,
        zorder=5,
        label="net",
    )
    for x, entry in zip(xs, entries):
        # A negative net sits inside the price bar, so the label needs its own
        # background to stay readable against the fill.
        axis.annotate(
            f"{entry['net']:+.0f}",
            (x, entry["net"]),
            textcoords="offset points",
            xytext=(11, -2),
            fontsize=8,
            fontweight="bold",
            color=NET,
            bbox=dict(boxstyle="round,pad=0.16", fc="#FFFFFF", ec="none", alpha=0.85),
        )
    axis.axhline(0.0, color=ZERO, linewidth=1.0)
    axis.set_xticks(list(xs))
    axis.set_xticklabels(names, fontsize=7.5)
    axis.set_ylabel("microseconds (+ = latent ahead)")
    axis.set_title("Dispatch budget: a linear saving against a quadratic price")
    axis.legend(frameon=False, fontsize=8, loc="lower left")


def plot_floor(rows: list[dict], alphas: list[int], axis, pal: Palette) -> None:
    by_key = index(rows)
    floor_from = 0
    for dim, experts, top_k in shapes(rows):
        # alphas ascend, so these run uncompressed -> narrowest.
        widths = [by_key[(dim, a, "gather")]["latent"] for a in alphas]
        times = [by_key[(dim, a, "gather")]["graph_us"] for a in alphas]
        color = pal.color(f"shape{dim}")
        axis.plot(
            widths,
            times,
            marker="o",
            color=color,
            label=f"d{dim} E{experts} k{top_k}",
        )
        # Linear-in-width is what a pure data movement is *supposed* to do, drawn
        # as a ray through the uncompressed point; the gap to it is the finding.
        axis.plot(
            [0, widths[0]],
            [0, times[0]],
            color=color,
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
        # Only segments that genuinely stopped responding to width get called
        # out; d1536 still scales at 384, so a shaded "floor region" would lie.
        for (w0, t0), (w1, t1) in zip(zip(widths, times), list(zip(widths, times))[1:]):
            # Flat means halving the width did NOT buy time back, so the test is
            # on t1 failing to fall -- not on t1 merely being no larger than t0,
            # which every decreasing segment satisfies.
            if t1 >= t0 * 0.95:
                floor_from = max(floor_from, w0)
                axis.annotate(
                    f"{w0}->{w1} wide: {t0:.1f}->{t1:.1f}us",
                    ((w0 + w1) / 2, t1),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )
    axis.annotate(
        f"halving the width below ~{floor_from} buys nothing:\n"
        "gather is launch- and occupancy-bound there,\n"
        "so the dispatch saving stops before alpha does",
        (0.03, 0.97),
        xycoords="axes fraction",
        fontsize=7.5,
        va="top",
    )
    finish_axis(
        axis,
        "expert input width (dim / alpha)",
        "gather (us, graph replay)",
        "The saving floors: dotted = linear in width",
        xticks=sorted({r["latent"] for r in rows}),
    )
    axis.set_xlim(0, None)
    axis.set_ylim(0, None)
    axis.tick_params(axis="x", labelsize=7.5, rotation=45)
    axis.legend(frameon=False, fontsize=7.5, loc="lower right")


def plot_efficiency(rows: list[dict], alphas: list[int], axis, pal: Palette) -> None:
    by_key = index(rows)
    shape_list = shapes(rows)
    xs = range(len(shape_list))
    width = 0.8 / len(alphas)
    for i, alpha in enumerate(alphas):
        vals = [
            by_key[(dim, alpha, "expert_gemm")]["tflops"] for dim, _, _ in shape_list
        ]
        offset = (i - (len(alphas) - 1) / 2) * width
        bars = axis.bar(
            [x + offset for x in xs],
            vals,
            width,
            color=pal.color(f"alpha{alpha}"),
            label=f"alpha={alpha}",
        )
        bar_labels(axis, bars, "{:.0f}")
        if alpha == 1:
            continue
        for x, (dim, _, _), val in zip(xs, shape_list, vals):
            base = by_key[(dim, 1, "expert_gemm")]["tflops"]
            axis.annotate(
                f"{100 * (val / base - 1):+.0f}%",
                (x + offset, val),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=7.5,
                fontweight="bold",
                color=NET,
                bbox=dict(
                    boxstyle="round,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.85
                ),
            )
    axis.axhline(MAMF, color=CEIL, linestyle="--", label=f"bf16 MAMF {MAMF:.0f}T")
    axis.set_xticks(list(xs))
    axis.set_xticklabels([label(s) for s in shape_list], fontsize=7.5)
    axis.set_ylabel("expert-GEMM achieved TFLOP/s")
    axis.set_ylim(0, MAMF * 1.12)
    axis.set_title("What narrow K costs the expert GEMMs")
    axis.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)


def plot_capacity(rows: list[dict], alphas: list[int], axis, pal: Palette) -> None:
    by_key = index(rows)
    shape_list = shapes(rows)
    xs = range(len(shape_list))
    width = 0.8 / len(alphas)
    for i, alpha in enumerate(alphas):
        params = [
            by_key[(dim, alpha, "gather")]["bank_params_m"] for dim, _, _ in shape_list
        ]
        offset = (i - (len(alphas) - 1) / 2) * width
        bars = axis.bar(
            [x + offset for x in xs],
            params,
            width,
            color=pal.color(f"alpha{alpha}"),
            label=f"alpha={alpha}",
        )
        bar_labels(axis, bars, "{:.0f}M")

    twin = axis.twinx()
    twin.grid(False)
    for i, alpha in enumerate(alphas):
        offset = (i - (len(alphas) - 1) / 2) * width
        wall = [
            by_key[(dim, alpha, "whole_layer")]["graph_us"] for dim, _, _ in shape_list
        ]
        twin.scatter(
            [x + offset for x in xs],
            wall,
            marker="_",
            s=190,
            linewidths=2.2,
            color=ZERO,
            zorder=4,
            label="whole layer (wall, launch-bound)" if i == 0 else None,
        )
        for x, value in zip(xs, wall):
            twin.annotate(
                f"{value:.0f}",
                (x + offset, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=6.5,
                color=ZERO,
            )
    twin.set_ylabel("whole layer (us wall clock)", fontsize=9)
    twin.set_ylim(0, None)

    axis.set_xticks(list(xs))
    axis.set_xticklabels([label(s) for s in shape_list], fontsize=7.5)
    axis.set_ylabel("expert-bank parameters (M)")
    axis.set_title("alpha is not capacity-neutral: 1/alpha of the bank")
    handles, labels = axis.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    axis.legend(
        handles + extra[0],
        labels + extra[1],
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )


def report(entries: list[dict], rows: list[dict], alphas: list[int]) -> None:
    by_key = index(rows)
    print(f"{'shape':<18}{'alpha':>6}{'saved':>9}{'price':>9}{'net':>9}{'GEMM':>10}")
    print("-" * 61)
    for entry in entries:
        base = by_key[(entry["dim"], 1, "expert_gemm")]["tflops"]
        got = by_key[(entry["dim"], entry["alpha"], "expert_gemm")]["tflops"]
        name = f"d{entry['dim']} E{entry['experts']} k{entry['top_k']}"
        print(
            f"{name:<18}{entry['alpha']:>6}{entry['saved']:>8.1f}u"
            f"{entry['price']:>8.1f}u{entry['net']:>+8.1f}u"
            f"{100 * (got / base - 1):>+9.0f}%"
        )
    print()
    for dim, _, _ in shapes(rows):
        widths = [by_key[(dim, a, "gather")]["latent"] for a in alphas]
        times = [by_key[(dim, a, "gather")]["graph_us"] for a in alphas]
        pairs = "  ".join(f"{w}->{t:.1f}us" for w, t in zip(widths, times))
        print(f"  gather d{dim}: {pairs}")
    print()
    for dim, _, _ in shapes(rows):
        for alpha in alphas:
            row = by_key[(dim, alpha, "whole_layer")]
            share = 100 * row["cpu_us"] / row["graph_us"]
            print(
                f"  whole_layer d{dim} a{alpha}: {row['graph_us']:7.1f}us wall, "
                f"{row['cpu_us']:7.1f}us CPU enqueue ({share:.0f}%)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/moe/moe_latent")
    cli = parser.parse_args()
    with open(os.path.join(cli.dir, "moe_latent.json")) as handle:
        payload = json.load(handle)
    rows = payload["rows"]
    alphas = sorted({r["alpha"] for r in rows})
    entries = budget(rows, alphas)
    report(entries, rows, alphas)

    pal = Palette()
    for key in SERIES_ORDER:
        pal.color(key)
    fig, axes = new_figure(2, 2, figsize=(15.5, 10.4))
    plot_budget(entries, axes[0], pal)
    plot_floor(rows, alphas, axes[1], pal)
    plot_efficiency(rows, alphas, axes[2], pal)
    plot_capacity(rows, alphas, axes[3], pal)

    worst = min(entries, key=lambda e: e["net"])
    best = max(entries, key=lambda e: e["net"])
    # The device comes from the payload, not the plotting host: this script runs
    # on CPU and must report where the numbers were actually measured.
    fig.suptitle(
        f"Latent MoE -- {payload['device']}  |  T={payload['config']['tokens']}, "
        f"hidden={payload['config']['hidden']}\n"
        "dispatch saving is linear in width, the down/up projections are quadratic "
        f"in dim -- net {worst['net']:+.0f} to {best['net']:+.0f} us\n"
        "no panel here is a speedup: alpha=2 halves the expert bank and alpha=4 "
        "quarters it, so any end-to-end gain is largely a smaller model",
        fontweight="bold",
    )
    path = os.path.join(cli.dir, "moe_latent.png")
    save_figure(fig, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
