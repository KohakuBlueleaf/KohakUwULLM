"""Scaling curves for the Kohaku 4-card sweep, judged by correlation not by slope.

Bars across rungs answer nothing -- of course 1.5B is slower than 200M. The question is
whether **one** power law describes the whole ladder, so every panel is log-log with a
fitted exponent *and* an r2, and the r2 is the verdict.

Why r2 and not "is the slope -1": -1 is only ideal when the x-axis is the true cost, and
no parameter count is. With untied embeddings the vocabulary table is a quarter of
Kohaku-200M's parameters and performs no multiply-accumulate, while the equally large LM
head is a GEMM. Measured with the closed-form model in
:mod:`kohakuwullm.bench.model.flops_analytic`, ``gemm_FLOPs / (6 * active_params)`` runs 0.75 at
Kohaku-200M to 0.95 at MoE-8B -- a monotonic distortion, which is the shape that bends a
fitted exponent. A high r2 with an unremarkable exponent means every rung behaves the
same way; a low r2 means one rung is doing something the others are not.

Panels:
  1. matrix FLOPs/token vs B tokens/day -- the honest compute axis, closed form.
  2. effective parameters ``sqrt(total * active)`` vs B tokens/day -- the quantity the
     ladder was designed in, so this is where the interleaved sequence should show up as
     one line through dense and sparse rungs alike.
  3. active parameters vs tokens/parameter/day -- how much data each parameter sees.
  4. achieved matrix TFLOP/s against the bf16 dense-GEMM ceiling -- utilisation, with a
     real denominator rather than 6ND. The ceiling comes from the sweep when the sweep
     recorded one, and the caption says which; a carried ceiling is a measurement from
     another day at another sustained clock, and it has already rejected a good row.

Usage::

    .venv/bin/python scripts/bench/e2e/kohaku_e2e_plot.py --dir out/bench/train/kohaku_e2e
"""

import argparse
import math
import os

from kohakuwullm.bench import load_sweep, planning_units
from kohakuwullm.bench.core.plotting import (
    Palette,
    finish_axis,
    new_figure,
    save_figure,
)
from kohakuwullm.bench.model.flops_analytic import budget
from kohakuwullm.bench.model.ladder import ladder_census

SPREAD_LIMIT = 0.02
# Fallback only, for rows written before the sweep recorded its own. A dense ceiling is
# a property of the card at its sustained clock, not a hardware bound: this same 227
# figure, carried into a later run, disqualified a legitimate 228 TF/s row that was
# 0.4% away from it. `ceiling_tflops` below prefers whatever the run measured.
BF16_CEILING_TFLOPS = 227.0


def ceiling_tflops(rows: list[dict]) -> tuple[float, bool]:
    """The bf16 dense-GEMM ceiling to draw, and whether the sweep measured it.

    A plotter cannot measure a ceiling -- it has no card and no run -- so the honest
    thing is to use the one the sweep recorded and say so when there is none. Returns
    the median across rows rather than the first: a sweep spans hours and this box's
    cards differ by 2.7% in sustained clock, so no single row owns the number.
    """
    measured = [
        r["matmul_ceiling_tflops"] for r in rows if r.get("matmul_ceiling_tflops")
    ]
    if not measured:
        return BF16_CEILING_TFLOPS, False
    return sorted(measured)[len(measured) // 2], True


def arm_of(row: dict, sparse: bool) -> str:
    """What the run actually did, from the payload -- never from the tag.

    Dense and sparse are separate series, not one line with two marker shapes. They are
    different machines: a sparse rung's arithmetic tracks its *active* parameters while
    its memory traffic and all-reduce tracks its *total*, so a single fit through both
    families measures neither and its r2 is diluted by a difference that is real.
    """
    parts = [str(row["dtype"])]
    if row.get("mxfp8"):
        parts.append("mxfp8")
    if row.get("rounding", "nearest") != "nearest":
        parts.append("sr")
    family = "MoE" if sparse else "dense"
    # The optimizer belongs in the arm: the AdamW bridge row is the same preset, dtype
    # and strategy as a Muon row, so without it that row joins the Muon series and adds
    # a fifth "dense rung" to a four-rung ladder.
    #
    # Gradient checkpointing does NOT: it is a memory technique the big rungs have no
    # choice about, not a different way of training, and splitting on it cuts the MoE
    # pipeline line in two exactly where it gets interesting.
    strategy = str(row["strategy"]).replace("_ckpt", "")
    return "+".join(parts) + f" {row['optimizer']} {strategy} · {family}"


def _pick_one_per_cell(rows: list[dict]) -> list[dict]:
    """One row per (preset, strategy, micro, arm), choosing checkpointed or not.

    Gradient checkpointing is a memory strategy, not a separate configuration to plot --
    splitting on it would cut each line in two exactly where the big rungs start needing
    it. But the OOM backfill means some cells now hold both variants, and two points at
    one x is worse than either.

    The rule is **whichever variant lets both arms be compared**. Preferring plain
    un-checkpointed rows everywhere would strand dense-1.5B, whose mxfp8 arm exists only
    checkpointed; preferring checkpointed everywhere would discard the clean
    un-checkpointed pairs the rest of the ladder is built on.
    """

    def cell(row):
        return (
            row["preset"],
            str(row["strategy"]).replace("_ckpt", ""),
            row["per_micro"],
        )

    def is_ckpt(row):
        return "_ckpt" in str(row["strategy"]) or "_ckpt" in str(row["tag"])

    def arm(row):
        return (bool(row.get("mxfp8")), str(row["dtype"]), row["optimizer"])

    paired: dict[tuple, set] = {}
    for row in rows:
        paired.setdefault((cell(row), is_ckpt(row)), set()).add(arm(row))

    kept: dict[tuple, dict] = {}
    for row in rows:
        key = (cell(row), arm(row))
        if key not in kept:
            kept[key] = row
            continue
        # Both variants present for this arm: keep the one whose *cell* has more arms
        # measured under it, so the pair survives.
        current = len(paired[(cell(kept[key]), is_ckpt(kept[key]))])
        candidate = len(paired[(cell(row), is_ckpt(row))])
        if candidate > current:
            kept[key] = row
    return list(kept.values())


def fit_loglog(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """``(exponent, r2)`` for ``y = a * x**b``, fitted in log space.

    Three points minimum: two always fit a line exactly and would report r2 = 1.0,
    which is the one number here that must never be flattering by construction.
    """
    n = len(xs)
    if n < 3:
        return None
    lx = [math.log10(v) for v in xs]
    ly = [math.log10(v) for v in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    syy = sum((v - my) ** 2 for v in ly)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    return sxy / sxx, (sxy * sxy) / (sxx * syy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/bench/train/kohaku_e2e")
    ap.add_argument("--micro", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    census = {c.name: c for c in ladder_census()}
    rows, notes, _ = load_sweep(args.dir, max_spread=SPREAD_LIMIT)
    rows = [
        r
        for r in rows
        if r["ok"] and r["per_micro"] == args.micro and r["preset"] in census
    ]
    # The drift control is the *same* rung measured again an hour later. Left in, it
    # becomes a second point at one x, inflating n and dragging the fit toward whichever
    # rung happened to be repeated. It is a check on the sweep, so it is reported as one.
    drift = [r for r in rows if "drift" in r["tag"]]
    rows = [r for r in rows if "drift" not in r["tag"]]
    for row in rows:
        row.update(planning_units(row))
    if not rows:
        print(f"no rows at micro={args.micro} in {args.dir}")
        for note in notes:
            print(f"  note: {note}")
        return

    rows = _pick_one_per_cell(rows)

    flops = {name: budget(name) for name in {r["preset"] for r in rows}}
    series: dict[str, list[dict]] = {}
    for row in rows:
        series.setdefault(arm_of(row, census[row["preset"]].sparse), []).append(row)

    palette = Palette()
    fig, axes = new_figure(2, 3, figsize=(19, 9.5))
    ax_flop, ax_eff, ax_total, ax_norm, ax_norm_total, ax_util = axes
    summary = []

    for arm, group in sorted(series.items()):
        rung = [census[r["preset"]] for r in group]
        order = sorted(range(len(group)), key=lambda i: rung[i].compute_active)
        group = [group[i] for i in order]
        rung = [rung[i] for i in order]
        colour = palette.color(arm)

        matrix_per_token = [flops[r["preset"]].matrix for r in group]
        # The ladder's own design quantity: dense rungs collapse to `total`, sparse rungs
        # land at the geometric mean, which is where the interleaved sequence was solved.
        effective = [math.sqrt(c.total * c.active) for c in rung]
        rate = [r["b_tok_day"] for r in group]
        per_param = [
            r["tokens_per_s"] * 86400 / c.compute_active for r, c in zip(group, rung)
        ]
        util = [f * r["tokens_per_s"] / 1e12 for f, r in zip(matrix_per_token, group)]

        fills = [colour if r["clean"] else "none" for r in group]
        marks = ["s" if c.sparse else "o" for c in rung]

        total_p = [c.total for c in rung]
        # Per *stored* parameter as well as per active one: for a sparse rung these
        # differ by 1/kappa, and which one matters depends on whether the constraint
        # is arithmetic or memory.
        per_total = [r["tokens_per_s"] * 86400 / c.total for r, c in zip(group, rung)]

        for ax, xs, ys in (
            (ax_flop, matrix_per_token, rate),
            (ax_eff, effective, rate),
            (ax_total, total_p, rate),
            (ax_norm, total_p, per_param),
            (ax_norm_total, total_p, per_total),
            (ax_util, effective, util),
        ):
            ax.plot(xs, ys, "-", color=colour, lw=1.2, alpha=0.75, label=arm)
            for x, y, fill, mark in zip(xs, ys, fills, marks):
                ax.plot(
                    x, y, mark, color=colour, markerfacecolor=fill, markersize=6, lw=0
                )

        for ax, xs, ys in (
            (ax_flop, matrix_per_token, rate),
            (ax_eff, effective, rate),
            (ax_total, total_p, rate),
        ):
            fit = fit_loglog(xs, ys)
            if fit is None:
                continue
            slope, r2 = fit
            ax.annotate(
                f"r$^2$={r2:.4f}  (exp {slope:+.2f})",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(-6, -13),
                ha="right",
                fontsize=7,
                color=colour,
            )
        fit_f = fit_loglog(matrix_per_token, rate)
        fit_e = fit_loglog(effective, rate)
        summary.append((arm, len(group), fit_f, fit_e))

    ceiling, from_run = ceiling_tflops(rows)
    ax_util.axhline(ceiling, color="crimson", ls="--", lw=1)
    ax_util.annotate(
        f"{ceiling:.0f} TF/s bf16 GEMM ceiling"
        + (" (this run)" if from_run else " (carried, not from this run)"),
        (min(ax_util.get_xlim()), ceiling),
        textcoords="offset points",
        xytext=(6, 4),
        color="crimson",
        fontsize=8,
    )

    for ax, xlabel, ylabel, title in (
        (
            ax_flop,
            "matrix FLOPs / token (closed form)",
            "B tokens / day",
            "Compute scaling",
        ),
        (
            ax_eff,
            "effective parameters  sqrt(total x active)",
            "B tokens / day",
            "The ladder's design axis",
        ),
        (
            ax_total,
            "total parameters (stored)",
            "B tokens / day",
            "Against what has to be stored",
        ),
        (
            ax_norm,
            "total parameters (stored)",
            "tokens / compute-active param / day",
            "Data seen per compute-active parameter",
        ),
        (
            ax_norm_total,
            "total parameters (stored)",
            "tokens / total param / day",
            "Data seen per stored parameter",
        ),
        (
            ax_util,
            "effective parameters  sqrt(total x active)",
            "achieved matrix TFLOP/s",
            "Utilisation vs the measured ceiling",
        ),
    ):
        finish_axis(ax, xlabel, ylabel, title=title, logx=True, logy=True, legend=True)

    fig.suptitle(
        f"Kohaku ladder, 4x RTX 5090, micro {args.micro}, 262144 tokens/step -- "
        "squares are MoE, hollow markers exceed 2% tail spread",
        fontsize=9,
    )
    path = args.out or os.path.join(args.dir, f"kohaku_scaling_{args.micro}.png")
    save_figure(fig, path)
    print(f"wrote {path}")

    print(
        f"{'arm':26s} {'n':>2s}  {'FLOP axis r2':>12s} {'exp':>6s}  {'eff-P r2':>9s} {'exp':>6s}"
    )
    for arm, n, fit_f, fit_e in summary:
        ff = f"{fit_f[1]:.4f} {fit_f[0]:+.2f}" if fit_f else "  (<3 rungs)"
        fe = f"{fit_e[1]:.4f} {fit_e[0]:+.2f}" if fit_e else "  (<3 rungs)"
        print(f"{arm:26s} {n:>2d}  {ff:>19s}  {fe:>16s}")
    # The drift check: same preset, same config, an hour of sustained load apart. A gap
    # here is thermal, and it bounds how much of any rung-to-rung difference is real.
    for row in drift:
        base = [
            r
            for r in rows
            if r["preset"] == row["preset"]
            and r["strategy"] == row["strategy"]
            and r["per_micro"] == row["per_micro"]
        ]
        if base:
            ratio = row["tokens_per_s"] / base[0]["tokens_per_s"]
            print(
                f"drift {row['preset']} {row['strategy']} micro {row['per_micro']}: "
                f"{base[0]['tokens_per_s']:,.0f} -> {row['tokens_per_s']:,.0f} tok/s "
                f"({ratio - 1:+.2%} over the sweep)"
            )
    for note in notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
