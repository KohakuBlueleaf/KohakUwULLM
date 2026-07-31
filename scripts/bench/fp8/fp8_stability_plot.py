"""Does MXFP8 cost stability margin? bf16 vs round-up across multiples of the base lr.

The main A/B answers "does fp8 track bf16 on the shipped recipe" and it can only
answer it at 651.8M tokens. The published MX divergence appeared at 300B, which
this hardware cannot reach, so the limitation is real and a longer run is not the
way around it. The mechanism behind that divergence is, though: quantization noise
is claimed to eat *stability margin*, and margin is measurable in 2000 steps by
putting the recipe at its edge and asking whether the two dtypes break in the same
place. A too-high lr fails in hundreds of steps, not hundreds of thousands.

What the figure must not do is let "fp8 survived every multiplier" read as "fp8 is
safe". If bf16 also survives every multiplier the sweep never found *either* edge,
and the result is an upper bound on the margin fp8 costs -- not a demonstration
that it costs none. ``margin_verdict`` refuses to conflate those. The fix for the
bound is to sweep *further up* until the baseline breaks, which turns fp8's edge
relative to it into a measurement; the multipliers are therefore discovered from the
directory rather than fixed, so extending the sweep needs no edit here.

One caveat the multiplier alone does not carry: grad clipping at 1.0 is itself a
stability mechanism, so a high-lr edge may be where *clipping* stops holding the run
together rather than where the dtype fails. Both arms get the identical clip, so a
difference between them is still the dtype -- but the report quotes the clip-firing
rate beside the edge so the distinction stays checkable.

Usage::

    .venv/bin/python scripts/bench/fp8/fp8_stability_plot.py --dir out/bench/train/fp8_stability
"""

import argparse
import json
import os

import numpy as np

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    contended_fraction,
    discover_lr_multipliers,
    finish_axis,
    load_arms,
    new_figure,
    save_figure,
    speedup,
    step_ms,
)
from kohakuwullm.bench.analysis.abtest import margin_verdict, trailing_mean

# Discovered from the directory, not hardcoded: the sweep is extended upward until
# the baseline actually breaks, and a hardcoded tuple silently drops the runs that
# were added to find that edge -- which is the one result the extension exists for.
DTYPES = ("bf16", "fp8_up")
LABELS = {"bf16": "bf16", "fp8_up": "MXFP8 round-up"}
SMOOTH = 50
# WARMUP_RATIO 0.01 of 2000 steps. Excursions inside it are not stability events:
# every arm's loss falls 11.3 -> 6 here, so a "spike" is the transient itself.
# Reported separately rather than dropped, because the largest fp8 excursion in
# the whole sweep lands at step 19 and hiding it would be the more misleading half.
WARMUP = 20


def load(
    directory: str, multipliers: tuple[int, ...]
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Finished sweep runs only, keyed by ``(dtype, multiplier)``.

    ``require_summary=True`` is load-bearing, not defensive: every number this figure
    produces -- final loss, divergence, per-octave increment, step time -- reduces a
    run to one value, and an in-progress CSV reduces to whatever it has reached so
    far. Admitting one made a 600-of-2000-step fp8 run read as a +0.77 loss
    regression against a completed bf16 arm.
    """
    tags = [f"{dtype}_lr{mult}x" for dtype in DTYPES for mult in multipliers]
    loaded = load_arms(directory, tags, require_summary=True)
    runs = {}
    for dtype in DTYPES:
        for mult in multipliers:
            tag = f"{dtype}_lr{mult}x"
            if tag in loaded:
                runs[(dtype, mult)] = loaded[tag] | {"tag": tag}
    return runs


def _smooth(values: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """This file's default window over ``abtest.trailing_mean``."""
    return trailing_mean(values, window)


def excursion(run: dict, after: int = WARMUP) -> tuple[float, int]:
    """Largest ``loss / trailing-min`` ratio, and the step it happened at.

    The step matters as much as the size. An excursion at step 19 is the warmup
    transient every arm has; the same number at step 1500 would be an instability.
    """
    loss = run["loss"]
    floor = np.minimum.accumulate(loss)
    ratio = loss[after:] / floor[after:]
    index = int(np.argmax(ratio))
    return float(ratio[index]), index + after


def _mults(runs: dict) -> tuple[int, ...]:
    """The multipliers actually loaded, ascending. Every axis and table derives its
    extent from this so an extended sweep needs no edits to plot."""
    return tuple(sorted({mult for _, mult in runs}))


def _verdict(runs: dict) -> str:
    """Shape the sweep's runs into the two loss dicts ``abtest.margin_verdict`` wants."""
    return margin_verdict(
        {m: runs["bf16", m]["loss"] for m in _mults(runs) if ("bf16", m) in runs},
        {m: runs["fp8_up", m]["loss"] for m in _mults(runs) if ("fp8_up", m) in runs},
    )


def draw(runs: dict, path: str) -> None:
    palette = Palette()
    for mult in _mults(runs):
        palette.color(f"lr{mult}x")
    fig, axes = new_figure(2, 2, figsize=(13, 9.2))
    lead = SMOOTH

    for (dtype, mult), run in runs.items():
        axes[0].plot(
            np.arange(len(run["loss"]))[lead:],
            _smooth(run["loss"])[lead:],
            color=palette.color(f"lr{mult}x"),
            linestyle="-" if dtype == "bf16" else "--",
            linewidth=1.6,
            label=f"{mult}x {LABELS[dtype]}",
        )
    finish_axis(
        axes[0],
        "step",
        f"cross-entropy (mean of {SMOOTH} steps)",
        f"Loss at {_mults(runs)[0]}x-{_mults(runs)[-1]}x base lr "
        "(solid bf16, dashed MXFP8)",
        si_x=False,
        legend=True,
    )
    axes[0].legend(ncol=2, fontsize=8)

    # The margin panel. Final loss against lr multiplier is the curve whose *shape*
    # answers the question: two dtypes degrading in parallel cost the same margin,
    # and a divergence would show as one curve turning up while the other does not.
    for dtype in DTYPES:
        present = [m for m in _mults(runs) if (dtype, m) in runs]
        finals = [runs[dtype, m]["loss"][-200:].mean() for m in present]
        axes[1].plot(
            present,
            finals,
            marker="o",
            color=palette.color(dtype),
            linestyle="-" if dtype == "bf16" else "--",
            label=LABELS[dtype],
        )
        for mult, value in zip(present, finals):
            axes[1].annotate(
                f"{value:.4f}",
                (mult, value),
                textcoords="offset points",
                xytext=(0, 7 if dtype == "bf16" else -13),
                ha="center",
                fontsize=8,
            )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(list(_mults(runs)))
    axes[1].set_xticklabels([f"{m}x" for m in _mults(runs)])
    finish_axis(
        axes[1],
        "learning rate, multiple of 3e-4",
        "final loss (mean of last 200 steps)",
        "Degradation with lr -- parallel curves mean equal margin",
        si_x=False,
        legend=True,
    )

    # Each bar carries the *step* it happened at, not just its height. The tallest
    # bar in this panel is fp8 at 4x lr and it occurs at step 19, inside warmup --
    # a 122 sigma-looking spike that is the initial transient, not an instability.
    # Without the step annotation this panel argues the opposite of what happened.
    width = 0.38
    peak_steps: list[int] = []
    peak_values: list[float] = []
    for offset, dtype in zip((-width / 2, width / 2), DTYPES):
        present = [m for m in _mults(runs) if (dtype, m) in runs]
        peaks = [runs[dtype, m]["grad_norm"].max() for m in present]
        steps = [int(np.argmax(runs[dtype, m]["grad_norm"])) for m in present]
        peak_steps += steps
        peak_values += peaks
        bars = axes[2].bar(
            np.arange(len(present)) + offset,
            peaks,
            width=width,
            color=palette.color(dtype),
            label=LABELS[dtype],
        )
        for bar, value, step in zip(bars, peaks, steps):
            axes[2].annotate(
                f"{value:.0f}\n@{step}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                fontsize=8,
            )
    axes[2].set_xticks(range(len(_mults(runs))))
    axes[2].set_xticklabels([f"{m}x" for m in _mults(runs)])
    axes[2].axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    # Room for the two-line bar labels, which on a log axis would otherwise sit on
    # top of the title.
    axes[2].set_ylim(0.85, max(peak_values) * 3.0)
    latest = max(peak_steps) if peak_steps else 0
    finish_axis(
        axes[2],
        "learning rate, multiple of 3e-4",
        "peak gradient L2 norm, pre-clip",
        (
            f"Worst spike, @step (all within warmup, <={latest})"
            if latest < WARMUP
            else f"Worst spike, @step (latest at {latest})"
        ),
        si_x=False,
        logy=True,
        legend=False,
    )
    # The 3x headroom above the tallest bar is what makes upper-left free; at the
    # default ylim the legend landed on the 1x bar's label.
    axes[2].legend(loc="upper left")

    # Throughput belongs on this figure and not the A/B's: these eight runs are the
    # only bf16/fp8 pairs in the experiment measured on the *same card*. The main
    # arms sit on GPU0 and GPU1, whose sustained clocks differ by 2.7%, so their
    # step-time ratio is a card comparison wearing a dtype label.
    # Through ``runlog.speedup``, not a bare division: it re-checks both sides for
    # exclusivity, so a co-tenant on one sweep run takes that bar off the chart
    # instead of quietly bending the mean the A/B quotes.
    present, ratios = [], []
    for mult in _mults(runs):
        if ("bf16", mult) not in runs or ("fp8_up", mult) not in runs:
            continue
        present.append(mult)
        ratios.append(speedup(runs["bf16", mult], runs["fp8_up", mult], same_card=True))
    bars = axes[3].bar(
        range(len(present)),
        ratios,
        width=0.6,
        color=[palette.color(f"lr{m}x") for m in present],
    )
    bar_labels(axes[3], bars, fmt="{:.4f}")
    mean = float(np.mean(ratios))
    axes[3].axhline(mean, color="#333333", linewidth=1.0, label=f"mean {mean:.4f}x")
    axes[3].set_xticks(range(len(present)))
    axes[3].set_xticklabels([f"{m}x" for m in present])
    axes[3].set_ylim(1.0, max(ratios) * 1.02)
    finish_axis(
        axes[3],
        "learning rate, multiple of 3e-4",
        "bf16 step time / MXFP8 step time",
        "Speedup, four same-card pairs",
        si_x=False,
        legend=True,
    )

    fig.suptitle(_verdict(runs), fontsize=10, y=1.0)
    save_figure(fig, path)
    print(f"wrote {path}")


def report(runs: dict, ceiling: float | None) -> None:
    """The numbers behind the figure, so a claim can cite one."""
    print(
        f"\n{'run':<16}{'final':>9}{'min':>9}{'|g| med':>9}{'|g| max':>9}"
        f"{'at':>6}{'clip%':>7}{'excursion':>11}{'at step':>9}{'ms/step':>9}"
        f"{'contended':>11}"
    )
    for mult in _mults(runs):
        for dtype in DTYPES:
            if (dtype, mult) not in runs:
                continue
            run = runs[dtype, mult]
            loss, norms = run["loss"], run["grad_norm"]
            ratio, step = excursion(run)
            # The step of the peak norm, not just its size: the sweep's worst spike
            # is fp8 at 4x lr and it lands at step 19, inside warmup. Same number
            # at step 1500 would be the headline; here it is a transient.
            print(
                f"{run['tag']:<16}{loss[-200:].mean():>9.4f}{loss.min():>9.4f}"
                f"{np.median(norms):>9.3f}{norms.max():>9.2f}"
                f"{int(np.argmax(norms)):>6}"
                f"{(norms > 1.0).mean():>6.1%}{ratio:>11.2f}{step:>9}"
                f"{step_ms(run):>9.3f}{contended_fraction(run):>10.4%}"
            )

    print(f"\n{'pair':<8}{'bf16 ms':>10}{'fp8 ms':>9}{'speedup':>10}")
    ratios = []
    for mult in _mults(runs):
        if ("bf16", mult) not in runs or ("fp8_up", mult) not in runs:
            continue
        b, f = step_ms(runs["bf16", mult]), step_ms(runs["fp8_up", mult])
        ratios.append(speedup(runs["bf16", mult], runs["fp8_up", mult], same_card=True))
        print(f"{mult}x{'':<6}{b:>10.3f}{f:>9.3f}{ratios[-1]:>10.4f}")
    if ratios:
        values = np.array(ratios)
        print(
            f"  speedup {values.mean():.4f}x +/- {values.std(ddof=1):.4f} (sd, n={len(values)}), "
            f"range {values.min():.4f}-{values.max():.4f}"
        )
        if ceiling:
            print(
                f"  against a {ceiling:.4f}x matmul ceiling: "
                f"{(values.mean() - 1) / (ceiling - 1):.1%} of the available headroom"
            )

    # Degradation per octave is the margin statement. Comparing the two dtypes'
    # slopes is what makes it a statement about fp8 rather than about the lr.
    print("")
    for dtype in DTYPES:
        present = [m for m in _mults(runs) if (dtype, m) in runs]
        finals = np.array([runs[dtype, m]["loss"][-200:].mean() for m in present])
        octaves = np.log2(present)
        slope = np.polyfit(octaves, finals, 1)[0]
        print(f"  {LABELS[dtype]:<16} +{slope:.4f} loss per octave of lr")

    # The *increments*, not just the fitted slope. A single slope hides the shape, and
    # the shape is the mechanism: if each octave costs less than the one before while
    # the clip rate stays flat, the recipe is not approaching an edge at all --
    # AdamW's per-parameter normalisation bounds the update by lr whatever the
    # gradient scale, so raising lr buys a worse optimum rather than a divergence.
    # A sweep that keeps climbing under those conditions will never find an edge, and
    # saying so is more useful than another octave.
    for dtype in DTYPES:
        present = [m for m in _mults(runs) if (dtype, m) in runs]
        if len(present) < 3:
            continue
        finals = [runs[dtype, m]["loss"][-200:].mean() for m in present]
        clips = [(runs[dtype, m]["grad_norm"] > 1.0).mean() for m in present]
        steps = [
            f"{a}x->{b}x {y - x:+.4f}"
            for a, b, x, y in zip(present, present[1:], finals, finals[1:])
        ]
        print(f"  {LABELS[dtype]:<16} increments: " + ", ".join(steps))
        print(
            f"  {'':<16} clip rate:  "
            + ", ".join(f"{m}x {c:.1%}" for m, c in zip(present, clips))
        )
    if all(("fp8_up", m) in runs and ("bf16", m) in runs for m in _mults(runs)):
        gaps = [
            runs["fp8_up", m]["loss"][-200:].mean()
            - runs["bf16", m]["loss"][-200:].mean()
            for m in _mults(runs)
        ]
        print(
            "  fp8 - bf16 final loss by multiplier: "
            + ", ".join(f"{m}x {g:+.4f}" for m, g in zip(_mults(runs), gaps))
        )
    print(f"\nverdict: {_verdict(runs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/train/fp8_stability")
    parser.add_argument(
        "--accounting",
        default="out/bench/train/fp8_training/accounting.json",
        help="matmul ceiling, for reading the speedup against",
    )
    args = parser.parse_args()
    multipliers = discover_lr_multipliers(args.dir, DTYPES)
    if not multipliers:
        raise SystemExit(f"no finished runs in {args.dir}")
    print(f"multipliers: {', '.join(f'{m}x' for m in multipliers)}")
    runs = load(args.dir, multipliers)
    if not runs:
        raise SystemExit(f"no run CSVs in {args.dir}")
    ceiling = None
    if os.path.exists(args.accounting):
        with open(args.accounting) as handle:
            ceiling = json.load(handle)["matmul_speedup_ceiling"]
    draw(runs, os.path.join(args.dir, "fp8_stability.png"))
    report(runs, ceiling)


if __name__ == "__main__":
    main()
