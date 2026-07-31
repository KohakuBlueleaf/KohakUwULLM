"""Draw the MXFP8 training A/B from the four ``*.csv`` files.

Split from the measurement half for the same reason as ``kernels.py`` /
``kernels_plot.py``: the runs cost GPU-hours and a rejected figure should not cost
them again.

**The acceptance criterion is the slope, not the offset.** A constant gap is
tolerable -- more tokens or a retuned lr absorbs it -- but a widening one is not,
because 651.8M tokens is 460x short of the 300B where the published MX divergence
appeared, so a slope barely resolvable here is large there. ``bf16_ctrl`` is
therefore not a pass threshold but the **null**: two identical bf16 runs already
drift apart through atomics in the embedding and attention backwards, and only drift
beyond theirs is attributable to quantization.

The panels: loss (do the arms track?), gap against bf16 with the replicate's band
drawn behind it, grad norm (a norm spike precedes a loss spike, so a flat gap with a
rising norm is still a warning), and throughput -- which comes from the lr sweep's
same-card pairs, never from these arms, which sit on two cards.

Usage::

    .venv/bin/python scripts/bench/fp8/fp8_training_plot.py --dir out/bench/train/fp8_training
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
    is_exclusive,
    load_arms,
    new_figure,
    pair_speedups,
    replay_identical,
    save_figure,
    speedup,
    step_ms,
)
from kohakuwullm.bench.analysis.abtest import (
    BLOCK,
    RESOLVABLE_RATIO,
    block_means,
    difference,
    factorial_read,
    gap_trend,
    near_miss_fraction,
    slope_vs_null,
    spike_overlap,
    trailing_mean,
)

# Drawing order, fixed here rather than read off the directory so the colours
# stay stable when an arm is still running.
ORDER = ["bf16", "bf16_ctrl", "fp8_up", "fp8_rtn"]
LABELS = {
    "bf16": "bf16",
    "bf16_ctrl": "bf16 (replicate)",
    "fp8_up": "MXFP8 round-up",
    "fp8_rtn": "MXFP8 round-to-nearest",
}
BASELINE = "bf16"
CONTROL = "bf16_ctrl"
# Column headers for the wide tables. Not ``arm[:4]``, which collapses ``bf16`` and
# ``bf16_ctrl`` onto the same string and made a replicate comparison look like a
# self-comparison.
SHORT = {"bf16": "bf16", "bf16_ctrl": "ctrl", "fp8_up": "up", "fp8_rtn": "rtn"}
SPIKE_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)
# The threshold the pairwise table uses. One is enough: the six pairs at four
# thresholds do not fit a line, and 5.0 is the repo's standing definition of a spike.
SPIKE_REFERENCE = 5.0
# Which card each arm ran on. Recorded here because nothing in the CSV can say:
# ``device_name()`` reads index 0 of the *visible* set, so all four arms report a
# byte-identical device string. The assignment is the orchestrator's
# (``bf16``/``bf16_ctrl`` on GPU0, ``fp8_up``/``fp8_rtn`` on GPU1) and it is a
# convenient one -- it puts both *within-family* pairs on one card, so the
# replicate comparison and the rounding comparison are the two that can carry a
# step-time reading at all.
CARD = {"bf16": 0, "bf16_ctrl": 0, "fp8_up": 1, "fp8_rtn": 1}
SMOOTH = 200
# The gap panel needs its own, much heavier window. A gap is a difference of two
# highly correlated series, so at 200 steps the per-step loss noise still swamps
# it visually -- a real offset of 0.003 is invisible next to a 0.02 swing. The
# trend is the thing being read, so it gets a window that resolves the trend.
GAP_SMOOTH = 2000


# The dtypes the lr sweep tags its runs with, for discovering which multipliers it
# actually holds. The sweep is the experiment's only admissible speedup source: all
# its runs are sequential on one card, so each ratio is within-card, whereas the A/B's
# own arms sit on GPU0 and GPU1 and ``runlog.speedup`` refuses that comparison.
SWEEP_DTYPES = ("bf16", "fp8_up")


def _smooth(values: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """This file's default window over ``abtest.trailing_mean``."""
    return trailing_mean(values, window)


def _common_length(arms: dict) -> int:
    return min(len(data["step"]) for data in arms.values())


def _token_ticks(total: int) -> list[int]:
    """Six round ticks over the token axis, so ``si_x`` renders them as 100M / 200M."""
    step = max(1, int(total / 6))
    magnitude = 10 ** max(0, len(str(step)) - 2)
    step = max(magnitude, (step // magnitude) * magnitude)
    return list(range(0, total + step, step))


def draw(
    arms: dict,
    path: str,
    pairs: list[tuple[str, float]] | None = None,
    ceiling: float | None = None,
) -> None:
    palette = Palette()
    for arm in ORDER:
        palette.color(arm)
    pairs = pairs or []
    fig, axes = new_figure(2, 2, figsize=(13, 9.2))
    ticks = _token_ticks(
        int(max(data["cumulative_tokens"][-1] for data in arms.values()))
    )
    # Every plotted point is a full trailing window, so the first one starts a
    # window in. `smooth` pads the head with values[0], and plotting that padding
    # would put a synthetic curve next to real ones -- and its huge chaotic
    # transient would set the y-scale for the whole panel.
    lead = min(SMOOTH, min(len(data["step"]) for data in arms.values()) // 4)

    for arm, data in arms.items():
        axes[0].plot(
            data["cumulative_tokens"][lead:],
            _smooth(data["loss"])[lead:],
            color=palette.color(arm),
            label=LABELS[arm],
            linestyle="--" if arm == CONTROL else "-",
        )
    finish_axis(
        axes[0],
        "tokens trained",
        f"cross-entropy (mean of {SMOOTH} steps)",
        "Loss, four arms on identical data",
        xticks=ticks,
        legend=True,
    )

    if BASELINE in arms:
        n = _common_length(arms)
        base = arms[BASELINE]["loss"][:n]
        # The gap panel carries a GAP_SMOOTH trend line, so it has to start a
        # *trend* window in, not a SMOOTH one, or the padded head of that line
        # sets the y-scale for everything drawn beside it.
        glead = min(GAP_SMOOTH, n // 4)
        floor = None
        span_of = lambda arm: np.abs(  # noqa: E731
            _smooth(arms[arm]["loss"][:n] - base)[glead:]
        )
        if CONTROL in arms:
            floor = np.abs(_smooth(arms[CONTROL]["loss"][:n] - base))
            axes[1].fill_between(
                arms[BASELINE]["cumulative_tokens"][glead:n],
                -floor[glead:],
                floor[glead:],
                color=palette.color(CONTROL),
                alpha=0.45,
                edgecolor=palette.color(CONTROL),
                linewidth=0.6,
                label="bf16 self-difference (noise floor)",
            )
        for arm in ("fp8_up", "fp8_rtn"):
            if arm not in arms:
                continue
            gap = arms[arm]["loss"][:n] - base
            axes[1].plot(
                arms[BASELINE]["cumulative_tokens"][glead:n],
                _smooth(gap)[glead:],
                color=palette.color(arm),
                linewidth=0.7,
                alpha=0.35,
            )
            axes[1].plot(
                arms[BASELINE]["cumulative_tokens"][glead:n],
                _smooth(gap, GAP_SMOOTH)[glead:],
                color=palette.color(arm),
                label=LABELS[arm],
                linewidth=2.0,
            )
        axes[1].axhline(0.0, color="#333333", linewidth=0.8)
        finish_axis(
            axes[1],
            "tokens trained",
            "loss - bf16",
            f"Gap against bf16 ({GAP_SMOOTH}-step trend), "
            "with what bf16 costs itself",
            xticks=ticks,
            legend=True,
        )
        if floor is not None:
            span = max(floor[glead:].max(), 1e-6)
            for arm in ("fp8_up", "fp8_rtn"):
                if arm in arms:
                    span = max(span, span_of(arm).max())
            axes[1].set_ylim(-1.6 * span, 1.6 * span)

    for arm, data in arms.items():
        axes[2].plot(
            data["cumulative_tokens"][lead:],
            _smooth(data["grad_norm"])[lead:],
            color=palette.color(arm),
            label=LABELS[arm],
            linestyle="--" if arm == CONTROL else "-",
        )
    finish_axis(
        axes[2],
        "tokens trained",
        f"gradient L2 norm, pre-clip (mean of {SMOOTH} steps)",
        "Gradient norm",
        xticks=ticks,
        logy=True,
        legend=True,
    )

    # This panel used to divide the arms' own step times by each other and report
    # 1.055x. It cannot: bf16 ran on GPU0, fp8_up on GPU1, and the fp8 arm is 3.7%
    # contended. The same-card lr-sweep pairs measure the same two dtypes on one
    # card with both sides certified exclusive, and give 1.071x. The invalid number
    # was low *and* drifting -- it read 1.062x earlier in the same run as contention
    # accumulated -- which is how a cross-card ratio hides: it looks stable enough.
    if pairs:
        labels = [name for name, _ in pairs]
        ratios = [value for _, value in pairs]
        bars = axes[3].bar(
            range(len(pairs)),
            ratios,
            color=[palette.color(f"pair{name}") for name in labels],
            width=0.62,
        )
        bar_labels(axes[3], bars, fmt="{:.4f}")
        mean = float(np.mean(ratios))
        axes[3].axhline(mean, color="#333333", linewidth=1.0, label=f"mean {mean:.4f}x")
        if ceiling:
            axes[3].axhline(
                ceiling,
                color="#333333",
                linewidth=1.0,
                linestyle="--",
                label=f"matmul ceiling {ceiling:.3f}x",
            )
        axes[3].set_xticks(range(len(pairs)))
        axes[3].set_xticklabels([f"{name} lr" for name in labels])
        axes[3].set_ylim(1.0, max(ceiling or 0.0, max(ratios)) * 1.03)
        finish_axis(
            axes[3],
            "lr sweep pair",
            "bf16 step time / MXFP8 step time",
            "Speedup, same card, both sides certified exclusive",
            si_x=False,
            legend=True,
        )
    else:
        axes[3].axis("off")
        axes[3].text(
            0.5,
            0.5,
            "no same-card pair available;\nthe A/B arms are on two cards\n"
            "and cannot supply a speedup",
            ha="center",
            va="center",
            fontsize=10,
        )

    save_figure(fig, path)
    print(f"wrote {path}")


def report(
    arms: dict,
    directory: str,
    pairs: list[tuple[str, float]] | None = None,
) -> None:
    """The numbers the figure is read against, so a claim can cite one."""
    if BASELINE not in arms:
        return
    n = _common_length(arms)
    base = arms[BASELINE]["loss"][:n]
    tail = slice(max(0, n - 2000), n)
    print(
        f"steps {n}, "
        f"{arms[BASELINE]['cumulative_tokens'][n - 1] / 1e6:.1f}M trained tokens"
    )

    # Which harness produced these rows. `log_every` decides whether the loop syncs
    # every step, and a sync serializes host issue against device execution -- which
    # costs fp8 more than bf16, so the *same arms* yield a different step-time ratio
    # under a different setting. Absent means 1: these runs predate the flag, and 1 was
    # the only behaviour then. Stated here so a step-time number can never be quoted
    # without the harness that produced it.
    settings = {
        arm: int(data.get("summary", {}).get("log_every", 1))
        for arm, data in arms.items()
    }
    unique = sorted(set(settings.values()))
    detail = ", ".join(f"{SHORT[a]}={settings[a]}" for a in ORDER if a in settings)
    print(
        f"harness: log_every {unique[0] if len(unique) == 1 else detail} "
        f"({'one device sync per step' if unique == [1] else 'batched device reads'})"
        + (
            ""
            if len(unique) == 1
            else "  <-- MIXED: step times are not comparable across these arms"
        )
    )

    # Printed before any number derived from it. The loss columns are valid whatever
    # this says; the ms/step column is not, and labelling it in place is what stops
    # the two being read with the same confidence.
    ok, why = replay_identical([arms[a] for a in ORDER if a in arms])
    print(f"replay: {'IDENTICAL' if ok else 'BROKEN'} -- {why}")

    print(
        f"\n{'arm':<24}{'final loss':>12}{'gap':>10}{'|gap| max':>11}"
        f"{'ms/step':>9}{'contended':>11}"
    )
    for arm in ORDER:
        if arm not in arms:
            continue
        loss = arms[arm]["loss"][:n]
        gap = loss - base
        frac = contended_fraction(arms[arm])
        flag = "" if is_exclusive(arms[arm]) else " !"
        print(
            f"{LABELS[arm]:<24}{loss[tail].mean():>12.4f}"
            f"{gap[tail].mean():>+10.4f}{np.abs(_smooth(gap)).max():>11.4f}"
            f"{step_ms(arms[arm]):>9.1f}{frac:>10.2%}{flag}"
        )
    for arm in ORDER:
        if arm not in arms:
            continue
        norms = arms[arm]["grad_norm"][:n]
        print(
            f"  {LABELS[arm]:<22} grad-norm mean {norms.mean():.3f} "
            f"median {np.median(norms):.3f} max {norms.max():.2f}  "
            f"spikes>5 {(norms > 5).sum()}  clipped {(norms > 1.0).mean():.1%}"
        )
    print(
        "  ms/step above is per-arm wall time, NOT a speedup: the arms sit on two "
        "cards (2.7% clock spread) and a '!' marks a contended clock."
    )

    # The measured speedup means nothing without the ceiling: bf16 WGRAD and a
    # bf16 LM head cap it far below the per-GEMM figure.
    accounting = os.path.join(directory, "accounting.json")
    if os.path.exists(accounting):
        with open(accounting) as handle:
            acc = json.load(handle)
        head_share = acc["mac_per_token"]["lm_head"] / acc["mac_per_token"]["total"]
        print(
            f"\nfp8 covers {acc['fp8_share_of_matmul']:.1%} of matmul MAC/token "
            f"(lm head {head_share:.1%} stays bf16)\n"
            f"  {acc['per_gemm_speedup']:.2f}x per GEMM -> "
            f"{acc['matmul_speedup_ceiling']:.3f}x matmul ceiling"
        )
        if pairs:
            values = np.array([v for _, v in pairs])
            print(
                f"  -> {values.mean():.4f}x +/- {values.std(ddof=1):.4f} measured "
                f"end to end (n={len(values)} same-card pairs, "
                f"range {values.min():.4f}-{values.max():.4f}), "
                f"{(values.mean() - 1) / (acc['matmul_speedup_ceiling'] - 1):.0%} "
                "of the headroom"
            )

    # The gap has to be read against what bf16 costs itself, so print the two
    # side by side rather than leaving it to the figure.
    if CONTROL in arms:
        floor = np.abs(_smooth(arms[CONTROL]["loss"][:n] - base))
        print(f"\nnoise floor (|bf16 - bf16|, smoothed): max {floor.max():.4f}")
        for arm in ("fp8_up", "fp8_rtn"):
            if arm not in arms:
                continue
            gap = np.abs(_smooth(arms[arm]["loss"][:n] - base))
            over = (gap > floor).mean()
            print(
                f"  {LABELS[arm]:<22} |gap| max {gap.max():.4f} "
                f"({gap.max() / max(floor.max(), 1e-9):.2f}x floor), "
                f"outside the floor on {over:.0%} of the run"
            )
    paired_differences(arms, n)
    gap_trends(arms, n)
    spike_alignment(arms, n)
    within_card_step_times(arms)
    factorial_verdict(arms, n)


def factorial_verdict(arms: dict, n: int) -> None:
    """Read the four arms as the 2x2 they are, and state what follows.

    The formatting lives here and the inference lives in ``bench.abtest``, which is
    where it can be tested against arms whose answer is known by construction.
    """
    bf16_arms = [a for a in ("bf16", "bf16_ctrl") if a in arms]
    fp8_arms = [a for a in ("fp8_up", "fp8_rtn") if a in arms]
    if not bf16_arms or not fp8_arms:
        print("\nverdict: needs at least one bf16 and one fp8 arm")
        return

    tail = slice(max(0, n - n // 2), n)
    read = factorial_read(
        [block_means(arms[a]["loss"][tail]) for a in bf16_arms],
        [block_means(arms[a]["loss"][tail]) for a in fp8_arms],
    )
    blocks = len(block_means(arms[BASELINE]["loss"][tail]))
    tokens = arms[BASELINE]["cumulative_tokens"][n - 1] / 1e6

    print(f"\n2x2 read over the last {n - tail.start} steps ({blocks} blocks)")
    print(
        f"  dtype effect  (fp8 - bf16, {len(fp8_arms)}v{len(bf16_arms)} arms)  "
        f"{read.effect.mean:+.5f} +/- {read.effect.err:.5f} "
        f"({read.effect.sigma:.1f} sigma)"
    )
    if read.noise is not None:
        print(
            f"  bf16 self-gap (replicate = the noise scale)   "
            f"{read.noise.mean:+.5f} +/- {read.noise.err:.5f} "
            f"({read.noise.sigma:.1f} sigma)"
        )
    if read.rounding is not None:
        print(
            f"  rounding      (round-to-nearest - round-up)   "
            f"{read.rounding.mean:+.5f} +/- {read.rounding.err:.5f} "
            f"({read.rounding.sigma:.1f} sigma)"
        )

    print(f"  pooled verdict: {read.sentence(tokens)}")

    # Per arm as well as pooled, because pooling misleads exactly when it matters. The
    # two fp8 arms are different *treatments*, not replicates, and here they differ by
    # 10x -- so the pooled effect is dominated by round-to-nearest and says almost
    # nothing about round-up, which is the configuration that would actually ship.
    if read.noise is None:
        thirds(arms, n)
        return
    floor = abs(read.noise.mean)
    base = np.mean([block_means(arms[a]["loss"][tail]) for a in bf16_arms], axis=0)
    for arm in fp8_arms:
        delta = difference(base, block_means(arms[arm]["loss"][tail]))
        ratio = abs(delta.mean) / max(floor, 1e-12)
        band = (
            "inside"
            if ratio <= 1.0
            else "comparable" if ratio <= RESOLVABLE_RATIO else "outside"
        )
        print(
            f"  {LABELS[arm]:<24}{delta.mean:+.5f} +/- {delta.err:.5f} = "
            f"{ratio:5.2f}x the floor ({band})"
        )
    thirds(arms, n)


def thirds(arms: dict, n: int) -> None:
    """Each arm's gap against bf16 in three equal stretches of the run.

    The noise floor is not a constant: two identical bf16 runs start bit-identical and
    drift apart as atomics accumulate, so a whole-run floor is too large early and too
    small late. Splitting into thirds makes "the floor grew to meet it" visible rather
    than inferred, and is the descriptive companion to the slope test above.
    """
    present = [a for a in ORDER if a in arms and a != BASELINE]
    if not present:
        return
    edges = [n * i // 3 for i in range(4)]
    print("\ngap against bf16 by third of the run (does the noise floor grow?)")
    print(f"  {'arm':<24}" + "".join(f"{f'third {i + 1}':>20}" for i in range(3)))
    for arm in present:
        cells = ""
        for lo, hi in zip(edges, edges[1:]):
            delta = difference(
                block_means(arms[BASELINE]["loss"][lo:hi]),
                block_means(arms[arm]["loss"][lo:hi]),
            )
            cells += f"{delta.mean:>+11.5f}+/-{delta.err:.5f}"
        marker = " <- floor" if arm == CONTROL else ""
        print(f"  {LABELS[arm]:<24}{cells}{marker}")


def gap_trends(arms: dict, n: int) -> None:
    """The primary result: each arm's gap slope, tested against the replicate's.

    Magnitude is negotiable and slope is not -- see ``abtest.slope_vs_null`` for why,
    and why the replicate rather than zero is the null. Reported per 100M trained
    tokens, a unit the 300B question could be extrapolated in, while remembering that
    extrapolating a null slope 460x is not evidence.
    """
    present = [a for a in ORDER if a in arms and a != BASELINE]
    if not present:
        return
    base = arms[BASELINE]["loss"][:n]
    centres = block_means(arms[BASELINE]["cumulative_tokens"][:n].astype(float)) / 1e8
    if len(centres) < 4:
        return
    trends = {
        arm: gap_trend(block_means(arms[arm]["loss"][:n] - base), centres)
        for arm in present
    }

    print(f"\ngap trend against bf16, per 100M trained tokens ({len(centres)} blocks)")
    for arm in present:
        trend = trends[arm]
        print(
            f"  {LABELS[arm]:<24}{trend.slope:>+9.5f} +/- {trend.err:.5f}  "
            f"({trend.sigma:4.1f} sigma vs 0)  mean gap {trend.mean_gap:+.5f}"
        )

    # The primary result. Sigma against *zero* above is the wrong test -- it asks
    # whether the gap drifts, when the question is whether it drifts more than two
    # identical bf16 runs already do through atomics alone.
    if CONTROL not in trends:
        print("  no replicate arm yet: no null slope, so no slope test")
        return
    null = trends[CONTROL]
    control = arms[CONTROL]["loss"][:n]
    print(
        f"\nPRIMARY: slope tested against the bf16 replicate's own slope "
        f"({null.slope:+.5f} +/- {null.err:.5f} per 100M tokens)"
    )
    for arm in present:
        if arm == CONTROL:
            continue
        # Fitted on treatment - control directly. Subtracting the two baseline-relative
        # slopes would combine correlated errors, since both share bf16's noise.
        excess = gap_trend(block_means(arms[arm]["loss"][:n] - control), centres)
        test = slope_vs_null(excess, trends[arm], null)
        print(
            f"  {LABELS[arm]:<24}excess {test.excess_slope:>+9.5f} "
            f"+/- {test.excess_err:.5f}  "
            f"({test.sigma:4.1f} sigma vs null)  {test.label}"
        )


def spike_alignment(arms: dict, n: int) -> None:
    """Spike rates, plus the Jaccard overlap with its artifact labelled.

    Read the rates -- they are robust. ``runlog.replay_identical`` is the real replay
    check; see ``abtest.spike_overlap`` for why the overlap column misleads.
    """
    present = [a for a in ORDER if a in arms]
    if len(present) < 2:
        return
    norms = {arm: arms[arm]["grad_norm"][:n] for arm in present}

    # Rates first and in their own table. With four arms the pairwise columns are
    # six per row, which at one line per threshold is unreadable -- and the rates are
    # the half that carries the finding.
    print("\ngrad-norm spike rates (share of steps above the threshold)")
    print("  threshold " + "".join(f"{SHORT[a]:>9}" for a in present))
    for threshold in SPIKE_THRESHOLDS:
        cells = "".join(f"{(norms[a] > threshold).mean():>9.3%}" for a in present)
        print(f"  |g|>{threshold:<6}{cells}")

    print(f"\npairwise spike-step overlap at |g|>{SPIKE_REFERENCE}")
    for i, left in enumerate(present):
        for right in present[i + 1 :]:
            _, _, jaccard = spike_overlap(norms[left], norms[right], SPIKE_REFERENCE)
            near = near_miss_fraction(norms[left], norms[right], SPIKE_REFERENCE)
            # nan means the two arms flagged exactly the same steps, so there is no
            # disagreement to attribute. Printed as "--" rather than "nan%", which
            # reads as a failed computation.
            share = "  --" if np.isnan(near) else f"{near:3.0%}"
            r = np.corrcoef(norms[left], norms[right])[0, 1]
            print(
                f"  {SHORT[left]:<4} / {SHORT[right]:<4} "
                f"J={jaccard:.2f}  near-miss {share}  r(|g|)={r:.5f}"
            )


def within_card_step_times(arms: dict) -> None:
    """Step-time ratios for the arm pairs that share a card, gate and all.

    Cross-family cannot be timed, but the two within-family pairs can: the replicate
    is the only check that a step time reproduces run to run, and round-up vs
    round-to-nearest asks whether the rounding costs throughput (it should not --
    one Triton instruction). A pair failing the gate is reported as refused, because
    "no number" is honest and a caveated one is what keeps getting un-published.
    """
    pairs = [("bf16", "bf16_ctrl"), ("fp8_up", "fp8_rtn")]
    if not any(a in arms and b in arms for a, b in pairs):
        return
    print("\nsame-card step-time ratios within each family")
    for left, right in pairs:
        if left not in arms or right not in arms:
            continue
        try:
            ratio = speedup(
                arms[left], arms[right], same_card=CARD[left] == CARD[right]
            )
        except ValueError as error:
            print(f"  {LABELS[left]} / {LABELS[right]}: REFUSED -- {error}")
            continue
        print(
            f"  {LABELS[left]} / {LABELS[right]} (both GPU{CARD[left]}): "
            f"{ratio:.4f}x  ({step_ms(arms[left]):.2f} vs "
            f"{step_ms(arms[right]):.2f} ms/step)"
        )


def paired_differences(arms: dict, n: int) -> None:
    """Every pair's loss difference over the run's tail, with a block standard error.

    A single ``max |gap|`` is one realization and cannot separate a real effect from
    a lucky draw; if the four *between* differences are not larger than the
    replicate difference, the arms are indistinguishable.
    """
    present = [a for a in ORDER if a in arms]
    if len(present) < 2:
        return
    tail = slice(max(0, n - n // 2), n)
    blocks = {arm: block_means(arms[arm]["loss"][tail]) for arm in present}

    # Three kinds of pair, and only one is a noise floor. The two fp8 arms are
    # different *treatments*, not replicates, so their difference is the rounding
    # answer -- calling it "within" would hide the discriminator among controls.
    kinds = {
        frozenset(("bf16", "bf16_ctrl")): "replicate",
        frozenset(("fp8_up", "fp8_rtn")): "ROUNDING",
    }
    count = len(next(iter(blocks.values())))
    print(
        f"\npaired loss difference over the last {n - tail.start} steps "
        f"({count} blocks of {BLOCK})"
    )
    for i, left in enumerate(present):
        for right in present[i + 1 :]:
            delta = difference(blocks[left], blocks[right])
            kind = kinds.get(frozenset((left, right)), "BETWEEN")
            print(
                f"  {kind:<10} {LABELS[right]:<24} - {LABELS[left]:<18} "
                f"{delta.mean:>+8.5f} +/- {delta.err:.5f}  ({delta.sigma:.1f} sigma)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/train/fp8_training")
    parser.add_argument(
        "--speedup-dir",
        default="out/bench/train/fp8_stability",
        help="lr-sweep runs, which supply the same-card speedup this figure needs",
    )
    args = parser.parse_args()
    arms = load_arms(args.dir, ORDER)
    if not arms:
        raise SystemExit(f"no arm CSVs in {args.dir}")
    pairs = pair_speedups(
        args.speedup_dir,
        [
            (f"{m}x", f"bf16_lr{m}x", f"fp8_up_lr{m}x")
            for m in discover_lr_multipliers(args.speedup_dir, SWEEP_DTYPES)
        ],
    )
    ceiling = None
    accounting = os.path.join(args.dir, "accounting.json")
    if os.path.exists(accounting):
        with open(accounting) as handle:
            ceiling = json.load(handle)["matmul_speedup_ceiling"]
    draw(arms, os.path.join(args.dir, "fp8_training.png"), pairs, ceiling)
    report(arms, args.dir, pairs)


if __name__ == "__main__":
    main()
