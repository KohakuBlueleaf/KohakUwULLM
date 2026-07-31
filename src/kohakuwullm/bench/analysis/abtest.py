"""Statistics for a multi-arm training A/B: block errors, gap trends, 2x2 reads.

Everything reduces non-overlapping block means before taking a standard error, and
every verdict is read against a replicate arm. See docs/performance/ab-testing.md.
"""

from typing import NamedTuple

import numpy as np

BLOCK = 200
RESOLVABLE_RATIO = 3.0
TREND_SIGMA = 3.0
DIVERGENCE_FACTOR = 1.5


class Difference(NamedTuple):
    """A mean paired difference and its block standard error."""

    mean: float
    err: float

    @property
    def sigma(self) -> float:
        return abs(self.mean) / max(self.err, 1e-15)


class GapTrend(NamedTuple):
    """Whether an arm's gap against the baseline grows with tokens."""

    slope: float
    err: float
    mean_gap: float

    @property
    def sigma(self) -> float:
        return abs(self.slope) / max(self.err, 1e-15)

    @property
    def growing(self) -> bool:
        return self.slope > 0 and self.sigma > TREND_SIGMA

    @property
    def label(self) -> str:
        return "GROWING" if self.growing else "flat"


class FactorialRead(NamedTuple):
    """The 2x2 read: a dtype effect, the noise scale it must beat, and a verdict.

    ``noise`` is ``None`` when no replicate arm is present, and ``verdict`` is then
    ``"no-replicate"``. Otherwise ``"inside"``, ``"comparable"`` or ``"outside"``.
    """

    effect: Difference
    noise: Difference | None
    rounding: Difference | None
    verdict: str

    @property
    def ratio(self) -> float | None:
        if self.noise is None:
            return None
        return abs(self.effect.mean) / max(abs(self.noise.mean), 1e-12)

    def sentence(self, tokens_m: float) -> str:
        """The conclusion in words, chosen by the same data that set ``verdict``."""
        match self.verdict:
            case "no-replicate":
                return (
                    "no replicate arm, so the dtype effect has no scale to be read "
                    "against -- inconclusive by construction"
                )
            case "inside":
                return (
                    f"the fp8 gap is {self.ratio:.2f}x bf16's own run-to-run gap -- "
                    f"INSIDE the noise floor, indistinguishable from a bf16 "
                    f"replicate at {tokens_m:.0f}M trained tokens"
                )
            case "comparable":
                return (
                    f"the fp8 gap is {self.ratio:.2f}x bf16's own run-to-run gap -- "
                    f"COMPARABLE to it. One replicate pair pins the noise scale to "
                    f"about a factor of two, so anything under "
                    f"{RESOLVABLE_RATIO:.0f}x needs more bf16 replicates to call"
                )
            case _:
                return (
                    f"the fp8 gap is {self.ratio:.2f}x bf16's own run-to-run gap -- "
                    f"OUTSIDE the noise floor, a real regression at {tokens_m:.0f}M "
                    f"tokens (see the gap trend for whether it grows)"
                )


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth with a trailing window, so the last point is real data.

    The head is padded with ``values[0]``; callers must skip the first ``window``
    points rather than plot the padding.
    """
    if len(values) < window:
        window = max(1, len(values) // 10)
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def block_means(values: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Non-overlapping ``block``-step means, dropping any short final block."""
    usable = len(values) // block * block
    if usable < block:
        return np.array([])
    return values[:usable].reshape(-1, block).mean(axis=1)


def difference(left: np.ndarray, right: np.ndarray) -> Difference:
    """``right - left`` block-mean difference, with the standard error of the mean."""
    delta = right - left
    return Difference(
        float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
    )


def gap_trend(gap_blocks: np.ndarray, token_blocks: np.ndarray) -> GapTrend:
    """Fit ``gap ~ tokens`` and return the slope with an OLS standard error.

    ``token_blocks`` is expected pre-scaled by the caller, which owns the unit.
    """
    slope, intercept = np.polyfit(token_blocks, gap_blocks, 1)
    residual = gap_blocks - (slope * token_blocks + intercept)
    spread = ((token_blocks - token_blocks.mean()) ** 2).sum()
    err = residual.std(ddof=2) / np.sqrt(max(spread, 1e-30))
    return GapTrend(float(slope), float(err), float(gap_blocks.mean()))


class SlopeTest(NamedTuple):
    """A treatment's excess drift over the replicate's, from a direct difference fit.

    ``sigma`` is the significance of the *excess*, not of either arm's own slope.
    """

    excess_slope: float
    excess_err: float
    treatment_slope: float
    null_slope: float

    @property
    def sigma(self) -> float:
        return abs(self.excess_slope) / max(self.excess_err, 1e-15)

    @property
    def growing(self) -> bool:
        """Widening faster than two identical runs do, beyond the noise."""
        return self.excess_slope > 0 and self.sigma > TREND_SIGMA

    @property
    def label(self) -> str:
        return "GROWING vs null" if self.growing else "flat vs null"


def slope_vs_null(excess: GapTrend, treatment: GapTrend, null: GapTrend) -> SlopeTest:
    """Excess drift of a treatment over the replicate, with the correct error.

    ``excess`` must be fitted on ``treatment - control`` **directly**, never obtained
    by subtracting two slopes measured against the baseline. ``treatment_slope`` and
    ``null_slope`` are carried for reporting only. See docs/performance/ab-testing.md.
    """
    return SlopeTest(excess.slope, excess.err, treatment.slope, null.slope)


def factorial_read(
    baseline_blocks: list[np.ndarray],
    treatment_blocks: list[np.ndarray],
) -> FactorialRead:
    """Average within each family, difference across, and judge against the replicate."""
    if not baseline_blocks or not treatment_blocks:
        raise ValueError("need at least one arm in each family")
    effect = difference(
        np.mean(baseline_blocks, axis=0), np.mean(treatment_blocks, axis=0)
    )
    noise = (
        difference(baseline_blocks[0], baseline_blocks[1])
        if len(baseline_blocks) > 1
        else None
    )
    rounding = (
        difference(treatment_blocks[0], treatment_blocks[1])
        if len(treatment_blocks) > 1
        else None
    )

    if noise is None:
        verdict = "no-replicate"
    else:
        ratio = abs(effect.mean) / max(abs(noise.mean), 1e-12)
        match ratio:
            case r if r <= 1.0:
                verdict = "inside"
            case r if r <= RESOLVABLE_RATIO:
                verdict = "comparable"
            case _:
                verdict = "outside"
    return FactorialRead(effect, noise, rounding, verdict)


def diverged(
    loss: np.ndarray, factor: float = DIVERGENCE_FACTOR, tail: int = 200
) -> bool:
    """Did this run leave and not come back?

    True on any non-finite loss, or when the final ``tail`` steps mean more than
    ``factor`` times the run's own trailing minimum. See docs/performance/ab-testing.md.
    """
    if not np.isfinite(loss).all():
        return True
    best = np.minimum.accumulate(loss)[-1]
    return bool(loss[-tail:].mean() > factor * best)


def break_point(losses: dict[int, np.ndarray]) -> int | None:
    """Smallest key whose run diverged, or ``None`` if none did."""
    broke = [key for key, loss in losses.items() if diverged(loss)]
    return min(broke) if broke else None


def margin_verdict(
    baseline: dict[int, np.ndarray], treatment: dict[int, np.ndarray]
) -> str:
    """What an lr sweep is allowed to conclude about stability margin.

    Derived rather than written as a caption so it cannot go stale against its data.
    """
    base_edge, treat_edge = break_point(baseline), break_point(treatment)
    top = max([*baseline, *treatment], default=0)
    match (base_edge, treat_edge):
        case (None, None):
            return (
                f"neither dtype diverged through {top}x base lr: the sweep bounds "
                "the margin cost, it does not show margin is intact"
            )
        case (None, edge):
            return f"treatment broke at {edge}x where baseline held to {top}x: it costs margin"
        case (edge, None):
            return f"baseline broke at {edge}x and treatment held to {top}x: no margin cost"
        case (b, t) if t < b:
            return (
                f"treatment broke at {t}x where baseline held to {b}x: it costs margin"
            )
        case (b, t) if t == b:
            return f"both broke at {b}x: no measurable margin cost at this scale"
        case (b, t):
            return f"baseline broke at {b}x, treatment held to {t}x: no margin cost"


def spike_overlap(
    left: np.ndarray, right: np.ndarray, threshold: float
) -> tuple[int, int, float]:
    """Spike counts for two arms and the Jaccard overlap of the steps they hit.

    Report with :func:`near_miss_fraction`; the Jaccard alone misleads.
    """
    a = set(np.flatnonzero(left > threshold).tolist())
    b = set(np.flatnonzero(right > threshold).tolist())
    union = a | b
    jaccard = len(a & b) / len(union) if union else 1.0
    return len(a), len(b), jaccard


def near_miss_fraction(
    left: np.ndarray, right: np.ndarray, threshold: float, tolerance: float = 0.2
) -> float:
    """Share of spike disagreements where the arm that missed was near the line.

    A high fraction means the two arms saw the same event and the metric split it.
    """
    a = set(np.flatnonzero(left > threshold).tolist())
    b = set(np.flatnonzero(right > threshold).tolist())
    disagreements = (a | b) - (a & b)
    if not disagreements:
        return float("nan")
    near = 0
    for step in disagreements:
        missed = right[step] if step in a else left[step]
        if missed > (1.0 - tolerance) * threshold:
            near += 1
    return near / len(disagreements)
