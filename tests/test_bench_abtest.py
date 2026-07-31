"""Checks for the statistics that decide what the MXFP8 A/B concluded.

The verdict these produce is the experiment's output, and in a plotting script it
could not be tested at all. Here each case is built with a known answer, so the
tests can pin the two failures that would matter most:

* calling a real regression "inside the noise floor" -- the false negative that
  would ship a bad dtype;
* calling a constant offset a divergence -- the false positive that would reject a
  good one. A slope test is the only thing separating those, so it gets both
  directions.
"""

import numpy as np
import pytest

from kohakuwullm.bench.analysis.abtest import (
    BLOCK,
    RESOLVABLE_RATIO,
    block_means,
    break_point,
    difference,
    diverged,
    factorial_read,
    gap_trend,
    margin_verdict,
    near_miss_fraction,
    slope_vs_null,
    spike_overlap,
)

BLOCKS = 100


def arm(offset: float = 0.0, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """A block-mean series: a constant loss level plus independent block noise."""
    rng = np.random.default_rng(seed)
    return 1.5 + offset + rng.normal(0.0, noise, BLOCKS)


def test_block_means_drops_a_short_final_block():
    """A partial block would be a mean over fewer steps, weighted as if it were full."""
    values = np.arange(BLOCK * 3 + 7, dtype=float)
    means = block_means(values)
    assert len(means) == 3
    assert means[0] == pytest.approx(np.arange(BLOCK).mean())
    assert len(block_means(np.arange(BLOCK - 1, dtype=float))) == 0


def test_factorial_read_calls_a_small_effect_inside_the_noise():
    """The whole point of the replicate arm: an effect below it is not a result."""
    # Two bf16 arms differing by 0.004, two fp8 arms offset by only 0.001 from them.
    read = factorial_read(
        [arm(0.000, seed=1), arm(0.004, seed=2)],
        [arm(0.001, seed=3), arm(0.003, seed=4)],
    )
    assert read.noise is not None
    assert read.rounding is not None
    assert read.ratio < 1.0
    assert read.verdict == "inside"


def test_factorial_read_calls_a_large_effect_outside_the_noise():
    """The false negative that would matter: a real regression called noise."""
    read = factorial_read(
        [arm(0.000, seed=1), arm(0.001, seed=2)],
        [arm(0.100, seed=3), arm(0.101, seed=4)],
    )
    assert read.ratio > RESOLVABLE_RATIO
    assert read.verdict == "outside"
    assert read.effect.mean == pytest.approx(0.100, abs=0.01)


def test_factorial_read_refuses_to_resolve_a_marginal_effect():
    """Between 1x and 3x the noise, one replicate pair cannot call it either way."""
    read = factorial_read(
        [arm(0.000, seed=1), arm(0.010, seed=2)],
        [arm(0.018, seed=3), arm(0.020, seed=4)],
    )
    assert 1.0 < read.ratio <= RESOLVABLE_RATIO
    assert read.verdict == "comparable"


def test_factorial_read_without_a_replicate_is_inconclusive_by_construction():
    """A 12-sigma effect and no noise scale is still not an answer."""
    read = factorial_read([arm(0.0, seed=1)], [arm(0.05, seed=2)])
    assert read.noise is None
    assert read.ratio is None
    assert read.verdict == "no-replicate"
    # The effect is overwhelmingly significant and that changes nothing.
    assert read.effect.sigma > 10


def test_factorial_read_needs_both_families():
    with pytest.raises(ValueError, match="each family"):
        factorial_read([], [arm()])


def test_gap_trend_separates_a_constant_offset_from_a_divergence():
    """A flat offset must read flat, and a growing one must read GROWING."""
    tokens = np.linspace(0.1, 6.0, BLOCKS)
    rng = np.random.default_rng(0)

    flat = 0.001 + rng.normal(0.0, 0.0005, BLOCKS)
    trend = gap_trend(flat, tokens)
    assert not trend.growing
    assert trend.label == "flat"
    assert trend.mean_gap == pytest.approx(0.001, abs=0.0003)

    diverging = 0.001 + 0.004 * tokens + rng.normal(0.0, 0.0005, BLOCKS)
    trend = gap_trend(diverging, tokens)
    assert trend.growing
    assert trend.label == "GROWING"
    assert trend.slope == pytest.approx(0.004, rel=0.1)


def test_gap_trend_does_not_call_a_shrinking_gap_growth():
    """Sign matters: a significant *negative* slope is not a divergence warning."""
    tokens = np.linspace(0.1, 6.0, BLOCKS)
    rng = np.random.default_rng(1)
    shrinking = 0.01 - 0.004 * tokens + rng.normal(0.0, 0.0005, BLOCKS)
    trend = gap_trend(shrinking, tokens)
    assert trend.sigma > 3.0
    assert not trend.growing


def three_arms(base_drift, treat_drift, treat_offset, seed, noise=0.0004):
    """Baseline, replicate and treatment loss series sharing one baseline noise draw.

    Built as three *arms* rather than two pre-made gaps, because the correlation this
    exercises only exists when both gaps are measured against the same baseline -- which
    is exactly how the real experiment measures them.
    """
    tokens = np.linspace(0.1, 6.0, BLOCKS)
    rng = np.random.default_rng(seed)
    base = 1.5 + rng.normal(0, noise, BLOCKS)
    control = base + 0.0004 + base_drift * tokens + rng.normal(0, noise, BLOCKS)
    treatment = (
        base + treat_offset + treat_drift * tokens + rng.normal(0, noise, BLOCKS)
    )
    return tokens, base, control, treatment


def build_test(tokens, base, control, treatment):
    """Assemble a SlopeTest the way the plotting script does."""
    null = gap_trend(control - base, tokens)
    treat = gap_trend(treatment - base, tokens)
    excess = gap_trend(treatment - control, tokens)
    return slope_vs_null(excess, treat, null)


def test_slope_vs_null_clears_drift_the_replicate_also_has():
    """The test the criterion turns on: drift only counts if it beats the null's.

    An fp8 slope can sit many sigma from *zero* and still be entirely what two
    identical bf16 runs produce from atomics. Testing against zero would call that a
    divergence; testing against the replicate does not.
    """
    drift = 0.0020
    tokens, base, control, treatment = three_arms(drift, drift, 0.0012, seed=3)
    test = build_test(tokens, base, control, treatment)

    # Against zero the treatment's own slope looks like a runaway divergence.
    assert gap_trend(treatment - base, tokens).growing
    assert gap_trend(treatment - base, tokens).sigma > 10

    assert not test.growing
    assert test.label == "flat vs null"
    assert abs(test.excess_slope) < 0.0005


def test_slope_vs_null_still_catches_drift_beyond_the_null():
    """The other direction: real excess drift must survive the comparison."""
    tokens, base, control, treatment = three_arms(0.0002, 0.0060, 0.0004, seed=4)
    test = build_test(tokens, base, control, treatment)

    assert test.growing
    assert test.label == "GROWING vs null"
    assert test.excess_slope == pytest.approx(0.0058, rel=0.15)


def test_slope_vs_null_does_not_flag_an_arm_drifting_less_than_the_null():
    """Negative excess is not growth however significant it is."""
    tokens, base, control, treatment = three_arms(0.0060, 0.0002, 0.0, seed=5)
    test = build_test(tokens, base, control, treatment)

    assert test.excess_slope < 0
    assert test.sigma > 3.0
    assert not test.growing


def test_excess_error_is_the_direct_fit_not_a_quadrature_sum():
    """The correlated-error bug: both gaps share the baseline, so quadrature is wrong.

    With a large shared baseline noise the two baseline-relative slopes are strongly
    correlated, and ``hypot(err_treatment, err_null)`` overstates the excess error --
    conservatively, which costs sensitivity silently. The direct fit on
    ``treatment - control`` cancels the baseline and is exact.
    """
    tokens, base, control, treatment = three_arms(
        0.0020, 0.0020, 0.0012, seed=7, noise=0.004
    )
    null = gap_trend(control - base, tokens)
    treat = gap_trend(treatment - base, tokens)
    direct = gap_trend(treatment - control, tokens)

    quadrature = float(np.hypot(treat.err, null.err))
    assert direct.err < quadrature
    # The test must carry the direct error, not the inflated one.
    assert build_test(tokens, base, control, treatment).excess_err == pytest.approx(
        direct.err
    )


def test_difference_error_grows_with_the_spread_it_is_given():
    a, b = arm(0.0, noise=0.001, seed=5), arm(0.0, noise=0.05, seed=6)
    assert difference(a, a).mean == pytest.approx(0.0)
    assert difference(a, b).err > difference(a, arm(0.0, 0.001, seed=7)).err


def test_spike_overlap_and_near_miss_explain_a_falling_jaccard():
    """A shared event straddling the threshold is scored as two private spikes."""
    left = np.zeros(100)
    right = np.zeros(100)
    # Steps 10-14: both clearly above. Steps 20-24: left above, right just below,
    # which is one event the metric splits rather than two different events.
    left[10:15] = 9.0
    right[10:15] = 9.0
    left[20:25] = 5.2
    right[20:25] = 4.9

    count_left, count_right, jaccard = spike_overlap(left, right, 5.0)
    assert (count_left, count_right) == (10, 5)
    assert jaccard == pytest.approx(0.5)
    # Every disagreement is a near-miss, so the Jaccard of 0.5 is entirely artifact.
    assert near_miss_fraction(left, right, 5.0) == pytest.approx(1.0)

    # Move the near-misses far below and the same Jaccard now means real divergence.
    right[20:25] = 0.1
    assert near_miss_fraction(left, right, 5.0) == pytest.approx(0.0)


def test_near_miss_fraction_is_nan_when_the_arms_agree_completely():
    values = np.zeros(50)
    values[5:10] = 9.0
    assert np.isnan(near_miss_fraction(values, values.copy(), 5.0))


def converging(final: float = 1.4, spike: float | None = None) -> np.ndarray:
    """A loss curve that falls 11.3 -> ``final``, optionally with a warmup spike."""
    loss = np.concatenate([np.linspace(11.3, 2.0, 20), np.linspace(2.0, final, 1980)])
    if spike is not None:
        loss[19] = spike
    return loss


def test_diverged_ignores_a_transient_that_recovers():
    """The real fp8 4x-lr arm spikes at step 19 and converges; that is not divergence.

    Testing the *maximum* instead of the final loss would call every warmup a
    divergence -- and every arm's worst spike in this sweep is in its first 20 steps.
    """
    assert not diverged(converging())
    assert not diverged(converging(spike=13.5))
    # The spike is nearly 10x the final loss, so a max-based test would fire here.
    assert converging(spike=13.5).max() / converging(spike=13.5)[-200:].mean() > 9


def test_diverged_catches_a_run_that_leaves_and_stays_out():
    loss = converging()
    loss[-300:] = 4.0
    assert diverged(loss)

    nan_run = converging()
    nan_run[1500] = np.nan
    assert diverged(nan_run)


def test_break_point_is_the_smallest_multiplier_that_broke():
    broken = converging()
    broken[-300:] = 5.0
    losses = {1: converging(), 2: converging(), 4: broken, 8: broken}
    assert break_point(losses) == 4
    assert break_point({1: converging(), 2: converging()}) is None


def test_margin_verdict_will_not_call_a_null_sweep_a_clean_bill():
    """The failure this exists to prevent: 'nothing broke' read as 'margin intact'."""
    clean = {m: converging() for m in (1, 2, 4, 8)}
    verdict = margin_verdict(clean, {m: converging() for m in (1, 2, 4, 8)})
    assert "bounds the margin cost" in verdict
    assert "does not show margin is intact" in verdict
    assert "no measurable margin cost" not in verdict


def test_margin_verdict_reports_a_real_margin_loss():
    broken = converging()
    broken[-300:] = 5.0
    baseline = {1: converging(), 2: converging(), 4: converging(), 8: broken}
    treatment = {1: converging(), 2: converging(), 4: broken, 8: broken}
    assert margin_verdict(baseline, treatment) == (
        "treatment broke at 4x where baseline held to 8x: it costs margin"
    )

    # Both breaking at the same multiplier is the "no cost" reading, and it is only
    # available because the sweep actually found an edge.
    assert "no measurable margin cost" in margin_verdict(treatment, treatment)
