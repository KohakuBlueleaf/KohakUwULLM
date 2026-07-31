"""The contention rules that gate every benchmark, on a box with no GPU.

``bench/contention.py`` is arithmetic over lists of floats and imports no torch.
That seam is the whole point: the rules deciding whether a measurement is
admissible have to stay testable where no measurement can be taken, or the tests
that pin the *reasoning* are the first ones a CPU-only box skips.

**This module imports no torch either, and that is load-bearing.** It is what
turns "contention.py has no device dependency" from a claim in a docstring into
something that fails if it stops being true.
"""

import pytest

from kohakuwullm.bench.core.contention import (
    contended_fraction_limit,
    contended_fraction_min_iters,
    sample_contended_fraction,
    sample_spread,
    spread_has_power,
    spread_no_power_below_ms,
    spread_threshold,
)


def test_spread_uses_percentiles_so_one_outlier_does_not_convict():
    """The metric change: ``max - min`` gets stricter as iterations grow.

    More samples mean more chances to catch one scheduling hiccup, so a fixed
    threshold silently tightens with ``iters`` -- on an idle card a single outlier
    put a vendor GEMM at 18.4% of range. A p90-p10 band ignores the tail while
    still widening under real contention, which shifts the whole distribution.
    """
    clean = [1.0] * 20
    assert sample_spread(clean) == pytest.approx(0.0)

    # One outlier in twenty: the range doubles, the percentile band does not move.
    spiked = clean[:-1] + [2.0]
    assert (max(spiked) - min(spiked)) / 1.0 == pytest.approx(1.0)
    assert sample_spread(spiked) == pytest.approx(0.0)

    # A genuinely wider distribution still registers.
    wide = [0.8, 0.9, 1.0, 1.1, 1.2] * 4
    assert sample_spread(wide) > 0.3

    assert sample_spread([]) != sample_spread([])  # nan for no samples
    assert sample_spread([2.5]) == pytest.approx(0.0)


def test_spread_threshold_scales_with_the_work_being_measured():
    """One flat percentage cannot serve a 3-second step and a 0.03 ms kernel.

    Jitter is a roughly fixed absolute cost, so as a fraction it grows as the work
    shrinks. Calibrated on an idle card: a flat 5% flagged five of seven shapes
    between 0.02 and 0.6 ms, all of them clean.
    """
    # Long measurements keep the established 5%, so step rows keep their meaning.
    assert spread_threshold(3000.0) == pytest.approx(0.05, abs=1e-4)
    assert spread_threshold(1.0) < spread_threshold(0.1) < spread_threshold(0.02)
    # A 0.6 ms kernel that read 0.6% on an idle card must pass.
    assert spread_threshold(0.6) > 0.006
    # And the test must admit where it has no power rather than pass silently.
    assert spread_has_power(1.0) and spread_has_power(0.1)
    assert not spread_has_power(0.02)
    assert spread_threshold(0.0) == pytest.approx(0.05)


def test_contended_fraction_sees_what_the_spread_band_is_blind_to():
    """Why two contention checks exist, as an executable demonstration.

    They answer different questions. ``sample_spread`` asks "is this number
    trustworthy"; ``sample_contended_fraction`` asks "did I have company". Those
    can legitimately disagree: at a few percent contamination the median is
    untouched, so a measurement is usable *and* was taken on a shared card.

    The first case is the one that motivated the second check -- **spread 0.0000
    against its threshold, contended fraction 0.05**. Discarding the top decile
    must hide contamination that lives inside a decile; no threshold on a p90-p10
    band can fix that, which is why a second statistic was needed rather than a
    retune.
    """
    quiet_ms = 153.0
    limit = contended_fraction_limit()

    # One slow sample in twenty: below the p10/p90 cut entirely.
    intermittent = [quiet_ms] * 19 + [750.0]
    assert sample_spread(intermittent) == pytest.approx(0.0)
    assert sample_spread(intermittent) < spread_threshold(quiet_ms), "band must miss it"
    assert sample_contended_fraction(intermittent) > limit, "frac must catch it"

    # Sustained degradation moves both, so the band is not useless -- just partial.
    sustained = [quiet_ms] * 13 + [400.0] * 7
    assert sample_spread(sustained) > spread_threshold(quiet_ms)
    assert sample_contended_fraction(sustained) > limit

    # A clean sample must fire neither. On 43 labelled clean windows of a real
    # training run this statistic was *exactly* zero -- not one sample in 8,000.
    clean = [153.0, 152.0, 154.0, 153.5, 152.5] * 4
    assert sample_spread(clean) < spread_threshold(quiet_ms)
    assert sample_contended_fraction(clean) == pytest.approx(0.0)

    assert sample_contended_fraction([]) != sample_contended_fraction([])  # nan
    assert sample_contended_fraction([2.5]) == pytest.approx(0.0)


def test_contended_fraction_is_robust_to_a_lone_hiccup():
    """The failure mode that disqualified ``max - min`` must not reappear here.

    A single scheduling hiccup is what made the old range metric convict idle
    cards. In a large sample this statistic barely moves, which is what lets its
    threshold sit two orders above a clean card's zero.
    """
    hiccup = [1.0] * 499 + [5.0]
    assert sample_contended_fraction(hiccup) == pytest.approx(0.002)
    assert sample_contended_fraction(hiccup) < contended_fraction_limit()
    # ... while the range metric it replaces reads 400% on the same sample.
    assert (max(hiccup) - min(hiccup)) / 1.0 == pytest.approx(4.0)

    # "In a large sample" is load-bearing, and the boundary is where callers get it
    # wrong: below `contended_fraction_min_iters` the same lone hiccup convicts. At 30
    # iterations one slow sample reads 3.3% against a 1% limit, so the verdict has no
    # state between "clean" and "shared" -- which flagged a GEMM row whose median was
    # reproducible to 0.87% across five processes, twice, on a different arm each time.
    floor = contended_fraction_min_iters()
    assert floor == 100
    # 30 samples was the iteration count a GEMM row actually used.
    too_few = [1.0] * 29 + [5.0]
    assert sample_contended_fraction(too_few) == pytest.approx(1.0 / 30)
    assert sample_contended_fraction(too_few) > contended_fraction_limit()
    # At the floor the same lone hiccup lands exactly on the limit and is admitted,
    # which is the whole point of deriving the count from the limit rather than
    # picking one: one slow sample in a clean run must not be a conviction.
    just_enough = [1.0] * (floor - 1) + [5.0]
    assert len(just_enough) == floor
    assert sample_contended_fraction(just_enough) == pytest.approx(1.0 / floor)
    assert sample_contended_fraction(just_enough) <= contended_fraction_limit()


def test_no_power_boundary_is_pinned_where_callers_think_it_is():
    """Which rows a reader trusts depends on this number, so it is pinned.

    It was quoted wrong twice in one session -- 0.2 ms and 0.03 ms, against an
    actual 0.05 -- because it was read off a table rather than derived. Both
    mistakes point the same way: they invite trusting a row whose spread check has
    no power. The closed form and the predicate are checked against each other so
    a constant change cannot leave a stale figure in a docstring.
    """
    boundary = spread_no_power_below_ms()
    assert boundary == pytest.approx(0.05)
    # Straddle it: the predicate must agree with its own closed form.
    assert not spread_has_power(boundary * 0.99)
    assert spread_has_power(boundary)
    assert spread_has_power(boundary * 1.01)
    # bench-hygiene's mxfp8 range (0.09-1 ms) must keep power throughout.
    assert spread_has_power(0.09) and spread_has_power(1.0)
