"""Whether a card was exclusive, and whether the check had the power to say so.

Torch-free arithmetic on sample times, so it is testable without a GPU.
See docs/performance/ab-testing.md.
"""

import math
import statistics


def _percentile(sorted_times: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if len(sorted_times) == 1:
        return sorted_times[0]
    position = fraction * (len(sorted_times) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_times) - 1)
    return sorted_times[lower] + (position - lower) * (
        sorted_times[upper] - sorted_times[lower]
    )


def sample_spread(times: list[float]) -> float:
    """``(p90 - p10) / median``. Judge against :func:`spread_threshold`, not a flat %."""
    if not times:
        return float("nan")
    ordered = sorted(times)
    median = statistics.median(ordered)
    if not median:
        return float("nan")
    return (_percentile(ordered, 0.9) - _percentile(ordered, 0.1)) / median


_SPREAD_BASE = 0.05
_SPREAD_FLOOR_MS = 0.010
_SPREAD_NO_POWER = 0.25

_CONTENDED_FACTOR = 1.1
_CONTENDED_LIMIT = 0.01


def sample_contended_fraction(
    times: list[float], factor: float = _CONTENDED_FACTOR
) -> float:
    """Fraction of samples over ``factor`` x median. Blind to a uniform slowdown."""
    if not times:
        return float("nan")
    median = statistics.median(times)
    if not median:
        return float("nan")
    return sum(t > factor * median for t in times) / len(times)


def contended_fraction_limit() -> float:
    """Threshold above which :func:`sample_contended_fraction` means "shared card"."""
    return _CONTENDED_LIMIT


def contended_fraction_min_iters() -> int:
    """Iterations a :func:`sample_contended_fraction` verdict needs to mean its limit."""
    return math.ceil(1.0 / _CONTENDED_LIMIT)


def spread_threshold(median_ms: float) -> float:
    """Largest :func:`sample_spread` a quiet card is expected to show at this median."""
    if not median_ms or median_ms != median_ms:
        return _SPREAD_BASE
    return _SPREAD_BASE + _SPREAD_FLOOR_MS / median_ms


def spread_no_power_below_ms() -> float:
    """The median below which :func:`spread_has_power` is False."""
    return _SPREAD_FLOOR_MS / (_SPREAD_NO_POWER - _SPREAD_BASE)


def spread_has_power(median_ms: float) -> bool:
    """Whether a spread test on work this short can distinguish contention at all."""
    return spread_threshold(median_ms) <= _SPREAD_NO_POWER
