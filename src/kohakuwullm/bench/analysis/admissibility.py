"""May this measurement be quoted, and which card produced it? See docs/performance/ab-testing.md."""

import os
import statistics

from kohakuwullm.bench.core.contention import (
    contended_fraction_limit,
    contended_fraction_min_iters,
    sample_contended_fraction,
    spread_has_power,
)
from kohakuwullm.bench.core.timing import bench_samples, cpu_enqueue_ms, graph_ms


def median_and_contention(fn, iters: int | None = None) -> tuple[float, float, bool]:
    """Median wall ms, contended fraction and ``has_power``, from one sample loop.

    ``iters`` may not be set below :func:`contended_fraction_min_iters`.
    """
    iters = iters or contended_fraction_min_iters()
    if iters < contended_fraction_min_iters():
        raise ValueError(
            f"iters={iters} cannot resolve a {contended_fraction_limit():.0%} "
            f"contended-fraction limit; needs {contended_fraction_min_iters()}"
        )
    samples = bench_samples(fn, iters=iters)
    median = statistics.median(samples)
    return median, sample_contended_fraction(samples), spread_has_power(median)


def net_and_graph(fn, graph_iters: int = 30) -> tuple[float, float, float, bool]:
    """``(wall minus host, graph replay, contended fraction, verdict has power)``."""
    wall, contended, power = median_and_contention(fn)
    return wall - cpu_enqueue_ms(fn), graph_ms(fn, iters=graph_iters), contended, power


def contention_notes(row: dict, keys: tuple[str, ...], certifiable: bool) -> list[str]:
    """Contention verdicts for the arms whose samples back *this* row's own number."""
    if not certifiable:
        return ["too short to certify exclusivity; not a clean verdict"]
    notes = []
    for key in keys:
        if row[key] > contended_fraction_limit():
            notes.append(f"{key}={row[key]:.2%}: card was shared, discard this row")
    return notes


def suspect_speedup(row: dict, keys: tuple[str, ...]) -> str:
    """Why this row's *ratio* is not evidence, when its own number still is."""
    notes = []
    for key in keys:
        if key.startswith("contended") and row[key] > contended_fraction_limit():
            notes.append(f"{key}={row[key]:.2%}: baseline arm was shared")
        if key.startswith("host_share") and row[key] > 0.5:
            notes.append(f"{key}={row[key]:.0%}: baseline net_ms is not evidence")
    return "; ".join(notes)


def visible_devices() -> str:
    """The card this process can see, as the launcher set it -- the only record."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "<unset: torch picked device 0>")
