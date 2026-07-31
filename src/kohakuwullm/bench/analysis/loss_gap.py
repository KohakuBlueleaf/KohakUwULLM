"""Is an arm's loss gap the dtype, or the gap bf16 has with itself?

Every gap is reported as a **multiple of the control arm's own gap**, over windows
rather than pointwise, never as an absolute number. See docs/performance/ab-testing.md.
"""

import statistics

from kohakuwullm.bench.analysis.runlog import load_arms

# The baseline, and the arm that differs from it in nothing.
BASELINE = "bf16"
CONTROL = "bf16_ctrl"


def _mean(values, lo: int, hi: int) -> float:
    return statistics.fmean(values[lo:hi])


def load_for_comparison(directory: str, names) -> dict[str, list[float]]:
    """Per-arm loss curves, truncated to the shortest arm and finished-only.

    ``loss`` is already the fp32 per-token mean the head reduced; do not divide it by
    ``trained`` again.
    """
    arms = load_arms(directory, list(names), require_summary=True)
    if BASELINE not in arms:
        raise ValueError(
            f"{directory} has no finished {BASELINE!r} arm; every gap is measured "
            "against it, so there is nothing to compare"
        )
    steps = min(len(run["loss"]) for run in arms.values())
    return {name: list(run["loss"][:steps]) for name, run in arms.items()}


def final_window(curves: dict[str, list[float]], window: int = 50) -> dict[str, float]:
    """Each arm's mean loss over its last ``window`` steps."""
    return {name: _mean(values, -window, None) for name, values in curves.items()}


def windowed_gaps(
    curves: dict[str, list[float]], window: int = 50, points: int = 4
) -> list[dict]:
    """Gap against the baseline per window, and its multiple of the control's gap.

    ``multiple`` is ``None`` rather than infinity when the control's gap is zero.
    """
    steps = len(curves[BASELINE])
    if steps < window:
        raise ValueError(f"{steps} steps cannot fill a {window}-step window")
    others = [n for n in curves if n not in (BASELINE, CONTROL)]
    starts = [round(i * (steps - window) / max(points - 1, 1)) for i in range(points)]

    rows = []
    for lo in sorted(set(starts)):
        hi = lo + window
        base = _mean(curves[BASELINE], lo, hi)
        floor = (
            abs(_mean(curves[CONTROL], lo, hi) - base) if CONTROL in curves else None
        )
        gaps = {n: abs(_mean(curves[n], lo, hi) - base) for n in others}
        rows.append(
            {
                "window": (lo, hi),
                "baseline_loss": base,
                "floor": floor,
                "gaps": gaps,
                "multiples": {
                    n: (g / floor if floor else None) for n, g in gaps.items()
                },
            }
        )
    return rows


def format_gaps(rows: list[dict]) -> str:
    """The table, with the floor column first -- every other column divides by it."""
    if not rows:
        return "no windows"
    others = list(rows[0]["gaps"])
    head = f"{'window':>13}  {'baseline':>9}  {'floor':>9}" + "".join(
        f"{n:>26}" for n in others
    )
    lines = [head]
    for row in rows:
        lo, hi = row["window"]
        floor = row["floor"]
        cells = [
            f"{f'{lo}-{hi}':>13}  {row['baseline_loss']:9.4f}  "
            f"{'n/a' if floor is None else f'{floor:9.4f}'}"
        ]
        for name in others:
            gap, mult = row["gaps"][name], row["multiples"][name]
            tail = "     n/a" if mult is None else f"{mult:6.0f}x floor"
            cells.append(f"{gap:14.4f} {tail}")
        lines.append("".join(cells))
    return "\n".join(lines)
