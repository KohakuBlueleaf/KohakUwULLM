"""Checks for the loss-gap reader every fp8 A/B conclusion is read off.

The negative cases are the point, and two of them pin mistakes that were live rather
than hypothetical:

* an unfinished arm must be **skipped**, not compared. A live run's CSV ends wherever it
  has reached, and reading one at 280 of 400 steps against finished arms is how a partial
  curve gets quoted as a result.
* a floor of exactly zero must report **no multiple**, not an infinite one. Two
  bit-identical arms say something about determinism and nothing about resolution, and a
  formatted ``infx floor`` reads as an enormous effect rather than as an absent
  denominator.
"""

import csv
import json

import pytest

from kohakuwullm.bench.analysis.loss_gap import (
    final_window,
    format_gaps,
    load_for_comparison,
    windowed_gaps,
)
from kohakuwullm.bench.analysis.runlog import HEADER

ARMS = ("bf16", "bf16_ctrl", "fp8_up")


def _write(directory, name, losses, finished=True):
    """One arm's CSV, and its summary unless the arm is meant to look still-running."""
    path = directory / f"{name}.csv"
    with open(path, "w", newline="") as handle:
        handle.write(HEADER + "\n")
        writer = csv.writer(handle)
        for step, loss in enumerate(losses):
            writer.writerow([step, 8192, 7000, f"{loss:.8f}", 1.0, 3e-4, 0.2 * step])
    if finished:
        with open(directory / f"{name}.json", "w") as handle:
            json.dump({"arm": name, "steps": len(losses)}, handle)


def test_gaps_are_multiples_of_the_control_and_ignore_unfinished_arms(tmp_path):
    """The whole contract, on curves whose answer is known by construction."""
    steps = 120
    base = [3.0 - 0.001 * i for i in range(steps)]
    _write(tmp_path, "bf16", base)
    # Control offset by 0.01, arm under test by 0.05: the multiple must read 5x.
    _write(tmp_path, "bf16_ctrl", [v + 0.01 for v in base])
    _write(tmp_path, "fp8_up", [v + 0.05 for v in base])

    curves = load_for_comparison(tmp_path, ARMS)
    assert set(curves) == set(ARMS)
    rows = windowed_gaps(curves, window=20, points=3)
    for row in rows:
        assert row["floor"] == pytest.approx(0.01, abs=1e-9)
        assert row["gaps"]["fp8_up"] == pytest.approx(0.05, abs=1e-9)
        assert row["multiples"]["fp8_up"] == pytest.approx(5.0, abs=1e-6)
    # The control is a reference, never a column of its own: reporting its gap beside
    # the arms would invite reading it as a third result.
    assert list(rows[0]["gaps"]) == ["fp8_up"]

    finals = final_window(curves, window=20)
    assert finals["fp8_up"] - finals["bf16"] == pytest.approx(0.05, abs=1e-9)

    # An arm still running is absent from the comparison entirely -- not truncated into
    # it, and not raising. Its CSV is *longer* here, so a reader that admitted it would
    # not even be shortened into agreement; it would compare 200 steps against 120.
    _write(tmp_path, "fp8_rtn", [v + 0.5 for v in base] * 2, finished=False)
    curves = load_for_comparison(tmp_path, ARMS + ("fp8_rtn",))
    assert "fp8_rtn" not in curves


def test_a_zero_floor_reports_no_multiple_rather_than_infinity(tmp_path):
    """Bit-identical arms mean no resolution to divide by, not an infinite effect."""
    base = [2.0] * 60
    _write(tmp_path, "bf16", base)
    _write(tmp_path, "bf16_ctrl", list(base))
    _write(tmp_path, "fp8_up", [v + 0.2 for v in base])

    rows = windowed_gaps(load_for_comparison(tmp_path, ARMS), window=20, points=2)
    for row in rows:
        assert row["floor"] == 0.0
        assert row["gaps"]["fp8_up"] == pytest.approx(0.2)
        assert row["multiples"]["fp8_up"] is None
    text = format_gaps(rows)
    assert "inf" not in text and "n/a" in text


def test_curves_truncate_to_the_shortest_arm_and_a_missing_baseline_raises(tmp_path):
    base = [2.0 - 0.001 * i for i in range(90)]
    _write(tmp_path, "bf16", base)
    _write(tmp_path, "fp8_up", base[:40])
    curves = load_for_comparison(tmp_path, ARMS)
    assert all(len(v) == 40 for v in curves.values())
    # No control present: the gaps still compute, and the multiple is simply absent.
    rows = windowed_gaps(curves, window=20, points=2)
    assert rows[0]["floor"] is None and rows[0]["multiples"]["fp8_up"] is None

    # Every gap is measured against `bf16`, so its absence is not a partial result.
    with pytest.raises(ValueError, match="no finished 'bf16' arm"):
        load_for_comparison(tmp_path / "empty", ARMS)


def test_a_window_longer_than_the_run_refuses(tmp_path):
    """Silently shrinking the window would report a mean over fewer steps than asked."""
    _write(tmp_path, "bf16", [2.0] * 10)
    curves = load_for_comparison(tmp_path, ("bf16",))
    with pytest.raises(ValueError, match="cannot fill"):
        windowed_gaps(curves, window=50)
