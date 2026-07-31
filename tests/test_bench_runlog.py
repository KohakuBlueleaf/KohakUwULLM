"""Checks for the training-run reader the fp8 figures draw their numbers from.

Two of these pin mistakes that were live in this repo's output rather than
hypothetical ones. The MXFP8 A/B published a 1.062x speedup computed from bf16 on
GPU0 against a 4.3%-contended fp8 arm on GPU1 -- a cross-card comparison of
contended data, reported to four significant figures. And the per-step median that
produced it is quantised to 1 ms by the CSV's 3-decimal ``elapsed`` column, which
is 0.7% at 150 ms/step against a ~7% effect.

So the negative cases are the point: ``speedup`` must *refuse*.
"""

import csv
import json

import numpy as np
import pytest
import torch

from kohakuwullm.bench.analysis.runlog import (
    COLUMNS,
    HEADER,
    contended_fraction,
    discover_lr_multipliers,
    is_exclusive,
    load_arms,
    load_run,
    replay_identical,
    speedup,
    spike_steps,
    step_diffs_ms,
    step_ms,
    write_step_rows,
)
from kohakuwullm.bench.core.contention import contended_fraction_limit

STEPS = 2000


def make_run(
    per_step_ms: float = 150.0,
    steps: int = STEPS,
    slow_every: int | None = None,
    slow_factor: float = 1.5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """A synthetic run whose ``elapsed`` is rounded exactly as the harness writes it.

    The rounding is the whole point of several of these tests, so it happens here
    rather than being approximated: ``f"{value:.3f}"`` in ``fp8_training.py`` is what
    puts the 1 ms floor under every per-step statistic.
    """
    rng = np.random.default_rng(seed)
    times = np.full(steps, per_step_ms) + rng.normal(0.0, 0.4, steps)
    if slow_every:
        times[::slow_every] *= slow_factor
    elapsed = np.round(np.cumsum(times) / 1e3, 3)
    return {
        "step": np.arange(steps, dtype=np.int64),
        "tokens": np.full(steps, 16384, dtype=np.int64),
        "trained": np.full(steps, 14500, dtype=np.int64),
        "loss": np.full(steps, 1.5),
        "grad_norm": np.full(steps, 0.4),
        "lr": np.full(steps, 3e-4),
        "elapsed": elapsed,
    }


def test_step_ms_resolves_what_the_per_step_median_quantises_away():
    """The endpoint mean recovers sub-ms detail; a median of diffs cannot express it."""
    true_ms = 150.37
    run = make_run(per_step_ms=true_ms)

    assert step_ms(run) == pytest.approx(true_ms, abs=0.05)

    # The median lands on a whole millisecond because every diff does, so it cannot
    # represent 150.37 even in principle -- and it is off by more than a third of
    # the ~7% effect the fp8 experiment is trying to measure.
    median = float(np.median(step_diffs_ms(run)))
    assert median == pytest.approx(round(median), abs=1e-6)
    assert abs(median - true_ms) > 0.3


def test_speedup_refuses_a_cross_card_pair():
    """The 2.7% clock spread between GPU0 and GPU1 is 25x the effect being measured."""
    baseline, treatment = make_run(155.0), make_run(145.0, seed=1)

    assert speedup(baseline, treatment, same_card=True) == pytest.approx(
        155.0 / 145.0, rel=0.01
    )
    with pytest.raises(ValueError, match="cross-card"):
        speedup(baseline, treatment, same_card=False)


def test_speedup_refuses_a_contended_run_and_says_which_side():
    """A co-tenant inflates the step time it lands on, biasing the ratio either way."""
    clean = make_run(155.0)
    dirty = make_run(145.0, slow_every=20, seed=1)

    assert is_exclusive(clean)
    assert contended_fraction(clean) <= contended_fraction_limit()
    assert not is_exclusive(dirty)
    assert contended_fraction(dirty) > contended_fraction_limit()

    with pytest.raises(ValueError, match="treatment run is"):
        speedup(clean, dirty, same_card=True)
    with pytest.raises(ValueError, match="baseline run is"):
        speedup(dirty, clean, same_card=True)


def test_replay_identical_catches_a_diverged_batch_order():
    """``trained`` comes from the memmap, so a mismatch is a broken replay, not fp8."""
    a, b = make_run(seed=0), make_run(seed=1)
    ok, why = replay_identical([a, b])
    assert ok and "identical" in why

    b["trained"] = b["trained"].copy()
    b["trained"][1234] += 1
    ok, why = replay_identical([a, b])
    assert not ok
    assert "trained differs from step 1234" == why


def test_replay_identity_survives_arms_of_different_length():
    """Round 1 is read while round 2 is still running, so the check must truncate."""
    a, short = make_run(steps=STEPS), make_run(steps=STEPS // 2)
    ok, why = replay_identical([a, short])
    assert ok
    assert f"over {STEPS // 2} steps" in why


def test_load_run_keeps_token_counts_in_int64(tmp_path):
    """A run passes 2^31 tokens; a float32 counter stops incrementing at 2^24."""
    path = tmp_path / "arm.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "tokens", "trained", "loss", "grad_norm", "lr", "elapsed"]
        )
        writer.writerow([0, 16384, 14500, 1.5, 0.4, 3e-4, 0.15])
        writer.writerow([1, 2**31 + 5, 14500, 1.4, 6.0, 3e-4, 0.3])

    run = load_run(str(path))
    assert run["tokens"].dtype == np.int64
    assert run["trained"].dtype == np.int64
    assert int(run["tokens"][1]) == 2**31 + 5
    assert spike_steps(run, 5.0).tolist() == [1]


def write_run(directory, name: str, steps: int, loss: float, summary: bool) -> None:
    """A CSV at a flat ``loss``, with the ``.json`` written only if ``summary``."""
    with open(directory / f"{name}.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "tokens", "trained", "loss", "grad_norm", "lr", "elapsed"]
        )
        for step in range(steps):
            writer.writerow(
                [step, 16384, 14500, loss, 0.4, 3e-4, round(step * 0.15, 3)]
            )
    if summary:
        with open(directory / f"{name}.json", "w") as handle:
            json.dump({"steps": steps}, handle)


def test_load_arms_can_exclude_runs_that_have_not_finished(tmp_path):
    """A live CSV ends where it has reached, so comparing final values needs the gate.

    This is the bug it exists for, with the real numbers: an fp8 sweep run stopped at
    600 of 2000 steps sat at loss 2.04 while the finished bf16 arm was at 1.57, and
    admitting it reported a +0.47 dtype regression that was purely the missing 1400
    steps. Divergence-sized, entirely an artifact of reading a run too early.
    """
    write_run(tmp_path, "bf16_lr16x", 2000, 1.57, summary=True)
    write_run(tmp_path, "fp8_up_lr16x", 600, 2.04, summary=False)
    names = ["bf16_lr16x", "fp8_up_lr16x"]

    # The default admits both, which is what a curves-over-time plot wants.
    assert set(load_arms(str(tmp_path), names)) == {"bf16_lr16x", "fp8_up_lr16x"}

    gated = load_arms(str(tmp_path), names, require_summary=True)
    assert set(gated) == {"bf16_lr16x"}

    # And the multiplier is still discovered, because bf16 finished at it -- so the
    # gate has to be in the loader, not only in discovery.
    assert discover_lr_multipliers(str(tmp_path), ("bf16", "fp8_up")) == (16,)


def test_load_arms_carries_cumulative_trained_tokens(tmp_path):
    write_run(tmp_path, "bf16", 300, 1.5, summary=True)
    run = load_arms(str(tmp_path), ["bf16"])["bf16"]
    assert run["cumulative_tokens"].dtype == np.int64
    assert int(run["cumulative_tokens"][-1]) == 300 * 14500
    assert run["summary"]["steps"] == 300


def test_write_step_rows_pairs_each_loss_with_its_own_grad_norm(tmp_path):
    """The stack-and-split must not scramble which grad norm belongs to which step.

    Both scalars for every buffered step go into one tensor so the batch costs one
    device transfer instead of two per step, and the halves are separated by position.
    An off-by-half there would pair step i's loss with step i+k's gradient norm, and
    the CSV would still parse, still look plausible, and still produce a smooth loss
    curve -- so nothing downstream would notice. Hence distinct, checkable values.
    """
    path = tmp_path / "arm.csv"
    trained = 1000
    # loss_i = i+1 (so per-token = (i+1)/1000), grad_i = 100+i. Distinct ranges mean a
    # swapped half is unmistakable rather than merely wrong.
    pending = [
        (
            i,
            16384,
            trained,
            torch.tensor(float(i + 1)),
            torch.tensor(100.0 + i),
            3e-4,
            0.5 * i,
        )
        for i in range(4)
    ]
    with open(path, "w") as handle:
        handle.write(HEADER + "\n")
        last = write_step_rows(handle, pending, elapsed_now=9.75)

    run = load_run(str(path))
    assert run["step"].tolist() == [0, 1, 2, 3]
    assert run["loss"].tolist() == [0.001, 0.002, 0.003, 0.004]
    assert run["grad_norm"].tolist() == [100.0, 101.0, 102.0, 103.0]
    # Only the final row's timestamp is re-taken after the transfer; earlier rows keep
    # the host-side value they were appended with.
    assert run["elapsed"].tolist() == [0.0, 0.5, 1.0, 9.75]
    assert last == (0.004, 103.0, 9.75)


def test_write_step_rows_matches_the_schema_the_reader_expects(tmp_path):
    """One batch of one row is the default path, and it must round-trip exactly."""
    path = tmp_path / "one.csv"
    pending = [(7, 16321, 14040, torch.tensor(2.5), torch.tensor(0.75), 1.5e-6, 4.076)]
    with open(path, "w") as handle:
        handle.write(HEADER + "\n")
        write_step_rows(handle, pending, elapsed_now=4.076)

    with open(path) as handle:
        header, row = handle.read().splitlines()
    assert header == "step,tokens,trained,loss,grad_norm,lr,elapsed"
    assert row == "7,16321,14040,0.00017806,0.750000,1.50000000e-06,4.076"
    assert set(load_run(str(path))) >= set(COLUMNS)


def test_step_ms_refuses_a_run_too_short_to_time():
    with pytest.raises(ValueError, match="no timing span"):
        step_ms(make_run(steps=150))
