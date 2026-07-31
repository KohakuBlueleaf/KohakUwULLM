"""Writing and reading a training run's per-step CSV, and judging its clock.

Step time is admissible only from an exclusive run on a known card; see
docs/performance/ab-testing.md.
"""

import csv
import json
import os
import re

import numpy as np
import torch

from kohakuwullm.bench.core.contention import (
    contended_fraction_limit,
    sample_contended_fraction,
)

# The per-step CSV schema, shared by the reader and the writer.
COLUMNS = ("step", "tokens", "trained", "loss", "grad_norm", "lr", "elapsed")
HEADER = ",".join(COLUMNS)

# Head excluded from timing only; the loss curve keeps these rows.
TIMING_HEAD = 100
MIN_TIMING_STEPS = 200


def write_step_rows(handle, pending: list[tuple], elapsed_now: float) -> tuple:
    """Write buffered ``COLUMNS``-ordered rows using **one** device transfer.

    ``pending`` carries loss and grad norm still on device; ``elapsed_now`` replaces
    the last row's timestamp. Returns that row's ``(loss, grad_norm, elapsed)``.
    """
    scalars = torch.stack(
        [row[3] for row in pending] + [row[4] for row in pending]
    ).tolist()
    half = len(pending)
    last = (0.0, 0.0, elapsed_now)
    for index, row in enumerate(pending):
        step, n_tokens, n_trained, _, _, lr_value, elapsed = row
        per_token = scalars[index] / max(n_trained, 1)
        grad_norm = scalars[half + index]
        if index == half - 1:
            elapsed = elapsed_now
            last = (per_token, grad_norm, elapsed_now)
        handle.write(
            f"{step},{n_tokens},{n_trained},{per_token:.8f},"
            f"{grad_norm:.6f},{lr_value:.8e},{elapsed:.3f}\n"
        )
    handle.flush()
    return last


def load_run(path: str) -> dict[str, np.ndarray]:
    """One arm's CSV as arrays. ``tokens``/``trained`` stay int64 by construction."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    return {
        "step": np.array([int(r["step"]) for r in rows], dtype=np.int64),
        "tokens": np.array([int(r["tokens"]) for r in rows], dtype=np.int64),
        "trained": np.array([int(r["trained"]) for r in rows], dtype=np.int64),
        "loss": np.array([float(r["loss"]) for r in rows]),
        "grad_norm": np.array([float(r["grad_norm"]) for r in rows]),
        "lr": np.array([float(r["lr"]) for r in rows]),
        "elapsed": np.array([float(r["elapsed"]) for r in rows]),
    }


def step_diffs_ms(run: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step wall times in ms. Quantised to 1 ms by the CSV's 3-decimal column."""
    return np.diff(run["elapsed"]) * 1e3


def contended_fraction(run: dict[str, np.ndarray]) -> float:
    """Share of steps slower than 1.1x the run's own median -- "did I have company?" """
    return sample_contended_fraction(step_diffs_ms(run).tolist())


def is_exclusive(run: dict[str, np.ndarray]) -> bool:
    """Whether this run's clock is admissible as a measurement."""
    return contended_fraction(run) <= contended_fraction_limit()


def step_ms(run: dict[str, np.ndarray], head: int = TIMING_HEAD) -> float:
    """Per-step wall time as an endpoint mean over the post-autotune span."""
    elapsed = run["elapsed"]
    if len(elapsed) - head < MIN_TIMING_STEPS:
        raise ValueError(
            f"{len(elapsed)} steps leaves no timing span after a {head}-step head"
        )
    return float((elapsed[-1] - elapsed[head]) / (len(elapsed) - 1 - head) * 1e3)


def speedup(
    baseline: dict[str, np.ndarray],
    treatment: dict[str, np.ndarray],
    same_card: bool,
) -> float:
    """``baseline`` step time over ``treatment``'s, or raise if it is not a measurement.

    ``same_card`` is not inferable from the CSVs, so it is mandatory.
    """
    if not same_card:
        raise ValueError(
            "cross-card step times are not a speedup: the two cards differ by 2.7% "
            "in sustained clock, ~1% on identical work, against a ~7% effect"
        )
    for name, run in (("baseline", baseline), ("treatment", treatment)):
        if not is_exclusive(run):
            raise ValueError(
                f"{name} run is {contended_fraction(run):.2%} contended "
                f"(limit {contended_fraction_limit():.0%}): discard and re-measure"
            )
    return step_ms(baseline) / step_ms(treatment)


def load_arms(
    directory: str, names: list[str], require_summary: bool = False
) -> dict[str, dict[str, np.ndarray]]:
    """Every named arm present in ``directory``, plus its cumulative trained tokens.

    Missing arms are skipped. ``require_summary`` also skips unfinished ones and is
    **mandatory for any comparison of final values** -- see docs/performance/ab-testing.md.
    """
    arms = {}
    for name in names:
        path = os.path.join(directory, f"{name}.csv")
        if not os.path.exists(path):
            continue
        if require_summary and not os.path.exists(
            os.path.join(directory, f"{name}.json")
        ):
            continue
        run = load_run(path)
        # Trained tokens, not tokens seen: prompt positions carry no label.
        run["cumulative_tokens"] = np.cumsum(run["trained"], dtype=np.int64)
        summary = os.path.join(directory, f"{name}.json")
        if os.path.exists(summary):
            with open(summary) as handle:
                run["summary"] = json.load(handle)
        arms[name] = run
    return arms


def discover_lr_multipliers(directory: str, dtypes: tuple[str, ...]) -> tuple[int, ...]:
    """Every ``{dtype}_lr{N}x`` multiplier with a finished (``.json``) run, ascending."""
    found = set()
    for name in os.listdir(directory):
        match = re.fullmatch(r"(.+)_lr(\d+)x\.json", name)
        if match and match.group(1) in dtypes:
            found.add(int(match.group(2)))
    return tuple(sorted(found))


def pair_speedups(
    directory: str, pairs: list[tuple[str, str, str]]
) -> list[tuple[str, float]]:
    """``(label, speedup)`` per pair present; asserts ``same_card`` for every one."""
    out = []
    for label, baseline, treatment in pairs:
        base_path = os.path.join(directory, f"{baseline}.csv")
        treat_path = os.path.join(directory, f"{treatment}.csv")
        if not (os.path.exists(base_path) and os.path.exists(treat_path)):
            continue
        out.append(
            (label, speedup(load_run(base_path), load_run(treat_path), same_card=True))
        )
    return out


def spike_steps(run: dict[str, np.ndarray], threshold: float) -> np.ndarray:
    """Steps whose pre-clip gradient norm exceeded ``threshold``."""
    return np.flatnonzero(run["grad_norm"] > threshold)


def replay_identical(runs: list[dict[str, np.ndarray]]) -> tuple[bool, str]:
    """Whether every run saw the same batches in the same order, and how it is known.

    Compares the four fields that come from the memmap and the schedule, never the
    model, so they are exactly equal or the replay is broken.
    """
    if len(runs) < 2:
        return True, "single run"
    n = min(len(run["step"]) for run in runs)
    for field in ("step", "tokens", "trained", "lr"):
        reference = runs[0][field][:n]
        for other in runs[1:]:
            if not np.array_equal(reference, other[field][:n]):
                bad = int(np.argmax(reference != other[field][:n]))
                return False, f"{field} differs from step {bad}"
    return True, f"tokens, trained and lr identical over {n} steps"
