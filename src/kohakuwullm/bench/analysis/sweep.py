"""Reading the end-to-end sweep: what a row is, and whether to trust it.

A row is ``clean``, ``contended`` or ``oom``; *never run* is the absence of a row and
must stay distinguishable from a zero. See docs/performance/ab-testing.md.
"""

import glob
import json
import os

# Tag flags the drivers emit. Anything else is reported rather than guessed at.
DTYPES = ("fp32", "bf16", "fp16")
OPTIMIZERS = ("adamw", "muon")
# `mxfp8` and `drift` are listed to keep `unknown` meaningful; nothing reads them here.
FLAGS = ("ckpt", "micro", "compiled", "mxfp8", "drift")


def parse_tag(tag: str) -> dict:
    """Strategy / dtype / optimizer / flags from a tag; ``strategy`` never holds dtype."""
    parts = tag.split("_")
    out = dict(strategy=parts[0], dtype=None, optimizer=None, flags=[], unknown=[])
    for part in parts[1:]:
        if part in DTYPES:
            out["dtype"] = part
        elif part in OPTIMIZERS:
            out["optimizer"] = part
        elif part in FLAGS:
            out["flags"].append(part)
        else:
            out["unknown"].append(part)
    if "ckpt" in out["flags"]:
        out["strategy"] = f"{parts[0]}_ckpt"
    return out


def resolve_dtype(payload: dict, parsed: dict) -> tuple[str | None, str | None]:
    """``(dtype, note)``: recorded config, then tag, then default; ``None`` if unknowable."""
    if "param_dtype" in payload:
        return payload["param_dtype"], None
    if parsed["dtype"]:
        return parsed["dtype"], None
    if "micro" in parsed["flags"]:
        return None, (
            "dtype not recoverable -- the micro driver forces bf16, the tag "
            "omits it, and the JSON predates config recording"
        )
    return "fp32", None


def classify(row: dict, max_spread: float) -> str:
    """``clean`` / ``contended`` / ``oom``; a missing or NaN spread is contended."""
    if not row.get("ok"):
        return "oom"
    spread = row.get("step_spread", float("nan"))
    # NaN fails every comparison, so this also catches the missing-field case.
    return "clean" if spread <= max_spread else "contended"


def load_sweep(directory: str, max_spread: float = 0.05) -> tuple[list, list, set]:
    """``(rows, notes, devices)`` over every ``<preset>_<tag>.json`` in a dir."""
    rows: list[dict] = []
    notes: list[str] = []
    devices: set[str] = set()
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        name = os.path.basename(path)[: -len(".json")]
        # A bare list of rows carries no preset, so it is skipped rather than raising.
        if not isinstance(payload, dict):
            notes.append(f"{name}: no preset/rows, skipped")
            continue
        preset = payload.get("preset")
        if not preset or "rows" not in payload:
            notes.append(f"{name}: no preset/rows, skipped")
            continue
        tag = payload.get("tag") or (
            name[len(preset) + 1 :] if name.startswith(preset) else name
        )
        parsed = parse_tag(tag)
        if parsed["unknown"]:
            notes.append(
                f"{name}: unrecognised tag part(s) {parsed['unknown']} -- "
                "config inferred only as far as the tag allows"
            )
        dtype, note = resolve_dtype(payload, parsed)
        if note:
            notes.append(f"{name}: {note}")
        if payload.get("device"):
            devices.add(payload["device"])
        for row in payload["rows"]:
            status = classify(row, max_spread)
            rows.append(
                dict(
                    preset=preset,
                    tag=tag,
                    strategy=parsed["strategy"],
                    world=payload.get("world", 1),
                    dtype=dtype,
                    optimizer=(
                        payload.get("optimizer") or parsed["optimizer"] or "adamw"
                    ),
                    per_micro=row["per_micro"],
                    n_micro=row["n_micro"],
                    tokens_per_s=row["tokens_per_s"],
                    peak_gib=row.get("peak_gib", float("nan")),
                    spread=row.get("step_spread", float("nan")),
                    # ``None`` where the row predates the steady-state check.
                    steady=row.get("steady"),
                    # From the payload, never the tag: what the run did, not its label.
                    mxfp8=bool(payload.get("mxfp8", False)),
                    rounding=payload.get("rounding", "nearest"),
                    # ``None`` where the sweep did not measure its own dense ceiling.
                    matmul_ceiling_tflops=payload.get("matmul_ceiling_tflops"),
                    status=status,
                    ok=status != "oom",
                    clean=status == "clean",
                )
            )
    return rows, notes, devices


def planning_units(row: dict) -> dict:
    """``tok/s`` restated in planning units, against the row's **own** batch."""
    budget = row["per_micro"] * row["n_micro"]
    per_day = row["tokens_per_s"] * 86400
    return dict(
        budget=budget,
        b_tok_day=per_day / 1e9,
        steps_day=per_day / budget,
        steps_hour=row["tokens_per_s"] * 3600 / budget,
    )


def ordered_presets(rows: list[dict]) -> list[str]:
    """Dense presets then MoE, each alphabetical -- the reading order."""
    names = {r["preset"] for r in rows}
    # Substring, not prefix: the Kohaku ladder names its sparse rungs `Kohaku-MoE-*`.
    dense = sorted(n for n in names if "MoE" not in n)
    return dense + sorted(n for n in names if "MoE" in n)


def strategy_coverage(rows: list[dict]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = {}
    for row in rows:
        coverage.setdefault(row["strategy"], set()).add(row["preset"])
    return coverage


def best_cell(rows: list[dict], preset: str, strategy: str) -> dict | None:
    """Fastest clean row, else least-contended, else OOM. ``None`` means never run."""
    pool = [r for r in rows if r["preset"] == preset and r["strategy"] == strategy]
    if not pool:
        return None
    clean = [r for r in pool if r["clean"]]
    if clean:
        return dict(max(clean, key=lambda r: r["tokens_per_s"]), status="clean")
    usable = [r for r in pool if r["ok"]]
    if usable:
        return dict(min(usable, key=lambda r: r["spread"]), status="contended")
    return dict(pool[0], status="oom", tokens_per_s=0.0, peak_gib=0.0)


def falling_throughput(rows: list[dict], tolerance: float = 0.9) -> list[tuple]:
    """Clean rows whose throughput drops as the microbatch grows."""
    by_run: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["clean"]:
            by_run.setdefault((row["preset"], row["tag"]), []).append(row)
    out = []
    for (preset, tag), group in sorted(by_run.items()):
        points = sorted(group, key=lambda r: r["per_micro"])
        for prev, nxt in zip(points, points[1:]):
            if nxt["tokens_per_s"] < prev["tokens_per_s"] * tolerance:
                out.append((preset, tag, prev, nxt))
    return out
