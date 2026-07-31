"""Report the Kohaku 4-card sweep in planning units: B tokens/day and steps/day.

tok/s answers "is this kernel fast"; a run is planned in days and checkpointed in
steps, and the conversion is where an hour of arithmetic gets done by hand and gets
done wrong.

Row loading and the clean/contended/oom verdict come from
:mod:`kohakuwullm.bench.analysis.sweep` rather than being re-derived here -- a plotter with its
own idea of which rows to trust is a second definition of the rule, and the mistake is
a single comparison operator wide.

Usage::

    .venv/bin/python scripts/bench/e2e/kohaku_e2e_table.py --dir out/bench/train/kohaku_e2e
"""

import argparse
import json
import os

from kohakuwullm.bench import load_sweep, planning_units
from kohakuwullm.bench.model.ladder import ladder_census
from kohakuwullm.models.presets import KOHAKU_LADDER


def rows_in_ladder_order(directory: str, max_spread: float):
    rows, notes, devices = load_sweep(directory, max_spread=max_spread)
    order = {name: i for i, name in enumerate(KOHAKU_LADDER)}
    rows = [r for r in rows if r["ok"]]
    for row in rows:
        row.update(planning_units(row))
    rows.sort(key=lambda r: (order.get(r["preset"], 99), r["tag"], r["per_micro"]))
    return rows, notes, devices


def write_markdown(
    path: str,
    rows: list[dict],
    notes: list[str],
    devices: set,
    family: str = "",
) -> None:
    """The traceable artifact: one row per measurement, with its own caveats attached.

    Written to disk rather than only printed because this is the table a run gets
    planned from, and a number quoted in conversation cannot be re-checked later.
    """
    budgets = sorted({r["budget"] for r in rows})
    lines = [
        f"# Kohaku ladder: 4-card end-to-end throughput{family and ' -- ' + family}",
        "",
        f"Cards: {', '.join(sorted(devices)) or 'unrecorded'} (4x, all owned by the sweep)",
        f"Batch: {', '.join(f'{b:,}' for b in budgets)} tokens/step",
        "Protocol: 32 steps per row, the last 16 timed, **median** of those; "
        "steady state verified by the caching allocator carving no new segment "
        "inside the measured window.",
        "",
        "`state` is that check: `ok` steady, `ALLOC` allocated mid-window, `?` predates "
        "the check. Rows marked either way are not sustained rates.",
        "",
        "`compute-active P` excludes the embedding *table*: a token reads one row of it, "
        "so the other 65535 do no arithmetic. The LM head stays counted -- it is a full "
        "GEMM. This removes 25% at Kohaku-200M and 5% at MoE-8B, an error that grows "
        "toward the small rungs and would flatten any scaling fit taken over it.",
        "",
        "| preset | arm | micro | compute-active P | total P | tok/s | B tok/day | "
        "tok/param/day | steps/day | steps/hr | GiB/rank | spread | state |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    census = {c.name: c for c in ladder_census()}
    for r in rows:
        rung = census.get(r["preset"])
        active = f"{rung.compute_active / 1e6:.0f}M" if rung else "?"
        total = f"{rung.total / 1e6:.0f}M" if rung else "?"
        # Per active parameter, not per stored one: active is what sets the arithmetic,
        # so this is the axis on which a dense and a sparse rung are comparable.
        per_param = (
            f"{r['tokens_per_s'] * 86400 / rung.compute_active:,.1f}" if rung else "?"
        )
        spread = "?" if r["spread"] != r["spread"] else f"{r['spread']:.1%}"
        # One verdict column, covering both checks: a reader scanning for `ok` must not
        # find it next to a 10% spread just because the allocator behaved.
        if r["steady"] is False:
            state = "**ALLOC**"
        elif not r["clean"]:
            state = "**SPREAD**"
        elif r["steady"] is None:
            state = "?"
        else:
            state = "ok"
        peak = "?" if r["peak_gib"] != r["peak_gib"] else f"{r['peak_gib']:.1f}"
        lines.append(
            f"| {r['preset']} | {r['tag']} | {r['per_micro']} | {active} | {total} | "
            f"{r['tokens_per_s']:,.0f} | **{r['b_tok_day']:.2f}** | {per_param} | "
            f"{r['steps_day']:,.0f} | {r['steps_hour']:,.0f} | {peak} | "
            f"{spread} | {state} |"
        )

    dirty = [r for r in rows if not r["clean"] or r["steady"] is False]
    if dirty:
        lines += ["", "## Not trustworthy as a sustained rate", ""]
        for r in dirty:
            why = [] if r["clean"] else [f"spread {spread_str(r)}"]
            if r["steady"] is False:
                why.append("allocated mid-window")
            lines.append(
                f"- `{r['preset']} {r['tag']} micro {r['per_micro']}`: {', '.join(why)}"
            )
    if notes:
        lines += ["", "## Loader notes", ""] + [f"- {n}" for n in notes]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def spread_str(row: dict) -> str:
    return "?" if row["spread"] != row["spread"] else f"{row['spread']:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/bench/train/kohaku_e2e")
    ap.add_argument("--max-spread", type=float, default=0.02)
    ap.add_argument("--json", default=None, help="also write the table as JSON")
    ap.add_argument(
        "--markdown",
        default=None,
        help="write a markdown table here (default: <dir>/RESULTS.md)",
    )
    args = ap.parse_args()

    rows, notes, devices = rows_in_ladder_order(args.dir, args.max_spread)
    if not rows:
        print(f"no completed rows in {args.dir}")
        for note in notes:
            print(f"  note: {note}")
        return

    head = (
        f"{'preset':16s} {'strategy':9s} {'micro':>6s} {'tok/s':>9s} "
        f"{'B tok/day':>10s} {'steps/day':>10s} {'steps/hr':>9s} "
        f"{'GiB':>6s} {'spread':>7s} {'state':>6s}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        # An unknown spread or an unknown steadiness is reported as unknown, never as
        # a pass: that distinction is the whole point of recording them.
        spread = "-" if r["spread"] != r["spread"] else f"{r['spread']:.1%}"
        state = {True: "ok", False: "ALLOC", None: "?"}[r["steady"]]
        peak = "-" if r["peak_gib"] != r["peak_gib"] else f"{r['peak_gib']:.1f}"
        print(
            f"{r['preset']:16s} {str(r['strategy']):9s} {r['per_micro']:>6d} "
            f"{r['tokens_per_s']:>9,.0f} {r['b_tok_day']:>10.2f} "
            f"{r['steps_day']:>10,.0f} {r['steps_hour']:>9,.0f} "
            f"{peak:>6s} {spread:>7s} {state:>6s}"
        )

    budgets = sorted({r["budget"] for r in rows})
    print(f"\nbudget(s): {', '.join(f'{b:,}' for b in budgets)} tokens/step")
    print(
        f"dtype: {', '.join(sorted(str(r['dtype']) for r in rows))[:60]}   "
        f"optimizer: {', '.join(sorted({r['optimizer'] for r in rows}))}"
    )
    if devices:
        print(f"device(s): {', '.join(sorted(devices))}")
    dirty = [r for r in rows if not r["clean"] or r["steady"] is False]
    if dirty:
        print(f"\n{len(dirty)} row(s) not trustworthy as a sustained rate:")
        for r in dirty:
            why = [] if r["clean"] else [f"spread {r['spread']:.1%}"]
            if r["steady"] is False:
                why.append("allocated mid-window")
            print(
                f"  {r['preset']:16s} {r['tag']:22s} {r['per_micro']:>6d}  {', '.join(why)}"
            )
    for note in notes:
        print(f"note: {note}")

    # Two files, not one with a divider: dense and sparse rungs scale differently
    # enough that reading them in one table invites comparing rows that are not
    # comparable, and each file is what gets pasted into a planning discussion.
    census = {c.name: c for c in ladder_census()}
    families = {
        "dense": [r for r in rows if not census[r["preset"]].sparse],
        "MoE": [r for r in rows if census[r["preset"]].sparse],
    }
    base = args.markdown or os.path.join(args.dir, "RESULTS")
    base = base[: -len(".md")] if base.endswith(".md") else base
    print()
    for family, subset in families.items():
        if not subset:
            continue
        path = f"{base}_{family}.md"
        write_markdown(path, subset, notes, devices, family=f"{family} rungs")
        print(f"wrote {path} ({len(subset)} rows)")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
