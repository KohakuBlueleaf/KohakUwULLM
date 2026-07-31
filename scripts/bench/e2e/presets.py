"""The Kohaku preset ladder, measured: does it hit its design targets?

Nine rungs, dense and sparse interleaved, whose effective capacities
``sqrt(active * total)`` were solved to sit on one smooth sequence. This script
does not trust that solve. Every count comes from a real ``LMBackbone`` built on
``torch.device("meta")`` -- exact shapes, no allocation, no GPU -- and is compared
against the target the ladder was designed to, because a closed form omits what it
did not model (the router matrix, the second embedding matrix, every norm) and
those omissions are what a figure would otherwise present as a fit.

Writes ``presets.json``; ``presets_plot.py`` draws the four figures from it, split
for the same reason as ``kernels.py`` / ``kernels_plot.py``.

Usage:
    .venv/bin/python scripts/bench/e2e/presets.py --out out/bench/model/presets
    .venv/bin/python scripts/bench/e2e/presets_plot.py --dir out/bench/model/presets
"""

import argparse
import json
import os

from kohakuwullm.bench.model.ladder import RungCensus, ladder_census
from kohakuwullm.models.presets import KOHAKU_LADDER

# What the solve predicted, per rung: (total M, active M, target capacity M).
# `active` is the solver's convention -- body parameters only, no embedding, no
# head, no router -- which is why `RungCensus` reports that quantity separately
# from the repo-wide `count_active_parameters`. Kept here rather than in the
# presets so a preset edit cannot quietly move the number it is checked against,
# and keyed by name rather than positional, so `--presets` with a subset still
# compares each rung against its own target.
SOLVED_M: dict[str, tuple[float, float, int]] = {
    "Kohaku-200M": (204.3, 204.3, 200),
    "Kohaku-MoE-1B": (990.0, 146.2, 350),
    "Kohaku-500M": (546.2, 546.2, 500),
    "Kohaku-MoE-2B": (1951.6, 292.8, 700),
    "Kohaku-1B": (981.5, 981.5, 1000),
    "Kohaku-MoE-3B": (2905.2, 480.9, 1250),
    "Kohaku-1.5B": (1514.1, 1514.1, 1500),
    "Kohaku-MoE-5B": (4931.6, 772.7, 1850),
    "Kohaku-MoE-8B": (7712.6, 1163.2, 2800),
}

# Kohaku-MoE-8B routes 16 of 128 experts where every other sparse rung routes 8
# of 64, so its expert width is 0.25*dim rather than 0.5*dim. Sparsity is
# unchanged. Marked in every rung label so a reader does not take it for an error.
GRANULARITY_BREAK = "Kohaku-MoE-8B"


def label_for(name: str) -> str:
    """Short display name. Every rung is a Kohaku, so the prefix is dead width.

    Computed here rather than in the plotter: which rung carries the asterisk is a
    fact about the ladder, and the JSON is the contract between the two halves.
    """
    label = name.replace("Kohaku-", "")
    return f"{label} *" if name == GRANULARITY_BREAK else label


def as_records(rows: list[RungCensus]) -> list[dict]:
    """The table view. Every number in every figure is reachable from here."""
    records = []
    for i, rung in enumerate(rows):
        design_active = rung.total if not rung.sparse else rung.routed_active
        total_m, active_m, target = SOLVED_M[rung.name]
        records.append(
            dict(
                name=rung.name,
                label=label_for(rung.name),
                ladder_index=i,
                sparse=rung.sparse,
                dim=rung.dim,
                depth=rung.depth,
                heads=rung.heads,
                kv_heads=rung.kv_heads,
                kv_out=rung.kv_out,
                gqa_ratio=rung.gqa_ratio,
                mlp_hidden=rung.mlp_hidden,
                num_experts=rung.num_experts,
                top_k=rung.top_k,
                kappa=rung.kappa,
                expert_hidden=rung.expert_hidden,
                expert_hidden_over_dim=(
                    rung.expert_hidden / rung.dim if rung.sparse else None
                ),
                num_shared=rung.num_shared,
                total=rung.total,
                active_design=design_active,
                active_full=rung.active,
                embedding_and_head=rung.embedding,
                router=rung.router,
                total_over_active=rung.total / design_active,
                embed_share=rung.embedding / rung.total,
                capacity_m=rung.capacity / 1e6,
                capacity_full_m=rung.capacity_full / 1e6,
                target_capacity_m=target,
                capacity_vs_target_pct=100 * (rung.capacity / 1e6 / target - 1),
                expected_total_m=total_m,
                expected_active_m=active_m,
                total_vs_expected_pct=100 * (rung.total / 1e6 / total_m - 1),
                active_vs_expected_pct=100 * (design_active / 1e6 / active_m - 1),
            )
        )
    return records


def print_table(records: list[dict]) -> None:
    header = (
        f"{'rung':>16} {'total M':>9} {'vs solved':>10} {'active M':>9} "
        f"{'vs solved':>10} {'capacity M':>11} {'target':>7} {'vs target':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in records:
        print(
            f"{row['name']:>16} {row['total'] / 1e6:9.2f} "
            f"{row['total_vs_expected_pct']:+9.3f}% {row['active_design'] / 1e6:9.2f} "
            f"{row['active_vs_expected_pct']:+9.3f}% {row['capacity_m']:11.1f} "
            f"{row['target_capacity_m']:7d} {row['capacity_vs_target_pct']:+9.2f}%"
        )
    worst_total = max(abs(r["total_vs_expected_pct"]) for r in records)
    worst_active = max(abs(r["active_vs_expected_pct"]) for r in records)
    print(
        f"\nworst disagreement with the solved table: total {worst_total:.3f}%, "
        f"active {worst_active:.3f}%"
    )
    off = {r["name"] for r in records if abs(r["total_vs_expected_pct"]) > 1.0}
    off |= {r["name"] for r in records if abs(r["active_vs_expected_pct"]) > 1.0}
    print("rows off by more than 1%: " + (", ".join(sorted(off)) or "none"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/model/presets")
    ap.add_argument("--presets", nargs="+", default=list(KOHAKU_LADDER))
    args = ap.parse_args()
    # Every figure reports a rung against its solved target, so a preset with no
    # entry in that table has nothing to be plotted against.
    unsolved = [name for name in args.presets if name not in SOLVED_M]
    if unsolved:
        ap.error(f"no solved target for {unsolved}; known rungs: {sorted(SOLVED_M)}")
    os.makedirs(args.out, exist_ok=True)

    records = as_records(ladder_census(tuple(args.presets)))
    print_table(records)
    path = os.path.join(args.out, "presets.json")
    with open(path, "w") as handle:
        json.dump(
            dict(granularity_break=GRANULARITY_BREAK, rungs=records), handle, indent=2
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
