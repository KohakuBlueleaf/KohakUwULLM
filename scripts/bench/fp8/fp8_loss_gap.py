"""Read the per-arm CSVs an fp8 A/B wrote and say whether a gap is real.

Separate from ``fp8_training.py`` because that file runs the arms and is at its line
budget, and because this is the step you re-run: the arms cost GPU minutes each and the
table gets read many times over. Every number is a multiple of the ``bf16_ctrl`` arm's
own gap -- see :mod:`kohakuwullm.bench.analysis.loss_gap` for why an absolute loss difference is
not a result.

The arm that makes a sparse run interpretable is ``fp8_linears_only``: fp8 in the
``nn.Linear`` projections with the routed experts left bf16. At MoE-1B-A280M it read 1x
the floor while the full fp8 arm read 41x, which attributes the whole gap to the routed
experts rather than to the swap.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/bench/fp8/fp8_training.py \\
        --stage train --arm fp8_linears_only --preset MoE-1B-A280M --steps 400 \\
        --cache out/bench/train/fp8_training/cache8192 --out out/bench/train/fp8_moe_loss
    .venv/bin/python scripts/bench/fp8/fp8_loss_gap.py --dir out/bench/train/fp8_moe_loss
"""

import argparse
import json
import os

from kohakuwullm.bench.analysis.loss_gap import (
    final_window,
    format_gaps,
    load_for_comparison,
    windowed_gaps,
)

ARMS = ("bf16", "bf16_ctrl", "fp8_linears_only", "fp8_up", "fp8_rtn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="out/bench/train/fp8_moe_loss")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        help="arms to include; missing or unfinished ones are skipped",
    )
    args = parser.parse_args()

    curves = load_for_comparison(args.dir, args.arms)
    rows = windowed_gaps(curves, window=args.window, points=args.points)
    finals = final_window(curves, window=args.window)

    print(f"{args.dir}: {len(curves)} finished arm(s), {len(curves['bf16'])} steps\n")
    print(f"mean loss over the last {args.window} steps")
    for name, value in finals.items():
        print(f"  {name:20s} {value:.4f}")
    print(f"\ngap against bf16, {args.window}-step windows")
    print(format_gaps(rows))

    path = os.path.join(args.dir, "loss_gap.json")
    with open(path, "w") as handle:
        json.dump({"final_window": finals, "windows": rows}, handle, indent=2)
    print(f"\nwrote {path}")


main()
