"""Plot pipelined decode against the single-card roofline.

    .venv/bin/python scripts/bench/gen/pipeline_plot.py

Reads the JSONs `scripts/dist/pp_generate_bench.py` writes. See
docs/guides/generation.md.
"""

import argparse
import glob
import json
import os

from kohakuwullm.bench import Palette, bar_labels, finish_axis, new_figure, save_figure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/bench/gen/pipeline")
    ap.add_argument("--out", default="out/bench/gen/pipeline/pipeline_decode.png")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src, "*.json")))
    if not paths:
        raise SystemExit(f"no benchmark JSON under {args.src}")

    palette = Palette()
    fig, axes = new_figure(1, 2, figsize=(12.0, 4.6))
    throughput, efficiency = axes

    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        label = f"{data['preset']} pp{data['stages']}"
        rows = data["rows_measured"]
        chunks = [str(r["microbatches"]) for r in rows]
        color = palette.color(label)

        bars = throughput.bar(chunks, [r["tokens_per_s"] for r in rows], color=color)
        bar_labels(throughput, bars)
        throughput.axhline(
            data["roofline_tokens_per_s"],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label=f"roofline {data['roofline_tokens_per_s']:.0f} tok/s",
        )
        pct = [100 * r["frac_peak"] for r in rows]
        marks = efficiency.bar(chunks, pct, color=color, label=label)
        bar_labels(efficiency, marks, fmt="{:.1f}%")

    throughput.set_yscale("log")
    throughput.legend(loc="upper right")
    finish_axis(
        throughput,
        "microbatches",
        "tokens / s",
        "Pipelined decode: splitting the batch only costs",
        si_x=False,
    )
    finish_axis(
        efficiency,
        "microbatches",
        "% of 1791 GB/s",
        "Bandwidth reached (single card compiled: ~23%)",
        si_x=False,
    )
    save_figure(fig, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
