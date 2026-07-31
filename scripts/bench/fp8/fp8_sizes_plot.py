"""Draw the MXFP8 size- and token-scaling result.

Split from the measurement for the reason the repo splits every bench: the runs cost
GPU time and a rejected figure should not cost them again.

The layout argues one claim in the order the evidence supports it. **What gates MXFP8
is device work per launch**, and model size and tokens/step are two ways of buying it:

* **step time against tokens** -- fp8's curve is *flat* to 8192 tokens, so below that it
  is paying launch cost and nothing else. This is the mechanism panel and it comes
  first because the other three are consequences of it.
* **speedup against tokens** -- where that plateau puts the crossover. The 1.0 line is
  the whole point: below ~16k tokens/step fp8 is a *slowdown* at this size.
* **speedup by preset** -- the size trend, with each preset's own matmul ceiling drawn
  beside it. The ceiling barely moves (1.202 -> 1.274) while the realised speedup moves
  from 0.63x to 1.25x, which is what shows the LM-head mechanism is not the driver.
* **ops per step** -- the launch count itself, the quantity the first panel is about
  and the one a fusion would move.

Usage::

    .venv/bin/python scripts/bench/fp8/fp8_sizes_plot.py --dir out/bench/train/fp8_sizes
"""

import argparse
import json
import os

import numpy as np

from kohakuwullm.bench import Palette, bar_labels, finish_axis, new_figure, save_figure

ARMS = ("bf16", "fp8_up")
LABELS = {"bf16": "bf16", "fp8_up": "MXFP8 round-up"}
SHORT = {"Nano-200M": "200M", "Nano-500M": "500M", "Nano-1B": "1B"}


def load(directory: str) -> tuple[dict[int, dict], dict | None]:
    """Size sweeps keyed by token count, plus the token sweep if it was run."""
    sizes: dict[int, dict] = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name, "fp8_sizes.json")
        if name.startswith("t") and os.path.exists(path):
            with open(path) as handle:
                report = json.load(handle)
            sizes[int(report["tokens"])] = report
    tokens_path = os.path.join(directory, "tokens", "token_sweep.json")
    token_sweep = None
    if os.path.exists(tokens_path):
        with open(tokens_path) as handle:
            token_sweep = json.load(handle)
    return sizes, token_sweep


def draw(sizes: dict[int, dict], token_sweep: dict | None, path: str) -> None:
    palette = Palette()
    for key in ("bf16", "fp8_up", "ceiling"):
        palette.color(key)
    fig, axes = new_figure(2, 2, figsize=(13, 9.2))

    if token_sweep:
        counts = sorted(int(k) for k in token_sweep["steady_ms"]["bf16"])
        for arm in ARMS:
            series = token_sweep["steady_ms"][arm]
            present = [t for t in counts if str(t) in series]
            axes[0].plot(
                present,
                [series[str(t)] for t in present],
                marker="o",
                color=palette.color(arm),
                linestyle="-" if arm == "bf16" else "--",
                label=LABELS[arm],
            )
        # The plateau is the finding, so it gets said on the panel rather than left to
        # the reader to notice from the flatness.
        flat = [t for t in counts if t <= 8192]
        if flat:
            axes[0].axvspan(
                min(flat),
                max(flat),
                color=palette.color("fp8_up"),
                alpha=0.10,
                label="MXFP8 dispatch-bound (flat in tokens)",
            )
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(counts)
        axes[0].set_xticklabels([f"{t // 1024}k" for t in counts])
        finish_axis(
            axes[0],
            "tokens per step",
            "steady-state step time (ms)",
            f"{token_sweep['preset']}: MXFP8 pays launch cost until the device catches up",
            si_x=False,
            legend=True,
        )

        ratios = {int(k): v for k, v in token_sweep["speedup"].items()}
        present = sorted(ratios)
        axes[1].plot(
            present,
            [ratios[t] for t in present],
            marker="o",
            color=palette.color("fp8_up"),
        )
        axes[1].axhline(1.0, color="#333333", linewidth=1.0, label="parity")
        low, high = token_sweep["crossover_bracket"]
        if low and high:
            axes[1].axvspan(
                low,
                high,
                color="#7F7F7F",
                alpha=0.18,
                label=f"crossover between {low // 1024}k and {high // 1024}k",
            )
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(present)
        axes[1].set_xticklabels([f"{t // 1024}k" for t in present])
        finish_axis(
            axes[1],
            "tokens per step",
            "bf16 step time / MXFP8 step time",
            "Below the crossover MXFP8 is a slowdown, not a small win",
            si_x=False,
            legend=True,
        )

    # Size trend. One group per preset, one bar per token count, ceiling as a marker so
    # a bar can be read against the ceiling it is a fraction of.
    if sizes:
        token_counts = sorted(sizes)
        presets = [row["preset"] for row in sizes[token_counts[0]]["sizes"]]
        width = 0.8 / max(len(token_counts), 1)
        for i, tokens in enumerate(token_counts):
            rows = {r["preset"]: r for r in sizes[tokens]["sizes"]}
            xs, ys = [], []
            for j, preset in enumerate(presets):
                row = rows.get(preset)
                if not row or row.get("speedup_steady") is None:
                    continue
                xs.append(j + (i - (len(token_counts) - 1) / 2) * width)
                ys.append(row["speedup_steady"])
            bars = axes[2].bar(
                xs,
                ys,
                width=width * 0.9,
                color=palette.color(f"t{tokens}"),
                label=f"{tokens // 1024}k tokens",
            )
            bar_labels(axes[2], bars, fmt="{:.3f}")
        ceilings = [
            r["accounting"]["matmul_speedup_ceiling"]
            for r in sizes[token_counts[0]]["sizes"]
        ]
        axes[2].plot(
            range(len(presets)),
            ceilings,
            marker="_",
            markersize=26,
            linestyle="none",
            color="#333333",
            label="matmul ceiling",
        )
        axes[2].axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
        axes[2].set_xticks(range(len(presets)))
        axes[2].set_xticklabels([SHORT.get(p, p) for p in presets])
        finish_axis(
            axes[2],
            "",
            "bf16 step time / MXFP8 step time",
            "The ceiling barely moves; the realised speedup moves a lot",
            si_x=False,
            legend=True,
        )

        # The sweep with the most rungs, not the largest token count: at 16k the 1B
        # rung OOMs, and 1B is the rung this panel most needs -- it is where the launch
        # count is amortised best and the speedup is highest.
        def complete(tokens: int) -> int:
            return sum(
                1
                for r in sizes[tokens]["sizes"]
                if r.get("bf16", {}).get("ops_per_step")
            )

        tokens = max(token_counts, key=complete)
        rows = [
            r for r in sizes[tokens]["sizes"] if r.get("bf16", {}).get("ops_per_step")
        ]
        idx = np.arange(len(rows))
        for offset, arm in zip((-0.2, 0.2), ARMS):
            bars = axes[3].bar(
                idx + offset,
                [r[arm]["ops_per_step"] for r in rows],
                width=0.38,
                color=palette.color(arm),
                label=LABELS[arm],
            )
            bar_labels(axes[3], bars, fmt="{:.0f}")
        axes[3].set_xticks(idx)
        axes[3].set_xticklabels([SHORT.get(r["preset"], r["preset"]) for r in rows])
        finish_axis(
            axes[3],
            "",
            "device ops issued per step",
            f"Launch count, the cost the first panel is about ({tokens // 1024}k tokens)",
            si_x=False,
            legend=True,
        )

    save_figure(fig, path)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/bench/train/fp8_sizes")
    args = parser.parse_args()
    sizes, token_sweep = load(args.dir)
    if not sizes and not token_sweep:
        raise SystemExit(f"no fp8_sizes.json or token_sweep.json under {args.dir}")
    draw(sizes, token_sweep, os.path.join(args.dir, "fp8_sizes.png"))


if __name__ == "__main__":
    main()
