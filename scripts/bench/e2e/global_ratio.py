"""How much global attention can a mixed local/global stack afford?

Sweeps the fraction of layers using full causal attention against sliding-window
attention, reporting throughput, KV-cache footprint and the four standard
metrics. The point is to pick the local/global mix from measurement rather than
copying a ratio from another model whose context length and head count differ.

The answer is **not** a single ratio, which is why this sweeps a matrix.
Attention is a larger share of a small model's cost than a large one's (the
per-token GEMMs grow with ``dim`` while the score matrix grows with ``dim`` too
but the *quadratic* term grows with sequence length), and windowing only saves
anything once the window is well below the context. At seq 2048 with a 1024
window there is at most 2x to save on a term that is already a minority of the
step; at 4096 with the same window there is 4x to save on a term that is a
larger minority. A ratio that is free at one point on that grid need not be free
at another.

Global-layer placement goes through ``window_pattern``, not
``global_layer_every``: the latter can only express ratios of the form
``depth / n``, which leaves everything between 0.5 and 1.0 unreachable. An
explicit pattern spreads exactly ``n_global`` global layers over the depth. The
last layer is always global (it feeds the head), so the minimum is 1/depth.

Usage:
    .venv/bin/python scripts/bench/e2e/global_ratio.py --out out/bench/train/global_ratio
    .venv/bin/python scripts/bench/e2e/global_ratio.py --preset Nano-1B --seq-len 4096
"""

import argparse
import json
import math
import os

import torch

from kohakuwullm.bench import (
    Palette,
    device_name,
    finish_axis,
    module_metrics,
    new_figure,
    save_figure,
)
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.components.seqinfo import SeqInfo

BYTES_PER_KV = 2  # bf16 cache


def global_pattern(depth: int, n_global: int) -> list[int | None]:
    """Indices of the ``n_global`` global layers, spread evenly, last one always global.

    Evenly spread rather than "every k-th from the top": a stack whose global
    layers are all bunched at one end has a different receptive-field profile
    from one that interleaves them, and the interleaved arrangement is what
    Gemma 3 and OLMo 3 actually ship.
    """
    n_global = max(1, min(n_global, depth))
    if n_global >= depth:
        return list(range(depth))
    # Anchor on the last layer and step backwards by depth/n_global.
    stride = depth / n_global
    return sorted({depth - 1 - round(i * stride) for i in range(n_global)})


def build(preset: str, vocab: int, n_global: int, window: int, seq_len: int, device):
    """A backbone with exactly ``n_global`` full-causal layers."""
    cfg = get_preset(preset, vocab_size=vocab, tie_embeddings=False)
    globals_at = set(global_pattern(cfg.depth, n_global))
    cfg.window_pattern = [None if i in globals_at else window for i in range(cfg.depth)]
    cfg.max_position = max(cfg.max_position, seq_len)
    model = LMBackbone(cfg).to(device=device, dtype=torch.float32)
    return cfg, model


def kv_bytes_per_token(cfg, seq_len: int) -> float:
    """KV cache bytes per token of context, charging each layer only its window.

    A windowed layer never holds more than ``window`` positions, so its share is
    ``window / seq_len`` of a global layer's. This is the number that decides
    whether a long-context deployment fits, and it is the half of the trade-off
    throughput alone cannot show.
    """
    per_layer = 2 * cfg.kv_heads * cfg.head_dim * BYTES_PER_KV
    return sum(
        per_layer * (seq_len if w is None else min(w, seq_len)) / seq_len
        for w in cfg.windows
    )


def attention_pairs(seq_len: int, window: int | None) -> int:
    """Number of (query, key) pairs one sequence attends over.

    Full causal is ``S(S+1)/2``. A window of ``W`` caps each query at ``W`` keys,
    so the early ``W`` queries are still triangular and the rest are constant --
    this is the entire FLOP saving windowing buys, and it is why the saving
    vanishes as ``W`` approaches ``S``.
    """
    if window is None or window >= seq_len:
        return seq_len * (seq_len + 1) // 2
    return window * seq_len - window * (window - 1) // 2


def step_flops(cfg, model, tokens: int, n_seq: int, seq_len: int) -> float:
    """Forward+backward FLOPs of one step, attention counted per layer's window.

    Backward is charged 2x forward, the standard convention. The linear part
    comes from the model's own active-parameter count rather than a re-derived
    formula, so it stays correct for MoE (where a token touches ``top_k`` of the
    experts) without this file having to know how routing works. The embedding
    *lookup* is subtracted because it is a gather, not a GEMM; the head is not,
    because it is.
    """
    summary = model.param_summary()
    gemm_params = summary["active"] - cfg.vocab_size * cfg.dim
    linear = 2 * tokens * gemm_params
    # Q@K and A@V each cost 2 flops per (pair, head, head-dim); GQA does not
    # reduce this, since the shared K/V are broadcast back to all query heads.
    per_pair = 4 * cfg.heads * cfg.head_dim
    attn = sum(per_pair * attention_pairs(seq_len, w) * n_seq for w in cfg.windows)
    return 3.0 * (linear + attn)


def active_hidden(mlp) -> int:
    """Feed-forward width one token actually flows through.

    For an MoE layer ``mlp.hidden`` is *one routed expert*, so a token's real
    width is ``top_k`` of those plus the always-on shared experts.
    """
    if isinstance(mlp, MoEMLP):
        return mlp.hidden * (mlp.top_k + mlp.num_shared)
    return mlp.hidden


def step_bytes(cfg, model, tokens: int) -> float:
    """A *floor* on DRAM traffic for one step, in bytes.

    Counted: every parameter read once in the forward and once in the backward
    plus one gradient write (fp32 master weights), and every activation the
    backward needs written once and read once (bf16 under autocast). Not
    counted: the attention score matrix, which flash never materialises, and any
    recompute or allocator traffic. So this is a lower bound, and the effective
    GB/s derived from it is one too -- which is the safe direction for a
    "% of peak" figure.
    """
    param_traffic = 3 * model.param_summary()["total"] * 4
    qkv = (cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim
    # Per layer the backward needs: block input, q/k/v, attention output, the
    # post-attention residual, and the SwiGLU gate/up/product.
    per_token = sum(
        4 * cfg.dim + qkv + 3 * active_hidden(block.mlp) for block in model.blocks
    )
    activations = tokens * per_token * 2
    return param_traffic + 2 * activations


def run(args, device) -> list[dict]:
    rows = []
    n_seq = max(1, args.tokens // args.seq_len)
    tokens = n_seq * args.seq_len
    info = SeqInfo.from_lengths(
        torch.full((n_seq,), args.seq_len, dtype=torch.int32), device
    )
    ids = torch.randint(0, args.vocab, (tokens,), device=device)
    labels = ids.roll(-1)

    depth = get_preset(args.preset).depth
    counts = sorted({max(1, min(depth, round(depth * r))) for r in args.ratios})
    for n_global in counts:
        cfg, model = build(
            args.preset, args.vocab, n_global, args.window, args.seq_len, device
        )

        def step(model=model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.loss(ids, labels, info, reduction="sum")
            (loss / tokens).backward()
            model.zero_grad(set_to_none=True)

        flops = step_flops(cfg, model, tokens, n_seq, args.seq_len)
        # From the built config, not the request: rounding collisions in the
        # placement can drop a layer, and the label must match the model.
        built = sum(1 for w in cfg.windows if w is None)
        row = dict(
            preset=args.preset,
            seq_len=args.seq_len,
            window=args.window,
            tokens=tokens,
            n_global=built,
            depth=cfg.depth,
            ratio=built / cfg.depth,
            params=model.param_summary()["total"],
            active_params=model.param_summary()["active"],
            kv_bytes_per_token=kv_bytes_per_token(cfg, args.seq_len),
            model_tflop=flops / 1e12,
        )
        try:
            metrics = module_metrics(
                step,
                flops=flops,
                bytes_moved=step_bytes(cfg, model, tokens),
                warmup=args.warmup,
                iters=args.iters,
                tensor_cores=True,
            )
            row.update(metrics, ok=True, tokens_per_s=tokens / metrics["ms"] * 1e3)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # RuntimeError too: kernels below the allocator raise a plain one
            # carrying the same text. Anything else is a bug, not a data point.
            if not isinstance(exc, torch.OutOfMemoryError) and (
                "out of memory" not in str(exc)
            ):
                raise
            torch.cuda.empty_cache()
            row.update(
                ms=float("nan"),
                tflops=0.0,
                gbps=0.0,
                peak_gib=float("nan"),
                compute_ceiling_tflops=0.0,
                compute_pct=0.0,
                bandwidth_pct=0.0,
                ok=False,
                tokens_per_s=0.0,
            )
        rows.append(row)

        head = f"{built:>2}/{cfg.depth} global"
        if row["ok"]:
            print(
                f"  {head}: {row['ms']:8.2f} ms  {row['tokens_per_s']:9,.0f} tok/s  "
                f"{row['tflops']:6.1f} TF/s ({row['compute_pct']:4.1f}%)  "
                f"{row['gbps']:6.0f} GB/s ({row['bandwidth_pct']:4.1f}%)  "
                f"{row['peak_gib']:6.2f} GiB  "
                f"KV {row['kv_bytes_per_token'] / 1024:5.1f} KiB/tok",
                flush=True,
            )
        else:
            print(f"  {head}: OOM", flush=True)
        del model
        torch.cuda.empty_cache()
    return rows


def plot(rows, out_dir, args):
    """Throughput and KV footprint on one pair of axes, plus where the time goes."""
    pal = Palette()
    fig, axes = new_figure(1, 4, figsize=(24, 5.6))
    good = [r for r in rows if r["ok"]]
    if not good:
        raise RuntimeError("every configuration OOMed; nothing to plot")
    xs = [r["ratio"] for r in good]
    base = max(r["tokens_per_s"] for r in good)

    tput = axes[0].plot(
        xs,
        [r["tokens_per_s"] / 1e3 for r in good],
        marker="o",
        color=pal.color("tput"),
        label="throughput",
    )
    kv_axis = axes[0].twinx()
    kv = kv_axis.plot(
        xs,
        [r["kv_bytes_per_token"] / 1024 for r in good],
        marker="s",
        linestyle="--",
        color=pal.color("kv"),
        label="KV cache",
    )
    kv_axis.set_ylabel("KV cache KiB / token of context")
    kv_axis.grid(False)
    finish_axis(
        axes[0],
        "fraction of layers global",
        "throughput (k tokens/s)",
        "What global attention costs, and what it buys back",
        si_x=False,
    )
    axes[0].legend(tput + kv, [h.get_label() for h in tput + kv], fontsize=8)

    # A Pareto scatter rather than a second ratio plot: the reader arrives with
    # a KV budget, not with a ratio, and this is the axis that answers them.
    axes[1].plot(
        [r["kv_bytes_per_token"] / 1024 for r in good],
        [r["tokens_per_s"] / 1e3 for r in good],
        marker="o",
        color=pal.color("tput"),
    )
    for row in good:
        axes[1].annotate(
            f"{row['n_global']}/{row['depth']}",
            (row["kv_bytes_per_token"] / 1024, row["tokens_per_s"] / 1e3),
            fontsize=6.5,
            xytext=(4, 3),
            textcoords="offset points",
        )
    budget = base * (1 - args.tolerance)
    axes[1].axhline(
        budget / 1e3,
        color="#c0392b",
        linestyle="-.",
        linewidth=1.2,
        label=f"{args.tolerance:.0%} of the fastest config",
    )
    finish_axis(
        axes[1],
        "KV cache KiB / token of context",
        "throughput (k tokens/s)",
        "Pick a cache budget, read off the stack",
        si_x=False,
    )
    axes[1].legend(fontsize=8)

    ceiling = good[0]["compute_ceiling_tflops"]
    comp = axes[2].plot(
        xs,
        [r["tflops"] for r in good],
        marker="o",
        color=pal.color("tflops"),
        label="achieved TFLOP/s",
    )
    axes[2].axhline(ceiling, color="#999999", linestyle="--", linewidth=1.1)
    axes[2].annotate(
        f"MAMF {ceiling:.0f}T",
        (xs[0], ceiling),
        fontsize=7,
        va="top",
        xytext=(2, -3),
        textcoords="offset points",
    )
    axes[2].set_ylim(0, ceiling * 1.1)
    bw_axis = axes[2].twinx()
    bw = bw_axis.plot(
        xs,
        [r["gbps"] for r in good],
        marker="s",
        linestyle="--",
        color=pal.color("kv"),
        label="effective GB/s (floor)",
    )
    bw_axis.set_ylabel("effective GB/s")
    bw_axis.set_ylim(0, max(r["gbps"] for r in good) * 1.6)
    bw_axis.grid(False)
    finish_axis(
        axes[2],
        "fraction of layers global",
        "achieved TFLOP/s (window-aware count)",
        "A windowed layer does less work, not faster work",
        si_x=False,
    )
    axes[2].legend(
        comp + bw, [h.get_label() for h in comp + bw], fontsize=8, loc="lower right"
    )

    lat = axes[3].plot(
        xs,
        [r["ms"] for r in good],
        marker="o",
        color=pal.color("ms"),
        label="step (ms)",
    )
    mem_axis = axes[3].twinx()
    mem = mem_axis.plot(
        xs,
        [r["peak_gib"] for r in good],
        marker="s",
        linestyle="--",
        color=pal.color("mem"),
        label="peak memory (GiB)",
    )
    mem_axis.set_ylabel("peak memory (GiB)")
    mem_axis.set_ylim(0, max(r["peak_gib"] for r in good) * 1.35)
    mem_axis.grid(False)
    oom = [r for r in rows if not r["ok"]]
    finish_axis(
        axes[3],
        "fraction of layers global",
        "fwd+bwd latency (ms)",
        "Latency and memory"
        + (f" -- {len(oom)} configuration(s) OOMed" if oom else ""),
        si_x=False,
    )
    axes[3].legend(lat + mem, [h.get_label() for h in lat + mem], fontsize=8)

    fig.suptitle(
        f"Global attention ratio -- {device_name()}  |  {args.preset}, "
        f"window {args.window}, seq {args.seq_len}, {good[0]['tokens']} tokens/step",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "global_ratio.png"))


def load_matrix(root: str) -> list[dict]:
    """Every per-case result under ``root``, tagged with the directory it came from."""
    cases = []
    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        path = os.path.join(directory, "global_ratio.json")
        if os.path.isfile(path):
            with open(path) as handle:
                cases.append({**json.load(handle), "out": directory})
    return cases


def _case_label(case: dict) -> str:
    cfg = case["config"]
    return f"{cfg['preset']} s{cfg['seq_len']}"


def plot_matrix(cases, out_path, tolerance: float):
    """One figure per the whole grid: how the answer moves with scale and context.

    Throughput is normalised per case so curves of very different absolute
    speeds share an axis -- the question is what *fraction* of throughput a
    global layer costs, and that is the only form in which a 200M model at 2048
    and a 1B model at 4096 are comparable at all.
    """
    pal = Palette()
    fig, axes = new_figure(1, 3, figsize=(19.5, 5.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    summary = []
    for i, case in enumerate(cases):
        rows = [r for r in case["rows"] if r["ok"]]
        if not rows:
            continue
        label = _case_label(case)
        color = pal.color(label)
        marker = markers[i % len(markers)]
        best = max(r["tokens_per_s"] for r in rows)
        xs = [r["ratio"] for r in rows]
        axes[0].plot(
            xs,
            [100 * r["tokens_per_s"] / best for r in rows],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1].plot(
            xs,
            [r["kv_bytes_per_token"] / 1024 for r in rows],
            marker=marker,
            color=color,
            label=label,
        )
        pick = recommend(case["rows"], tolerance)
        summary.append((label, color, pick, rows))

    axes[0].axhline(
        100 * (1 - tolerance),
        color="#c0392b",
        linestyle="-.",
        linewidth=1.2,
        label=f"{tolerance:.0%} budget",
    )
    finish_axis(
        axes[0],
        "fraction of layers global",
        "% of that case's best throughput",
        "Cost of global attention, per scale and context",
        si_x=False,
    )
    axes[0].legend(fontsize=7, ncol=2)

    finish_axis(
        axes[1],
        "fraction of layers global",
        "KV cache KiB / token of context",
        "What windowing buys: cache footprint",
        si_x=False,
    )
    axes[1].legend(fontsize=7, ncol=2)

    ys = range(len(summary))
    # A config that OOMed gets a zero-length bar and a label saying so, rather
    # than being dropped: an absent row reads as "not measured", not "did not fit".
    costs = [
        0.0 if math.isnan(pick["all_global_cost"]) else 100 * pick["all_global_cost"]
        for _, _, pick, _ in summary
    ]
    bars = axes[2].barh(
        list(ys), costs, color=[c for _, c, _, _ in summary], height=0.6
    )
    for bar, (label, _, pick, _) in zip(bars, summary):
        cost = (
            "all-global OOM"
            if math.isnan(pick["all_global_cost"])
            else f"{bar.get_width():.1f}%"
        )
        axes[2].annotate(
            f"{cost}  -> recommend {pick['n_global']}/{pick['depth']}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    axes[2].axvline(
        100 * tolerance,
        color="#c0392b",
        linestyle="-.",
        linewidth=1.2,
        label=f"{tolerance:.0%} budget",
    )
    axes[2].set_yticks(list(ys))
    axes[2].set_yticklabels([label for label, _, _, _ in summary], fontsize=8)
    axes[2].set_xlabel("throughput lost by making every layer global (%)")
    axes[2].set_title("The decision")
    axes[2].set_xlim(
        left=min(0.0, min(costs) * 1.2), right=max(costs + [tolerance * 100]) * 2.4
    )
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Global attention ratio across scales -- {device_name()}",
        fontweight="bold",
    )
    save_figure(fig, out_path)
    return summary


def recommend(rows, tolerance: float) -> dict:
    """The largest global share whose throughput stays within ``tolerance`` of the best.

    Reported as a recommendation because more global layers is strictly better
    for quality at equal speed; the only reason to keep a layer windowed is the
    KV cache it saves, so the useful answer is the *most* global stack the
    throughput budget allows.
    """
    good = [r for r in rows if r["ok"]]
    if not good:
        return {}
    best = max(r["tokens_per_s"] for r in good)
    affordable = [r for r in good if r["tokens_per_s"] >= best * (1 - tolerance)]
    pick = max(affordable, key=lambda r: r["ratio"])
    cheapest = min(good, key=lambda r: r["ratio"])
    return {
        "n_global": pick["n_global"],
        "depth": pick["depth"],
        "ratio": pick["ratio"],
        "tokens_per_s": pick["tokens_per_s"],
        "best_tokens_per_s": best,
        "kv_kib_per_token": pick["kv_bytes_per_token"] / 1024,
        "kv_vs_min_layers": pick["kv_bytes_per_token"] / cheapest["kv_bytes_per_token"],
        "all_global_cost": (
            1 - next(r["tokens_per_s"] for r in good if r["ratio"] == 1.0) / best
            if any(r["ratio"] == 1.0 for r in good)
            else float("nan")
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/global_ratio")
    ap.add_argument("--preset", default="Nano-500M")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="throughput a recommendation may give up against the fastest config",
    )
    ap.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    )
    ap.add_argument(
        "--summarize",
        metavar="ROOT",
        default=None,
        help="skip measurement; draw the cross-scale figure from ROOT/*/global_ratio.json",
    )
    args = ap.parse_args()

    if args.summarize:
        cases = load_matrix(args.summarize)
        if not cases:
            raise RuntimeError(f"no per-case results under {args.summarize}")
        # Per-case figures are redrawn from their JSON rather than re-measured:
        # a plotting change must never cost an hour of GPU time to apply.
        for case in cases:
            case_args = argparse.Namespace(**case["config"])
            case_args.tolerance = args.tolerance
            plot(case["rows"], case["out"], case_args)
        out_path = os.path.join(args.summarize, "global_ratio_matrix.png")
        summary = plot_matrix(cases, out_path, args.tolerance)
        print(
            f"{'case':>26}  {'recommend':>10}  {'all-global':>10}  {'KV KiB/tok':>10}"
        )
        for label, _, pick, _ in summary:
            print(
                f"{label:>26}  {pick['n_global']:>4}/{pick['depth']:<5}  "
                f"{pick['all_global_cost']:>9.1%}  {pick['kv_kib_per_token']:>10.1f}"
            )
        print(f"wrote {out_path}")
        return

    os.makedirs(args.out, exist_ok=True)
    print(
        f"device: {device_name()}  |  {args.preset} seq {args.seq_len} "
        f"window {args.window} tokens {args.tokens}"
    )

    rows = run(args, torch.device("cuda"))
    pick = recommend(rows, args.tolerance)
    payload = dict(config=vars(args), rows=rows, recommendation=pick)
    with open(os.path.join(args.out, "global_ratio.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    plot(rows, args.out, args)

    if pick:
        print(
            f"\nrecommendation: {pick['n_global']}/{pick['depth']} global "
            f"({pick['ratio']:.0%}) -- {pick['tokens_per_s']:,.0f} tok/s, "
            f"within {args.tolerance:.0%} of the best "
            f"({pick['best_tokens_per_s']:,.0f}); KV "
            f"{pick['kv_kib_per_token']:.1f} KiB/token, "
            f"{pick['kv_vs_min_layers']:.2f}x the all-windowed cache"
        )
        print(f"all-global costs {pick['all_global_cost']:.1%} throughput")
    print(f"wrote {args.out}/global_ratio.png")


if __name__ == "__main__":
    main()
