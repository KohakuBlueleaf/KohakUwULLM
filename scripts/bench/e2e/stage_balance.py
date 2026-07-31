"""Per-stage runtime of a pipeline split, measured on one GPU.

Each stage is built exactly as ``plan_for`` would place it, then timed in
isolation for forward and backward on one microbatch. A 4-stage split is only
as fast as its slowest stage, so the spread reported here is the pipeline's
efficiency ceiling before any communication cost.

Usage:
    .venv/bin/python scripts/bench/e2e/stage_balance.py --preset Nano-1B
    .venv/bin/python scripts/bench/e2e/stage_balance.py --preset Nano-1B --micro-tokens 8192
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    bench_ms,
    device_name,
    make_info,
    new_figure,
    save_figure,
)
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training.optim.lowbit import cast_parameters_
from kohakuwullm.training.parallel.pipeline import LMStage, plan_for

PARAM_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def time_stage(backbone, plan, cfg, info, per_micro, device, args):
    stage = LMStage(backbone, plan).to(device=device, dtype=torch.float32)
    if args.param_dtype != "fp32":
        cast_parameters_(stage, PARAM_DTYPES[args.param_dtype])
    stage.set_seq_info([info])

    if plan.has_embed:
        inp = torch.randint(0, cfg.vocab_size, (per_micro,), device=device)
    else:
        inp = torch.randn(
            per_micro,
            cfg.dim,
            device=device,
            dtype=PARAM_DTYPES[args.param_dtype],
            requires_grad=True,
        )
    targets = torch.randint(0, cfg.vocab_size, (per_micro,), device=device)

    def fwd_only(stage=stage, info=info, inp=inp, targets=targets):
        stage.set_seq_info([info])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = stage(inp)
            if plan.has_head:
                out, _ = stage.loss(out, targets)
        return out

    def fwd_bwd(stage=stage):
        stage.zero_grad(set_to_none=True)
        out = fwd_only()
        (out if plan.has_head else out.float().sum()).backward()

    try:
        fwd_bwd()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fwd_ms = bench_ms(lambda: fwd_only(), warmup=args.warmup, iters=args.iters)
        both_ms = bench_ms(fwd_bwd, warmup=args.warmup, iters=args.iters)
        peak = torch.cuda.max_memory_allocated() / 2**30
        ok = True
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(
            exc
        ):
            raise
        fwd_ms = both_ms = peak = float("nan")
        ok = False
        torch.cuda.empty_cache()

    params = sum(p.numel() for p in stage.parameters())
    del stage
    torch.cuda.empty_cache()
    return dict(
        stage=plan.index,
        layers=f"{plan.start_layer}..{plan.end_layer}",
        num_layers=plan.num_layers,
        has_embed=plan.has_embed,
        has_head=plan.has_head,
        params_m=params / 1e6,
        predicted_cost=plan.cost,
        fwd_ms=fwd_ms,
        fwd_bwd_ms=both_ms,
        peak_gib=peak,
        ok=ok,
    )


def run(args, device):
    cfg = get_preset(args.preset, vocab_size=args.vocab, tie_embeddings=False)
    cfg.max_position = max(cfg.max_position, args.micro_tokens, args.len_max)
    backbone = LMBackbone(cfg)
    plans = plan_for(cfg, args.stages, seq_len=args.plan_seq_len or args.micro_tokens)
    info = make_info(args.micro_tokens, args.len_min, args.len_max, device, args.seed)

    rows = []
    for plan in plans:
        row = time_stage(backbone, plan, cfg, info, args.micro_tokens, device, args)
        rows.append(row)
        print(
            f"  stage {row['stage']} layers {row['layers']:>7} "
            f"({row['num_layers']:2d}){'  +embed' if row['has_embed'] else ''}"
            f"{'  +head' if row['has_head'] else '       '} "
            f"{row['params_m']:7.1f}M  fwd {row['fwd_ms']:7.2f}ms  "
            f"fwd+bwd {row['fwd_bwd_ms']:8.2f}ms  peak {row['peak_gib']:5.2f} GiB",
            flush=True,
        )
    return cfg, rows


def summarize(rows):
    good = [r for r in rows if r["ok"]]
    if not good:
        return {}
    times = [r["fwd_bwd_ms"] for r in good]
    slowest, mean = max(times), sum(times) / len(times)
    return dict(
        slowest_ms=slowest,
        mean_ms=mean,
        balance=mean / slowest,
        bound_stage=int(max(good, key=lambda r: r["fwd_bwd_ms"])["stage"]),
        ideal_speedup=sum(times) / slowest,
    )


def plot(rows, summary, out_dir, args):
    pal = Palette()
    good = [r for r in rows if r["ok"]]
    labels = [f"s{r['stage']}\n{r['layers']}" for r in good]
    fig, axes = new_figure(1, 3, figsize=(18, 5.2))

    fwd = [r["fwd_ms"] for r in good]
    bwd = [r["fwd_bwd_ms"] - r["fwd_ms"] for r in good]
    axes[0].bar(labels, fwd, color=pal.color("fwd"), label="forward")
    axes[0].bar(labels, bwd, bottom=fwd, color=pal.color("bwd"), label="backward")
    if summary:
        axes[0].axhline(
            summary["mean_ms"], color="#999999", linestyle="--", label="mean"
        )
    axes[0].legend(frameon=False)
    axes[0].set_ylabel("ms per microbatch")
    axes[0].set_title(f"Per-stage runtime (balance {summary.get('balance', 0):.2f})")
    axes[0].tick_params(axis="x", labelsize=8)

    pred = [r["predicted_cost"] for r in good]
    scale = sum(r["fwd_bwd_ms"] for r in good) / sum(pred) if sum(pred) else 0.0
    bars = axes[1].bar(
        labels, [p * scale for p in pred], color=pal.color("pred"), width=0.55
    )
    axes[1].plot(
        labels,
        [r["fwd_bwd_ms"] for r in good],
        marker="o",
        color=pal.color("meas"),
        label="measured",
    )
    bar_labels(axes[1], bars, "{:.0f}")
    axes[1].legend(frameon=False)
    axes[1].set_ylabel("ms per microbatch")
    axes[1].set_title("Cost model vs measurement")
    axes[1].tick_params(axis="x", labelsize=8)

    bars = axes[2].bar(
        labels, [r["peak_gib"] for r in good], color=pal.color("mem"), width=0.55
    )
    bar_labels(axes[2], bars, "{:.1f}")
    axes[2].set_ylabel("peak GiB")
    axes[2].set_title("Per-stage activation memory")
    axes[2].tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"Pipeline stage balance -- {device_name()}  |  {args.preset}, "
        f"{args.stages} stages, {args.micro_tokens} tok/microbatch, "
        f"docs {args.len_min}-{args.len_max}",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, f"stage_balance_{args.preset}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/stage_balance")
    ap.add_argument("--preset", default="Nano-1B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--len-min", type=int, default=512)
    ap.add_argument("--len-max", type=int, default=4096)
    ap.add_argument("--plan-seq-len", type=int, default=None)
    ap.add_argument("--param-dtype", default="fp32", choices=list(PARAM_DTYPES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    print(f"device: {device_name()}  |  {args.preset}, {args.stages} stages")
    cfg, rows = run(args, device)
    summary = summarize(rows)
    if summary:
        print(
            f"\nslowest stage {summary['bound_stage']} at "
            f"{summary['slowest_ms']:.2f}ms; balance {summary['balance']:.3f} "
            f"(1.0 is perfect); compute-only speedup ceiling "
            f"{summary['ideal_speedup']:.2f}x of {args.stages}x",
            flush=True,
        )

    path = os.path.join(args.out, f"stage_balance_{args.preset}.json")
    with open(path, "w") as handle:
        json.dump(
            dict(preset=args.preset, rows=rows, summary=summary), handle, indent=2
        )
    plot(rows, summary, args.out, args)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
