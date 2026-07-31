"""End-to-end training throughput: does pipelining actually help, and when?

Everything is reported in **tokens/s** -- the only metric that compares across
model sizes, batch shapes and parallelism strategies. TFLOP/s flatters a model
that does more work per token; step time flatters a small batch. Tokens/s is
what decides when the run finishes.

Four strategies, each measured on a real forward+backward+step over packed
varlen batches:

* ``single``   -- one GPU, the baseline.
* ``ddp``      -- N GPUs, full replica each, gradients all-reduced.
* ``pipeline`` -- N GPUs, cost-balanced stage split, 1F1B schedule.
* ``pp+ckpt``  -- pipeline with activation checkpointing.

The interesting question is not "which is fastest" (DDP usually is, when it
fits) but **where the crossover is**. DDP needs the whole model plus its
optimizer state on one card; at ~16 bytes/parameter a 3B model needs ~48 GB and
a 5090 has 32. So the honest comparison is: DDP wins until it OOMs, and the
plot's job is to show what pipelining costs when DDP still fits, and that it
runs at all where DDP does not.

Launch with torchrun for the multi-GPU strategies:

    torchrun --standalone --nproc_per_node=4 scripts/bench/_archive/e2e.py \\
        --out out/bench/train/e2e --presets Nano-500M MoE-3B-A500M

Single-process (runs the `single` strategy only, plus the memory model):

    .venv/bin/python scripts/bench/_archive/e2e.py --out out/bench/train/e2e
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    device_name,
    new_figure,
    save_figure,
)
from kohakuwullm.data.packing import IGNORE_INDEX
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.training.parallel.pipeline import (
    plan_for,
)
from kohakuwupipe import describe

BYTES_PER_PARAM_ADAMW = 4 + 4 + 4 + 2  # fp32 master + 2 moments + bf16 copy


def env_int(name, default=0):
    return int(os.environ.get(name, default))


def make_batch(tokens: int, mean_len: int, vocab: int, device, seed: int = 0):
    """A packed varlen batch with a realistic lognormal length spread."""
    generator = torch.Generator().manual_seed(seed)
    lengths = []
    total = 0
    while total < tokens:
        remaining = tokens - total
        draw = int(mean_len * torch.randn(1, generator=generator).mul(0.6).exp().item())
        # `remaining` clamps last, not first: a bare `max(16, ...)` overshoots
        # `tokens` when fewer than 16 remain and leaves `_pad_to` a negative fill.
        draw = min(max(16, min(draw, 4096)), remaining)
        lengths.append(draw)
        total += draw
    lengths_t = torch.tensor(lengths, dtype=torch.int32)
    seq_info = SeqInfo.from_lengths(lengths_t, device)
    ids = torch.randint(0, vocab, (total,), device=device)
    labels = ids.roll(-1)
    labels[seq_info.cu_seqlens[1:].long() - 1] = IGNORE_INDEX
    return ids, labels, seq_info, total


def _pad_to(batch, target: int, vocab: int, device):
    """Pad a packed batch up to exactly `target` tokens with one masked filler doc."""
    ids, labels, seq_info, total = batch
    if total == target:
        return batch
    fill = target - total
    ids = torch.cat([ids, torch.zeros(fill, dtype=ids.dtype, device=device)])
    labels = torch.cat(
        [labels, torch.full((fill,), IGNORE_INDEX, dtype=labels.dtype, device=device)]
    )
    lengths = torch.cat(
        [seq_info.seqlens.cpu(), torch.tensor([fill], dtype=torch.int32)]
    )
    return ids, labels, SeqInfo.from_lengths(lengths, device), target


def estimate_memory(config) -> dict:
    """Static memory model: what DDP would need per card, before activations."""
    model = LMBackbone(config)
    params = sum(p.numel() for p in model.parameters())
    del model
    return {
        "params": params,
        "ddp_state_gib": params * BYTES_PER_PARAM_ADAMW / 2**30,
    }


def run_single(config, args, device, grad_ckpt=False):
    config.grad_ckpt = grad_ckpt
    model = LMBackbone(config).to(device=device, dtype=torch.float32)
    opt = build_optimizer(model, lr=1e-4)
    ids, labels, seq_info, total = make_batch(
        args.tokens_per_step, args.mean_len, config.vocab_size, device
    )

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = model.loss(ids, labels, seq_info, reduction="sum")
        (loss / total).backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    model = opt = None
    torch.cuda.empty_cache()
    return {"tokens_per_s": total * args.iters / elapsed, "peak_gib": peak}


def run_ddp(config, args, device, rank, world, grad_ckpt=False):
    from torch.nn.parallel import DistributedDataParallel

    config.grad_ckpt = grad_ckpt
    model = LMBackbone(config).to(device=device, dtype=torch.float32)
    ddp = DistributedDataParallel(
        model,
        device_ids=[device.index],
        find_unused_parameters=config.moe_every > 0,
        gradient_as_bucket_view=True,
    )
    opt = build_optimizer(ddp, lr=1e-4)
    # Each rank gets its own shard of the global batch, as in real training.
    ids, labels, seq_info, total = make_batch(
        args.tokens_per_step, args.mean_len, config.vocab_size, device, seed=rank
    )

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = ddp.module.loss(ids, labels, seq_info, reduction="sum")
        (loss / total).backward()
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    ddp = model = opt = None
    torch.cuda.empty_cache()
    # Global throughput: every rank processed its own shard.
    return {"tokens_per_s": total * world * args.iters / elapsed, "peak_gib": peak}


def run_pipeline(config, args, device, rank, world, grad_ckpt=False):
    """Pipeline over cost-balanced stages, on ``torch.distributed.pipelining``.

    Microbatches are built here rather than by the runtime's tensor chunking:
    chunking a packed batch along the token axis would slice through documents.
    Every microbatch carries the same padded token count, which is also what
    ``PipelineStage``'s fixed boundary-activation shape requires.
    """
    from kohakuwullm.training.parallel.pipeline_lightning import (
        build_schedule,
        build_stage,
        run_step,
    )

    config.grad_ckpt = grad_ckpt
    plans = plan_for(config, world, seq_len=args.mean_len)
    if rank == 0:
        print(describe(plans), flush=True)

    n_micro = args.microbatches
    per_micro = args.tokens_per_step // n_micro
    stage_module, stage, _, _ = build_stage(
        config, rank, world, device, per_micro, seq_len=per_micro
    )
    opt = build_optimizer(stage_module, lr=1e-4)
    micro = [
        _pad_to(
            make_batch(
                per_micro, args.mean_len, config.vocab_size, device, seed=100 + i
            ),
            per_micro,
            config.vocab_size,
            device,
        )
        for i in range(n_micro)
    ]
    total = sum(m[3] for m in micro)
    inputs = [m[0] for m in micro]
    targets = [m[1] for m in micro]
    seq_infos = [m[2] for m in micro]

    def loss_fn(hidden, target):
        loss, _ = stage_module.loss(hidden, target)
        return loss / total

    schedule = build_schedule(stage, n_micro, loss_fn=loss_fn, kind=args.pp_schedule)

    def step():
        opt.zero_grad(set_to_none=True)
        stage_module.set_seq_info(seq_infos)
        run_step(
            schedule,
            rank,
            world,
            tokens=torch.cat(inputs),
            targets=torch.cat(targets),
        )
        opt.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 2**30
    stage_module = stage = opt = None
    torch.cuda.empty_cache()
    # One pipeline processes one global batch per step, not one per rank.
    return {"tokens_per_s": total * args.iters / elapsed, "peak_gib": peak}


def plot(rows, out_dir, args):
    pal = Palette()
    fig, axes = new_figure(1, 3, figsize=(18, 5.2))
    presets = sorted({r["preset"] for r in rows}, key=lambda p: rows[0]["preset"] != p)
    strategies = [
        s
        for s in ["single", "ddp", "ddp+ckpt", "pipeline", "pp+ckpt"]
        if any(r["strategy"] == s for r in rows)
    ]

    width = 0.8 / len(strategies)
    for i, strategy in enumerate(strategies):
        vals, mems = [], []
        for preset in presets:
            row = next(
                (
                    r
                    for r in rows
                    if r["preset"] == preset and r["strategy"] == strategy
                ),
                None,
            )
            vals.append(row["tokens_per_s"] / 1e3 if row and row.get("ok") else 0)
            mems.append(row["peak_gib"] if row and row.get("ok") else 0)
        xs = [j + (i - (len(strategies) - 1) / 2) * width for j in range(len(presets))]
        bars = axes[0].bar(
            xs, vals, width=width, color=pal.color(strategy), label=strategy
        )
        bar_labels(axes[0], bars, "{:.0f}")
        axes[1].bar(xs, mems, width=width, color=pal.color(strategy), label=strategy)

    for ax, ylabel, title in (
        (axes[0], "throughput (k tokens/s)", "End-to-end training throughput"),
        (axes[1], "peak memory per GPU (GiB)", "Peak memory"),
    ):
        ax.set_xticks(range(len(presets)))
        ax.set_xticklabels(presets, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    # The card's capacity is the line every bar is really competing against.
    axes[1].axhline(args.gpu_gib, color="#D55E00", linestyle="--", linewidth=1.2)
    axes[1].annotate(
        f"{args.gpu_gib:.0f} GiB card",
        (0, args.gpu_gib),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color="#D55E00",
    )

    # Panel 3: the static memory model -- why pipelining is not optional at 3B.
    model_rows = [r for r in rows if r["strategy"] == "memory_model"]
    if model_rows:
        names = [r["preset"] for r in model_rows]
        ddp_state = [r["ddp_state_gib"] for r in model_rows]
        pp_state = [r["ddp_state_gib"] / max(args.world, 1) for r in model_rows]
        xs = range(len(names))
        axes[2].bar(
            [x - 0.2 for x in xs],
            ddp_state,
            width=0.4,
            color=pal.color("ddp"),
            label="DDP (full replica)",
        )
        axes[2].bar(
            [x + 0.2 for x in xs],
            pp_state,
            width=0.4,
            color=pal.color("pipeline"),
            label=f"pipeline / {args.world} stages",
        )
        axes[2].axhline(args.gpu_gib, color="#D55E00", linestyle="--", linewidth=1.2)
        axes[2].set_xticks(list(xs))
        axes[2].set_xticklabels(names, fontsize=8)
        axes[2].set_ylabel("optimizer + weight state (GiB)")
        axes[2].set_title("Why pipelining: state per GPU, before activations")
        axes[2].legend(fontsize=8)

    fig.suptitle(
        f"End-to-end training -- {device_name()} x{args.world}  |  "
        f"{args.tokens_per_step} tokens/step, bf16 autocast",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "e2e.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/e2e")
    ap.add_argument(
        "--presets", nargs="+", default=["Nano-200M", "Nano-500M", "MoE-1B-A200M"]
    )
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--tokens-per-step", type=int, default=16384)
    ap.add_argument("--mean-len", type=int, default=256)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--pp-schedule", default="1f1b", choices=["1f1b", "gpipe"])
    ap.add_argument(
        "--run-id",
        default=None,
        help="single (preset, strategy) run; results append to <out>/parts/. "
        "Each combination gets its own process because mixing DDP's "
        "collective communicators with the pipeline's lazily-created P2P "
        "communicators in one process throws an internal NCCL error, and "
        "because a shared process carries allocator fragmentation from the "
        "previous strategy into the next one's peak-memory number.",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--gpu-gib", type=float, default=31.4)
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=["ddp", "pipeline", "pp+ckpt"],
        help="multi-GPU strategies to measure; drop 'pipeline' to skip it",
    )
    args = ap.parse_args()

    rank = env_int("RANK", 0)
    world = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    args.world = world
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.makedirs(args.out, exist_ok=True)

    # One process per (preset, strategy): see --run-id.
    if args.run_id:
        preset, strategy = args.run_id.split(":")
        presets = [preset]
        wanted = [strategy]
    else:
        presets = args.presets
        wanted = args.strategies

    rows = []
    for preset in presets:
        config = get_preset(preset, vocab_size=args.vocab)
        if rank == 0 and not args.run_id:
            stats = estimate_memory(config)
            rows.append(
                {
                    "preset": preset,
                    "strategy": "memory_model",
                    "ok": True,
                    "params": stats["params"],
                    "ddp_state_gib": stats["ddp_state_gib"],
                }
            )
            print(
                f"\n=== {preset}: {stats['params'] / 1e6:.0f}M params, "
                f"DDP state {stats['ddp_state_gib']:.1f} GiB/card ===",
                flush=True,
            )

        available = {
            "single": (run_single, {}),
            "ddp": (run_ddp, {}),
            "ddp+ckpt": (run_ddp, {"grad_ckpt": True}),
            "pipeline": (run_pipeline, {}),
            "pp+ckpt": (run_pipeline, {"grad_ckpt": True}),
        }
        todo = [(n, *available[n]) for n in wanted if n in available]
        if world == 1:
            todo = [("single", run_single, {})]

        for name, fn, kwargs in todo:
            torch.cuda.reset_peak_memory_stats()
            try:
                if name == "single":
                    result = fn(config, args, device, **kwargs)
                else:
                    result = fn(config, args, device, rank, world, **kwargs)
                ok, err = True, None
            except torch.OutOfMemoryError:
                result, ok, err = {"tokens_per_s": 0, "peak_gib": 0}, False, "OOM"
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                result, ok, err = (
                    {"tokens_per_s": 0, "peak_gib": 0},
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
                torch.cuda.empty_cache()
            if rank == 0:
                rows.append(
                    {
                        "preset": preset,
                        "strategy": name,
                        "ok": ok,
                        "error": err,
                        **result,
                    }
                )
                status = (
                    f"{result['tokens_per_s'] / 1e3:8.1f}k tok/s  "
                    f"{result['peak_gib']:5.2f} GiB"
                    if ok
                    else f"FAILED: {err}"
                )
                print(f"  {name:10s} {status}", flush=True)
            if world > 1:
                dist.barrier()

    if args.run_id:
        if rank == 0:
            parts = os.path.join(args.out, "parts")
            os.makedirs(parts, exist_ok=True)
            safe = args.run_id.replace(":", "__").replace("/", "_")
            with open(os.path.join(parts, f"{safe}.json"), "w") as handle:
                json.dump(rows, handle)
        if world > 1:
            dist.destroy_process_group()
        return

    if rank == 0:
        with open(os.path.join(args.out, "e2e.json"), "w") as handle:
            json.dump(rows, handle, indent=2)
        plot(rows, args.out, args)
        print(f"\nwrote {args.out}/e2e.png")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
