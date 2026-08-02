"""Throughput at a realistic optimizer-step batch size.

Earlier sweeps measured a single forward/backward, which flatters whichever
configuration happens to fit in one shot and tells you nothing about a real
training step. Here the *step* is fixed -- a token budget, 256k by default --
and the only free variable is how that budget is split into microbatches. Both
parallelism strategies are then measured against the same step:

* single GPU accumulates gradients over ``budget / per_micro`` microbatches
* the 4-stage pipeline runs the same count through ``Schedule1F1B``

so the speedup reported is between two runs that consume identical data and
produce one optimizer step each.

Sequences are ragged by default, because pretraining is: document lengths are
drawn uniformly from ``[--len-min, --len-max]`` and packed to exactly
``per_micro`` tokens, trimming the document that straddles the boundary. Every
microbatch therefore holds the same token count -- required, since the boundary
activation shape is fixed at stage-build time -- while the documents inside it
differ, which is what exercises the varlen attention path.

Usage (``--gpus`` sets the rank count; 0 uses every GPU):
    .venv/bin/python scripts/bench/e2e/step_throughput.py --preset Nano-1B
    .venv/bin/python scripts/bench/e2e/step_throughput.py --gpus 4 \
        --preset Nano-1B --out out/bench/train/step
"""

import argparse
import contextlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.nn.parallel import DistributedDataParallel

from kohakuwullm.bench import make_packs, make_tokens, trained_tokens
from kohakuwullm.bench.vendor.vendor_moe import dense_ceilings
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights, swap_mxfp8
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.training.optim.lowbit import cast_parameters_

PARAM_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
AUTOCAST_DTYPES = {
    "fp32": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 1


def _scope(args) -> tuple[str, ...] | None:
    """The swap scope, as a tuple the report can record verbatim."""
    return tuple(args.mxfp8_scope) if args.mxfp8_scope else None


def _select_moe_path(model, args) -> None:
    """Bind the routed experts' fp8 kernels before the swap reads the choice.

    Before `swap_mxfp8`, not after: `MoEMLP.enable_mxfp8` resolves the path once and
    holds the bound method, so setting the attribute afterwards would change nothing
    and the run would silently be the other arm.
    """
    for module in model.modules():
        if isinstance(module, MoEMLP):
            module.mxfp8_expert_path = args.mxfp8_moe


def _opt_kwargs(args) -> dict:
    """Optimizer kwargs the dtype arms need, empty for the default arm.

    Passed only when non-default so an optimizer with no rounding rule keeps working;
    when one is asked for and unsupported, the ``TypeError`` is the right outcome --
    a silently ignored ``--rounding stochastic`` would label an arm that never ran.
    """
    return {} if args.rounding == "nearest" else {"rounding": args.rounding}


def compile_blocks(model, enabled):
    """Compile transformer blocks only, in ``reduce-overhead`` mode.

    Whole-model compilation is not an option: the chunked linear-cross-entropy
    head is hostile to it. Per-block compilation leaves the head alone, and
    ``reduce-overhead`` is the mode that pays off here -- it is CUDA-graph
    capture against launch overhead, not the aggressive fusion of ``default``.
    """
    if not enabled:
        return model
    for i, block in enumerate(model.blocks):
        model.blocks[i] = torch.compile(block, mode="reduce-overhead")
    return model


def run_single(cfg, args, device, ddp_rank=0, ddp_world=1):
    """Gradient accumulation on one GPU, or sharded across ranks under DDP.

    Under DDP each rank takes ``budget / world`` tokens and syncs once per step
    (``no_sync`` on every microbatch but the last), which is what a real
    accumulation loop does. At 1B parameters that single all-reduce is not free,
    so it is measured rather than assumed away.
    """
    is_ddp = ddp_world > 1
    shard = args.budget // ddp_world
    rows = []
    for per_micro in args.micro_tokens:
        if per_micro > shard:
            continue
        torch.manual_seed(1234)
        model = LMBackbone(cfg).to(device=device, dtype=torch.float32)
        if args.param_dtype != "fp32":
            cast_parameters_(model, PARAM_DTYPES[args.param_dtype])
        # After the cast, before compile: the swap copies weights, so it must see the
        # dtype the run will train in, and Inductor cannot lower the fp8 swizzles.
        # Not named `report`: that is the module-level printer this function calls
        # later, and binding the name here makes it local for the whole scope -- so the
        # non-mxfp8 path died on UnboundLocalError and the mxfp8 path would have called
        # a SwapReport. Cost one dense DDP row before it was caught.
        expect_refresh = 0
        if args.mxfp8:
            _select_moe_path(model, args)
            swapped = swap_mxfp8(model, scope=_scope(args))
            if swapped.blocking:
                raise RuntimeError(f"mxfp8 swap incomplete: {swapped.blocking}")
            expect_refresh = len(swapped.modules)
        model = compile_blocks(model, args.compile)
        infos = make_packs(
            shard, per_micro, args.len_min, args.len_max, device, args.seed + ddp_rank
        )
        ids, labels = make_tokens(infos, cfg.vocab_size, device, args.seed + ddp_rank)
        total = trained_tokens(labels)
        if is_ddp:
            total = int(all_reduce_scalar(total, device))
            wrapped = DistributedDataParallel(
                _LossModule(model, args.autocast_dtype), device_ids=[device.index]
            )
            opt = build_optimizer(
                wrapped, name=args.optimizer, lr=1e-4, **_opt_kwargs(args)
            )
        else:
            wrapped, opt = None, build_optimizer(
                model, name=args.optimizer, lr=1e-4, **_opt_kwargs(args)
            )

        def step(opt=opt, model=model, wrapped=wrapped, infos=infos):
            opt.zero_grad(set_to_none=True)
            for i, info in enumerate(infos):
                lo, hi = i * per_micro, (i + 1) * per_micro
                if is_ddp:
                    last = i == len(infos) - 1
                    ctx = contextlib.nullcontext() if last else wrapped.no_sync()
                    with ctx:
                        loss = wrapped(ids[lo:hi], labels[lo:hi], info)
                        (loss * ddp_world / total).backward()
                else:
                    with torch.autocast("cuda", dtype=args.autocast_dtype):
                        loss, _ = model.loss(
                            ids[lo:hi], labels[lo:hi], info, reduction="sum"
                        )
                    (loss / total).backward()
            opt.step()
            # Every step, exactly as the trainer does. `MXFP8Linear.forward` quantizes
            # only when its cache is empty and never invalidates on a weight update, so
            # omitting this measures a step that trains on the *previous* step's weights
            # and skips the requantization a real fp8 step pays for -- faster than the
            # thing it claims to measure, and wrong in a way the loss would not show.
            if args.mxfp8:
                got = refresh_mxfp8_weights(model)
                # The count is checked, not discarded: this function returns it exactly
                # so a caller can prove the refresh reached every module. A silent zero
                # is the failure it exists to catch, and it has no symptom in the loss.
                if got != expect_refresh:
                    raise RuntimeError(
                        f"refreshed {got} modules, expected {expect_refresh}"
                    )

        row = timed(step, per_micro, len(infos) * ddp_world, args, device)
        row["budget"] = args.budget
        # Recorded so the table can state what a row actually trained on: packing is
        # exact, so `budget` is all real tokens, and `trained` is lower only by the one
        # label per document that has no successor.
        row["trained"] = int(total)
        row["tokens_per_s"] = (args.budget / row["seconds"]) if row["ok"] else 0.0
        rows.append(row)
        if ddp_rank == 0:
            report(f"ddp{ddp_world}" if is_ddp else "1gpu", row)
        write_rows(rows, args, ddp_world, device, ddp_rank)
        del model, opt, wrapped
        torch.cuda.empty_cache()
    return rows


class _LossModule(torch.nn.Module):
    """Routes the loss through ``forward`` so DDP's gradient hooks actually fire.

    Calling ``ddp.module.loss(...)`` reaches the same parameters but bypasses the
    entry point where the sync hooks are installed, so no all-reduce is ever
    issued and each rank silently keeps a local gradient.
    """

    def __init__(self, backbone, autocast_dtype=torch.bfloat16):
        super().__init__()
        self.backbone = backbone
        self.autocast_dtype = autocast_dtype

    def forward(self, ids, labels, seq_info):
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            loss, _ = self.backbone.loss(ids, labels, seq_info, reduction="sum")
        return loss


def all_reduce_scalar(value, device):
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t)
    return t.item()


def run_pipeline(cfg, args, device, rank, world):
    """4-stage 1F1B over the same step, through torch.distributed.pipelining."""
    from kohakuwullm.training.parallel.pipeline_lightning import (
        build_schedule,
        build_stage,
        run_step,
        stage_local_optimizer,
    )

    rows = []
    for per_micro in args.micro_tokens:
        if per_micro > args.budget or (args.budget // per_micro) < world:
            continue
        torch.manual_seed(1234)
        module, stage, plan, _ = build_stage(
            cfg,
            rank,
            world,
            device,
            per_micro,
            autocast_dtype=args.autocast_dtype,
            param_dtype=PARAM_DTYPES[args.param_dtype],
            seq_len=per_micro,
        )
        inner = getattr(module, "module", module)
        # Per stage, not per model: each rank owns a disjoint slice, so the swap has to
        # run on all four or three of them silently stay bf16 and the arm is mislabelled.
        expect_refresh = 0
        if args.mxfp8:
            _select_moe_path(inner, args)
            swapped = swap_mxfp8(inner, scope=_scope(args))
            if swapped.blocking:
                raise RuntimeError(f"mxfp8 swap incomplete: {swapped.blocking}")
            expect_refresh = len(swapped.modules)
        compile_blocks(inner, args.compile)
        opt = stage_local_optimizer(
            module, name=args.optimizer, lr=1e-4, **_opt_kwargs(args)
        )
        infos = make_packs(
            args.budget, per_micro, args.len_min, args.len_max, device, args.seed
        )
        ids, labels = make_tokens(infos, cfg.vocab_size, device, args.seed)
        total = trained_tokens(labels)
        n_micro = len(infos)

        def loss_fn(hidden, target, module=module, total=total):
            loss, _ = module.loss(hidden, target)
            return loss / total

        schedule = build_schedule(stage, n_micro, loss_fn=loss_fn, kind=args.schedule)

        def step(opt=opt, module=module, schedule=schedule, infos=infos):
            opt.zero_grad(set_to_none=True)
            module.set_seq_info(infos)
            run_step(schedule, rank, world, tokens=ids, targets=labels)
            opt.step()
            if args.mxfp8:
                got = refresh_mxfp8_weights(module)
                if got != expect_refresh:
                    raise RuntimeError(
                        f"refreshed {got} modules, expected {expect_refresh}"
                    )

        rows.append(timed(step, per_micro, n_micro, args, device))
        rows[-1]["stage"] = f"{plan.start_layer}..{plan.end_layer}"
        rows[-1]["trained"] = int(total)
        if rank == 0:
            report(f"pp{world}", rows[-1])
        write_rows(rows, args, world, device, rank)
        del module, stage, schedule, opt
        torch.cuda.empty_cache()
    return rows


def timed(step, per_micro, n_micro, args, device=None):
    """Step a fixed number of times and measure only the steady-state tail.

    A constant warm-up count is the wrong control for a varlen stream: how many
    steps it takes to compile every shape depends on the data, so a fixed small
    number either charges Triton compilation as throughput or burns device time
    guessing. Step through a whole window instead and measure its tail, with
    steady state **verified** rather than assumed -- the caching allocator must
    have stopped carving new segments across the measured steps, which is what
    "no further compile or allocation" looks like from outside the process.

    The tail's **median**, not its mean: one descheduled step should not move the
    number a configuration is judged by. Spread over the tail is reported beside
    it, so a window that never settled is visible instead of averaged away.

    OOM is recorded, not raised -- a rung that does not fit is a result.
    """
    per_step = []
    steady = False
    try:
        for _ in range(max(args.steps - args.measure_last, 0)):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_stats()
        for _ in range(args.measure_last):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            per_step.append(time.perf_counter() - start)
        after = torch.cuda.memory_stats()
        seconds = statistics.median(per_step)
        peak = torch.cuda.max_memory_allocated() / 2**30
        # A new segment means a cudaMalloc landed inside the measured window; a
        # retry means the allocator had to free and try again. Either way the
        # window was not steady and its rate is not the sustained one.
        steady = after.get("segment.all.allocated", 0) == before.get(
            "segment.all.allocated", 0
        ) and after.get("num_alloc_retries", 0) == before.get("num_alloc_retries", 0)
        ok = True
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.OutOfMemoryError) and (
            "out of memory" not in str(exc) and "OutOfMemoryError" not in str(exc)
        ):
            raise
        seconds, peak, ok = float("nan"), float("nan"), False
        torch.cuda.empty_cache()

    # Under any multi-rank strategy the step is only as fast as its slowest
    # rank, and the VRAM that matters is the worst stage's, not rank 0's.
    if device is not None and dist.is_available() and dist.is_initialized():
        stats = torch.tensor(
            [seconds if ok else 1e9, peak if ok else 0.0, 1.0 if ok else 0.0],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        peaks = torch.tensor([peak if ok else 0.0], device=device)
        dist.all_reduce(peaks)
        seconds, peak = stats[0].item(), stats[1].item()
        total_peak = peaks.item()
    else:
        total_peak = peak
    spread = (max(per_step) - min(per_step)) / seconds if ok and per_step else 0.0
    return dict(
        per_micro=per_micro,
        n_micro=n_micro,
        budget=per_micro * n_micro,
        seconds=seconds,
        tokens_per_s=(per_micro * n_micro / seconds) if ok else 0.0,
        peak_gib=peak,
        total_peak_gib=total_peak,
        step_spread=spread,
        per_step=per_step,
        steady=steady,
        steps=args.steps,
        measured=args.measure_last,
        ok=ok,
    )


def write_rows(rows, args, world, device, rank) -> None:
    """Persist what has been measured so far, after every row.

    Written incrementally, not once at the end: a rung that fits at 4096 and OOMs at
    8192 kills the process on the larger row, and a single end-of-run write throws away
    the smaller row that already succeeded. That cost MoE-3B's whole record -- both its
    pipeline and DDP 4096 rows had completed cleanly.
    """
    if rank != 0:
        return
    os.makedirs(args.out, exist_ok=True)
    default_tag = "1gpu" if world == 1 else f"{args.mode}{world}"
    suffix = ("_compiled" if args.compile else "") + (
        f"_{args.param_dtype}" if args.param_dtype != "fp32" else ""
    )
    tag = args.tag or default_tag + suffix
    path = os.path.join(args.out, f"{args.preset}_{tag}.json")
    # The run config is recorded, not just encoded in --tag: the microbatch
    # drivers pass --param-dtype bf16 without putting it in the tag, so a
    # plotter that parses the filename reports those rows as fp32.
    with open(path, "w") as handle:
        json.dump(
            dict(
                preset=args.preset,
                world=world,
                mode=args.mode,
                param_dtype=args.param_dtype,
                optimizer=args.optimizer,
                # Recorded, not left to the tag: a mislabelled arm is the one
                # failure mode a throughput comparison cannot survive.
                # The dense bf16 rate this card reached in this process, so a
                # plotter can say "this run" instead of drawing another day's clock.
                matmul_ceiling_tflops=args.matmul_ceiling_tflops,
                mxfp8=args.mxfp8,
                rounding=args.rounding,
                grad_ckpt=args.grad_ckpt,
                compiled=args.compile,
                schedule=args.schedule,
                budget=args.budget,
                len_min=args.len_min,
                len_max=args.len_max,
                steps=args.steps,
                measure_last=args.measure_last,
                device=torch.cuda.get_device_name(device),
                tag=tag,
                rows=rows,
            ),
            handle,
            indent=2,
        )
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def report(tag, row):
    if not row["ok"]:
        print(f"  {tag} {row['n_micro']:3d}x{row['per_micro']:6d}: OOM", flush=True)
        return
    print(
        f"  {tag} {row['n_micro']:3d}x{row['per_micro']:6d} = {row['budget']}: "
        f"{row['seconds']:6.2f}s {row['tokens_per_s']:9,.0f} tok/s "
        f"peak {row['peak_gib']:5.1f} GiB (all ranks {row['total_peak_gib']:5.1f}) "
        f"spread {row['step_spread'] * 100:.1f}%"
        f"{'' if row.get('steady', True) else '  ALLOCATED-MID-WINDOW'}",
        flush=True,
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, default=GPUS)
    ap.add_argument("--out", default="out/bench/train/step")
    ap.add_argument("--preset", default="Nano-1B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--budget", type=int, default=262144)
    ap.add_argument("--micro-tokens", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument(
        "--micro-counts",
        type=int,
        nargs="+",
        default=None,
        help="microbatch counts; sizes are budget//count, so they need not be "
        "powers of two. Varlen packing fills any size, and the memory ceiling "
        "rarely lands on a power of two.",
    )
    ap.add_argument("--len-min", type=int, default=512)
    ap.add_argument("--len-max", type=int, default=4096)
    ap.add_argument("--uniform", action="store_true", help="equal-length docs instead")
    ap.add_argument("--schedule", default="1f1b")
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "ddp"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--param-dtype", default="fp32", choices=list(PARAM_DTYPES))
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--rounding",
        default="nearest",
        choices=["nearest", "stochastic"],
        help="parameter writeback rounding; stochastic requires 16-bit params",
    )
    ap.add_argument(
        "--mxfp8",
        action="store_true",
        help="swap eligible matmuls to MXFP8 before timing",
    )
    ap.add_argument(
        "--mxfp8-moe",
        default="fused",
        choices=["fused", "unfused"],
        help="which MXFP8 routed-expert path a sparse preset uses; ignored by a dense "
        "one. 'unfused' issues every expert GEMM through the vendor-verified grouped "
        "primitive instead of the four fused kernels",
    )
    ap.add_argument(
        "--mxfp8-scope",
        nargs="*",
        default=None,
        help="qualified-name substrings to restrict the swap to, e.g. attn. "
        "(default: everything eligible)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="total steps to run; the tail is what gets measured (0: warmup+iters)",
    )
    ap.add_argument(
        "--measure-last",
        type=int,
        default=0,
        help="how many of the final steps to time (0: --iters)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.micro_counts:
        args.micro_tokens = sorted({args.budget // n for n in args.micro_counts})
    if args.uniform:
        args.len_min = args.len_max = min(args.len_max, min(args.micro_tokens))
    return args


def main(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    cfg = get_preset(args.preset, vocab_size=args.vocab, tie_embeddings=False)
    cfg.max_position = max(cfg.max_position, max(args.micro_tokens), args.len_max)
    cfg.grad_ckpt = args.grad_ckpt
    args.autocast_dtype = AUTOCAST_DTYPES[args.param_dtype]
    # Resolved once here so `timed` knows only the window, not the two ways of
    # spelling it: the legacy --warmup/--iters pair maps onto the same protocol.
    args.steps = args.steps or (args.warmup + args.iters)
    args.measure_last = min(args.measure_last or args.iters, args.steps)

    if rank == 0:
        print(
            f"{args.preset}  budget {args.budget} tok/step  "
            f"docs {args.len_min}-{args.len_max}  world {world}",
            flush=True,
        )

    # Measured here, not carried as a constant: a stored ceiling is a measurement from
    # another day, and one disqualified a legitimate 228 TF/s row on this box by being
    # 0.4% stale. Taken before the rows so cuBLAS is warm, and the cache is released so
    # the 4096-square operands do not inflate the peak-memory column that follows.
    args.matmul_ceiling_tflops = None
    if rank == 0:
        try:
            args.matmul_ceiling_tflops = dense_ceilings()["bf16"]
        except Exception as exc:
            print(
                f"dense ceiling unavailable ({exc}); rows will say 'carried'",
                flush=True,
            )
        torch.cuda.empty_cache()

    if world == 1:
        rows = run_single(cfg, args, device)
    elif args.mode == "ddp":
        rows = run_single(cfg, args, device, ddp_rank=rank, ddp_world=world)
    else:
        rows = run_pipeline(cfg, args, device, rank, world)

    write_rows(rows, args, world, device, rank)

    if world > 1:
        dist.destroy_process_group()


def launch() -> None:
    """Run ``main`` on ``--gpus`` spawned ranks, or in place for a single one."""
    args = parse_args()
    nproc = args.gpus or torch.cuda.device_count()
    if os.environ.get("RANK") is not None or nproc <= 1:
        main(args)
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="step_throughput",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)(args)


if __name__ == "__main__":
    launch()
