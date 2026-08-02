"""Do DDP and pipeline produce the same gradients as a single-GPU reference?

This is a *correctness* harness, not a benchmark. A parallelism strategy that is
fast and subtly wrong is worse than no strategy at all, because the symptom is a
slightly worse loss curve that nobody attributes to the plumbing. So before any
throughput number is taken, both paths are held against the same yardstick: one
GPU, one batch, one backward.

The comparison that matters is **gradients**, not loss. Loss agreeing only says
the forward matched; gradients agreeing says the whole backward -- including the
pipeline's stage-boundary gradient handoff and DDP's all-reduce -- is faithful.

Normalization is what makes the three comparable. Every path computes a
*summed* loss and divides by the same global trained-token count:

* reference: one backward over the whole batch
* DDP: each rank takes a shard, backward, then gradients are all-reduced. DDP
  averages, so each rank's local loss is scaled by ``world`` to undo it.
* pipeline: each stage backwards its microbatches; gradients accumulate locally
  because a stage's parameters live on exactly one rank.

Run (the script spawns its own ranks):
    .venv/bin/python scripts/dist/parallel_equivalence.py
    .venv/bin/python scripts/dist/parallel_equivalence.py --mode ddp
"""

import argparse
import contextlib
import os

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.parallel.pipeline_lightning import (
    build_schedule,
    build_stage,
    run_step,
)

IGNORE = -100
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 3


def build_batches(cfg, n_micro, per_micro, device, seed=0):
    """Identical microbatches on every rank, from a fixed seed."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    micro = []
    for _ in range(n_micro):
        lengths = torch.full((max(1, per_micro // 128),), 128, dtype=torch.int32)
        lengths = lengths[: per_micro // 128]
        info = SeqInfo.from_lengths(lengths, device)
        ids = torch.randint(0, cfg.vocab_size, (per_micro,), generator=gen).to(device)
        labels = ids.roll(-1)
        labels[info.cu_seqlens[1:].long() - 1] = IGNORE
        micro.append((ids, labels, info))
    return micro


def reference_grads(cfg, micro, device):
    """Single-GPU: every microbatch through one model, one summed backward."""
    torch.manual_seed(1234)
    model = LMBackbone(cfg).to(device=device, dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    total = sum((lab != IGNORE).sum().item() for _, lab, _ in micro)
    for ids, labels, info in micro:
        loss, _ = model.loss(ids, labels, info, reduction="sum")
        (loss / total).backward()
    grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return model, grads, total


def rel_err(a, b):
    a = a.double()
    b = b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def run_pipeline(cfg, micro, device, rank, world, total):
    """Uses torch.distributed.pipelining, matching the production path."""
    torch.manual_seed(1234)
    per_micro = micro[0][0].numel()
    module, stage, plan, _ = build_stage(
        cfg,
        rank,
        world,
        device,
        per_micro,
        autocast_dtype=None,
        seq_len=per_micro,
    )
    module.zero_grad(set_to_none=True)
    module.set_seq_info([m[2] for m in micro])

    def loss_fn(hidden, target):
        loss, _ = module.loss(hidden, target)
        return loss / total

    schedule = build_schedule(stage, len(micro), loss_fn=loss_fn, kind="1f1b")
    losses = [] if rank == world - 1 else None
    run_step(
        schedule,
        rank,
        world,
        tokens=torch.cat([m[0] for m in micro]),
        targets=torch.cat([m[1] for m in micro]),
        losses=losses,
    )
    return module, plan


def compare_pipeline(cfg, micro, device, rank, world):
    ref_model, ref_grads, total = reference_grads(cfg, micro, device)
    stage, plan = run_pipeline(cfg, micro, device, rank, world, total)

    # Map stage-local parameter names back onto the reference model's names.
    # Only `blocks.N` is renumbered; embed / final_norm / head keep their names.
    worst = 0.0
    checked = 0
    missing = []
    for name, param in stage.named_parameters():
        if param.grad is None:
            missing.append(name)
            continue
        if name.startswith("blocks."):
            parts = name.split(".")
            parts[1] = str(int(parts[1]) + plan.start_layer)
            ref_name = ".".join(parts)
        else:
            ref_name = name
        if ref_name not in ref_grads:
            # The untied head copy has no counterpart under a tied reference.
            print(f"[rank{rank}] no reference for {name} -> {ref_name}", flush=True)
            continue
        err = rel_err(param.grad, ref_grads[ref_name])
        worst = max(worst, err)
        checked += 1
    print(
        f"[rank{rank}] stage {plan.start_layer}..{plan.end_layer} "
        f"checked={checked} worst_rel_err={worst:.3e} "
        f"no_grad={len(missing)}",
        flush=True,
    )
    if missing:
        print(f"[rank{rank}] params with NO gradient: {missing[:6]}", flush=True)
    del ref_model
    return worst, checked, len(missing)


class _LossModule(torch.nn.Module):
    """Exposes the loss through ``forward`` so DDP can actually see it.

    This wrapper is not decoration. ``DistributedDataParallel`` installs its
    gradient-synchronization hooks during **its own** ``forward``; calling
    ``ddp.module.loss(...)`` reaches the same parameters but skips that entry
    point, so no all-reduce is ever issued and every rank silently keeps its own
    local gradient. The symptom is a model that trains but does not converge like
    the single-GPU run, with nothing in the logs to say why.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, ids, labels, seq_info):
        loss, _ = self.backbone.loss(ids, labels, seq_info, reduction="sum")
        return loss


def compare_ddp(cfg, micro, device, rank, world):
    """Each rank takes a disjoint shard; all-reduced grads must match the reference."""
    from torch.nn.parallel import DistributedDataParallel

    ref_model, ref_grads, total = reference_grads(cfg, micro, device)

    torch.manual_seed(1234)
    model = LMBackbone(cfg).to(device=device, dtype=torch.float32)
    ddp = DistributedDataParallel(_LossModule(model), device_ids=[device.index])
    ddp.zero_grad(set_to_none=True)
    shard = micro[rank::world]
    for i, (ids, labels, info) in enumerate(shard):
        # Sync only on the final micro-batch of the shard: DDP would otherwise
        # all-reduce after every backward, which is correct but wasteful, and
        # `no_sync` is what a real accumulation loop uses.
        ctx = ddp.no_sync() if i < len(shard) - 1 else contextlib.nullcontext()
        with ctx:
            loss = ddp(ids, labels, info)
            # DDP averages across ranks; scale by `world` to undo that so the
            # result is the plain sum over all micro-batches, as the reference is.
            (loss * world / total).backward()

    # Weights first: if init diverged, comparing gradients tells you nothing.
    ref_params = dict(ref_model.named_parameters())
    wmax = max(
        (
            rel_err(p.detach(), ref_params[n].detach())
            for n, p in ddp.module.backbone.named_parameters()
            if n in ref_params
        ),
        default=0.0,
    )
    print(f"[rank{rank}] ddp weight rel_err vs reference = {wmax:.3e}", flush=True)

    worst, checked = 0.0, 0
    for name, param in ddp.module.backbone.named_parameters():
        if param.grad is None or name not in ref_grads:
            continue
        worst = max(worst, rel_err(param.grad, ref_grads[name]))
        checked += 1
    print(f"[rank{rank}] ddp checked={checked} worst_rel_err={worst:.3e}", flush=True)
    return worst, checked, 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "ddp"])
    ap.add_argument("--preset", default="Nano-25M")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--microbatches", type=int, default=6)
    ap.add_argument("--per-microbatch", type=int, default=256)
    ap.add_argument("--tol", type=float, default=2e-4)
    return ap.parse_args()


def main(args):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    # fp32 and untied: this is an exactness check, so nothing that costs
    # precision or complicates the parameter mapping belongs in it.
    cfg = get_preset(args.preset, vocab_size=args.vocab, tie_embeddings=False)
    micro = build_batches(cfg, args.microbatches, args.per_microbatch, device)

    if args.mode == "pipeline":
        worst, checked, missing = compare_pipeline(cfg, micro, device, rank, world)
    else:
        worst, checked, missing = compare_ddp(cfg, micro, device, rank, world)

    verdict = torch.tensor(
        [1.0 if (worst < args.tol and checked > 0 and missing == 0) else 0.0],
        device=device,
    )
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    if rank == 0:
        ok = bool(verdict.item())
        print(
            f"\n{'PASS' if ok else 'FAIL'}: {args.mode} gradients match the "
            f"single-GPU reference (tol {args.tol:g})",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if verdict.item() else 1


def launch() -> int:
    """Run ``main`` on ``GPUS`` spawned ranks, or in place when ranks exist.

    Returns the worst exit status any rank reported.
    """
    args = parse_args()
    if os.environ.get("RANK") is not None:
        return main(args)
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=GPUS or torch.cuda.device_count(),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="parallel_equivalence",
        max_restarts=0,
        start_method="spawn",
    )
    return max(elastic_launch(config, main)(args).values())


if __name__ == "__main__":
    raise SystemExit(launch())
