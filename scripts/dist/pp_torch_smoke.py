"""Smoke test for torch.distributed.pipelining, without Lightning in the path.

The script spawns its own ranks::

    .venv/bin/python scripts/dist/pp_torch_smoke.py
"""

import os
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.models import get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.parallel.pipeline_lightning import (
    build_schedule,
    build_stage,
    run_step,
    stage_local_optimizer,
)

IGNORE = -100
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


def log(rank: int, msg: str) -> None:
    print(f"[rank{rank} {time.time() % 1000:8.2f}] {msg}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    preset = os.environ.get("PP_PRESET", "Nano-25M")
    vocab = int(os.environ.get("PP_VOCAB", "4096"))
    n_micro = int(os.environ.get("PP_MICRO", "4"))
    per_micro = int(os.environ.get("PP_PER_MICRO", "256"))
    schedule_kind = os.environ.get("PP_SCHEDULE", "1f1b")

    cfg = get_preset(preset, vocab_size=vocab, tie_embeddings=False)
    module, stage, plan, _ = build_stage(
        cfg, rank, world, device, per_micro, seq_len=per_micro
    )
    log(rank, f"stage {plan.start_layer}..{plan.end_layer} built")

    total_tokens = per_micro * n_micro
    n_seq = max(1, per_micro // 128)
    lengths = torch.full((n_seq,), per_micro // n_seq, dtype=torch.int32)
    infos = [SeqInfo.from_lengths(lengths, device) for _ in range(n_micro)]
    module.set_seq_info(infos)

    tokens = torch.randint(0, vocab, (total_tokens,), device=device)
    targets = tokens.roll(-1)

    def loss_fn(hidden, target):
        loss, _ = module.loss(hidden, target)
        return loss / total_tokens

    schedule = build_schedule(stage, n_micro, loss_fn=loss_fn, kind=schedule_kind)
    opt = stage_local_optimizer(module, lr=1e-4)
    log(rank, f"schedule {schedule_kind} ready, {n_micro} microbatches")

    for it in range(3):
        opt.zero_grad(set_to_none=True)
        module.set_seq_info(infos)
        losses = [] if rank == world - 1 else None
        start = time.perf_counter()
        run_step(schedule, rank, world, tokens=tokens, targets=targets, losses=losses)
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        mean = (
            sum(x.item() for x in losses) / max(len(losses), 1)
            if losses
            else float("nan")
        )
        log(rank, f"iter {it}: {elapsed:.3f}s loss={mean:.4f}")

    dist.barrier()
    log(rank, "ALL OK")
    dist.destroy_process_group()


def launch() -> None:
    """Run ``main`` on ``GPUS`` spawned ranks, or in place when ranks exist."""
    if os.environ.get("RANK") is not None:
        main()
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=GPUS or torch.cuda.device_count(),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="pp_torch_smoke",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)()


if __name__ == "__main__":
    launch()
