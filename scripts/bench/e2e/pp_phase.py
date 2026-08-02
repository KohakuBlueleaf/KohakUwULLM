"""Time one pipeline stage's compute in isolation, with no communication.

Separates stage compute from pipeline overhead: if the sum of per-stage times is
already close to the observed step time, the schedule is not the problem.

    .venv/bin/python scripts/bench/e2e/pp_phase.py
"""

import os
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.parallel.pipeline import AutocastStage, LMStage, plan_for

# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    preset = os.environ.get("PP_PRESET", "Kohaku-1B")
    per = int(os.environ.get("PP_PER_MICRO", "1024"))
    cfg = get_preset(preset, vocab_size=65536, tie_embeddings=False)
    backbone = LMBackbone(cfg)
    plans = plan_for(cfg, world, seq_len=per)
    plan = plans[rank]
    stage = AutocastStage(
        LMStage(backbone, plan).to(device=device, dtype=torch.float32)
    )

    n_seq = max(1, per // 256)
    lengths = torch.full((n_seq,), per // n_seq, dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, device)

    if plan.has_embed:
        x = torch.randint(0, 65536, (per,), device=device)
    else:
        x = torch.randn(per, cfg.dim, device=device, requires_grad=True)

    def once() -> None:
        # Every iteration measures a write, not an accumulate.
        stage.zero_grad(set_to_none=True)
        x.grad = None
        stage.set_seq_info([info])
        stage(x).sum().backward()

    for _ in range(2):
        once()
    torch.cuda.synchronize()
    dist.barrier()

    iters = 5
    start = time.perf_counter()
    for _ in range(iters):
        once()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    print(
        f"[rank{rank}] layers {plan.start_layer}..{plan.end_layer} "
        f"({plan.num_layers}L embed={plan.has_embed} head={plan.has_head}): "
        f"{elapsed * 1000:.1f} ms per microbatch fwd+bwd",
        flush=True,
    )
    dist.barrier()
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
        run_id="pp_phase",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)()


if __name__ == "__main__":
    launch()
