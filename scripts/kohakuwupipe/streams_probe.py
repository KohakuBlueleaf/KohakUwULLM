"""Do multi-stream boundaries work, and does each stream get the right gradient?

    .venv/bin/python scripts/kohakuwupipe/streams_probe.py

Three shapes, each checked against what it must satisfy:

1. **accumulator** -- every stage adds a scalar; each stage's term must end with
   gradient 1, the same as an independent ``.backward()``.
2. **constant** -- a tensor a later stage reads. Checks whether the runtime
   needs the grad-carrier trick or accepts a dead boundary.
3. **skip** -- an activation carried past intermediate stages, U-Net style.
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from kohakuwupipe import get_logger, init_pipeline, shutdown
from kohakuwupipe.parallel.streams import GradCarrier

log = get_logger("streams_probe")
DIM, ROWS = 128, 64
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


class StreamStage(nn.Module):
    """Carries (hidden, aux, context, skip) and contributes to each."""

    def __init__(self, first: bool, last: bool, carrier: bool) -> None:
        super().__init__()
        self.first, self.last = first, last
        self.block = nn.Linear(DIM, DIM)
        # A per-stage term the accumulator must give gradient 1.
        self.term_scale = nn.Parameter(torch.ones(()))
        self.carrier = GradCarrier() if carrier else None

    def forward(self, hidden, aux, context, skip):
        hidden = hidden + torch.tanh(self.block(hidden))
        # Every stage adds its own scalar; grad must reach term_scale as 1.
        aux = aux + self.term_scale
        if self.carrier is not None:
            context = self.carrier(context)
        if self.first:
            skip = hidden * 1.0
        return hidden, aux, context, skip

    def loss(self, hidden, target):
        return (hidden - target).pow(2).mean(), {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--microbatches", type=int, default=4)
    ap.add_argument("--carrier", action="store_true", default=True)
    ap.add_argument("--no-carrier", dest="carrier", action="store_false")
    return ap.parse_args()


def main(args: argparse.Namespace) -> None:
    ranks = init_pipeline()
    first, last = ranks.rank == 0, ranks.rank == ranks.world - 1
    stage_mod = StreamStage(first, last, args.carrier).to(ranks.device)

    dev = ranks.device
    example = (
        torch.zeros(ROWS, DIM, device=dev),  # hidden
        torch.zeros(1, 1, device=dev),  # accumulator, one slot per microbatch
        torch.zeros(ROWS, DIM, device=dev),  # constant context
        torch.zeros(ROWS, DIM, device=dev),  # skip
    )
    stage = PipelineStage(stage_mod, ranks.rank, ranks.world, dev, input_args=example)

    def loss_fn(output, target):
        hidden, aux, context, skip = output
        base = (hidden - target).pow(2).mean()
        # The accumulator enters the loss directly: grad 1 to every term.
        return base + aux.sum() + 0.0 * context.sum() + 0.0 * skip.sum()

    schedule = ScheduleGPipe(
        stage, n_microbatches=args.microbatches, loss_fn=loss_fn, scale_grads=False
    )
    n = ROWS * args.microbatches
    hidden = torch.randn(n, DIM, device=dev, requires_grad=True)
    # One slot per microbatch, contiguous so each chunk is its own dense tensor:
    # a chunk that is a view into a shared buffer is refused by NCCL P2P.
    aux0 = torch.zeros(args.microbatches, 1, device=dev).contiguous()
    ctx = torch.randn(n, DIM, device=dev)
    skip0 = torch.zeros(n, DIM, device=dev)
    target = torch.randn(n, DIM, device=dev)

    stage_mod.zero_grad(set_to_none=True)
    try:
        if first:
            schedule.step(hidden, aux0, ctx, skip0)
        elif last:
            schedule.step(target=target, losses=[])
        else:
            schedule.step()
    except Exception as exc:  # noqa: BLE001
        log.error("multi-stream step failed", err=f"{type(exc).__name__}: {exc}")
        shutdown()
        return

    grad = stage_mod.term_scale.grad
    log.info(
        "accumulator gradient",
        term_grad=None if grad is None else float(grad),
        expect=float(args.microbatches),
        carrier=args.carrier,
    )
    if grad is None:
        log.error("accumulator term received NO gradient")
    elif abs(float(grad) - args.microbatches) < 1e-3:
        log.info("OK: each stage's term got gradient 1 per microbatch")
    else:
        log.error("accumulator gradient wrong", got=float(grad))
    shutdown()


def launch() -> None:
    """Run ``main`` on ``GPUS`` spawned ranks, or in place when ranks exist."""
    args = parse_args()
    if os.environ.get("RANK") is not None:
        main(args)
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=GPUS or torch.cuda.device_count(),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="streams_probe",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)(args)


if __name__ == "__main__":
    launch()
