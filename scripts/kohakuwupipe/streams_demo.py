"""Three multi-stream pipelines that work, and the caveat each one hits.

    torchrun --standalone --nproc_per_node=4 scripts/kohakuwupipe/streams_demo.py --case aux
    torchrun --standalone --nproc_per_node=4 scripts/kohakuwupipe/streams_demo.py --case dit
    torchrun --standalone --nproc_per_node=4 scripts/kohakuwupipe/streams_demo.py --case unet

* ``aux``  -- every stage adds its own auxiliary loss to an accumulator stream.
* ``dit``  -- every stage cross-attends the same text context, carried as a stream.
* ``unet`` -- an early stage's activation is carried to a later one as a skip.

Pseudo-models, real boundaries: each asserts the gradient its caveat is about.
See docs/streams.md.
"""

import argparse

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

from kohakuwupipe import get_logger, init_pipeline, shutdown
from kohakuwupipe.parallel.streams import GradCarrier, reduce_accumulator

log = get_logger("streams_demo")
DIM, ROWS, CTX = 64, 32, 8


class AuxStage(nn.Module):
    """Hidden state plus an accumulator every stage contributes to."""

    def __init__(self, **_):
        super().__init__()
        self.blk = nn.Linear(DIM, DIM)
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, hidden, aux):
        hidden = hidden + torch.tanh(self.blk(hidden))
        # This stage's aux term. Its gradient must come back as 1.
        return hidden, aux + self.gate

    def examples(self, dev):
        return (torch.zeros(ROWS, DIM, device=dev), torch.zeros(ROWS, 1, device=dev))

    def check(self, world, microbatches):
        return "gate", float(ROWS * microbatches)


class DiTStage(nn.Module):
    """Every stage cross-attends one shared text context, carried as a stream."""

    def __init__(self, carrier: bool, **_):
        super().__init__()
        self.q = nn.Linear(DIM, DIM)
        self.kv = nn.Linear(DIM, DIM)
        self.gate = nn.Parameter(torch.ones(()))
        # Without this the context boundary has no backward edge at all.
        self.carrier = GradCarrier() if carrier else None

    def forward(self, hidden, context, aux):
        attn = torch.softmax(self.q(hidden) @ self.kv(context).transpose(-1, -2), -1)
        hidden = hidden + attn @ context
        if self.carrier is not None:
            context = self.carrier(context)
        return hidden, context, aux + self.gate

    def examples(self, dev):
        return (
            torch.zeros(ROWS, DIM, device=dev),
            torch.zeros(ROWS, DIM, device=dev),
            torch.zeros(ROWS, 1, device=dev),
        )

    def check(self, world, microbatches):
        return "gate", float(ROWS * microbatches)


class UNetStage(nn.Module):
    """An early stage's activation is carried past the middle to a later one."""

    def __init__(self, first: bool, last: bool, **_):
        super().__init__()
        self.first, self.last = first, last
        self.blk = nn.Linear(DIM, DIM)
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, hidden, skip, aux):
        hidden = hidden + torch.tanh(self.blk(hidden))
        if self.first:
            skip = hidden * 1.0
        elif self.last:
            hidden = hidden + skip
        return hidden, skip, aux + self.gate

    def examples(self, dev):
        return (
            torch.zeros(ROWS, DIM, device=dev),
            torch.zeros(ROWS, DIM, device=dev),
            torch.zeros(ROWS, 1, device=dev),
        )

    def check(self, world, microbatches):
        return "gate", float(ROWS * microbatches)


CASES = {"aux": AuxStage, "dit": DiTStage, "unet": UNetStage}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), default="aux")
    ap.add_argument("--microbatches", type=int, default=4)
    ap.add_argument("--no-carrier", dest="carrier", action="store_false", default=True)
    args = ap.parse_args()

    ranks = init_pipeline()
    first, last = ranks.rank == 0, ranks.rank == ranks.world - 1
    module = CASES[args.case](first=first, last=last, carrier=args.carrier)
    module = module.to(ranks.device)

    dev = ranks.device
    examples = module.examples(dev)
    stage = PipelineStage(module, ranks.rank, ranks.world, dev, input_args=examples)

    def loss_fn(output, target):
        streams = output if isinstance(output, tuple) else (output,)
        loss = (streams[0] - target).pow(2).mean()
        for stream in streams[1:]:
            # Accumulators join the loss; a dense gradient is mandatory.
            if stream.shape[-1] == 1:
                loss = loss + reduce_accumulator(stream)
        return loss

    schedule = ScheduleGPipe(
        stage, n_microbatches=args.microbatches, loss_fn=loss_fn, scale_grads=False
    )
    n = ROWS * args.microbatches
    seeds = [torch.randn(n, e.shape[-1], device=dev) for e in examples]
    seeds[0] = seeds[0].requires_grad_()
    target = torch.randn(n, DIM, device=dev)

    module.zero_grad(set_to_none=True)
    try:
        if first:
            schedule.step(*seeds)
        elif last:
            schedule.step(target=target, losses=[])
        else:
            schedule.step()
    except Exception as exc:  # noqa: BLE001
        log.error(f"{args.case} failed", err=f"{type(exc).__name__}: {str(exc)[:110]}")
        shutdown()
        return

    name, expect = module.check(ranks.world, args.microbatches)
    grad = getattr(module, name).grad
    got = None if grad is None else float(grad)
    log.info(f"{args.case}", param=name, grad=got, expect=expect)
    if got is None:
        log.error("no gradient reached this stage's term")
    elif abs(got - expect) < 1e-2:
        log.info(f"OK: {args.case} — every stage's term got gradient 1 per element")
    else:
        log.error("gradient wrong", got=got, expect=expect)
    shutdown()


if __name__ == "__main__":
    main()
