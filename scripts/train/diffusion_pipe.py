"""Flow-matching diffusion on kohakuwupipe: the worked example for ``target``.

Launched and configured by KohakuEngine; the script spawns its own ranks::

    kogine run scripts/train/diffusion_pipe.py
    kogine run scripts/train/diffusion_pipe.py --set MAX_STEPS=50 --set GPUS=2

An LM is the degenerate case of the pipeline contract -- one input, one target,
both derived from the same token array. Diffusion is not, and it is the shortest
example that forces every part of the contract into the open:

* the model is conditioned on **two** tensors, the noised sample and the
  timestep, so ``inputs`` is a tuple;
* the thing the loss compares against is **neither** of them, so ``target``
  carries a dict of velocity and per-sample loss weight;
* the timestep is an input rather than part of the target, because the stage
  needs it to compute anything at all.

See docs/kohakuwupipe/module.md for the contract this demonstrates.
"""

import math
import os
import sys

import torch
import torch.nn as nn
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwupipe import (
    LossLog,
    PipelineModule,
    PipelineTrainer,
    ProgressBar,
    StagePlan,
    Throughput,
    get_logger,
    init_pipeline,
    plan_stages,
    shutdown,
)

torch.set_float32_matmul_precision("high")
log = get_logger("diffusion_pipe")

DIM = 512
DEPTH = 16
DATA_DIM = 64
BATCH = 256
NUM_MICROBATCHES = 8
LR = 3e-4
GRAD_CLIP = 1.0
MAX_STEPS = 200
SEED = 20090220
LOG_INTERVAL = 1
CONSOLE_INTERVAL = 25
THROUGHPUT_INTERVAL = 25
PROGRESS_BAR = True
GPUS = 0


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features, projected to the model width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        angles = t.float()[:, None] * freqs[None, :]
        return self.proj(torch.cat([angles.cos(), angles.sin()], dim=-1))


class Block(nn.Module):
    """Pre-norm MLP block, modulated by the timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x) * (1 + self.scale(cond)))


class DiffusionStage(nn.Module):
    """One pipeline stage: some blocks, plus the ends when it owns them.

    The boundary is ``(hidden, cond)`` on every stage, so the timestep
    conditioning reaches the blocks that are not on rank 0. See
    docs/kohakuwupipe/streams.md.
    """

    def __init__(self, plan: StagePlan, dim: int, data_dim: int) -> None:
        super().__init__()
        self.plan = plan
        self.time_embed = TimestepEmbedding(dim) if plan.has_embed else None
        self.in_proj = nn.Linear(data_dim, dim) if plan.has_embed else None
        self.blocks = nn.ModuleList(Block(dim) for _ in range(plan.num_layers))
        self.out_norm = nn.LayerNorm(dim) if plan.has_head else None
        self.out_proj = nn.Linear(dim, data_dim) if plan.has_head else None

    def loss(self, hidden: torch.Tensor, target: dict):
        """Flow-matching regression, weighted per sample.

        ``target`` is whatever the step carried, with its structure intact --
        here a dict, which a plain schedule target could not be.
        """
        error = nn.functional.mse_loss(
            hidden.float(), target["velocity"].float(), reduction="none"
        )
        return (error.mean(-1) * target["weight"]).sum(), {}

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        """``(noised, timestep) -> (hidden, cond)``; the last stage returns velocity."""
        if self.plan.has_embed:
            cond = self.time_embed(t)
            x = self.in_proj(x)
        else:
            x, cond = x, t
        for block in self.blocks:
            x = block(x, cond)
        if not self.plan.has_head:
            return x, cond
        return self.out_proj(self.out_norm(x))


class DiffusionModule(PipelineModule):
    """The kohakuwupipe contract for a flow-matching model."""

    def __init__(self, dim: int, data_dim: int, batch: int, lr: float) -> None:
        super().__init__()
        self.dim = dim
        self.data_dim = data_dim
        self.batch = batch
        self.lr = lr
        self.inner = None

    def configure_model(self, plan, rank, world, device) -> nn.Module:
        self.inner = DiffusionStage(plan, self.dim, self.data_dim).to(device)
        return self.inner

    def boundary_example(self, plan, device):
        """What this rank receives. Rank 0 takes the pair the caller sends."""
        rows = self.batch // NUM_MICROBATCHES
        if plan.has_embed:
            return (
                torch.zeros(rows, self.data_dim, device=device),
                torch.zeros(rows, dtype=torch.long, device=device),
            )
        return (
            torch.zeros(rows, self.dim, device=device),
            torch.zeros(rows, self.dim, device=device),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.stage_module.parameters(), lr=self.lr)


def flow_batch(batch: int, data_dim: int, device, generator):
    """One flow-matching step: ``((noised, timestep), {velocity, weight})``.

    ``x1`` stands in for data and ``x0`` for noise; the model learns the
    straight-line velocity ``x1 - x0`` at the interpolated point.
    """
    x1 = torch.randn(batch, data_dim, device=device, generator=generator)
    x0 = torch.randn(batch, data_dim, device=device, generator=generator)
    t = torch.rand(batch, device=device, generator=generator)
    noised = (1 - t)[:, None] * x0 + t[:, None] * x1
    velocity = x1 - x0
    weight = 1.0 + t
    return (noised, (t * 1000).long()), {"velocity": velocity, "weight": weight}


class FlowStep:
    """A :class:`kohakuwupipe.MicrobatchStep` for the flow-matching batch above.

    ``inputs`` is a tuple because the stage takes two arguments; ``target`` is
    a dict because the loss needs both the velocity and its weight.
    """

    def __init__(self, inputs, target, trained: int) -> None:
        self.inputs = inputs
        self.target = target
        self.layout = None
        self.trained = trained


def stream(batch: int, data_dim: int, device, seed: int):
    """An endless supply of flow-matching steps, identical on every rank."""
    generator = torch.Generator(device=device).manual_seed(seed)
    while True:
        inputs, target = flow_batch(batch, data_dim, device, generator)
        yield FlowStep(inputs, target, target["velocity"].numel())


def build_reporter():
    """Log on ``CONSOLE_INTERVAL``; the progress bar owns stdout."""

    def report(row: dict) -> None:
        step = int(row.get("step", 0))
        if CONSOLE_INTERVAL > 0 and step % CONSOLE_INTERVAL == 0:
            log.info("metrics", **row)

    return report


def main() -> None:
    ranks = init_pipeline()
    torch.manual_seed(SEED)

    # Uniform blocks and no LM head: every layer costs the same, the ends nothing.
    plans = plan_stages(DEPTH, ranks.world, layer_cost=1.0, head_cost=0.0)
    if ranks.rank == 0:
        log.info("stages", split=[p.num_layers for p in plans])

    module = DiffusionModule(DIM, DATA_DIM, BATCH, LR)
    report = build_reporter() if ranks.rank == 0 else None
    callbacks = [
        Throughput(every_n_steps=THROUGHPUT_INTERVAL, warmup_steps=4, report=report),
        LossLog(every_n_steps=LOG_INTERVAL, report=report),
    ]
    if PROGRESS_BAR:
        callbacks.append(ProgressBar(postfix=()))

    trainer = PipelineTrainer(
        module,
        ranks,
        plans,
        micro_tokens=BATCH // NUM_MICROBATCHES,
        num_microbatches=NUM_MICROBATCHES,
        grad_clip=GRAD_CLIP,
        callbacks=callbacks,
    )
    trainer.fit(stream(BATCH, DATA_DIM, ranks.device, SEED), max_steps=MAX_STEPS)
    log.info("done", steps=trainer.loop.global_step)
    shutdown()


def launch() -> None:
    """Run ``main`` under torchrun when the caller did not."""
    if os.environ.get("RANK") is not None:
        main()
        return
    nproc = GPUS or torch.cuda.device_count()
    if nproc <= 1:
        main()
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="diffusion_pipe",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, _worker)(
        {k: v for k, v in vars(sys.modules[__name__]).items() if k.isupper()}
    )


def _worker(overrides: dict) -> None:
    module = sys.modules[__name__]
    for key, value in overrides.items():
        setattr(module, key, value)
    main()


if __name__ == "__main__":
    launch()
