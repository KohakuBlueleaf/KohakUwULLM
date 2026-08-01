"""Flow-matching diffusion on kohakuwupipe: the worked example for the contract.

Launched and configured by KohakuEngine; the script spawns its own ranks::

    kogine run scripts/train/diffusion_pipe.py
    kogine run scripts/train/diffusion_pipe.py --set MAX_STEPS=50 --set GPUS=2
    kogine run scripts/train/diffusion_pipe.py --set SCHEDULE=gpipe --set AUTOCAST_DTYPE=fp16

An LM is the degenerate case of the contract -- one input, one target, both from
the same token array. Diffusion is not, and it exercises every part.
``DiffusionModule.training_step`` has the shape a ``LightningModule`` does::

    clean = batch.inputs                       # whatever the loader gave
    noised, t = ...                            # preprocess, on the fly
    loss = self.forward_backward(              # the only part this package owns
        inputs=(noised, t),                    # a tuple: the stage takes two
        target={"velocity": ..., "weight": ...},   # any structure
    )
    self.log("weighted_mse", loss)             # post-process and log

The one place it cannot match Lightning is that ``training_step`` never holds
the model output: a 1F1B schedule interleaves microbatch forwards and backwards
across ranks, so the objective arrives as :meth:`DiffusionModule.loss`, once per
microbatch. Anything differentiable goes there; ``forward_backward`` returns the
step's reduced loss for logging only.

Every ``PipelineModule`` hook and every trainer knob is used below, so this file
doubles as the API tour. See docs/kohakuwupipe/module.md.
"""

import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from anyschedule import AnySchedule
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwupipe import (
    Callback,
    Checkpoint,
    LossLog,
    MicrobatchStep,
    PipelineGradScaler,
    PipelineModule,
    PipelineTrainer,
    ProgressBar,
    StagePlan,
    Throughput,
    describe,
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
# "1f1b" or "gpipe".
SCHEDULE = "1f1b"
PARAM_DTYPE = "fp32"
# Forward autocast; "" runs in PARAM_DTYPE. fp16 here turns the scaler on.
AUTOCAST_DTYPE = ""
LR = 3e-4
# AnySchedule config; "end" and "warmup" are filled from MAX_STEPS below.
SCHEDULER_CONFIG: dict = {"lr": {"mode": "cosine", "min_value": 0.05}}
SCHED_WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
MAX_STEPS = 200
VAL_EVERY = 50
VAL_BATCHES = 4
SEED = 20090220
CKPT_DIR = ""
CKPT_INTERVAL = 100
CHECKPOINT_PATH: str | None = None
LOG_INTERVAL = 1
CONSOLE_INTERVAL = 25
THROUGHPUT_INTERVAL = 25
PROGRESS_BAR = True
GPUS = 0

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass(frozen=True)
class DiffusionStagePlan(StagePlan):
    """A stage plan that names this model's ends, as an LM names its own."""

    @property
    def has_time_embed(self) -> bool:
        return self.is_first

    @property
    def has_out_proj(self) -> bool:
        return self.is_last


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
        features = torch.cat([angles.cos(), angles.sin()], dim=-1)
        # The sinusoid is built in fp32 whatever the parameters are stored in.
        return self.proj(features.to(self.proj[0].weight.dtype))


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
    conditioning reaches blocks that are not on rank 0.
    See docs/kohakuwupipe/streams.md.
    """

    def __init__(
        self,
        plan: DiffusionStagePlan,
        dim: int,
        data_dim: int,
        autocast_dtype: torch.dtype | None = None,
        boundary_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.autocast_dtype = autocast_dtype
        self.boundary_dtype = boundary_dtype
        self.time_embed = TimestepEmbedding(dim) if plan.has_time_embed else None
        self.in_proj = nn.Linear(data_dim, dim) if plan.has_time_embed else None
        self.blocks = nn.ModuleList(Block(dim) for _ in range(plan.num_layers))
        self.out_norm = nn.LayerNorm(dim) if plan.has_out_proj else None
        self.out_proj = nn.Linear(dim, data_dim) if plan.has_out_proj else None
        self.dropout_p = 0.0

    def set_seq_info(self, layout) -> None:
        """Per-step layout, whatever the module chooses to carry.

        The loop calls this with ``batch.layout`` before every step; here it
        switches dropout, which is the smallest thing that proves it arrives.
        """
        self.dropout_p = 0.0 if layout is None else float(layout.get("dropout", 0.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        """``(noised, timestep) -> (hidden, cond)``; the last stage returns velocity.

        Runs under the stage's autocast, and leaves the boundary in the dtype
        ``boundary_example`` declared.
        """
        context = (
            nullcontext()
            if self.autocast_dtype is None
            else torch.autocast("cuda", dtype=self.autocast_dtype)
        )
        with context:
            out = self._blocks(x, t)
        if self.boundary_dtype is None or self.out_proj is not None:
            return out
        return tuple(stream.to(self.boundary_dtype) for stream in out)

    def _blocks(self, x: torch.Tensor, t: torch.Tensor | None):
        if self.time_embed is not None:
            cond = self.time_embed(t)
            x = self.in_proj(x)
        else:
            x, cond = x, t
        for block in self.blocks:
            x = block(x, cond)
            if self.dropout_p:
                x = nn.functional.dropout(x, self.dropout_p, self.training)
        if self.out_proj is None:
            return x, cond
        return self.out_proj(self.out_norm(x))


class DiffusionModule(PipelineModule):
    """The kohakuwupipe contract, with every hook it offers used at least once."""

    def __init__(self, dim: int, data_dim: int, batch: int, lr: float) -> None:
        super().__init__()
        self.dim = dim
        self.data_dim = data_dim
        self.batch = batch
        self.lr = lr
        self.inner: DiffusionStage | None = None
        self.samples_seen = 0

    # -- construction ----------------------------------------------------- #

    def configure_model(self, plan, rank, world, device) -> nn.Module:
        """Build this rank's slice and put it in the run's parameter dtype."""
        stage = DiffusionStage(
            plan,
            self.dim,
            self.data_dim,
            autocast_dtype=_DTYPES[AUTOCAST_DTYPE] if AUTOCAST_DTYPE else None,
            boundary_dtype=_DTYPES[PARAM_DTYPE],
        ).to(device)
        if _DTYPES[PARAM_DTYPE] is not torch.float32:
            stage = stage.to(dtype=_DTYPES[PARAM_DTYPE])
        self.inner = stage
        return stage

    def configure_optimizers(self):
        """``(optimizer, scheduler)``; the loop steps the scheduler each step.

        AnySchedule reads one config per parameter-group key and scales each
        group from its own base, so a per-group lr split survives the schedule.
        """
        optimizer = torch.optim.AdamW(self.stage_module.parameters(), lr=self.lr)
        config = {
            key: {
                **sub,
                "end": MAX_STEPS,
                "warmup": int(MAX_STEPS * SCHED_WARMUP_RATIO),
            }
            for key, sub in SCHEDULER_CONFIG.items()
        }
        return optimizer, AnySchedule(optimizer, config=config)

    def boundary_example(self, plan, device):
        """The shape and dtype ``PipelineStage`` freezes for this rank's input."""
        rows = self.batch // NUM_MICROBATCHES
        dtype = _DTYPES[PARAM_DTYPE]
        if plan.has_time_embed:
            return (
                torch.zeros(rows, self.data_dim, dtype=dtype, device=device),
                torch.zeros(rows, dtype=torch.long, device=device),
            )
        return (
            torch.zeros(rows, self.dim, dtype=dtype, device=device),
            torch.zeros(rows, self.dim, dtype=dtype, device=device),
        )

    @property
    def block_prefix(self) -> str:
        """Where the blocks live in the stage's state dict, for checkpoints."""
        return "blocks"

    # -- the step --------------------------------------------------------- #

    def training_step(self, batch, batch_idx: int) -> None:
        """Preprocess, run the model, log -- the LightningModule shape.

        Everything before :meth:`forward_backward` is ordinary eager code on
        this rank: a real trainer would VAE-encode here, or sample a schedule.
        """
        clean = batch.inputs
        noise = torch.randn_like(clean)
        t = torch.rand(clean.shape[0], device=clean.device)
        noised = (1 - t)[:, None] * noise + t[:, None] * clean
        # The subclass's own field arrived on the device with the rest.
        self.log("cond_mean", batch.cond.float().mean())

        loss = self.forward_backward(
            inputs=(noised.to(clean.dtype), (t * 1000).long()),
            target={"velocity": clean - noise, "weight": 1.0 + t},
        )

        self.samples_seen += clean.shape[0]
        self.log("t_mean", t.mean())
        # Only the last stage computes a loss; the others get None.
        if loss is not None:
            self.log("weighted_mse", loss)

    def loss(self, hidden: torch.Tensor, target: dict):
        """The objective, one microbatch at a time.

        Anything differentiable between the model output and the number goes
        here, because the schedule hangs backward off this call. Per-term
        breakdowns go through :meth:`log`, which averages over the microbatches.
        """
        error = nn.functional.mse_loss(
            hidden.float(), target["velocity"].float(), reduction="none"
        ).mean(-1)
        self.log("mse_unweighted", error.mean().detach())
        return (error * target["weight"]).sum()

    def validation_step(self, batch, batch_idx: int):
        """The same pass with backward off. ``t`` is fixed so runs compare.

        ``no_grad`` is the knob: ``forward_backward`` sees it and runs the
        schedule forward-only.
        """
        clean = batch.inputs
        with torch.no_grad():
            noise = torch.randn_like(clean)
            t = torch.full((clean.shape[0],), 0.5, device=clean.device)
            noised = (1 - t)[:, None] * noise + t[:, None] * clean
            return self.forward_backward(
                inputs=(noised.to(clean.dtype), (t * 1000).long()),
                target={"velocity": clean - noise, "weight": torch.ones_like(t)},
            )

    def post_step(self, stage_module) -> dict[str, torch.Tensor]:
        """After ``optimizer.step``: device tensors that join the step's metrics."""
        norm = sum(
            p.detach().float().pow(2).sum() for p in stage_module.parameters()
        ).sqrt()
        return {"param_norm": norm}

    # -- checkpoint and lifecycle ----------------------------------------- #

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["samples_seen"] = self.samples_seen

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.samples_seen = int(checkpoint.get("samples_seen", 0))

    def on_train_start(self) -> None:
        log.info("training", rank=self.rank, seen=self.samples_seen)

    def on_train_end(self) -> None:
        log.info("finished", rank=self.rank, seen=self.samples_seen)


@dataclass
class DiffusionStep(MicrobatchStep):
    """A step that also carries a class label, as a conditioned model would.

    ``cond`` is an ordinary field: :meth:`MicrobatchStep.to` and ``pin_memory``
    move it with the rest, and ``training_step`` reads it like any other.
    """

    cond: torch.Tensor | None = None


class Validate(Callback):
    """Run a validation pass every ``every_n_steps`` and report its mean loss.

    Only the last stage's :meth:`DiffusionModule.validation_step` returns a
    number, so the mean goes through ``loop.broadcast_loss`` to reach every rank.
    Collective: every rank runs every pass.
    """

    def __init__(self, batches, every_n_steps: int, count: int) -> None:
        self.batches = batches
        self.every_n_steps = every_n_steps
        self.count = count
        self._seen: list[torch.Tensor] = []

    def on_train_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        loop.validate(next(self.batches) for _ in range(self.count))

    def on_validation_epoch_start(self, loop) -> None:
        self._seen = []

    def on_validation_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        self._seen.append(out)

    def on_validation_epoch_end(self, loop) -> None:
        losses = [x for x in self._seen if x is not None]
        mean = loop.broadcast_loss(losses or None, scale=max(self.count, 1))
        if mean is not None and loop.rank == 0:
            log.info("validation", step=loop.global_step, val_loss=float(mean))


def stream(batch: int, data_dim: int, device, seed: int, dropout: float = 0.0):
    """Clean samples and their labels, identical on every rank.

    Built on the CPU and moved with :meth:`MicrobatchStep.to`, which is what a
    real loader does and what makes the extra field worth having.
    """
    generator = torch.Generator().manual_seed(seed)
    dtype = _DTYPES[PARAM_DTYPE]
    while True:
        clean = torch.randn(batch, data_dim, generator=generator)
        yield DiffusionStep(
            inputs=clean.to(dtype),
            layout={"dropout": dropout},
            # What `loss` is normalized by: one term per sample, not per element.
            trained=clean.shape[0],
            cond=torch.randint(0, 10, (batch,), generator=generator),
        ).to(device)


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
    plans = plan_stages(
        DEPTH, ranks.world, layer_cost=1.0, head_cost=0.0, plan_cls=DiffusionStagePlan
    )
    if ranks.rank == 0:
        log.info("stage split\n%s", describe(plans))

    module = DiffusionModule(DIM, DATA_DIM, BATCH, LR)
    report = build_reporter() if ranks.rank == 0 else None
    callbacks = [
        Throughput(every_n_steps=THROUGHPUT_INTERVAL, warmup_steps=4, report=report),
        LossLog(every_n_steps=LOG_INTERVAL, report=report),
    ]
    if PROGRESS_BAR:
        callbacks.append(ProgressBar(postfix=("param_norm",)))
    if VAL_EVERY:
        validation = stream(BATCH, DATA_DIM, ranks.device, SEED + 1)
        callbacks.append(Validate(validation, VAL_EVERY, VAL_BATCHES))
    if CKPT_DIR:
        callbacks.append(
            Checkpoint(
                CKPT_DIR,
                plans[ranks.rank].start_layer,
                CKPT_INTERVAL,
                block_attr=module.block_prefix,
            )
        )

    trainer = PipelineTrainer(
        module,
        ranks,
        plans,
        micro_tokens=BATCH // NUM_MICROBATCHES,
        num_microbatches=NUM_MICROBATCHES,
        schedule=SCHEDULE,
        grad_clip=GRAD_CLIP,
        # fp16 gradients underflow without it; disabled costs nothing.
        scaler=PipelineGradScaler(enabled="fp16" in (PARAM_DTYPE, AUTOCAST_DTYPE)),
        callbacks=callbacks,
    )
    if CHECKPOINT_PATH:
        trainer.load_checkpoint(CHECKPOINT_PATH)

    trainer.fit(stream(BATCH, DATA_DIM, ranks.device, SEED, 0.1), max_steps=MAX_STEPS)

    if CKPT_DIR:
        trainer.save_checkpoint(f"{CKPT_DIR}/last.ckpt")
    log.info(
        "done",
        steps=trainer.loop.global_step,
        samples=trainer.loop.tokens_trained,
    )
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
