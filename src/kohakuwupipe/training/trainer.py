"""``PipelineTrainer``: builds the stage, the schedule and the loop, then fits.

The ``pl.Trainer`` equivalent. It owns construction order: cast before the
stage, stage before the schedule. See docs/kohakuwupipe/module.md.
"""

from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleDualPipeV,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
    ScheduleInterleavedZeroBubble,
    ScheduleLoopedBFS,
    ScheduleZBVZeroBubble,
)

from kohakuwupipe.io import checkpoint
from kohakuwupipe.parallel.distributed import PipelineRanks, warn_on_eager_init
from kohakuwupipe.training.chunks import local_stages
from kohakuwupipe.training.hooks import CallbackList
from kohakuwupipe.training.loop import PipelineLoop, build_loss_fn
from kohakuwupipe.training.module import PipelineModule
from kohakuwupipe.utils.logging import configure, get_logger

log = get_logger(__name__)


def _objective(module):
    """The module's own ``loss``, or ``None`` when it leaves it to the stage."""
    if type(module).loss is PipelineModule.loss:
        return None
    return module.loss


# One stage per rank.
SINGLE_SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}

# Several stages per rank, with the placement style torch maps them by. The "v"
# ones take exactly two. See docs/kohakuwupipe/schedules.md.
MULTI_SCHEDULES = {
    "interleaved-1f1b": (ScheduleInterleaved1F1B, "loop"),
    "looped-bfs": (ScheduleLoopedBFS, "loop"),
    "interleaved-zb": (ScheduleInterleavedZeroBubble, "loop"),
    "zbv": (ScheduleZBVZeroBubble, "v"),
    "dualpipe-v": (ScheduleDualPipeV, "v"),
}

SCHEDULES = {**SINGLE_SCHEDULES, **MULTI_SCHEDULES}


def stage_style(schedule: str) -> str | None:
    """``"v"`` / ``"loop"`` for a multi-stage schedule, ``None`` for a single one."""
    entry = MULTI_SCHEDULES.get(schedule)
    return None if entry is None else entry[1]


def stages_per_rank(schedule: str, virtual: int = 2) -> int:
    """How many chunks each rank owns under ``schedule``."""
    style = stage_style(schedule)
    if style is None:
        return 1
    return 2 if style == "v" else virtual


class PipelineTrainer:
    """Drives a :class:`PipelineModule` across stages.

    Args:
        module: the model definition.
        ranks: from :func:`kohakuwupipe.init_pipeline`.
        plans: one :class:`StagePlan` per stage.
        micro_tokens / num_microbatches: the step's shape.
        schedule: a key of :data:`SCHEDULES`.
        grad_clip: max grad norm; 0 disables.
        callbacks: see :class:`kohakuwupipe.Callback`.
    """

    def __init__(
        self,
        module,
        ranks: PipelineRanks,
        plans,
        micro_tokens: int,
        num_microbatches: int,
        schedule: str = "1f1b",
        grad_clip: float = 0.0,
        scaler=None,
        callbacks=(),
    ) -> None:
        configure(rank=ranks.rank)
        if problem := warn_on_eager_init():
            log.warning(problem)
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; have {sorted(SCHEDULES)}")

        self.module = module
        self.ranks = ranks
        self.plans = plans
        self.style = stage_style(schedule)
        self.micro_tokens = micro_tokens
        self.num_microbatches = num_microbatches
        self.callbacks = CallbackList(callbacks)
        self.scaler = scaler
        self._denom = 1
        # Where the blocks sit in the stage's state dict, wrapper included.
        self.block_attr = getattr(module, "block_prefix", "blocks")

        if self.style is None:
            if len(plans) != ranks.world:
                raise ValueError(
                    f"schedule {schedule!r} takes one stage per rank, so it needs "
                    f"{ranks.world} plans, got {len(plans)}"
                )
            self.plan = plans[ranks.rank]
            self.local_indices = [ranks.rank]
            self.local_plans = [self.plan]
            stage_module = module.setup(
                self.plan, ranks.rank, ranks.world, ranks.device
            )
            stages = [self._build_stage(module, self.plan, stage_module, len(plans))]
        else:
            self.local_indices = local_stages(
                len(plans), ranks.rank, ranks.world, self.style
            )
            local = [plans[index] for index in self.local_indices]
            self.plan = local[0]
            self.local_plans = local
            stage_module = module.setup_chunks(
                local, ranks.rank, ranks.world, ranks.device
            )
            stages = [
                self._build_stage(module, plan, chunk, len(plans))
                for plan, chunk in zip(local, stage_module.chunks)
            ]
        optimizers = module.configure_optimizers()
        optimizer, scheduler = (
            optimizers if isinstance(optimizers, tuple) else (optimizers, None)
        )
        self.optimizer = optimizer
        # One list, shared with the loop, which refills it every step.
        pieces: list = [None] * num_microbatches
        factory = (
            SINGLE_SCHEDULES[schedule]
            if self.style is None
            else MULTI_SCHEDULES[schedule][0]
        )
        self.schedule = factory(
            stages[0] if self.style is None else stages,
            n_microbatches=num_microbatches,
            loss_fn=build_loss_fn(
                stage_module,
                lambda: self._denom,
                num_microbatches,
                scaler,
                pieces,
                _objective(module),
            ),
            scale_grads=False,
        )
        self.loop = PipelineLoop(
            stage_module,
            self.schedule,
            optimizer,
            ranks.rank,
            ranks.world,
            micro_tokens=micro_tokens,
            num_microbatches=num_microbatches,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
            post_step=module.post_step,
            callbacks=self.callbacks.callbacks,
            is_first=any(plan.is_first for plan in self.local_plans),
            is_last=any(plan.is_last for plan in self.local_plans),
        )
        self.loop.target_pieces = pieces
        self.loop.trainer = self
        self.loop.module = module
        module._loop = self.loop
        # One list with the loop, so a later `append` reaches what calls it.
        self.callbacks = self.loop.callbacks

    def _build_stage(self, module, plan, stage_module, num_stages: int):
        """One torch ``PipelineStage`` for a chunk, at its own global index."""
        # A tuple boundary is a multi-stream stage; a tensor is the plain case.
        example = module.boundary_example(plan, self.ranks.device)
        input_args = example if isinstance(example, tuple) else (example,)
        return PipelineStage(
            stage_module,
            plan.index,
            num_stages,
            self.ranks.device,
            input_args=input_args,
        )

    def fit(self, batches, max_steps: int) -> None:
        """Train for ``max_steps``. Every rank consumes the same stream."""
        self.module.on_train_start()
        self.loop.fit(self._with_denominator(batches), max_steps)
        self.module.on_train_end()

    def _with_denominator(self, batches):
        """Set the loss divisor from each step before the schedule sees it."""
        for batch in batches:
            self._denom = max(int(batch.trained), 1)
            yield batch

    def save_checkpoint(self, path: str) -> bool:
        """Whole-model checkpoint, Lightning-shaped. Collective.

        Carries everything a resume needs beyond the weights: the loop's
        progress, the LR schedule, the loss scaler, the module's and the
        callbacks' own state. See docs/kohakuwupipe/checkpoint.md.
        """
        if self.style is not None:
            raise NotImplementedError(
                "checkpointing a multi-stage-per-rank schedule is not wired: "
                "`global_names` maps local to global parameter names from one "
                "`start_layer`, and this rank holds chunks "
                f"{self.local_indices}, which are not contiguous"
            )
        payload: dict = {}
        self.callbacks.call("on_save_checkpoint", self.loop, payload)
        self.module.on_save_checkpoint(payload)
        payload["callbacks"] = self.callbacks.state_dict()
        payload["progress"] = self.loop.progress_state()
        if self.loop.scheduler is not None:
            payload["lr_schedulers"] = [self.loop.scheduler.state_dict()]
        if self.scaler is not None:
            payload["grad_scaler"] = self.scaler.state_dict()
        wrote = checkpoint.save(
            path,
            self.loop.stage_module,
            self.optimizer,
            self.plan.start_layer,
            self.loop.global_step,
            self.ranks.rank,
            block_attr=self.block_attr,
            extra=payload,
        )
        if wrote:
            log.info("checkpoint written", path=path, step=self.loop.global_step)
        return wrote

    def load_checkpoint(self, path: str, strict: bool = True) -> dict:
        """Restore this rank's slice and everything :meth:`save_checkpoint` wrote."""
        payload = checkpoint.load(
            path,
            self.loop.stage_module,
            self.optimizer,
            self.plan.start_layer,
            self.ranks.rank,
            block_attr=self.block_attr,
            strict=strict,
        )
        self.loop.global_step = int(payload.get("global_step", 0))
        self.loop.load_progress_state(payload.get("progress"))
        schedules = payload.get("lr_schedulers") or []
        if self.loop.scheduler is not None and schedules:
            self.loop.scheduler.load_state_dict(schedules[0])
        elif self.loop.scheduler is not None:
            log.warning(
                "checkpoint carries no LR schedule; this run restarts its own "
                "from step 0, warmup included",
                step=self.loop.global_step,
            )
        if self.scaler is not None and "grad_scaler" in payload:
            self.scaler.load_state_dict(payload["grad_scaler"])
        self.callbacks.load_state_dict(payload.get("callbacks", {}))
        self.callbacks.call("on_load_checkpoint", self.loop, payload)
        self.module.on_load_checkpoint(payload)
        log.info("checkpoint restored", path=path, step=self.loop.global_step)
        return payload
