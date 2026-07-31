"""``PipelineTrainer``: builds the stage, the schedule and the loop, then fits.

The ``pl.Trainer`` equivalent. It owns construction order -- cast before the
stage, stage before the schedule -- which is what a caller most often gets
wrong. See docs/kohakuwupipe/module.md.
"""

from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

from kohakuwupipe.io import checkpoint
from kohakuwupipe.parallel.distributed import PipelineRanks, warn_on_eager_init
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


# Interleaved schedules want several stage chunks per rank; this trainer
# builds one. See docs/internals/pipeline.md.
SCHEDULES = {"1f1b": Schedule1F1B, "gpipe": ScheduleGPipe}


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
        self.plan = plans[ranks.rank]
        self.plans = plans
        self.micro_tokens = micro_tokens
        self.num_microbatches = num_microbatches
        self.callbacks = CallbackList(callbacks)
        self.scaler = scaler
        self._denom = 1
        # Where the blocks sit in the stage's state dict, wrapper included.
        self.block_attr = getattr(module, "block_prefix", "blocks")

        stage_module = module.setup(self.plan, ranks.rank, ranks.world, ranks.device)
        # A tuple boundary is a multi-stream stage; a tensor is the plain case.
        example = module.boundary_example(self.plan, ranks.device)
        input_args = example if isinstance(example, tuple) else (example,)
        stage = PipelineStage(
            stage_module, ranks.rank, ranks.world, ranks.device, input_args=input_args
        )
        optimizers = module.configure_optimizers()
        optimizer, scheduler = (
            optimizers if isinstance(optimizers, tuple) else (optimizers, None)
        )
        self.optimizer = optimizer
        # One list, shared with the loop, which refills it every step.
        pieces: list = [None] * num_microbatches
        self.schedule = SCHEDULES[schedule](
            stage,
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
        )
        self.loop.target_pieces = pieces
        self.loop.trainer = self
        self.loop.module = module
        module._loop = self.loop
        # One list, not two copies: a callback appended after construction has to
        # reach the loop, which is the thing that calls it.
        self.callbacks = self.loop.callbacks

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
        """Whole-model checkpoint, Lightning-shaped. Collective."""
        payload: dict = {}
        self.callbacks.call("on_save_checkpoint", self.loop, payload)
        self.module.on_save_checkpoint(payload)
        payload["callbacks"] = self.callbacks.state_dict()
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
        """Restore this rank's slice, its optimizer entry and the step count."""
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
        self.callbacks.load_state_dict(payload.get("callbacks", {}))
        self.callbacks.call("on_load_checkpoint", self.loop, payload)
        self.module.on_load_checkpoint(payload)
        log.info("checkpoint restored", path=path, step=self.loop.global_step)
        return payload
