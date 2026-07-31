"""A pipelining-first trainer: the module, the loop, the split and the checkpoint.

Depends on ``torch`` alone. Nothing here knows what a layer is -- costs, layouts
and post-step work arrive from the caller -- so a model of any architecture can
train by supplying a :class:`PipelineModule`. See docs/kohakuwupipe/README.md.
"""

from kohakuwupipe.io.checkpoint import gather_state_dict, load, save
from kohakuwupipe.parallel.distributed import (
    PipelineRanks,
    init_pipeline,
    shutdown,
    warn_on_eager_init,
)
from kohakuwupipe.parallel.plan import (
    StagePlan,
    describe,
    plan_from_layers,
    plan_stages,
)
from kohakuwupipe.training.callbacks import Checkpoint, LossLog, Throughput
from kohakuwupipe.training.hooks import Callback, CallbackList
from kohakuwupipe.training.loop import PipelineLoop, StepOutput, build_loss_fn
from kohakuwupipe.training.module import PipelineModule
from kohakuwupipe.training.trainer import SCHEDULES, PipelineTrainer
from kohakuwupipe.utils.logging import configure, get_logger

__version__ = "0.1.0"

__all__ = [
    "SCHEDULES",
    "Callback",
    "CallbackList",
    "Checkpoint",
    "LossLog",
    "PipelineLoop",
    "PipelineModule",
    "PipelineRanks",
    "PipelineTrainer",
    "StagePlan",
    "StepOutput",
    "Throughput",
    "build_loss_fn",
    "configure",
    "describe",
    "gather_state_dict",
    "get_logger",
    "init_pipeline",
    "load",
    "plan_from_layers",
    "plan_stages",
    "save",
    "shutdown",
    "warn_on_eager_init",
]
