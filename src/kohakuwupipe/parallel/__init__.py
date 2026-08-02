"""Rank wiring and the stage split: everything that decides who holds what."""

from kohakuwupipe.parallel.distributed import (
    PipelineRanks,
    eager_init_in_use,
    init_pipeline,
    shutdown,
    warn_on_eager_init,
)
from kohakuwupipe.parallel.plan import (
    StagePlan,
    describe,
    partition,
    plan_from_layers,
    plan_stages,
)

__all__ = [
    "PipelineRanks",
    "StagePlan",
    "describe",
    "eager_init_in_use",
    "init_pipeline",
    "partition",
    "plan_from_layers",
    "plan_stages",
    "shutdown",
    "warn_on_eager_init",
]

from kohakuwupipe.parallel.streams import (  # noqa: E402
    GradCarrier,
    accumulate,
    accumulator,
    reduce_accumulator,
    split_streams,
)

__all__ += [
    "GradCarrier",
    "accumulate",
    "accumulator",
    "reduce_accumulator",
    "split_streams",
]
