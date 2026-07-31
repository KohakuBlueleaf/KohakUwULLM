"""Training package: the Lightning trainer, callbacks, optimizers and pipelining.

``LMTrainer`` pulls in Lightning, so it is exposed lazily -- importing a pure
submodule (e.g. ``kohakuwullm.training.optim.build``) does not drag Lightning in.
"""

from kohakuwullm.training.loop.tokens import TokenSnapshot
from kohakuwullm.training.optim.build import build_optimizer, group_parameters
from kohakuwullm.training.parallel.pipeline import (
    LMStage,
    plan_for,
)

__all__ = [
    "LMTrainer",
    "PipelineOnlyStrategy",
    "PipelinedLMTrainer",
    "SampleLogCallback",
    "StepProgressBar",
    "ThroughputCallback",
    "TokenSnapshot",
    "build_optimizer",
    "group_parameters",
    "plan_for",
    "LMStage",
]

_LAZY = {
    "LMTrainer": ("kohakuwullm.training.loop.trainer", "LMTrainer"),
    "PipelinedLMTrainer": (
        "kohakuwullm.training.loop.pipeline_trainer",
        "PipelinedLMTrainer",
    ),
    "PipelineOnlyStrategy": (
        "kohakuwullm.training.parallel.strategy",
        "PipelineOnlyStrategy",
    ),
    "SampleLogCallback": ("kohakuwullm.training.loop.callbacks", "SampleLogCallback"),
    "StepProgressBar": ("kohakuwullm.training.loop.callbacks", "StepProgressBar"),
    "ThroughputCallback": ("kohakuwullm.training.loop.callbacks", "ThroughputCallback"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
