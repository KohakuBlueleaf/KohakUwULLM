"""Autoregressive generation, for a whole model or a pipeline of stages.

See docs/guides/generation.md.
"""

from kohakuwullm.generation.engine import (
    Generator,
    LocalGenerator,
    PipelineGenerator,
    build_generator,
    filter_min_p,
    filter_top_k,
    filter_top_p,
    sample,
)

__all__ = [
    "Generator",
    "LocalGenerator",
    "PipelineGenerator",
    "build_generator",
    "filter_min_p",
    "filter_top_k",
    "filter_top_p",
    "sample",
]
