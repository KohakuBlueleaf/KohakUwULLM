"""Sampling for the in-training preview callback.

KV-cached by default, and drawing from its own RNG stream rather than the
default one. See docs/guides/training.md.
"""

import torch

from kohakuwullm.generation.engine import SAMPLE_SEED, LocalGenerator


class PreviewSampler(LocalGenerator):
    """A :class:`LocalGenerator` whose backbone arrives per call.

    ``generate(backbone, prompt_ids, **kwargs)``; ``kwargs`` are
    :meth:`LocalGenerator.generate`'s.
    """

    def __init__(self, seed: int = SAMPLE_SEED, static: bool = False) -> None:
        super().__init__(None, seed=seed, static=static)

    def generate(self, backbone, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone = backbone
        return super().generate(prompt_ids, **kwargs)
