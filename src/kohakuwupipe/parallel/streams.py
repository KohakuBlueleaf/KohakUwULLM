"""Multi-stream stage boundaries: ship more than the hidden state between ranks.

A stage's boundary is a tuple, not one tensor, so a pipeline can carry side
channels alongside the activation. Three shapes are supported and each has its
own helper here:

* **accumulator** -- a scalar every stage adds to, summed into the final loss.
  ``d(total)/d(acc) == 1`` at every stage, so each stage's contribution gets the
  gradient it would have had from its own ``.backward()``.
* **constant** -- a tensor a later stage reads but no stage differentiates
  (a text context for cross-attention). It still has to *carry* grad or the
  runtime treats the boundary as dead, hence :class:`GradCarrier`.
* **skip** -- an activation handed to a stage further down (a U-Net skip).

See docs/kohakuwupipe/streams.md.
"""

import torch
import torch.nn as nn


class GradCarrier(nn.Module):
    """Multiplies a tensor by a learnable 1.0 so it carries gradient.

    A boundary tensor that never requires grad has no backward edge, and
    ``PipelineStage`` cannot ship a gradient for it. Scaling by a parameter
    initialized to one leaves the value unchanged and makes the edge exist.
    """

    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=True)
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale if self.trainable else self.scale.detach() + 0.0 * self.scale
        return x * scale


def accumulator(device, dtype=torch.float32) -> torch.Tensor:
    """A ``(1,)`` zero for the first stage to start an accumulator stream.

    It requires grad, so the boundary has a backward edge even on a stage with
    no terms of its own. Shape ``(1,)`` rather than ``()``: the loss reduction
    recognizes an accumulator by its trailing unit axis. See docs/kohakuwupipe/streams.md.
    """
    return torch.zeros(1, device=device, dtype=dtype, requires_grad=True)


def accumulate(carried: torch.Tensor | None, terms) -> torch.Tensor | None:
    """Add this stage's terms to the incoming accumulator.

    ``terms`` is any iterable of scalars, or empty. Returns ``carried``
    unchanged when there is nothing to add, so a stage with no terms is free.
    The result is ``(1,)``, matching :func:`accumulator`.
    """
    stack = [t for t in terms if t is not None]
    if not stack:
        return carried
    total = torch.cat([t.float().reshape(1) for t in stack]).sum().reshape(1)
    return total if carried is None else carried + total


class StreamSpec:
    """What one boundary stream is, and how to build its example tensor.

    Args:
        name: for diagnostics.
        shape: per-microbatch shape; ``(1,)`` for an accumulator.
        dtype: boundary dtype, which ``PipelineStage`` freezes.
        kind: ``"hidden"``, ``"accumulator"``, ``"constant"`` or ``"skip"``.
    """

    __slots__ = ("name", "shape", "dtype", "kind")

    def __init__(self, name: str, shape: tuple, dtype: torch.dtype, kind: str) -> None:
        if kind not in ("hidden", "accumulator", "constant", "skip"):
            raise ValueError(f"unknown stream kind {kind!r}")
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.kind = kind

    def example(self, device) -> torch.Tensor:
        """The zero tensor declared to ``PipelineStage(input_args=...)``."""
        return torch.zeros(self.shape, device=device, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"StreamSpec({self.name!r}, {self.shape}, {self.dtype}, {self.kind!r})"


def boundary_examples(specs, device) -> tuple:
    """Example tensors for a whole boundary, in stream order."""
    return tuple(spec.example(device) for spec in specs)


def reduce_accumulator(stream: torch.Tensor) -> torch.Tensor:
    """Sum an accumulator stream so its gradient is dense.

    ``stream.sum()`` builds its gradient by expanding a scalar to stride 0, and
    NCCL rejects a non-dense tensor for P2P. See docs/kohakuwupipe/streams.md.
    """
    return (stream * torch.ones_like(stream)).sum()


def split_streams(value):
    """Normalize a stage's output to a tuple, whatever it returned."""
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(value)
