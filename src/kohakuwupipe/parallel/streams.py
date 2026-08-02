"""Multi-stream stage boundaries: ship more than the hidden state between ranks.

A boundary is a tuple of tensors. Beside the hidden state a stream is either an
**accumulator** -- a scalar every stage adds to, summed into the final loss --
or a **constant** a later stage reads without differentiating, which still needs
:class:`GradCarrier` to carry a backward edge.
See docs/kohakuwupipe/streams.md.
"""

import torch
import torch.nn as nn


class GradCarrier(nn.Module):
    """Multiplies a tensor by a parameter fixed at 1.0, so it carries gradient.

    The value is unchanged; ``trainable=False`` keeps the parameter itself fixed.
    See docs/kohakuwupipe/streams.md.
    """

    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=True)
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale if self.trainable else self.scale.detach() + 0.0 * self.scale
        return x * scale


def accumulator(device, dtype=torch.float32) -> torch.Tensor:
    """A ``(1,)`` zero that requires grad: the first stage's accumulator stream.

    The trailing unit axis is what the loss reduction recognizes it by.
    See docs/kohakuwupipe/streams.md.
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


def reduce_accumulator(stream: torch.Tensor) -> torch.Tensor:
    """Sum an accumulator stream to a scalar, with a dense gradient.

    See docs/kohakuwupipe/streams.md.
    """
    return (stream * torch.ones_like(stream)).sum()


def split_streams(value):
    """Normalize a stage's output to a tuple, whatever it returned."""
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(value)
