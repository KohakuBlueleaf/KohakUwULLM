"""Feed-forward blocks.

All of them are last-dim ops, so they serve packed ``(T, D)`` and padded
``(B, S, D)`` hidden states unchanged. See docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.registry import MLP


def resolve_hidden(
    dim: int,
    ratio: float,
    hidden: int | None = None,
    multiple_of: int = 128,
    glu: bool = True,
) -> int:
    """Feed-forward width, rounded up to a multiple of ``multiple_of``.

    ``glu=True`` applies the conventional ``2/3`` correction, which holds the
    parameter count equal to a plain ``ratio * dim`` MLP. ``hidden`` overrides
    the computation entirely.
    """
    if hidden is not None:
        return hidden
    width = dim * ratio * (2 / 3) if glu else dim * ratio
    return multiple_of * ((int(width) + multiple_of - 1) // multiple_of)


class GLUMLP(nn.Module):
    """Gated linear unit feed-forward: ``w_out(act(gate) * value)``.

    Args:
        dim: model width.
        ratio: expansion ratio relative to a plain 2-matrix MLP.
        hidden: explicit feed-forward width (overrides ``ratio``).
        multiple_of: round ``hidden`` up to this multiple.
        bias: bias on the projections.
        fused_gate: keep gate and value in one ``w_in`` matrix; off splits them
            into ``gate_proj`` / ``up_proj``, as most HF checkpoints store them.
    """

    act = staticmethod(F.silu)

    def __init__(
        self,
        dim: int,
        ratio: float = 4.0,
        hidden: int | None = None,
        multiple_of: int = 128,
        bias: bool = False,
        fused_gate: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = resolve_hidden(dim, ratio, hidden, multiple_of, glu=True)
        self.fused_gate = fused_gate
        if fused_gate:
            self.w_in = nn.Linear(dim, 2 * self.hidden, bias=bias)
        else:
            self.gate_proj = nn.Linear(dim, self.hidden, bias=bias)
            self.up_proj = nn.Linear(dim, self.hidden, bias=bias)
        self.w_out = nn.Linear(self.hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate:
            gate, value = self.w_in(x).chunk(2, dim=-1)
        else:
            gate, value = self.gate_proj(x), self.up_proj(x)
        return self.w_out(self.act(gate) * value)


@MLP.register("swiglu")
class SwiGLU(GLUMLP):
    """SwiGLU (Llama / Qwen / DeepSeek / Gemma)."""

    act = staticmethod(F.silu)


@MLP.register("geglu")
class GeGLU(GLUMLP):
    """GeGLU (tanh-approximate GELU gate)."""

    @staticmethod
    def act(x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


@MLP.register("swiglu_triton")
class SwiGLUTriton(GLUMLP):
    """SwiGLU whose ``silu(gate) * value`` is one fused Triton kernel.

    Saves the intermediate ``silu(gate)`` activation. Wins only on an uncompiled
    path; see docs/concepts/architecture.md.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_gate:
            gate, value = self.w_in(x).chunk(2, dim=-1)
        else:
            gate, value = self.gate_proj(x), self.up_proj(x)
        return self.w_out(swiglu_mul(gate, value))


@MLP.register("gelu")
class GELUMLP(nn.Module):
    """Plain two-layer GELU MLP (no gating)."""

    def __init__(
        self,
        dim: int,
        ratio: float = 4.0,
        hidden: int | None = None,
        multiple_of: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = resolve_hidden(dim, ratio, hidden, multiple_of, glu=False)
        self.fc1 = nn.Linear(dim, self.hidden, bias=bias)
        self.fc2 = nn.Linear(self.hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))
