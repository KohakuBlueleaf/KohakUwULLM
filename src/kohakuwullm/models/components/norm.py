"""Normalization layers operating on channel-last tensors ``(..., D)``.

Every norm here is shape-agnostic in all but the last dimension, so the same
module serves the packed layout ``(T, D)`` and the padded layout ``(B, S, D)``
without a reshape. ``rmsnorm`` is the default; see docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm as triton_rms_norm
from kohakuwullm.registry import NORM


@NORM.register("rmsnorm")
class RMSNorm(nn.Module):
    """RMSNorm over the last dim. ``affine=False`` is used for QK-norm."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape[0]}, eps={self.eps}, affine={self.weight is not None}"


@NORM.register("rmsnorm_triton")
class RMSNormTriton(RMSNorm):
    """RMSNorm backed by the fused Triton kernel (see ``kernels/rmsnorm.py``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.weight, self.eps)


@NORM.register("layernorm")
class LayerNorm(nn.Module):
    """LayerNorm over the last dim."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


@NORM.register("gemma_rmsnorm")
class GemmaRMSNorm(nn.Module):
    """RMSNorm with a zero-centered weight (``(1 + w)``), as in Gemma."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        if affine:
            self.weight = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = None if self.weight is None else (1.0 + self.weight)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


@NORM.register("dyt")
class DynamicTanh(nn.Module):
    """Dynamic Tanh: ``w * tanh(a * x) + b``, a normalization-free drop-in."""

    def __init__(
        self, dim: int, eps: float = 1e-6, affine: bool = True, alpha: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), alpha))
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        if self.weight is None:
            return x
        return x * self.weight + self.bias
