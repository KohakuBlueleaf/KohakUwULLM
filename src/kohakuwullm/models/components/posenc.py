"""Position encodings for a decoder LM.

The backbone builds the carrier once per forward. Two contracts:

* ``prepare(position_ids, device, dtype)`` -> a carrier, or ``None``.
* the carrier exposes ``apply(q, k)`` over ``(..., H, D)``, with the position
  axis immediately left of the head axis.

See docs/concepts/architecture.md.
"""

import math

import torch
import torch.nn as nn

from kohakuwullm.kernels.attention.rope import DEFAULT_ROPE_IMPL, resolve_rope
from kohakuwullm.registry import POSENC


@POSENC.register("none")
class NoPE(nn.Module):
    """No position encoding, for a per-layer NoPE interleave."""

    def __init__(self, head_dim: int | None = None, **_unused) -> None:
        super().__init__()

    def prepare(self, position_ids, device, dtype):
        return None


class RotaryCache:
    """Carries ``(cos, sin)`` for one batch and rotates ``q``/``k``.

    ``cos``/``sin`` are ``(*pos_shape, rotary_dim)``, already doubled.

    ``impl`` names the rotation implementation -- ``"triton"``, ``"compiled"`` or
    ``"eager"`` -- and is resolved to a callable here, once, when the carrier is
    built. An unknown name raises. See docs/internals/kernels.md.
    """

    __slots__ = ("cos", "sin", "rotary_dim", "impl", "rotate")

    def __init__(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_dim: int,
        impl: str = DEFAULT_ROPE_IMPL,
    ) -> None:
        self.cos = cos
        self.sin = sin
        self.rotary_dim = rotary_dim
        self.impl = impl
        self.rotate = resolve_rope(impl)

    def apply(self, q: torch.Tensor, k: torch.Tensor):
        """Rotate ``q``, ``k`` of shape ``(..., H, D)``; head axis broadcasts."""
        # Every implementation takes the half table; cos/sin here are the doubled one.
        half = self.rotary_dim // 2
        cos = self.cos[..., :half]
        sin = self.sin[..., :half]
        return (
            self.rotate(q, cos, sin, self.rotary_dim),
            self.rotate(k, cos, sin, self.rotary_dim),
        )


def _yarn_ramp(low: float, high: float, dim: int, device) -> torch.Tensor:
    if low == high:
        high += 1e-3
    idx = torch.arange(dim, dtype=torch.float32, device=device)
    return ((idx - low) / (high - low)).clamp(0, 1)


def _yarn_correction_dim(
    rotations: float, dim: int, base: float, orig_ctx: int
) -> float:
    return (dim * math.log(orig_ctx / (rotations * 2 * math.pi))) / (2 * math.log(base))


@POSENC.register("rope")
class RoPE(nn.Module):
    """Rotary position embedding with optional context-extension scaling.

    Args:
        head_dim: per-head width.
        theta: RoPE base.
        partial_rotary_factor: fraction of each head that is rotated; the rest
            passes through unchanged.
        scaling: ``None`` | ``"linear"`` | ``"ntk"`` | ``"yarn"``.
        factor: extension factor for the scaling mode.
        original_context: context length the base model was trained at
            (``yarn``/``ntk`` need it to place the interpolation ramp).
        beta_fast / beta_slow: YaRN ramp endpoints, in rotations.
        impl: rotation implementation -- ``"triton"``, ``"compiled"`` or ``"eager"``.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        scaling: str | None = None,
        factor: float = 1.0,
        original_context: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        impl: str = DEFAULT_ROPE_IMPL,
        **_unused,
    ) -> None:
        super().__init__()
        # Reject an unknown name here rather than at the first forward.
        resolve_rope(impl)
        self.impl = impl
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary dim must be even; head_dim={head_dim} * "
                f"partial_rotary_factor={partial_rotary_factor} -> {rotary_dim}"
            )
        self.rotary_dim = rotary_dim
        self.scaling = scaling
        self.factor = factor
        self.attn_scale = 1.0

        half = rotary_dim // 2
        exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
        match scaling:
            case None | "none":
                inv_freq = 1.0 / (theta**exponent)
            case "linear":
                inv_freq = 1.0 / (theta**exponent) / factor
            case "ntk":
                # Raise the base, leaving the highest frequency untouched.
                adjusted = theta * factor ** (rotary_dim / (rotary_dim - 2))
                inv_freq = 1.0 / (adjusted**exponent)
            case "yarn":
                inv_extra = 1.0 / (theta**exponent)
                inv_inter = inv_extra / factor
                low = math.floor(
                    _yarn_correction_dim(beta_fast, rotary_dim, theta, original_context)
                )
                high = math.ceil(
                    _yarn_correction_dim(beta_slow, rotary_dim, theta, original_context)
                )
                ramp = _yarn_ramp(max(low, 0), min(high, half - 1), half, None)
                # ramp == 1 -> extrapolate; 0 -> interpolate.
                inv_freq = inv_inter * (1 - ramp) + inv_extra * ramp
                # YaRN also rescales attention logits.
                self.attn_scale = 0.1 * math.log(factor) + 1.0
            case _:
                raise ValueError(f"unknown rope scaling {scaling!r}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def prepare(self, position_ids: torch.Tensor, device, dtype) -> RotaryCache:
        """``position_ids`` is ``(T,)`` packed or ``(B, S)`` padded."""
        inv_freq = self.inv_freq.to(device)
        # (..., half) outer product; fp32 throughout, cast only at the end.
        angles = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
        angles = torch.cat([angles, angles], dim=-1)
        cos = angles.cos()
        sin = angles.sin()
        if self.attn_scale != 1.0:
            cos = cos * self.attn_scale
            sin = sin * self.attn_scale
        return RotaryCache(cos.to(dtype), sin.to(dtype), self.rotary_dim, self.impl)
