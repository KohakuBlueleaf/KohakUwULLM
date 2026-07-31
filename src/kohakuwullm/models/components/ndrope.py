"""N-dimensional and Golden-Gate rotary position embeddings.

Channel pair ``i`` rotates by ``w_i * <u_i, t>`` for a unit direction ``u_i``
and a position vector ``t`` in R^N; ggRoPE places those directions at successive
golden angles. See docs/concepts/architecture.md.
"""

import math

import torch
import torch.nn as nn

from kohakuwullm.kernels.attention.rope import DEFAULT_ROPE_IMPL, resolve_rope
from kohakuwullm.models.components.posenc import RotaryCache
from kohakuwullm.registry import POSENC

GOLDEN_RATIO = (1.0 + 5.0**0.5) / 2.0


def golden_directions(num_pairs: int, n_dims: int, device=None) -> torch.Tensor:
    """``(num_pairs, n_dims)`` unit directions from a low-discrepancy sequence.

    2-D uses the golden angle directly; higher dimensions use the generalized
    golden ratio as a Kronecker sequence. See docs/concepts/architecture.md.
    """
    if n_dims == 1:
        return torch.ones(num_pairs, 1, device=device)
    if n_dims == 2:
        angles = torch.arange(num_pairs, dtype=torch.float64, device=device)
        angles = angles * (math.pi / GOLDEN_RATIO)
        return torch.stack([angles.cos(), angles.sin()], dim=-1).float()

    # Generalized golden ratio: the unique positive root of x^(N+1) = x + 1.
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / (n_dims + 1))
    alpha = torch.tensor(
        [phi ** -(i + 1) for i in range(n_dims)], dtype=torch.float64, device=device
    )
    idx = torch.arange(1, num_pairs + 1, dtype=torch.float64, device=device)
    points = (idx[:, None] * alpha[None, :]) % 1.0
    # Inverse Gaussian CDF turns the uniform lattice isotropic.
    normal = torch.erfinv(2 * points.clamp(1e-6, 1 - 1e-6) - 1) * math.sqrt(2)
    return (normal / normal.norm(dim=-1, keepdim=True)).float()


def log_spaced_frequencies(
    num_pairs: int, omega_min: float, omega_max: float, zero_ratio: float = 0.0
) -> torch.Tensor:
    """``(num_pairs,)`` log-spaced magnitudes, with a zero-frequency tail."""
    freqs = torch.logspace(
        math.log10(omega_min), math.log10(omega_max), num_pairs, dtype=torch.float32
    )
    n_zero = int(num_pairs * zero_ratio)
    if n_zero > 0:
        # The lowest frequencies, which carry the least position to lose.
        freqs[:n_zero] = 0.0
    return freqs


@POSENC.register("ndrope")
class NDRoPE(nn.Module):
    """N-dimensional RoPE with configurable directions.

    Args:
        head_dim: per-head width.
        n_dims: dimensionality of the position vector.
        directions: ``"golden"`` (ggRoPE), ``"axial"``, or ``"learned"``.
        omega_min / omega_max: frequency magnitude range.
        zero_freq_ratio: fraction of pairs given zero frequency (NoPE channels).
        partial_rotary_factor: rotate only a prefix of each head.
        impl: rotation implementation -- ``"triton"``, ``"compiled"`` or ``"eager"``.

    ``prepare`` accepts ``(T,)`` for 1-D positions or ``(T, n_dims)`` for N-D.
    """

    def __init__(
        self,
        head_dim: int,
        n_dims: int = 1,
        directions: str = "golden",
        omega_min: float = 0.5,
        omega_max: float = 50.0,
        zero_freq_ratio: float = 0.0,
        partial_rotary_factor: float = 1.0,
        impl: str = DEFAULT_ROPE_IMPL,
        **_unused,
    ) -> None:
        super().__init__()
        # Reject an unknown name here rather than at the first forward.
        resolve_rope(impl)
        self.impl = impl
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {rotary_dim}")
        self.rotary_dim = rotary_dim
        self.n_dims = n_dims
        num_pairs = rotary_dim // 2

        freqs = log_spaced_frequencies(num_pairs, omega_min, omega_max, zero_freq_ratio)
        match directions:
            case "golden":
                dirs = golden_directions(num_pairs, n_dims)
            case "axial":
                # Round-robin over the standard basis.
                dirs = torch.zeros(num_pairs, n_dims)
                dirs[torch.arange(num_pairs), torch.arange(num_pairs) % n_dims] = 1.0
            case "learned":
                dirs = golden_directions(num_pairs, n_dims)
            case _:
                raise ValueError(f"unknown directions {directions!r}")

        # Fold the magnitude into the direction; only the product is ever used.
        weighted = dirs * freqs[:, None]
        if directions == "learned":
            self.freq_dirs = nn.Parameter(weighted)
        else:
            self.register_buffer("freq_dirs", weighted, persistent=False)

    def prepare(self, position_ids: torch.Tensor, device, dtype) -> RotaryCache:
        positions = position_ids.to(torch.float32)
        if positions.dim() == 1 or self.n_dims == 1:
            positions = positions.reshape(*positions.shape, 1)[..., :1]
        if positions.shape[-1] != self.n_dims:
            raise ValueError(
                f"position_ids last dim is {positions.shape[-1]}, expected {self.n_dims}"
            )
        freq_dirs = self.freq_dirs.to(device=device, dtype=torch.float32)
        # angle[..., i] = <freq_i * u_i, t>
        angles = positions @ freq_dirs.T
        angles = torch.cat([angles, angles], dim=-1)
        return RotaryCache(
            angles.cos().to(dtype), angles.sin().to(dtype), self.rotary_dim, self.impl
        )


@POSENC.register("ggrope")
class GGRoPE(NDRoPE):
    """ggRoPE: N-D RoPE with golden-angle directions. The recommended N-D default."""

    def __init__(self, head_dim: int, **kwargs) -> None:
        kwargs.setdefault("directions", "golden")
        super().__init__(head_dim, **kwargs)
