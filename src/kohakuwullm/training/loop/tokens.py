"""Token accounting: the globally-reduced progress snapshot and its rates.

*Seen* is what the GPU computed on, *trained* what carried a gradient. Counts are
int64 end to end and FLOPs float64. See docs/guides/training.md.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class TokenSnapshot:
    """Cumulative progress at one instant; subtract two to get an interval."""

    seen: int
    trained: int
    model_flops: float
    hardware_flops: float
    elapsed: float

    def __sub__(self, other: "TokenSnapshot") -> "TokenSnapshot":
        return TokenSnapshot(
            seen=self.seen - other.seen,
            trained=self.trained - other.trained,
            model_flops=self.model_flops - other.model_flops,
            hardware_flops=self.hardware_flops - other.hardware_flops,
            elapsed=self.elapsed - other.elapsed,
        )

    @property
    def tokens_per_sec(self) -> float:
        return self.seen / max(self.elapsed, 1e-9)

    @property
    def trained_tokens_per_sec(self) -> float:
        return self.trained / max(self.elapsed, 1e-9)

    @property
    def trained_frac(self) -> float:
        return self.trained / max(self.seen, 1)

    @property
    def b_tokens_per_day(self) -> float:
        """Billions of tokens per day."""
        return self.tokens_per_sec * SECONDS_PER_DAY / 1e9

    @property
    def b_trained_tokens_per_day(self) -> float:
        return self.trained_tokens_per_sec * SECONDS_PER_DAY / 1e9

    def mfu(self, peak_flops: float) -> float:
        """Model FLOPs utilization: the arithmetic the *architecture* owes."""
        return self.model_flops / max(self.elapsed * peak_flops, 1e-9)

    def hfu(self, peak_flops: float) -> float:
        """Hardware FLOPs utilization: adds the forward that checkpointing repeats."""
        return self.hardware_flops / max(self.elapsed * peak_flops, 1e-9)


def all_reduce_int(value: int, device: torch.device | None = None) -> int:
    """Sum one int64 scalar across ranks. Costs a collective and a host sync."""
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    total = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return int(total.item())


def all_reduce_(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across ranks in place, in its own dtype."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
