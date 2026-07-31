"""Loss scaling for a pipeline, where only one stage owns the loss.

``torch.amp.GradScaler`` records ``scale()`` per optimizer inside one process,
so a rank that never computes a loss cannot later ``unscale_``. Here the scale
is a plain number every rank holds, and the overflow check is a collective so
the ranks skip or step together. See docs/kohakuwupipe/scaler.md.
"""

import torch
import torch.distributed as dist


class PipelineGradScaler:
    """Dynamic loss scaling shared by every stage of a pipeline.

    Args:
        enabled: ``False`` makes every method a pass-through, so a bf16 or fp32
            run pays nothing and needs no branch at the call site.
        init_scale: starting multiplier applied to the loss.
        growth / backoff: multipliers on a clean run and on an overflow.
        growth_interval: clean steps before the scale grows.
        min_scale: refuse to back off below this; see docs/kohakuwupipe/scaler.md.
    """

    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 2.0**16,
        growth: float = 2.0,
        backoff: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self.scale_value = float(init_scale)
        self.growth = growth
        self.backoff = backoff
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.clean_steps = 0
        self.overflows = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply the loss so its gradients land inside fp16's range."""
        if not self.enabled:
            return loss
        return loss * self.scale_value

    def unscale_(self, parameters) -> torch.Tensor:
        """Divide every gradient by the scale; return a device flag, 1 if any is not finite.

        Reads nothing on the host, so the caller decides when to synchronize.
        """
        grads = [p.grad for p in parameters if p.grad is not None]
        found = torch.zeros(1, device=grads[0].device if grads else "cpu")
        if not self.enabled or not grads:
            return found
        inverse = 1.0 / self.scale_value
        for grad in grads:
            grad.mul_(inverse)
            found += (~torch.isfinite(grad)).any().to(found.dtype)
        return (found > 0).to(found.dtype)

    def agree(self, found: torch.Tensor) -> torch.Tensor:
        """Make the overflow flag unanimous. Collective.

        A rank that stepped while another skipped leaves the stages holding
        updates from different batches.
        """
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.all_reduce(found, op=dist.ReduceOp.MAX)
        return found

    def update(self, overflow: bool) -> None:
        """Back off on an overflow, grow after ``growth_interval`` clean steps."""
        if not self.enabled:
            return
        if overflow:
            self.scale_value = max(self.scale_value * self.backoff, self.min_scale)
            self.clean_steps = 0
            self.overflows += 1
            return
        self.clean_steps += 1
        if self.clean_steps >= self.growth_interval:
            self.scale_value *= self.growth
            self.clean_steps = 0

    def state_dict(self) -> dict:
        return {
            "scale_value": self.scale_value,
            "clean_steps": self.clean_steps,
            "overflows": self.overflows,
        }

    def load_state_dict(self, state: dict) -> None:
        self.scale_value = float(state.get("scale_value", self.scale_value))
        self.clean_steps = int(state.get("clean_steps", 0))
        self.overflows = int(state.get("overflows", 0))
