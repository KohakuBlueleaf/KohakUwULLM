"""Lightning strategy for pipeline parallelism.

See docs/internals/pipeline.md.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from lightning.fabric.utilities.optimizer import _optimizer_to_device
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.strategies import DDPStrategy
from torch.optim import Optimizer

# Marks a per-rank optimizer state list, so a plain state dict stays loadable.
STAGE_STATES = "pipeline_stage_optimizer_states"


class PipelineOnlyStrategy(DDPStrategy):
    """DDP launch and process-group init, without wrapping the module.

    Checkpointing is redirected through the module's gather/scatter hooks when it
    has them, so the file holds the whole model rather than rank 0's stage.
    See docs/internals/pipeline.md.
    """

    def lightning_module_state_dict(self) -> dict[str, Any]:
        """Every stage's slice, keyed by whole-backbone names. Collective."""
        gather = getattr(self.lightning_module, "gathered_state_dict", None)
        if not callable(gather):
            return super().lightning_module_state_dict()
        return gather()

    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        """Hand this rank's stage its own slice of a whole-backbone checkpoint."""
        scatter = getattr(self.lightning_module, "load_gathered_state_dict", None)
        if not callable(scatter):
            super().load_model_state_dict(checkpoint, strict=strict)
            return
        scatter(checkpoint["state_dict"], strict=strict)

    def optimizer_state(self, optimizer: Optimizer) -> dict[str, Any]:
        """Every stage's optimizer state, gathered onto rank 0. Collective.

        Stages own disjoint parameters, so the states are kept as a per-rank list
        rather than merged. See docs/internals/pipeline.md.
        """
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        local = _to_cpu(optimizer.state_dict())
        if not (dist.is_available() and dist.is_initialized()):
            return local
        parts = [None] * dist.get_world_size() if self.is_global_zero else None
        dist.gather_object(local, parts, dst=0)
        return {STAGE_STATES: parts} if self.is_global_zero else {}

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """Give each rank its own stage's moments out of the gathered list."""
        for optimizer, state in zip(self.optimizers, checkpoint["optimizer_states"]):
            if STAGE_STATES in state:
                state = state[STAGE_STATES][self.global_rank]
            optimizer.load_state_dict(state)
            _optimizer_to_device(optimizer, self.root_device)

    def configure_ddp(self) -> None:
        return


def _to_cpu(value):
    """Every tensor in a nested container, moved to the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(v) for v in value)
    return value
