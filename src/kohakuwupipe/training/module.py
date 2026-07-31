"""``PipelineModule``: what a ``LightningModule`` is, for a split model.

A subclass owns its stage, its optimizer and its loss, so the loop stays a
driver. The hook names match ``LightningModule`` where the concept survives
being split across ranks. See docs/kohakuwupipe/module.md.
"""

from typing import Any

import torch
import torch.nn as nn

from kohakuwupipe.utils.logging import get_logger

log = get_logger(__name__)


class PipelineModule(nn.Module):
    """One rank's stage, plus everything the loop needs to drive it.

    A subclass must provide :meth:`configure_model` and
    :meth:`configure_optimizers`; :meth:`loss` is required on the last stage
    only. ``self.stage_module`` is what the schedule calls and what the
    checkpoint reads, so an autocast wrapper belongs there rather than around
    ``forward``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.plan = None
        self.stage_module: nn.Module | None = None
        self.rank = 0
        self.world = 1
        self.global_step = 0
        self._metrics: dict[str, Any] = {}

    # -- construction ----------------------------------------------------- #

    def configure_model(self, plan, rank: int, world: int, device) -> nn.Module:
        """Build and return this rank's stage for ``plan``. Called once."""
        raise NotImplementedError

    def configure_optimizers(self):
        """Return ``optimizer`` or ``(optimizer, scheduler)`` over this stage."""
        raise NotImplementedError

    def boundary_example(self, plan, device) -> torch.Tensor:
        """The zero tensor whose shape ``PipelineStage`` freezes as the boundary."""
        raise NotImplementedError

    # -- the step --------------------------------------------------------- #

    def loss(self, hidden: torch.Tensor, target: Any):
        """``(loss, logs)`` on the last stage. ``loss`` is a sum, not a mean.

        ``target`` is whatever the step carried, unexamined by this package:
        labels for a supervised objective, ``None`` for a self-supervised one.
        It must be a tensor or ``None`` -- the schedule splits it along dim 0
        and a tuple or dict raises. Pack several tensors into one, or stack them.

        This is where a ``LightningModule.training_step`` body goes: the
        schedule owns forward and backward across ranks, so a stage supplies
        the objective rather than driving the pass. See docs/kohakuwupipe/module.md.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int):
        """Extra per-step work, run before the schedule. Returns nothing.

        Unlike ``LightningModule.training_step`` this does **not** compute the
        loss or call backward -- :meth:`loss` does, once per microbatch, from
        inside the schedule. Override for bookkeeping that needs the raw batch.
        """

    def validation_step(self, batch, batch_idx: int):
        """One validation batch. Whatever it returns reaches the callbacks."""

    def set_seq_info(self, layout) -> None:
        """Register per-microbatch layout before the schedule runs."""

    def post_step(self, stage_module: nn.Module) -> dict[str, torch.Tensor]:
        """Work after ``optimizer.step``; return device tensors to log."""
        return {}

    def log(self, name: str, value) -> None:
        """Record a metric for the current step, without reading the device."""
        self._metrics[name] = value

    def pop_metrics(self) -> dict[str, Any]:
        """Take and clear what :meth:`log` recorded."""
        metrics, self._metrics = self._metrics, {}
        return metrics

    # -- checkpoint ------------------------------------------------------- #

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add anything the module needs on resume."""

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Read back what :meth:`on_save_checkpoint` added."""

    # -- lifecycle -------------------------------------------------------- #

    def on_train_start(self) -> None: ...

    def on_train_end(self) -> None: ...

    def setup(self, plan, rank: int, world: int, device) -> nn.Module:
        """Build the stage and record the rank wiring. Returns the stage."""
        self.plan = plan
        self.rank = rank
        self.world = world
        self.stage_module = self.configure_model(plan, rank, world, device)
        log.info(
            "stage built",
            layers=f"{plan.start_layer}..{plan.end_layer - 1}",
            embed=plan.has_embed,
            head=plan.has_head,
            params=sum(p.numel() for p in self.stage_module.parameters()),
        )
        return self.stage_module

    def forward(self, *args, **kwargs):
        """Delegates to the stage, so the module is usable as one."""
        return self.stage_module(*args, **kwargs)
