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
        self._loop = None

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
        """The step's objective on the last stage. A sum, not a mean.

        ``target`` is this microbatch's slice of whatever the step carried,
        unexamined by this package and with its structure intact: a tensor, a
        tuple, a dict, ``None``. Tensors split along dim 0; anything else is
        passed whole. See docs/kohakuwupipe/module.md.

        Anything differentiable that a ``LightningModule`` would do between the
        model output and the loss belongs here, not in :meth:`training_step`.
        Per-term breakdowns go through :meth:`log` here: this is the only place
        a microbatch's internals exist. See docs/kohakuwupipe/module.md.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int) -> None:
        """One step, in the shape a ``LightningModule`` writes it.

        Preprocess, call :meth:`forward_backward` exactly once, then log or
        post-process. The default is the plain case: hand the step's own
        ``inputs`` and ``target`` straight to the model.

        Returns nothing -- the loss is not available here. A 1F1B schedule
        interleaves microbatch forwards and backwards across ranks, so there is
        no point at which one rank holds "the output"; the objective arrives as
        :meth:`loss`, once per microbatch. See docs/kohakuwupipe/module.md.
        """
        self.forward_backward(batch.inputs, batch.target)

    def forward_backward(self, inputs, target=None, layout=None):
        """Run the model over one step. Collective; call it once per step.

        Under ``torch.no_grad()`` it runs forward only. Returns the step's loss
        on the last stage, ``None`` elsewhere.
        """
        return self._loop.forward_backward(inputs, target, layout)

    def validation_step(self, batch, batch_idx: int):
        """One validation batch. Whatever it returns reaches the callbacks."""

    def set_seq_info(self, layout) -> None:
        """Register per-microbatch layout before the schedule runs."""

    def post_step(self, stage_module: nn.Module) -> dict[str, torch.Tensor]:
        """Work after ``optimizer.step``; return device tensors to log."""
        return {}

    def log(self, name: str, value) -> None:
        """Record a metric for the current step, without reading the device.

        Accumulates: :meth:`loss` runs once per microbatch, so assigning would
        report the last microbatch rather than the step.
        """
        total, count = self._metrics.get(name, (0.0, 0))
        self._metrics[name] = (total + value, count + 1)

    def pop_metrics(self) -> dict[str, Any]:
        """Take and clear what :meth:`log` recorded, averaged over its calls."""
        metrics = {
            name: total / count for name, (total, count) in self._metrics.items()
        }
        self._metrics = {}
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
            first=plan.is_first,
            last=plan.is_last,
            params=sum(p.numel() for p in self.stage_module.parameters()),
        )
        return self.stage_module

    def forward(self, *args, **kwargs):
        """Delegates to the stage, so the module is usable as one."""
        return self.stage_module(*args, **kwargs)
