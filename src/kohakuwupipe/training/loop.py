"""The pipeline training loop: the step, and nothing between it and the GPU.

Nothing in :meth:`PipelineLoop.step` reads a device tensor on the host, so a
throughput regression is a regression in this file. Model-specific work -- an
fp8 weight refresh, a router-bias update -- arrives as ``post_step``.

See docs/kohakuwupipe/loop.md.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.distributed as dist

from kohakuwupipe.parallel.streams import reduce_accumulator, split_streams
from kohakuwupipe.training import hooks


class MicrobatchStep(Protocol):
    """One optimizer step's data, already split into microbatches.

    ``inputs`` is read on the first stage only and ``target`` on the last;
    ``layout`` is whatever the stage module's ``set_seq_info`` accepts, and
    ``trained`` is what the loss is normalized by.

    Neither is examined here: ``inputs`` goes to the stage and ``target`` to
    ``loss``. ``target`` need not be labels, but it must be a tensor or ``None``
    -- the schedule splits it along dim 0. See docs/kohakuwupipe/module.md.
    """

    inputs: Any
    target: Any
    layout: Any
    trained: int


@dataclass
class StepOutput:
    """One step's results, device-resident.

    Reading a tensor here costs a synchronization, so a callback does that on
    its own cadence rather than the loop doing it every step.
    """

    index: int
    loss: torch.Tensor | None
    seen: int
    trained: int
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


Callback = hooks.Callback


class PipelineLoop:
    """Drives a ``torch.distributed.pipelining`` schedule over a step stream.

    Args:
        stage_module: this rank's stage. ``set_seq_info(layout)`` is called when
            present; ``parameters()`` must cover the stage.
        schedule: a built schedule whose ``loss_fn`` owns loss normalization.
        optimizer: over this stage's parameters, disjoint from every other rank's.
        rank / world: pipeline position and size.
        micro_tokens / num_microbatches: tokens per microbatch and their count.
        scheduler: stepped once per optimizer step, or ``None``.
        grad_clip: max grad norm; 0 disables.
        post_step: ``(stage_module) -> dict[str, Tensor]``, run after
            ``optimizer.step``. Values must stay on the device.
        callbacks: see :class:`Callback`.
    """

    def __init__(
        self,
        stage_module,
        schedule,
        optimizer,
        rank: int,
        world: int,
        micro_tokens: int,
        num_microbatches: int,
        scheduler=None,
        grad_clip: float = 0.0,
        post_step: Callable[[Any], dict[str, torch.Tensor]] | None = None,
        callbacks: Iterable[Callback] = (),
    ) -> None:
        self.stage_module = stage_module
        self.schedule = schedule
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.rank = rank
        self.world = world
        self.micro_tokens = micro_tokens
        self.num_microbatches = num_microbatches
        self.grad_clip = grad_clip
        self.post_step = post_step
        self.callbacks = hooks.CallbackList(callbacks)
        self.global_step = 0
        self.epoch = 0
        self.is_first = rank == 0
        self.is_last = rank == world - 1
        self._set_layout = getattr(stage_module, "set_seq_info", None)
        self.module = None

    def step(self, batch: MicrobatchStep, batch_idx: int = 0) -> StepOutput:
        """One optimizer step. Collective: no rank may return early."""
        self.callbacks.call("on_train_batch_start", self, batch, batch_idx)
        if self.module is not None:
            self.module.training_step(batch, batch_idx)
        self.callbacks.call("on_before_zero_grad", self, self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if self._set_layout is not None:
            self._set_layout(batch.layout)

        # The schedule runs forward and backward for every microbatch.
        losses = [] if self.is_last else None
        self.callbacks.call("on_before_backward", self)
        if self.is_first:
            self.schedule.step(batch.inputs)
        elif self.is_last:
            self.schedule.step(target=batch.target, losses=losses)
        else:
            self.schedule.step()
        self.callbacks.call("on_after_backward", self)

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.stage_module.parameters(), self.grad_clip
            )
        self.callbacks.call("on_before_optimizer_step", self, self.optimizer)
        self.optimizer.step()
        extra = self.post_step(self.stage_module) if self.post_step else {}
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        out = StepOutput(
            index=self.global_step,
            loss=self.broadcast_loss(losses),
            seen=self.micro_tokens * self.num_microbatches,
            trained=int(batch.trained),
            extra=extra,
        )
        self.callbacks.call("on_train_batch_end", self, out, batch, batch_idx)
        return out

    def broadcast_loss(self, losses) -> torch.Tensor | None:
        """The step's loss, from the last stage to every rank, as a device tensor.

        Microbatch losses **sum**: ``loss_fn`` already divided each one by the
        step's token count. See docs/kohakuwupipe/loop.md.
        """
        if losses is not None and not dist.is_initialized():
            return torch.stack(losses).sum().detach() if losses else None
        if not dist.is_initialized():
            return None
        device = next(self.stage_module.parameters()).device
        buf = torch.zeros(1, device=device)
        if losses:
            buf[0] = torch.stack([x.detach().float() for x in losses]).sum()
        dist.broadcast(buf, src=self.world - 1)
        return buf[0]

    def fit(self, batches: Iterator[MicrobatchStep], max_steps: int) -> None:
        """Step until ``max_steps``. Every rank must consume the same stream.

        ``on_exception`` fires on every rank before the error propagates, so a
        callback can tear its own state down.
        """
        self.callbacks.call("setup", self, "fit")
        self.callbacks.call("on_fit_start", self)
        self.callbacks.call("on_train_start", self)
        self.callbacks.call("on_train_epoch_start", self)
        try:
            for batch_idx, batch in enumerate(batches):
                if self.global_step >= max_steps:
                    break
                self.step(batch, batch_idx)
        except BaseException as exception:
            self.callbacks.call("on_exception", self, exception)
            raise
        self.callbacks.call("on_train_epoch_end", self)
        self.callbacks.call("on_train_end", self)
        self.callbacks.call("on_fit_end", self)
        self.callbacks.call("teardown", self, "fit")

    def validate(self, batches: Iterator[MicrobatchStep], step_fn=None) -> None:
        """Run the module's ``validation_step`` over ``batches``, under its hooks.

        ``step_fn(loop, batch)`` overrides the module's. Collective: every rank
        must supply the same number of batches.
        """
        if step_fn is None:

            def step_fn(loop, batch, idx=0):
                return loop.module.validation_step(batch, idx)

        self.callbacks.call("on_validation_start", self)
        self.callbacks.call("on_validation_epoch_start", self)
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                self.callbacks.call("on_validation_batch_start", self, batch, batch_idx)
                out = step_fn(self, batch)
                self.callbacks.call(
                    "on_validation_batch_end", self, out, batch, batch_idx
                )
        self.callbacks.call("on_validation_epoch_end", self)
        self.callbacks.call("on_validation_end", self)

    # Kept so an existing caller of the old name keeps working.
    run = fit


def build_loss_fn(
    stage_module, denom: Callable[[], int], num_microbatches: int = 1
) -> Callable:
    """A schedule ``loss_fn`` that normalizes by the step's trained tokens.

    Takes ``stage_module``, not the module it wraps: the schedule calls this
    outside the stage forward, so only the wrapper carries any autocast.

    With a multi-stream boundary the first stream is the hidden state and any
    scalar stream is an accumulator added to the loss **after** normalization.
    An accumulator carries a per-microbatch *mean*, so it is averaged over
    ``num_microbatches`` rather than summed with them. See docs/kohakuwupipe/streams.md.
    """
    aux_scale = 1.0 / max(num_microbatches, 1)

    def loss_fn(output, target):
        streams = split_streams(output)
        loss, _ = stage_module.loss(streams[0], target)
        loss = loss / max(denom(), 1)
        for stream in streams[1:]:
            if stream is not None and stream.shape[-1] == 1:
                loss = loss + aux_scale * reduce_accumulator(stream)
        return loss

    return loss_fn
