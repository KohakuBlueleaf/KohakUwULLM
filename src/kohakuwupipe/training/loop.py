"""The pipeline training loop: the step, and nothing between it and the GPU.

Nothing in :meth:`PipelineLoop.step` reads a device tensor on the host, so a
throughput regression is a regression in this file. Model-specific work -- an
fp8 weight refresh, a router-bias update -- arrives as ``post_step``.

See docs/kohakuwupipe/loop.md.
"""

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch
import torch.distributed as dist

from kohakuwupipe.parallel.streams import reduce_accumulator, split_streams
from kohakuwupipe.training import hooks


@dataclass
class MicrobatchStep:
    """One optimizer step's data, already split into microbatches.

    ``inputs`` is read on the first stage only and ``target`` on the last;
    ``layout`` is whatever the stage module's ``set_seq_info`` accepts, and
    ``trained`` is what the loss is normalized by.

    Neither is examined here: ``inputs`` goes to the stage and ``target`` to
    ``loss``. A tuple ``inputs`` is the first stage's argument list, so a model
    conditioned on more than one tensor passes them there; ``target`` may be any
    structure. Subclass to add fields. See docs/kohakuwupipe/module.md.
    """

    inputs: Any
    target: Any = None
    layout: Any = None
    trained: int = 0

    def map(self, fn):
        """A copy with ``fn`` applied to every tensor, in every field.

        Recurses through tuples, lists and dicts, so a subclass's own fields
        are carried without overriding anything.
        """
        return replace(
            self,
            **{f.name: _map_tensors(getattr(self, f.name), fn) for f in fields(self)},
        )

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        """This step on ``device``, subclass fields included."""
        return self.map(
            lambda t: t.to(device=device, dtype=dtype, non_blocking=non_blocking)
        )

    def pin_memory(self):
        """This step in pinned memory, for an overlapped host-to-device copy."""
        return self.map(lambda t: t.pin_memory())

    def tensors(self):
        """Every tensor this step carries, in field order."""
        found: list[torch.Tensor] = []
        self.map(lambda t: (found.append(t), t)[1])
        return found


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


def _map_tensors(value, fn):
    """``fn`` over every tensor in a nested container, structure preserved."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, fn) for item in value)
    return value


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
        scaler=None,
        grad_clip: float = 0.0,
        post_step: Callable[[Any], dict[str, torch.Tensor]] | None = None,
        callbacks: Iterable[Callback] = (),
    ) -> None:
        self.stage_module = stage_module
        self.schedule = schedule
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.rank = rank
        self.world = world
        self.micro_tokens = micro_tokens
        self.num_microbatches = num_microbatches
        self.grad_clip = grad_clip
        self.post_step = post_step
        self.callbacks = hooks.CallbackList(callbacks)
        self.global_step = 0
        self.epoch = 0
        self.max_steps = 0
        # This step's target, one entry per microbatch. Held, never sent.
        self.target_pieces: list = [None] * num_microbatches
        self._index = None
        # Python ints: a run passes 2^31 tokens in under an hour.
        self.tokens_seen = 0
        self.tokens_trained = 0
        self._elapsed_offset = 0.0
        self._clock_start: float | None = None
        self.is_first = rank == 0
        self.is_last = rank == world - 1
        self._set_layout = getattr(stage_module, "set_seq_info", None)
        self.module = None

    def step(self, batch: MicrobatchStep, batch_idx: int = 0) -> StepOutput:
        """One optimizer step. Collective: no rank may return early."""
        self.callbacks.call("on_train_batch_start", self, batch, batch_idx)
        self.callbacks.call("on_before_zero_grad", self, self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if self._set_layout is not None:
            self._set_layout(batch.layout)

        # The module decides what reaches the schedule, and when.
        self._losses = [] if self.is_last else None
        self.callbacks.call("on_before_backward", self)
        if self.module is not None:
            self.module.training_step(batch, batch_idx)
        else:
            self.forward_backward(batch.inputs, batch.target)
        losses = self._losses
        self.callbacks.call("on_after_backward", self)

        # One host sync per step, and the only one: a scaler has to branch.
        overflow = False
        scale = 1.0
        if self.scaler is not None and self.scaler.enabled:
            scale = self.scaler.scale_value
            found = self.scaler.unscale_(list(self.stage_module.parameters()))
            overflow = bool(self.scaler.agree(found).item())
            self.scaler.update(overflow)
        grad_norm = None
        if not overflow:
            if self.grad_clip:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.stage_module.parameters(), self.grad_clip
                )
            self.callbacks.call("on_before_optimizer_step", self, self.optimizer)
            self.optimizer.step()
        extra = self.post_step(self.stage_module) if self.post_step else {}
        if self.module is not None:
            extra.update(self.module.pop_metrics())
        if grad_norm is not None:
            extra["grad_norm"] = grad_norm.detach()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.scaler is not None and self.scaler.enabled:
            extra["scale"] = torch.tensor(scale)
            extra["overflow"] = torch.tensor(float(overflow))
        self.global_step += 1
        # Every rank draws the identical batch, so these are already global.
        seen = self.micro_tokens * self.num_microbatches
        self.tokens_seen += seen
        self.tokens_trained += int(batch.trained)
        out = StepOutput(
            index=self.global_step,
            loss=self.broadcast_loss(losses, scale),
            seen=seen,
            trained=int(batch.trained),
            extra=extra,
        )
        self.callbacks.call("on_train_batch_end", self, out, batch, batch_idx)
        return out

    def broadcast_loss(self, losses, scale: float = 1.0) -> torch.Tensor | None:
        """The step's loss, from the last stage to every rank, as a device tensor.

        Microbatch losses **sum**: ``loss_fn`` already divided each one by the
        step's token count. ``scale`` divides the loss scaler back out, so what
        is reported is the loss the model actually has.
        See docs/kohakuwupipe/loop.md.
        """
        if losses is not None and not dist.is_initialized():
            return torch.stack(losses).sum().detach() / scale if losses else None
        if not dist.is_initialized():
            return None
        device = next(self.stage_module.parameters()).device
        buf = torch.zeros(1, device=device)
        if losses:
            buf[0] = torch.stack([x.detach().float() for x in losses]).sum() / scale
        dist.broadcast(buf, src=self.world - 1)
        return buf[0]

    def fit(self, batches: Iterator[MicrobatchStep], max_steps: int) -> None:
        """Step until ``max_steps``. Every rank must consume the same stream.

        ``on_exception`` fires on every rank before the error propagates, so a
        callback can tear its own state down.
        """
        self.max_steps = max_steps
        self._clock_start = time.perf_counter()
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

    def forward_backward(self, inputs, target=None, layout=None):
        """Run the whole model over one step: forward and backward, all stages.

        The only part of a step this package owns. Call it once per step from
        ``training_step``, with whatever the module computed. A tuple ``inputs``
        is the first stage's argument list; ``target`` may be any structure and
        arrives at ``loss`` one microbatch at a time.

        Under ``torch.no_grad()`` this runs forward only -- the schedule's
        ``step`` refuses a disabled-grad context, so ``eval`` is used instead.
        That makes ``no_grad`` the knob, as it is everywhere else in torch.

        Returns the step's loss on the last stage and ``None`` elsewhere -- a
        device tensor, already normalized and unscaled, so logging it costs no
        synchronization until something reads it.

        Collective: every rank must call it the same number of times.
        See docs/kohakuwupipe/module.md.
        """
        if layout is not None and self._set_layout is not None:
            self._set_layout(layout)
        run = self.schedule.step if torch.is_grad_enabled() else self.schedule.eval
        if self._losses is None:
            self._losses = [] if self.is_last else None
        if self.is_first:
            args = inputs if isinstance(inputs, tuple) else (inputs,)
            run(*args)
        elif self.is_last:
            self.target_pieces[:] = split_target(target, self.num_microbatches)
            run(target=self._target_index(), losses=self._losses)
        else:
            run()
        if not self._losses:
            return None
        scale = 1.0
        if self.scaler is not None and self.scaler.enabled:
            scale = self.scaler.scale_value
        return torch.stack([x.detach() for x in self._losses]).sum() / scale

    def _target_index(self) -> torch.Tensor:
        """``arange(num_microbatches)``, the handle the schedule splits for us."""
        if self._index is None:
            device = next(self.stage_module.parameters()).device
            self._index = torch.arange(self.num_microbatches, device=device)
        return self._index

    def elapsed(self) -> float:
        """Training wall-clock, excluding the gap between a crash and a restart."""
        if self._clock_start is None:
            return self._elapsed_offset
        return self._elapsed_offset + (time.perf_counter() - self._clock_start)

    def progress_state(self) -> dict:
        """Cumulative token and time totals, for a checkpoint."""
        return {
            "tokens_seen": self.tokens_seen,
            "tokens_trained": self.tokens_trained,
            "elapsed": self.elapsed(),
        }

    def load_progress_state(self, state: dict | None) -> None:
        """Continue a restored run's totals instead of restarting them at zero."""
        if not state:
            return
        self.tokens_seen = int(state.get("tokens_seen", 0))
        self.tokens_trained = int(state.get("tokens_trained", 0))
        self._elapsed_offset = float(state.get("elapsed", 0.0))

    # Kept so an existing caller of the old name keeps working.
    run = fit


def split_target(target, chunks: int) -> list:
    """``target`` split into ``chunks`` microbatch pieces, whatever its structure.

    Tensors split along dim 0; tuples, lists and dicts split element-wise and
    keep their type; anything else rides along whole, once per microbatch.
    See docs/kohakuwupipe/module.md.
    """
    if target is None:
        return [None] * chunks
    if torch.is_tensor(target):
        return list(torch.tensor_split(target, chunks))
    if isinstance(target, dict):
        parts = {key: split_target(value, chunks) for key, value in target.items()}
        return [{key: value[i] for key, value in parts.items()} for i in range(chunks)]
    if isinstance(target, (tuple, list)):
        parts = [split_target(value, chunks) for value in target]
        return [type(target)(part[i] for part in parts) for i in range(chunks)]
    return [target] * chunks


def build_loss_fn(
    stage_module,
    denom: Callable[[], int],
    num_microbatches: int = 1,
    scaler=None,
    pieces: list | None = None,
    objective: Callable | None = None,
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
    objective = objective or stage_module.loss

    def loss_fn(output, target):
        streams = split_streams(output)
        # `target` is the microbatch index; the batch itself never left the rank.
        batch = target if pieces is None else pieces[int(target)]
        result = objective(streams[0], batch)
        loss = result[0] if isinstance(result, tuple) else result
        loss = loss / max(denom(), 1)
        for stream in streams[1:]:
            if stream is not None and stream.shape[-1] == 1:
                loss = loss + aux_scale * reduce_accumulator(stream)
        return loss if scaler is None else scaler.scale(loss)

    return loss_fn
