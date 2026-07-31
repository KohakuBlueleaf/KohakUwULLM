"""Callbacks: throughput, periodic logging, and checkpointing.

Every one is interval-gated, because reading a device tensor costs a
synchronization and the loop deliberately leaves them unread. See docs/kohakuwupipe/loop.md.
"""

import os
import time

import torch
import torch.distributed as dist

from kohakuwupipe.io import checkpoint
from kohakuwupipe.training.hooks import Callback
from kohakuwupipe.training.loop import PipelineLoop, StepOutput
from kohakuwupipe.utils.logging import get_logger


class Throughput(Callback):
    """Tokens/s and step time over the window since the last report.

    Trailing, not cumulative: a checkpoint or a preview costs tens of seconds,
    and a running average carries that into every later number. See docs/kohakuwupipe/loop.md.

    Args:
        every_n_steps: report cadence, and the window width.
        warmup_steps: steps excluded from the first window; a varlen stream
            compiles new shapes for a while and that is not throughput.
        report: ``(dict) -> None``; rank 0 only.
    """

    def __init__(self, every_n_steps: int = 50, warmup_steps: int = 8, report=None):
        self.every_n_steps = every_n_steps
        self.warmup_steps = warmup_steps
        self.report = report or _print_rank0
        self._start = None
        self._seen = 0
        self._trained = 0
        self._steps = 0

    def on_train_batch_end(
        self, loop: PipelineLoop, out: StepOutput, batch=None, batch_idx=0
    ) -> None:
        # `>=`: a resumed run's first index is the restored step + 1.
        if self._start is None:
            if out.index >= self.warmup_steps:
                self._reset()
            return
        self._seen += out.seen
        self._trained += out.trained
        self._steps += 1
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        _sync()
        elapsed = time.perf_counter() - self._start
        steps = max(self._steps, 1)
        if loop.rank == 0:
            self.report(
                {
                    "step": out.index,
                    "tokens_per_s": self._seen / max(elapsed, 1e-9),
                    "trained_tokens_per_s": self._trained / max(elapsed, 1e-9),
                    "ms_per_step": 1e3 * elapsed / steps,
                }
            )
        self._reset()

    def _reset(self) -> None:
        """Open a fresh window, excluding whatever the report itself cost."""
        _sync()
        self._start = time.perf_counter()
        self._seen = self._trained = self._steps = 0


class LossLog(Callback):
    """Read the step's loss on an interval and hand it to ``report``."""

    def __init__(self, every_n_steps: int = 10, report=None, ema: float = 0.9):
        self.every_n_steps = every_n_steps
        self.report = report or _print_rank0
        self.ema_decay = ema
        self.ema = None

    def on_train_batch_end(
        self, loop: PipelineLoop, out: StepOutput, batch=None, batch_idx=0
    ) -> None:
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        if out.loss is None:
            return
        loss = float(out.loss)
        self.ema = (
            loss
            if self.ema is None
            else (self.ema_decay * self.ema + (1 - self.ema_decay) * loss)
        )
        row = {"step": out.index, "loss": loss, "loss_ema": self.ema}
        row.update({k: float(v) for k, v in out.extra.items()})
        if loop.rank == 0:
            self.report(row)


class Checkpoint(Callback):
    """Write a whole-model checkpoint every ``every_n_steps``. Collective."""

    def __init__(
        self,
        dirpath: str,
        start_layer: int,
        every_n_steps: int = 1000,
        block_attr: str = "blocks",
        keep_last: bool = True,
    ):
        self.dirpath = dirpath
        self.start_layer = start_layer
        self.every_n_steps = every_n_steps
        self.block_attr = block_attr
        self.keep_last = keep_last

    def on_train_batch_end(
        self, loop: PipelineLoop, out: StepOutput, batch=None, batch_idx=0
    ) -> None:
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        self.write(loop, f"step-{out.index}.ckpt")
        if self.keep_last:
            self.write(loop, "last.ckpt")

    def write(self, loop: PipelineLoop, name: str) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        checkpoint.save(
            os.path.join(self.dirpath, name),
            loop.stage_module,
            loop.optimizer,
            self.start_layer,
            loop.global_step,
            loop.rank,
            block_attr=self.block_attr,
        )


log = get_logger(__name__)


def _sync() -> None:
    """Drain the device before reading a clock, where there is one."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _print_rank0(row: dict) -> None:
    """Default reporter: one structured log record."""
    if dist.is_initialized() and dist.get_rank():
        return
    log.info("metrics", **row)
