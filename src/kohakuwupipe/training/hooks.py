"""The callback protocol: Lightning's hook names, on a pipeline loop.

Names and order follow ``lightning.pytorch.Callback`` so an existing callback
drops in unchanged wherever its hooks exist here. Hooks the pipeline has no
analogue for are absent rather than silently never called. See docs/kohakuwupipe/loop.md.
"""

from typing import Any


class Callback:
    """Every hook the loop can call. Override only what you need.

    Every rank calls every hook in the same order, so a hook that runs a
    collective is safe and a hook that returns early on one rank only is not.
    """

    # -- lifecycle -------------------------------------------------------- #

    def setup(self, loop, stage: str) -> None: ...

    def teardown(self, loop, stage: str) -> None: ...

    def on_fit_start(self, loop) -> None: ...

    def on_fit_end(self, loop) -> None: ...

    def on_train_start(self, loop) -> None: ...

    def on_train_end(self, loop) -> None: ...

    def on_train_epoch_start(self, loop) -> None: ...

    def on_train_epoch_end(self, loop) -> None: ...

    # -- the step --------------------------------------------------------- #

    def on_train_batch_start(self, loop, batch: Any, batch_idx: int) -> None: ...

    def on_train_batch_end(self, loop, out, batch: Any, batch_idx: int) -> None: ...

    def on_before_zero_grad(self, loop, optimizer) -> None: ...

    def on_before_backward(self, loop) -> None: ...

    def on_after_backward(self, loop) -> None: ...

    def on_before_optimizer_step(self, loop, optimizer) -> None: ...

    # -- validation ------------------------------------------------------- #

    def on_validation_start(self, loop) -> None: ...

    def on_validation_end(self, loop) -> None: ...

    def on_validation_epoch_start(self, loop) -> None: ...

    def on_validation_epoch_end(self, loop) -> None: ...

    def on_validation_batch_start(self, loop, batch: Any, batch_idx: int) -> None: ...

    def on_validation_batch_end(
        self, loop, out, batch: Any, batch_idx: int
    ) -> None: ...

    # -- checkpoints ------------------------------------------------------ #

    def on_save_checkpoint(self, loop, checkpoint: dict) -> None: ...

    def on_load_checkpoint(self, loop, checkpoint: dict) -> None: ...

    def state_dict(self) -> dict:
        """State to carry in the checkpoint under this callback's class name."""
        return {}

    def load_state_dict(self, state: dict) -> None: ...

    # -- exceptions ------------------------------------------------------- #

    def on_exception(self, loop, exception: BaseException) -> None: ...


class CallbackList:
    """Fans one hook call out to every callback, in registration order.

    Unknown hook names raise rather than silently doing nothing, so a typo in a
    caller is a failure and not a callback that never fires.
    """

    HOOKS = frozenset(
        name for name in vars(Callback) if name.startswith(("on_", "setup", "teardown"))
    )

    def __init__(self, callbacks=()) -> None:
        self.callbacks = list(callbacks)

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def call(self, hook: str, *args, **kwargs) -> None:
        """Invoke ``hook`` on every callback that defines it."""
        if hook not in self.HOOKS:
            raise AttributeError(f"unknown callback hook {hook!r}")
        for callback in self.callbacks:
            fn = getattr(callback, hook, None)
            if fn is not None:
                fn(*args, **kwargs)

    def state_dict(self) -> dict:
        """Every callback's state, keyed by class name."""
        return {
            type(cb).__qualname__: state
            for cb in self.callbacks
            if (state := cb.state_dict())
        }

    def load_state_dict(self, state: dict) -> None:
        for callback in self.callbacks:
            entry = state.get(type(callback).__qualname__)
            if entry is not None:
                callback.load_state_dict(entry)
