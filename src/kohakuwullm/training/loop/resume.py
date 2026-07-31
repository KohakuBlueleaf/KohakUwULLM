"""RNG and dataloader position -- the run state a Lightning checkpoint misses.

Everything here must survive ``torch.load(weights_only=True)``. See docs/guides/training.md.
"""

import random
import warnings

import numpy as np
import torch


def rng_state() -> dict:
    """Snapshot every RNG a training step can consume."""
    keys, pos, has_gauss, cached = np.random.get_state()[1:]
    state = {
        "python": random.getstate(),
        "numpy_key": torch.from_numpy(keys.astype(np.int64)),
        "numpy_rest": (int(pos), int(has_gauss), float(cached)),
        "torch": torch.get_rng_state(),
    }
    # `torch.cuda.get_rng_state` initializes CUDA as a side effect.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def load_rng_state(state: dict) -> None:
    """Restore what :func:`rng_state` captured, skipping anything absent."""
    if "python" in state:
        python_state = state["python"]
        # Re-tupled: a json/yaml round trip returns lists, which `setstate` rejects.
        random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    if "numpy_key" in state:
        pos, has_gauss, cached = state["numpy_rest"]
        keys = state["numpy_key"].numpy().astype(np.uint32)
        np.random.set_state(("MT19937", keys, pos, has_gauss, cached))
    if "torch" in state:
        torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].to(torch.uint8).cpu())


def _stateful(obj) -> bool:
    """Whether ``obj`` exposes both halves of the state-dict protocol."""
    return (
        obj is not None
        and callable(getattr(obj, "state_dict", None))
        and callable(getattr(obj, "load_state_dict", None))
    )


def loader_state(loader) -> dict | None:
    """Position of the train dataloader; ``None`` if it tracks none. Never its dataset."""
    return None if not _stateful(loader) else {"state": loader.state_dict()}


def load_loader_state(loader, state: dict | None) -> bool:
    """Push ``state`` back into the loader. True if it was applied."""
    if not state:
        return False
    if not _stateful(loader):
        warnings.warn(
            "the checkpoint carries a dataloader position but the loader supplied "
            "at resume has no load_state_dict to restore it into; the run will "
            "repeat data it has already trained on",
            stacklevel=2,
        )
        return False
    loader.load_state_dict(state["state"])
    return True


def resumable(loader) -> bool:
    """Whether the loader will carry its position across a restart."""
    return _stateful(loader)
