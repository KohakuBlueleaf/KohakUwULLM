"""Small shared helpers, kept dependency-light. None run in the hot training loop."""

import importlib
from typing import Any

import torch


def import_class(path: Any) -> Any:
    """Import a dotted path ``"package.module.Name"`` and return that attribute.

    Non-string inputs are returned unchanged.
    """
    if not isinstance(path, str):
        return path
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"{path!r} is not a dotted import path")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total number of parameters in ``module``."""
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


def count_active_parameters(module: torch.nn.Module) -> int:
    """Parameters touched by one token's forward pass.

    Equals :func:`count_parameters` for a dense model. Modules opt in by exposing an
    ``active_parameters()`` method; anything else contributes all of its own
    (non-child) parameters.
    """
    active = getattr(module, "active_parameters", None)
    if callable(active):
        return active()
    total = sum(p.numel() for p in module.parameters(recurse=False))
    return total + sum(count_active_parameters(c) for c in module.children())


def autofill_schedule_steps(
    scheduler_config: dict,
    train_steps: int,
    warmup_ratio: float = 0.0,
    end_from_steps: bool = True,
    warmup_from_steps: bool = True,
    warmup_steps: int | None = None,
) -> dict:
    """Fill AnySchedule ``end`` / ``warmup`` from a computed ``train_steps``.

    ``warmup_steps`` is an absolute count and wins over ``warmup_ratio``.
    Mutates and returns the dict.
    """
    if end_from_steps:
        scheduler_config["end"] = train_steps
    if warmup_from_steps:
        warmup = (
            int(warmup_steps)
            if warmup_steps is not None
            else int(train_steps * warmup_ratio)
        )
        # A composer takes its warmup on the first phase. See docs/guides/training.md.
        phases = scheduler_config.get("schedules")
        if scheduler_config.get("mode") == "composer" and phases:
            phases[0]["warmup"] = warmup
        else:
            scheduler_config["warmup"] = warmup
    return scheduler_config


def human_count(n: int) -> str:
    """``1234567 -> '1.23M'`` for log lines."""
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)
