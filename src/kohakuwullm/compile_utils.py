"""Configurable ``torch.compile``: whole-model OR per-module, with exclusions.

The compile *spec* is a dict, or ``None`` to disable::

    {"mode": "module" | "model" | "off",
     "targets": ["blocks"],     # (module mode) containers whose children get compiled
     "exclude": ["blocks.0"],   # (module mode) dotted child names to skip
     "compile_mode": "reduce-overhead",   # torch.compile's own `mode=`
     **compile_kwargs}          # forwarded to torch.compile, e.g. dynamic=, fullgraph=

``mode="model"`` prefixes every checkpoint key with ``_orig_mod.``; ``"module"``
keeps them clean. See docs/guides/writing-configs.md.
"""

import torch
import torch.nn as nn


def raise_recompile_limit(needed: int) -> None:
    """Let Dynamo hold ``needed`` cache entries against one code object.

    Raised only, never lowered; the config is process-global.
    """
    config = torch._dynamo.config
    config.recompile_limit = max(config.recompile_limit, needed)
    config.accumulated_recompile_limit = max(
        config.accumulated_recompile_limit, 8 * needed
    )


def apply_compile(model: nn.Module, spec: dict | None) -> nn.Module:
    """Apply ``torch.compile`` per ``spec``; return the (possibly wrapped) model."""
    if not spec:
        return model
    opts = dict(spec)
    mode = opts.pop("mode", "module")
    targets = opts.pop("targets", ["blocks"])
    exclude = set(opts.pop("exclude", []))
    # `mode` is this spec's own; torch.compile's is reached as `compile_mode`.
    if "compile_mode" in opts:
        opts["mode"] = opts.pop("compile_mode")
    match mode:
        case "off":
            return model
        case "model":
            raise_recompile_limit(16)
            return torch.compile(model, **opts)
        case "module":
            raise_recompile_limit(_compile_children(model, targets, exclude, opts) + 8)
            return model
        case _:
            raise ValueError(f"unknown compile mode {mode!r}")


def _compile_children(model, targets, exclude, opts) -> int:
    """Compile each target container's children; return how many were compiled."""
    compiled = 0
    for target in targets:
        container = model.get_submodule(target) if target else model
        for name, child in container.named_children():
            full = f"{target}.{name}" if target else name
            if full in exclude:
                continue
            child.compile(**opts)
            compiled += 1
    return compiled
