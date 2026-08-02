"""Whole-model checkpoints from a split model, in the Lightning file layout.

Weights and optimizer state are both stored under whole-model names, so a file
one split writes loads into another that wraps its stage the same way. Saving is
collective on every rank; rank 0 writes. See docs/kohakuwupipe/checkpoint.md.
"""

from typing import Any

import torch
import torch.distributed as dist

# Legacy per-rank optimizer layout: readable, never written.
STAGE_STATES = "pipeline_stage_optimizer_states"
NAMED_STATES = "named_optimizer_state"


def global_names(stage_module, start_layer: int, block_attr: str = "blocks") -> dict:
    """This stage's state-dict keys mapped to their whole-model names.

    Only ``<block_attr>.<i>`` moves: a stage numbers its blocks from zero, the
    whole model from ``start_layer``.
    """
    prefix = f"{block_attr}."
    names = {}
    for key in stage_module.state_dict():
        if key.startswith(prefix):
            index, _, tail = key[len(prefix) :].partition(".")
            names[key] = f"{prefix}{int(index) + start_layer}.{tail}"
        else:
            names[key] = key
    return names


def gather_state_dict(stage_module, start_layer: int, block_attr: str = "blocks"):
    """Every stage's slice under whole-model names. Collective: all ranks call."""
    local = stage_module.state_dict()
    names = global_names(stage_module, start_layer, block_attr)
    mine = {name: local[key].detach().cpu() for key, name in names.items()}
    if not dist.is_initialized():
        return mine
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, mine)
    merged: dict[str, torch.Tensor] = {}
    for part in parts:
        merged.update(part)
    return merged


def load_state_dict(
    stage_module,
    state: dict,
    start_layer: int,
    block_attr: str = "blocks",
    strict: bool = True,
) -> None:
    """Take this stage's slice out of a whole-model state dict."""
    names = global_names(stage_module, start_layer, block_attr)
    missing = [name for name in names.values() if name not in state]
    if missing and strict:
        raise KeyError(
            f"checkpoint is missing {len(missing)} tensors this stage needs, "
            f"first: {missing[:3]}"
        )
    local = {key: state[name] for key, name in names.items() if name in state}
    stage_module.load_state_dict(local, strict=strict)


def parameter_names(stage_module, start_layer: int, block_attr: str = "blocks"):
    """``id(parameter) -> whole-model parameter name`` for this stage's slice."""
    names = global_names(stage_module, start_layer, block_attr)
    return {
        id(param): names[key]
        for key, param in stage_module.named_parameters()
        if key in names
    }


def _ordered_params(optimizer):
    """Optimizer state indices in the order ``state_dict`` numbers them."""
    return [param for group in optimizer.param_groups for param in group["params"]]


def gather_optimizer_state(optimizer, name_of: dict[int, str]) -> dict[str, Any]:
    """Per-parameter optimizer state under whole-model names, merged across ranks.

    Collective: every rank calls. Param groups are not stored; each rank rebuilds
    its own. See docs/kohakuwupipe/checkpoint.md.
    """
    local = optimizer.state_dict()
    index_to_name = {
        index: name_of[id(param)]
        for index, param in enumerate(_ordered_params(optimizer))
        if id(param) in name_of
    }
    mine = {
        index_to_name[i]: to_cpu(entry)
        for i, entry in local["state"].items()
        if i in index_to_name
    }
    if not dist.is_initialized():
        return {NAMED_STATES: mine}
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, mine)
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return {NAMED_STATES: merged}


def load_optimizer_state(
    optimizer, state: dict[str, Any], rank: int, name_of: dict[int, str] | None = None
) -> None:
    """Restore this rank's slice of a whole-model optimizer state.

    Falls back to the legacy per-rank list, which raises unless this run has the
    world size that wrote it. See docs/kohakuwupipe/checkpoint.md.
    """
    if NAMED_STATES not in state:
        if STAGE_STATES in state:
            parts = state[STAGE_STATES]
            world = dist.get_world_size() if dist.is_initialized() else 1
            if len(parts) != world:
                raise ValueError(
                    f"this checkpoint carries {STAGE_STATES} for {len(parts)} "
                    f"stages and this run has {world}: the legacy per-rank layout "
                    "loads only under the split that wrote it. Re-save under "
                    f"{len(parts)} ranks to get a {NAMED_STATES} file, which any "
                    "split can read"
                )
            state = parts[rank]
        optimizer.load_state_dict(state)
        return
    if name_of is None:
        raise ValueError("a name-keyed optimizer state needs `name_of`")
    whole = state[NAMED_STATES]
    # Only `state` is portable; this rank keeps the groups it built.
    local = optimizer.state_dict()
    local["state"] = {
        index: whole[name_of[id(param)]]
        for index, param in enumerate(_ordered_params(optimizer))
        if id(param) in name_of and name_of[id(param)] in whole
    }
    optimizer.load_state_dict(local)


def to_cpu(value):
    """Every tensor in a nested container, moved to the CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def save(
    path: str,
    stage_module,
    optimizer,
    start_layer: int,
    global_step: int,
    rank: int,
    block_attr: str = "blocks",
    extra: dict | None = None,
) -> bool:
    """Write a Lightning-shaped checkpoint. Collective; only rank 0 writes.

    Returns whether this rank wrote the file.
    """
    names = parameter_names(stage_module, start_layer, block_attr)
    payload = {
        "state_dict": gather_state_dict(stage_module, start_layer, block_attr),
        "optimizer_states": [gather_optimizer_state(optimizer, names)],
        "global_step": global_step,
        "epoch": 0,
        "pytorch-lightning_version": "kohakuwupipe",
    }
    if extra:
        payload.update(extra)
    if rank != 0:
        return False
    torch.save(payload, path)
    return True


def load(
    path: str,
    stage_module,
    optimizer,
    start_layer: int,
    rank: int,
    block_attr: str = "blocks",
    strict: bool = True,
) -> dict:
    """Restore this rank's slice and its optimizer entry; returns the payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    load_state_dict(
        stage_module, payload["state_dict"], start_layer, block_attr, strict
    )
    states = payload.get("optimizer_states") or []
    if optimizer is not None and states:
        load_optimizer_state(
            optimizer,
            states[0],
            rank,
            parameter_names(stage_module, start_layer, block_attr),
        )
    return payload
