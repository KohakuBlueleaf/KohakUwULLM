"""Whole-model checkpoints from a split model, in the Lightning file layout.

Each rank owns a disjoint slice, so a rank-0-only save keeps one stage and its
blocks are renumbered from zero -- a file that loads without error and is wrong.
Weights are gathered under whole-model names; optimizer state stays a per-rank
list, since stages share no parameters. See docs/kohakuwupipe/checkpoint.md.
"""

from typing import Any

import torch
import torch.distributed as dist

STAGE_STATES = "pipeline_stage_optimizer_states"


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


def gather_optimizer_state(optimizer) -> dict[str, Any]:
    """Every rank's optimizer state as a list, gathered onto rank 0."""
    local = to_cpu(optimizer.state_dict())
    if not dist.is_initialized():
        return local
    is_zero = dist.get_rank() == 0
    parts = [None] * dist.get_world_size() if is_zero else None
    dist.gather_object(local, parts, dst=0)
    return {STAGE_STATES: parts} if is_zero else {}


def load_optimizer_state(optimizer, state: dict[str, Any], rank: int) -> None:
    """Give this rank its own entry out of a gathered optimizer state."""
    if STAGE_STATES in state:
        state = state[STAGE_STATES][rank]
    optimizer.load_state_dict(state)


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
    payload = {
        "state_dict": gather_state_dict(stage_module, start_layer, block_attr),
        "optimizer_states": [gather_optimizer_state(optimizer)],
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
        load_optimizer_state(optimizer, states[0], rank)
    return payload
