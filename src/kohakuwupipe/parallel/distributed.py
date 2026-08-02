"""Process-group setup for a pipeline: rank wiring, and non-eager NCCL init.

``init_pipeline(...)`` initializes without ``device_id``, which is what gives
each send/recv pair its own communicator.
See docs/kohakuwupipe/distributed.md.
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from kohakuwupipe.utils.logging import configure


@dataclass(frozen=True)
class PipelineRanks:
    """This process's place in the pipeline."""

    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.world - 1


def init_pipeline(backend: str = "nccl", timeout=None) -> PipelineRanks:
    """Initialize the default process group from the launcher's environment.

    Deliberately passes no ``device_id``: see the module docstring.
    """
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        kwargs = {"timeout": timeout} if timeout is not None else {}
        dist.init_process_group(backend, rank=rank, world_size=world, **kwargs)
    # Logging is inert until a handler exists, so wire it at the entry point.
    configure(rank=rank)
    return PipelineRanks(rank=rank, world=world, local_rank=local_rank, device=device)


def eager_init_in_use() -> bool:
    """Whether the default group was built with a ``device_id``.

    True means P2P shares one communicator and serializes.
    See docs/kohakuwupipe/distributed.md.
    """
    if not dist.is_initialized():
        return False
    group = dist.distributed_c10d._get_default_group()
    return getattr(group, "bound_device_id", None) is not None


def warn_on_eager_init() -> str:
    """A message naming the throughput cost, or ``""`` when init is lazy."""
    if not eager_init_in_use():
        return ""
    return (
        "the default process group was initialized with a device_id, so NCCL "
        "routes every pipeline send/recv through one communicator and serializes "
        "them; initialize without device_id (see kohakuwupipe.init_pipeline)"
    )


def shutdown() -> None:
    """Tear the default group down; safe to call when it was never built."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
