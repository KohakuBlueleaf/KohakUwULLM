"""Restarting a run mid-epoch, exactly where it stopped.

A position is a pass number, a shard offset, and the documents the packer has
drawn but not yet emitted. It is pushed by the worker, recorded on consumption,
and all-gathered so every rank's position is in every rank's checkpoint.

**Do not rely on Lightning to restore it** -- ``LMTrainer`` does it itself, so
``load_state_dict`` must stay idempotent. See docs/internals/data.md.
"""

import os
import warnings

import torch.distributed as dist
import torch.utils.data as data

from kohakuwullm.data.loader.iterative import TokenBudgetIterableDataset, make_loader


def resolve_rank_world(
    rank: int | None = None, world_size: int | None = None
) -> tuple[int, int]:
    """This process's data-parallel identity, for sharding.

    Explicit arguments win, then a live process group, then ``RANK`` /
    ``WORLD_SIZE`` from the launcher's environment.
    """
    if rank is not None and world_size is not None:
        return rank, world_size
    if dist.is_available() and dist.is_initialized():
        found = (dist.get_rank(), dist.get_world_size())
    else:
        found = (
            int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))),
            int(os.environ.get("WORLD_SIZE", 1)),
        )
    return (
        found[0] if rank is None else rank,
        found[1] if world_size is None else world_size,
    )


class ResumablePackedDataset(TokenBudgetIterableDataset):
    """A token-budget dataset that yields ``(batch, position)`` pairs."""

    def __iter__(self):
        return self._iter_stateful()


class ResumableLoader:
    """DataLoader wrapper that keeps the stream position checkpointable.

    Args:
        loader: the underlying ``DataLoader`` over a dataset yielding
            ``(batch, position)`` pairs.
        dataset: that dataset, held so a restored position can be armed on it.
        rank / world_size: the data-parallel identity the shards were cut with.
        num_workers: shards are ``rank * num_workers + worker_id``, so a resume
            into a different worker count is a different partition entirely.
        steps_per_epoch: batches this rank yields per epoch when the count is
            pinned, or ``None``. Already per-rank -- unlike a map-style loader's
            ``__len__``, it must not be divided by the world size again.

    A restart rotates each worker cycle (:meth:`_rotation`) so the continuation
    matches an uninterrupted run batch for batch, and commits the position one
    batch behind the hand-off when the batch count is not pinned. See docs/internals/data.md.
    """

    def __init__(
        self,
        loader: data.DataLoader,
        dataset: TokenBudgetIterableDataset,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.loader = loader
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.steps_per_epoch = steps_per_epoch
        self._positions: dict[int, dict] = {}
        self._rotate = 0
        # Selected once, on the same condition Lightning's fetcher prefetches on.
        self._lag = 1 if steps_per_epoch is None else 0
        self._pending: dict | None = None

    def __len__(self) -> int:
        """Batches this rank yields per epoch; raises unless the count is pinned."""
        if self.steps_per_epoch is None:
            raise TypeError(
                "this loader packs to a token budget, so its batch count is "
                "data-dependent; pass batches_per_epoch to pin it"
            )
        return self.steps_per_epoch

    def __iter__(self):
        """Batches with their positions stripped off and recorded on the way past."""
        hold, self._rotate = self._rotate, 0
        # An abandoned epoch's uncommitted position belongs to a pass this
        # iterator is past, so it is dropped rather than carried.
        self._pending = None
        if not hold:
            for arrival in self.loader:
                yield self._consume(arrival)
            self._flush()
            return
        # Off-phase resume: buffer the whole cycle, since a cycle that ends short
        # must be left in arrival order and it is short only once it ends.
        width = max(self.num_workers, 1)
        cycle: list[tuple] = []
        for arrival in self.loader:
            cycle.append(arrival)
            if len(cycle) < width:
                continue
            for rotated in cycle[hold:] + cycle[:hold]:
                yield self._consume(rotated)
            cycle = []
        for trailing in cycle:
            yield self._consume(trailing)
        self._flush()

    def _consume(self, arrival: tuple):
        """Record a batch's position as it is handed on, and hand it on.

        On consumption, not arrival: a checkpoint describes what the trainer saw.
        """
        batch, position = arrival
        if self._lag:
            position, self._pending = self._pending, position
        if position is not None:
            self._positions[position["shard"]] = position
        return batch

    def _flush(self) -> None:
        """Commit the held position once the underlying loader is exhausted."""
        if self._pending is not None:
            self._positions[self._pending["shard"]] = self._pending
            self._pending = None

    def _rotation(self) -> int:
        """How far to rotate each worker cycle so a resumed run stays in phase.

        The next batch belongs to the worker with the fewest delivered.
        """
        workers = max(self.num_workers, 1)
        delivered = [
            self._positions.get(self.rank * workers + worker, {}).get("emitted", 0)
            for worker in range(workers)
        ]
        return delivered.index(min(delivered))

    def set_epoch(self, epoch: int) -> None:
        """Forward the epoch to the dataset, which is where the shuffle keys off it."""
        self.dataset.set_epoch(epoch)

    def state_dict(self) -> dict:
        """Every rank's position, on every rank. Collective: call on all ranks."""
        local = {shard: dict(pos) for shard, pos in self._positions.items()}
        ranks: list[dict | None] = [None] * self.world_size
        if (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() == self.world_size
        ):
            dist.all_gather_object(ranks, local)
        else:
            ranks[self.rank] = local
        return {
            "world_size": self.world_size,
            "num_workers": self.num_workers,
            "ranks": ranks,
        }

    def load_state_dict(self, state: dict) -> None:
        """Arm the dataset to continue from ``state``. Idempotent.

        Call before the loader is first iterated: a worker reads the position
        from the dataset copy it was forked with. Every mismatch raises.
        """
        if state["world_size"] != self.world_size:
            raise ValueError(
                f"checkpoint was written by a {state['world_size']}-rank run and "
                f"this one has {self.world_size}: every shard would move"
            )
        if state["num_workers"] != self.num_workers:
            raise ValueError(
                f"checkpoint was written with {state['num_workers']} dataloader "
                f"workers and this one has {self.num_workers}: shards are cut per "
                "worker, so every shard would move"
            )
        shards = state["ranks"][self.rank]
        if shards is None:
            raise ValueError(
                f"checkpoint holds no loader position for rank {self.rank}: it was "
                "written with no process group up, so only the writing rank's "
                "position survived"
            )
        self._positions = {int(shard): dict(pos) for shard, pos in shards.items()}
        self._rotate = self._rotation()
        self.dataset.set_resume_state(self._positions)


def build_ddp_loader(
    records,
    renderer,
    tokenizer,
    k: int = 262144,
    m: int = 512,
    ctx_max: int = 2048,
    num_workers: int = 16,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    batches_per_epoch: int | None = None,
    pad_to_multiple: int = 0,
    max_retry: int = 64,
) -> ResumableLoader:
    """Token-budget loader for DDP: disjoint shard per rank, resumable position.

    ``rank`` / ``world_size`` default to the launcher's. Sharding is nested:
    shard ``rank * num_workers + worker_id`` of ``world_size * num_workers``, all
    striding one permutation of ``(seed, epoch)``.

    ``batches_per_epoch`` is close to mandatory above one rank; see docs/internals/data.md.
    """
    rank, world_size = resolve_rank_world(rank, world_size)
    if world_size > 1 and batches_per_epoch is None:
        warnings.warn(
            f"build_ddp_loader over {world_size} ranks without batches_per_epoch: "
            "the batch count is data-dependent, so the ranks will disagree on when "
            "the epoch ends and the run will hang in the following collective",
            stacklevel=2,
        )
    dataset = ResumablePackedDataset(
        records,
        renderer,
        tokenizer,
        k=k,
        m=m,
        ctx_max=ctx_max,
        seed=seed,
        rank=rank,
        world_size=world_size,
        max_retry=max_retry,
        drop_last=drop_last,
        batches_per_epoch=batches_per_epoch,
        pad_to_multiple=pad_to_multiple,
    )
    loader = make_loader(dataset, num_workers, prefetch_factor, pin_memory)
    return ResumableLoader(
        loader,
        dataset,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        steps_per_epoch=(
            None
            if batches_per_epoch is None
            else batches_per_epoch * max(num_workers, 1)
        ),
    )
