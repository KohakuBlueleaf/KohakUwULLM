"""Fixed-size microbatches, for pipeline parallelism.

Every microbatch is exactly ``k`` tokens, the tail padded out with one
fully-masked filler document. This loader does **not** shard across the pipeline
dimension: under PP+DDP pass the data-parallel group's rank and size.
See docs/internals/data.md and docs/internals/pipeline.md.
"""

import warnings
from dataclasses import dataclass

import torch

from kohakuwullm.data.loader.iterative import TokenBudgetIterableDataset, make_loader
from kohakuwullm.data.loader.resume import ResumableLoader
from kohakuwullm.data.packing import PackedBatch, collate_packed
from kohakuwullm.models.components.seqinfo import SeqInfo


@dataclass
class MicroBatchedStep:
    """One optimizer step: ``num_microbatches`` packs of exactly ``k`` tokens.

    ``tokens`` and ``labels`` are flat over the whole step, so a pipeline schedule
    can chunk them along dim 0 and get the microbatch boundaries back;
    :attr:`microbatches` is the same data as views. ``seq_infos`` is per
    microbatch with offsets local to it.

    Attributes:
        tokens: ``(num_microbatches * k,)`` int64 input ids.
        labels: ``(num_microbatches * k,)`` int64 targets, already shifted.
        seq_infos: one packed :class:`SeqInfo` per microbatch.
        microbatch_tokens: ``k``; every microbatch is exactly this long.
        trained: per-microbatch count of tokens with a real label.
    """

    tokens: torch.Tensor
    labels: torch.Tensor
    seq_infos: list[SeqInfo]
    microbatch_tokens: int
    trained: tuple[int, ...]

    @classmethod
    def from_chunks(cls, chunks: list[PackedBatch], k: int) -> "MicroBatchedStep":
        """Flatten equal-sized packed chunks into one step."""
        return cls(
            tokens=torch.cat([chunk.tokens for chunk in chunks]),
            labels=torch.cat([chunk.labels for chunk in chunks]),
            seq_infos=[chunk.seq_info for chunk in chunks],
            microbatch_tokens=k,
            trained=tuple(chunk.num_trained for chunk in chunks),
        )

    @property
    def num_microbatches(self) -> int:
        return len(self.seq_infos)

    @property
    def num_tokens(self) -> int:
        return self.microbatch_tokens * len(self.seq_infos)

    @property
    def num_trained(self) -> int:
        return sum(self.trained)

    @property
    def microbatches(self) -> list[PackedBatch]:
        """Per-microbatch views onto the flat tensors; no copy."""
        k = self.microbatch_tokens
        return [
            PackedBatch(
                tokens=self.tokens[i * k : (i + 1) * k],
                labels=self.labels[i * k : (i + 1) * k],
                seq_info=info,
                num_tokens=k,
                num_trained=self.trained[i],
            )
            for i, info in enumerate(self.seq_infos)
        ]

    def to(self, device: torch.device) -> "MicroBatchedStep":
        """Move every tensor to ``device``, asynchronously where it can."""
        return MicroBatchedStep(
            tokens=self.tokens.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            seq_infos=[info.to(device) for info in self.seq_infos],
            microbatch_tokens=self.microbatch_tokens,
            trained=self.trained,
        )

    def pin_memory(self) -> "MicroBatchedStep":
        """Pin every tensor, so the ``to()`` above can overlap with compute."""
        return MicroBatchedStep(
            tokens=self.tokens.pin_memory(),
            labels=self.labels.pin_memory(),
            seq_infos=[_pin_seq_info(info) for info in self.seq_infos],
            microbatch_tokens=self.microbatch_tokens,
            trained=self.trained,
        )


def _pin_seq_info(info: SeqInfo) -> SeqInfo:
    """Pin a packed layout's tensors. Packed only: neither aliases storage."""
    return SeqInfo(
        cu_seqlens=info.cu_seqlens.pin_memory(),
        max_seqlen=info.max_seqlen,
        position_ids=info.position_ids.pin_memory(),
        num_seqs=info.num_seqs,
    )


class MicroBatchedDataset(TokenBudgetIterableDataset):
    """Shard -> exactly-``k``-token microbatches, ``num_microbatches`` per step.

    Args:
        k: tokens per *microbatch*, not per step. The step is
            ``k * num_microbatches``.
        num_microbatches: microbatches the pipeline schedule expects per step.
        batches_per_epoch: counts *steps* here, and is per shard as in the base
            class.

    Other arguments are
    :class:`~kohakuwullm.data.loader.iterative.TokenBudgetIterableDataset`'s. Yields
    ``(step, position)`` pairs for
    :class:`~kohakuwullm.data.loader.resume.ResumableLoader` to strip. A trailing
    partial step is dropped, never padded out with empty microbatches.
    """

    def __init__(
        self,
        records,
        renderer,
        tokenizer,
        k: int = 8192,
        num_microbatches: int = 32,
        m: int = 512,
        ctx_max: int = 2048,
        seed: int = 0,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
        max_retry: int = 64,
        batches_per_epoch: int | None = None,
    ) -> None:
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")
        super().__init__(
            records,
            renderer,
            tokenizer,
            k=k,
            m=m,
            ctx_max=ctx_max,
            seed=seed,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            max_retry=max_retry,
            drop_last=False,
            batches_per_epoch=batches_per_epoch,
            # Rounding up to the budget itself makes every microbatch exactly k.
            pad_to_multiple=k,
        )
        self.num_microbatches = num_microbatches

    def __iter__(self):
        """Yield ``(step, position)`` once ``num_microbatches`` chunks have filled."""
        shard_id, num_shards, resume = self._begin()
        emitted = 0 if resume is None else resume["emitted"]
        chunks: list[PackedBatch] = []
        for group, position in self._groups(shard_id, num_shards, resume):
            # Checked here, not at the yield: past the cap the remaining groups
            # of a step would be packed and then thrown away.
            if self.batches_per_epoch is not None and emitted >= self.batches_per_epoch:
                return
            chunk = collate_packed(group, pad_to_multiple=self.k)
            if chunk.num_tokens != self.k:
                raise ValueError(
                    f"microbatch came out {chunk.num_tokens} tokens, not {self.k}; "
                    "a single document must not exceed the budget or the stage's "
                    "declared activation shape stops matching"
                )
            chunks.append(chunk)
            if len(chunks) < self.num_microbatches:
                continue
            emitted += 1
            position["emitted"] = emitted
            yield MicroBatchedStep.from_chunks(chunks, self.k), position
            chunks = []


def build_pipeline_loader(
    records,
    renderer,
    tokenizer,
    k: int = 8192,
    num_microbatches: int = 32,
    m: int = 512,
    ctx_max: int = 2048,
    num_workers: int = 16,
    seed: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    batches_per_epoch: int | None = None,
    max_retry: int = 64,
) -> ResumableLoader:
    """Loader of :class:`MicroBatchedStep`, resumable, for pipeline schedules.

    ``rank`` / ``world_size`` describe the **data-parallel** group and default to
    no sharding. See docs/internals/data.md.
    """
    rank = 0 if rank is None else rank
    world_size = 1 if world_size is None else world_size
    if world_size > 1 and batches_per_epoch is None:
        warnings.warn(
            f"build_pipeline_loader over {world_size} data-parallel ranks without "
            "batches_per_epoch: the step count is data-dependent, so the ranks will "
            "disagree on when the epoch ends and the run will hang",
            stacklevel=2,
        )
    dataset = MicroBatchedDataset(
        records,
        renderer,
        tokenizer,
        k=k,
        num_microbatches=num_microbatches,
        m=m,
        ctx_max=ctx_max,
        seed=seed,
        rank=rank,
        world_size=world_size,
        max_retry=max_retry,
        batches_per_epoch=batches_per_epoch,
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
