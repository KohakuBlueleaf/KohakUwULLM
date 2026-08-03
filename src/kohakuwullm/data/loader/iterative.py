"""Token-budget iterative packing: varlen batches with a near-constant token count.

    k        per-batch token budget, e.g. 262144 (= 256 x 1024)
    m        "long content" threshold, and the bound on how far under budget a
             batch may land
    ctx_max  hard truncation for a single document

Nothing is padded, and the per-batch total lands in ``(k - m, k]``.
See docs/internals/data.md.
"""

import random
from collections import deque
from collections.abc import Iterable, Iterator
from itertools import chain

import torch.utils.data as data

from kohakuwullm.data.filters import build_filter
from kohakuwullm.data.loader.permute import ShardIndices
from kohakuwullm.data.packing import PackedBatch, collate_packed, encode_sample


def pack_to_budget(
    samples: Iterable[dict],
    k: int,
    m: int,
    max_retry: int = 64,
    drop_last: bool = False,
) -> Iterator[list[dict]]:
    """Group tokenized samples into lists whose token total sits just under ``k``.

    Yields lists of samples, preserving every input sample exactly once, and
    always with the queue drained.

    Args:
        samples: iterable of ``{"input_ids": [...], "labels": [...]}``.
        k: per-batch token budget.
        m: documents longer than this trigger a retry.
        max_retry: hard cap on rejections per batch.
        drop_last: drop a trailing batch that never reached the budget.
    """
    stream = iter(samples)
    carry: list[dict] = []
    exhausted = False

    while True:
        # The batch under construction drains `pending` and pushes rejections
        # into a fresh `carry`, never back into its own queue.
        pending = deque(carry)
        carry = []
        batch: list[dict] = []
        total = 0
        retries = 0
        reached_budget = False

        while True:
            if pending:
                sample = pending.popleft()
            elif exhausted:
                break
            else:
                sample = next(stream, None)
                if sample is None:
                    exhausted = True
                    break

            length = len(sample["input_ids"])
            # An empty batch accepts unconditionally, so a document longer than
            # `k` is emitted alone rather than rejected forever.
            if batch and total + length > k:
                carry.append(sample)
                reached_budget = True
                retries += 1
                if length > m and retries <= max_retry:
                    continue
                break
            batch.append(sample)
            total += length

        # An empty batch means nothing was available and nothing was rejected.
        if not batch:
            return
        if reached_budget or not drop_last:
            yield batch
        if exhausted and not carry:
            return


class TokenBudgetIterableDataset(data.IterableDataset):
    """Records -> rendered, tokenized, token-budget-packed :class:`PackedBatch`.

    Args:
        records: record source (``__len__`` + ``__getitem__``), as in
            :mod:`kohakuwullm.data.sources.vault`.
        renderer: callable ``(record, rng) -> (user_text, output_text)``.
        tokenizer: a ``transformers`` tokenizer.
        k: per-batch token budget.
        m: "long content" threshold; also the bound on how far under ``k`` a
            batch may land.
        ctx_max: hard truncation for one document.
        seed: base seed for both the shuffle and the per-record render rng.
        epoch: mixed into both.
        rank / world_size: distributed sharding; single-process by default.
        max_retry: see :func:`pack_to_budget`.
        drop_last: drop a trailing under-budget batch.
        batches_per_epoch: stop each *shard* after this many batches. Required
            under DDP; set it below ``tokens / (k * world_size * num_workers)``.
        pad_to_multiple: round the packed token count up with a fully-masked
            filler document.

    Batches are built **per worker**, each from its own shard.
    """

    def __init__(
        self,
        records,
        renderer,
        tokenizer,
        k: int = 262144,
        m: int = 512,
        ctx_max: int = 2048,
        seed: int = 0,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
        max_retry: int = 64,
        drop_last: bool = False,
        batches_per_epoch: int | None = None,
        pad_to_multiple: int = 0,
        val_frac: float = 0.0,
        split: str = "train",
        doc_filter=None,
    ) -> None:
        if k <= 0 or m < 0 or ctx_max <= 0:
            raise ValueError("k and ctx_max must be positive and m non-negative")
        if ctx_max > k:
            raise ValueError(
                f"ctx_max={ctx_max} exceeds the budget k={k}: a single document "
                "would then be able to overshoot the batch budget"
            )
        self.records = records
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.k = k
        self.m = m
        self.ctx_max = ctx_max
        self.seed = seed
        self.epoch = epoch
        self.rank = rank
        self.world_size = world_size
        self.max_retry = max_retry
        self.drop_last = drop_last
        self.batches_per_epoch = batches_per_epoch
        self.pad_to_multiple = pad_to_multiple
        self.val_frac = val_frac
        self.split = split
        self.doc_filter = build_filter(doc_filter)
        self._pass = 0
        self._resume: dict[int, dict] = {}

    def set_epoch(self, epoch: int) -> None:
        """Start a new epoch, resetting the per-process pass counter with it."""
        self.epoch = epoch
        self._pass = 0

    def set_resume_state(self, positions: dict[int, dict]) -> None:
        """Arm each shard to continue from a saved position, popped on first use."""
        self._resume = {int(shard): dict(pos) for shard, pos in positions.items()}

    def _stream(
        self,
        indices,
        epoch: int,
        progress: list[int] | None = None,
        outstanding: dict[int, None] | None = None,
        start: int = 0,
    ) -> Iterator[dict]:
        """Records -> tokenized samples, in shard order.

        ``progress`` is how far into the shard this stream has read;
        ``outstanding`` is what it handed over but no batch has emitted yet.
        """
        for offset, index in enumerate(indices, start=1):
            index = int(index)
            rec = self.records[index]
            if progress is not None:
                progress[0] = start + offset
            # A missing or rejected row is skipped: this path has no length to
            # preserve, and the packer simply draws the next document.
            if rec is None:
                continue
            if self.doc_filter is not None and self.doc_filter(rec):
                continue
            # Same derivation as RenderedDataset, so a given (seed, epoch,
            # index) renders the identical example in either path.
            rng = random.Random((self.seed * 1_000_003 + epoch) * 1_000_003 + index)
            user, output = self.renderer(rec, rng=rng)
            sample = encode_sample(self.tokenizer, user, output, self.ctx_max)
            if len(sample["input_ids"]) > 1:
                sample["index"] = index
                if outstanding is not None:
                    outstanding[index] = None
                yield sample

    def _begin(self) -> tuple[int, int, dict | None]:
        """This worker's shard, and the resume position armed for it."""
        worker = data.get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        resume = self._resume.pop(shard_id, None)
        if resume is not None and resume["shards"] != num_shards:
            raise ValueError(
                f"resume position was written with {resume['shards']} shards and "
                f"this run has {num_shards}: the shuffle is strided per shard, so "
                "continuing would skip some documents and replay others"
            )
        return shard_id, num_shards, resume

    def _groups(
        self, shard_id: int, num_shards: int, resume: dict | None
    ) -> Iterator[tuple[list[dict], dict]]:
        """Packed groups, each paired with the shard position that follows it."""
        if resume is not None:
            self._pass = resume["pass"]
        # `_pass` advances per iterator created in this process, which is how a
        # persistent worker sees a new epoch without `set_epoch` reaching it.
        epoch = self.epoch + self._pass
        self._pass += 1

        indices = ShardIndices(
            len(self.records),
            shard_id,
            num_shards,
            self.seed,
            epoch,
            val_frac=self.val_frac,
            split=self.split,
        )
        # Insertion-ordered, so `tuple(outstanding)` comes out in draw order.
        outstanding: dict[int, None] = {}
        start = 0 if resume is None else resume["cursor"]
        progress = [start]
        stream = self._stream(indices[start:], epoch, progress, outstanding, start)
        if resume is not None:
            # Carried documents are re-rendered ahead of the shard's tail and
            # left out of `progress`, which restores the packer's queue.
            stream = chain(
                self._stream(resume["carry"], epoch, outstanding=outstanding), stream
            )

        groups = pack_to_budget(
            stream,
            self.k,
            self.m,
            max_retry=self.max_retry,
            drop_last=self.drop_last,
        )
        this_pass = self._pass - 1
        for group in groups:
            for sample in group:
                del outstanding[sample["index"]]
            yield group, {
                "shard": shard_id,
                "shards": num_shards,
                "pass": this_pass,
                "cursor": progress[0],
                "carry": tuple(outstanding),
            }

    def _iter_stateful(self) -> Iterator[tuple[PackedBatch, dict]]:
        """Collated batches, each paired with the position that follows it."""
        shard_id, num_shards, resume = self._begin()
        emitted = 0 if resume is None else resume["emitted"]
        for group, position in self._groups(shard_id, num_shards, resume):
            if self.batches_per_epoch is not None and emitted >= self.batches_per_epoch:
                return
            emitted += 1
            position["emitted"] = emitted
            yield collate_packed(group, pad_to_multiple=self.pad_to_multiple), position

    def __iter__(self) -> Iterator[PackedBatch]:
        for batch, _ in self._iter_stateful():
            yield batch


def _passthrough(batch):
    """Identity collate: the dataset already yields finished batches."""
    return batch


def make_loader(
    dataset: data.IterableDataset,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> data.DataLoader:
    """DataLoader over a dataset that already yields whole batches.

    ``persistent_workers`` follows ``num_workers``. See docs/internals/data.md.
    """
    kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_passthrough,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return data.DataLoader(dataset, **kwargs)


def build_iterative_loader(
    records,
    renderer,
    tokenizer,
    k: int = 262144,
    m: int = 512,
    ctx_max: int = 2048,
    num_workers: int = 16,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    batches_per_epoch: int | None = None,
    pad_to_multiple: int = 0,
    max_retry: int = 64,
    val_frac: float = 0.0,
    split: str = "train",
    doc_filter=None,
) -> data.DataLoader:
    """DataLoader over :class:`TokenBudgetIterableDataset`, single-process.

    For DDP use :func:`kohakuwullm.data.loader.resume.build_ddp_loader`.
    """
    dataset = TokenBudgetIterableDataset(
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
        val_frac=val_frac,
        split=split,
        doc_filter=doc_filter,
    )
    return make_loader(dataset, num_workers, prefetch_factor, pin_memory)
