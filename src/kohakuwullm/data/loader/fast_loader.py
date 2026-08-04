"""The map-style per-sample loader, and the grouped one it beats.

**The per-sample DataLoader is already near-optimal. Do not group.**
:class:`BatchRenderedDataset` is the right shape only for a *pre-tokenization*
pass, where there is no DataLoader parallelism to lose.

See docs/internals/data.md.
"""

import random

import torch
import torch.utils.data as data

from kohakuwullm.data.loader.padded import collate_padded
from kohakuwullm.data.packing import (
    RenderedDataset,
    assemble_sample,
    collate_packed,
    nonempty_segments,
)


class BatchRenderedDataset(data.Dataset):
    """Renders and tokenizes a *group* of records per ``__getitem__``.

    Drive the DataLoader with ``batch_size=None`` and a sampler that yields
    groups, so one worker item already is a batch.

    Args:
        records: record source (``__len__`` + ``__getitem__``).
        renderer: ``(record, rng) -> `` rendered segments.
        tokenizer: a fast (Rust-backed) tokenizer.
        group_size: records rendered per item.
        max_length: truncation.
        seed / epoch: per-sample rng derivation, as in ``RenderedDataset``.
    """

    def __init__(
        self,
        records,
        renderer,
        tokenizer,
        group_size: int = 32,
        max_length: int = 2048,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.records = records
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_length = max_length
        self.seed = seed
        self.epoch = epoch
        self._n_groups = (len(records) + group_size - 1) // group_size

    def __len__(self) -> int:
        return self._n_groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _indices(self, group: int) -> list[int]:
        start = group * self.group_size
        stop = min(start + self.group_size, len(self.records))
        return list(range(start, stop))

    def __getitem__(self, group: int) -> list[dict]:
        """Render one group and encode it in a single batched tokenizer call."""
        rendered = []
        for index in self._indices(group):
            rec = self.records[index]
            if rec is None:
                continue
            rng = random.Random(
                (self.seed * 1_000_003 + self.epoch) * 1_000_003 + index
            )
            rendered.append(nonempty_segments(self.renderer(rec, rng=rng)))
        if not rendered:
            return []

        # One batched call over every segment of the group, then regrouped.
        texts = [text for segments in rendered for text, _ in segments]
        pieces = (
            self.tokenizer(texts, add_special_tokens=False)["input_ids"]
            if texts
            else []
        )

        samples, cursor = [], 0
        for segments in rendered:
            stop = cursor + len(segments)
            samples.append(
                assemble_sample(
                    self.tokenizer,
                    segments,
                    pieces[cursor:stop],
                    self.max_length,
                )
            )
            cursor = stop
        return samples


class GroupSampler(data.Sampler):
    """Shuffled group indices, so grouping does not fix which records share a batch."""

    def __init__(self, n_groups: int, shuffle: bool = True, seed: int = 0) -> None:
        self.n_groups = n_groups
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if not self.shuffle:
            yield from range(self.n_groups)
            return
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.n_groups, generator=generator).tolist()

    def __len__(self) -> int:
        return self.n_groups


def build_fast_loader(
    records,
    renderer,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 16,
    max_length: int = 2048,
    seed: int = 0,
    collate=None,
    prefetch_factor: int = 4,
    shuffle: bool = True,
    layout: str = "packed",
    pad_token_id: int = 0,
    pad_to_multiple: int = 0,
):
    """The measured-best online loader: per-sample, sharded across workers.

    ``layout`` selects ``"packed"`` (varlen) or ``"padded"`` (``(B, S)``); both
    yield a :class:`~kohakuwullm.data.packing.PackedBatch`, and an explicit
    ``collate`` overrides the choice.
    """
    if collate is None:
        if layout == "padded":

            def collate(samples):
                return collate_padded(
                    samples,
                    pad_token_id=pad_token_id,
                    max_length=max_length,
                    pad_to_multiple=max(pad_to_multiple, 8),
                )

        elif layout == "packed":

            def collate(samples):
                return collate_packed(samples, pad_to_multiple=pad_to_multiple)

        else:
            raise ValueError(f"layout must be 'packed' or 'padded', got {layout!r}")

    dataset = RenderedDataset(
        records, renderer, tokenizer, max_length=max_length, seed=seed
    )
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return data.DataLoader(dataset, **kwargs)


def _grouped_loader_do_not_use(
    records,
    renderer,
    tokenizer,
    batch_size=32,
    num_workers=16,
    max_length=2048,
    seed=0,
    collate=None,
    prefetch_factor=4,
    shuffle=True,
):
    """The grouped loader, kept only so the regression it caused stays reproducible.

    **Do not wire this into a config.** See docs/internals/data.md for what it measures.
    """
    dataset = BatchRenderedDataset(
        records,
        renderer,
        tokenizer,
        group_size=batch_size,
        max_length=max_length,
        seed=seed,
    )
    sampler = GroupSampler(len(dataset), shuffle=shuffle, seed=seed)
    collate = collate or collate_packed

    kwargs = dict(
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    loader = data.DataLoader(dataset, **kwargs)
    # The caller applies the collate: batch_size=None bypasses collate_fn.
    return loader, collate
