"""Tokenization, loss masking and varlen packing.

Every sequence is concatenated into one flat token axis, with boundaries in a
:class:`~kohakuwullm.models.components.seqinfo.SeqInfo`; nothing is padded. A
renderer marks which spans carry loss, and the rest get ``-100``.
See docs/internals/data.md.
"""

import random
from dataclasses import dataclass

import torch
import torch.utils.data as data

from kohakuwullm.models.components.seqinfo import SeqInfo

IGNORE_INDEX = -100


@dataclass
class PackedBatch:
    """One training step's worth of tokens.

    Attributes:
        tokens: ``(T,)`` int64 input ids, all documents concatenated.
        labels: ``(T,)`` int64 next-token targets, ``-100`` where masked.
        seq_info: document boundaries + per-document position ids.
        num_tokens: total tokens (the denominator for throughput).
        num_trained: tokens with a real label (the denominator for the loss).
    """

    tokens: torch.Tensor
    labels: torch.Tensor
    seq_info: SeqInfo
    num_tokens: int
    num_trained: int

    def to(self, device: torch.device) -> "PackedBatch":
        """Move every tensor to ``device``, asynchronously where it can."""
        return PackedBatch(
            tokens=self.tokens.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            seq_info=self.seq_info.to(device),
            num_tokens=self.num_tokens,
            num_trained=self.num_trained,
        )

    def pin_memory(self) -> "PackedBatch":
        """Pin every tensor, so the ``to()`` above can overlap with compute."""
        cu = self.seq_info.cu_seqlens
        return PackedBatch(
            tokens=self.tokens.pin_memory(),
            labels=self.labels.pin_memory(),
            seq_info=SeqInfo(
                # None in the padded layout, where boundaries are implicit.
                cu_seqlens=None if cu is None else cu.pin_memory(),
                max_seqlen=self.seq_info.max_seqlen,
                # .contiguous() first: the padded layout's position_ids alias
                # one row of storage through expand(), and pinning rejects that.
                position_ids=self.seq_info.position_ids.contiguous().pin_memory(),
                num_seqs=self.seq_info.num_seqs,
            ),
            num_tokens=self.num_tokens,
            num_trained=self.num_trained,
        )


def as_segments(rendered) -> list[tuple[str, bool]]:
    """A renderer's return value as ``[(text, is_target), ...]``.

    Accepts the segment list itself, or the ``(user_text, output_text)`` pair
    that the document renderers return.
    """
    if len(rendered) == 2 and isinstance(rendered[0], str):
        user, output = rendered
        return [(user, False), (output, True)]
    return list(rendered)


def nonempty_segments(rendered) -> list[tuple[str, bool]]:
    """:func:`as_segments`, with the empty texts dropped."""
    return [(text, bool(target)) for text, target in as_segments(rendered) if text]


def assemble_sample(
    tokenizer,
    segments: list[tuple[str, bool]],
    pieces: list[list[int]],
    max_length: int = 2048,
    add_bos: bool = True,
    add_eos: bool = True,
) -> dict:
    """Concatenate pre-tokenized segments into ids + a per-token loss mask.

    ``pieces`` holds the ids of each entry of ``segments``, in the same order.
    Returns ``input_ids`` and ``labels`` of equal length; ``labels`` is the
    **unshifted** target, and the shift happens at pack time.
    """
    ids: list[int] = []
    labels: list[int] = []
    if add_bos and tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(IGNORE_INDEX)
    for (_, is_target), piece in zip(segments, pieces):
        ids.extend(piece)
        labels.extend(piece if is_target else [IGNORE_INDEX] * len(piece))
    if add_eos and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return {"input_ids": ids[:max_length], "labels": labels[:max_length]}


def encode_sample(
    tokenizer,
    rendered,
    max_length: int = 2048,
    add_bos: bool = True,
    add_eos: bool = True,
) -> dict:
    """Tokenize one rendered sample into ids + a per-token loss mask.

    ``rendered`` is anything :func:`as_segments` accepts; each segment is
    tokenized on its own, never sliced out of a whole-sample encoding.
    """
    segments = nonempty_segments(rendered)
    pieces = (
        tokenizer([text for text, _ in segments], add_special_tokens=False)["input_ids"]
        if segments
        else []
    )
    return assemble_sample(
        tokenizer, segments, pieces, max_length, add_bos=add_bos, add_eos=add_eos
    )


class RenderedDataset(data.Dataset):
    """Record source + renderer + tokenizer -> tokenized samples.

    Args:
        records: any object with ``__len__`` and ``__getitem__`` returning a
            normalized record (see :mod:`kohakuwullm.data.sources.vault`).
        renderer: callable ``(record, rng) -> `` anything :func:`as_segments`
            accepts.
        tokenizer: a ``transformers`` tokenizer (used only to encode).
        max_length: truncate a single sample here.
        seed: base seed; the per-sample rng is derived from ``seed`` and the
            index so an epoch is reproducible but each index gets its own draw.
        epoch: mixed into the per-sample seed, so re-visiting a record in a
            later epoch renders a *different* example.
    """

    def __init__(
        self,
        records,
        renderer,
        tokenizer,
        max_length: int = 2048,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.records = records
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        """Render and tokenize record ``index`` under its own derived rng."""
        rec = self.records[index]
        if rec is None:
            # An empty sample, so the dataset length does not shift; the collate
            # drops it.
            return {"input_ids": [], "labels": []}
        rng = random.Random((self.seed * 1_000_003 + self.epoch) * 1_000_003 + index)
        rendered = self.renderer(rec, rng=rng)
        return encode_sample(self.tokenizer, rendered, self.max_length)


class WeightedConcatDataset(data.Dataset):
    """Concatenate sources with integer repeats, e.g. ``danbooru x3 + cc12m x1``.

    A repeat is a re-render, not a duplicate: the copies get distinct indices and
    :class:`RenderedDataset` seeds its rng from the index.
    """

    def __init__(
        self, datasets: list[data.Dataset], repeats: list[int] | None = None
    ) -> None:
        repeats = repeats or [1] * len(datasets)
        if len(repeats) != len(datasets):
            raise ValueError("datasets and repeats must be the same length")
        self.datasets = datasets
        self.repeats = repeats
        self._offsets = [0]
        for dataset, repeat in zip(datasets, repeats):
            self._offsets.append(self._offsets[-1] + len(dataset) * repeat)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index: int):
        for i, dataset in enumerate(self.datasets):
            if index < self._offsets[i + 1]:
                local = (index - self._offsets[i]) % len(dataset)
                return dataset[local]
        raise IndexError(index)


def collate_packed(samples: list[dict], pad_to_multiple: int = 0) -> PackedBatch:
    """Pack tokenized samples into one varlen batch.

    ``pad_to_multiple`` appends a single fully-masked filler document to round
    the total token count up. It contributes no gradient.
    """
    kept = [s for s in samples if len(s["input_ids"]) > 1]
    if not kept:
        empty = torch.zeros(0, dtype=torch.long)
        return PackedBatch(empty, empty, SeqInfo.from_lengths(torch.zeros(0)), 0, 0)

    token_chunks, label_chunks, lengths = [], [], []
    for sample in kept:
        ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        # Shift inside the document: position i predicts token i+1, and the last
        # position has nothing to predict.
        shifted = torch.empty_like(labels)
        shifted[:-1] = labels[1:]
        shifted[-1] = IGNORE_INDEX
        token_chunks.append(ids)
        label_chunks.append(shifted)
        lengths.append(ids.numel())

    if pad_to_multiple > 0:
        total = sum(lengths)
        remainder = total % pad_to_multiple
        if remainder:
            filler = pad_to_multiple - remainder
            token_chunks.append(torch.zeros(filler, dtype=torch.long))
            label_chunks.append(torch.full((filler,), IGNORE_INDEX, dtype=torch.long))
            lengths.append(filler)

    tokens = torch.cat(token_chunks)
    labels = torch.cat(label_chunks)
    seq_info = SeqInfo.from_lengths(torch.tensor(lengths, dtype=torch.int32))
    return PackedBatch(
        tokens=tokens,
        labels=labels,
        seq_info=seq_info,
        num_tokens=int(tokens.numel()),
        num_trained=int((labels != IGNORE_INDEX).sum()),
    )


def split_packed(batch: PackedBatch, num_chunks: int) -> list[PackedBatch]:
    """Split a packed batch into ``num_chunks`` document-aligned micro-batches.

    Always on document boundaries, and dealt longest-first onto the currently
    smallest chunk so the token counts stay close. See docs/internals/data.md.
    """
    if num_chunks <= 1 or batch.seq_info.num_seqs <= 1:
        return [batch]

    cu = batch.seq_info.cu_seqlens
    lengths = (cu[1:] - cu[:-1]).tolist()
    starts = cu[:-1].tolist()
    num_chunks = min(num_chunks, len(lengths))

    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    buckets: list[list[int]] = [[] for _ in range(num_chunks)]
    sizes = [0] * num_chunks
    for doc in order:
        target = min(range(num_chunks), key=lambda c: sizes[c])
        buckets[target].append(doc)
        sizes[target] += lengths[doc]

    chunks = []
    for bucket in buckets:
        if not bucket:
            continue
        bucket.sort()
        token_parts, label_parts, chunk_lengths = [], [], []
        for doc in bucket:
            start, length = starts[doc], lengths[doc]
            token_parts.append(batch.tokens[start : start + length])
            label_parts.append(batch.labels[start : start + length])
            chunk_lengths.append(length)
        tokens = torch.cat(token_parts)
        labels = torch.cat(label_parts)
        chunks.append(
            PackedBatch(
                tokens=tokens,
                labels=labels,
                seq_info=SeqInfo.from_lengths(
                    torch.tensor(chunk_lengths, dtype=torch.int32), tokens.device
                ),
                num_tokens=int(tokens.numel()),
                num_trained=int((labels != IGNORE_INDEX).sum()),
            )
        )
    return chunks
