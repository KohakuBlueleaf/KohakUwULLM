"""Padded batch collation, the alternative to varlen packing.

Interchangeable with packing at the collate boundary: both produce a
:class:`~kohakuwullm.data.packing.PackedBatch`, and ``seq_info.packed`` is the
only thing that ever branches. Packing stays the default; see docs/internals/data.md.
"""

import torch

from kohakuwullm.data.packing import IGNORE_INDEX, PackedBatch
from kohakuwullm.models.components.seqinfo import SeqInfo


def collate_padded(
    samples: list[dict],
    pad_token_id: int = 0,
    max_length: int | None = None,
    pad_to_multiple: int = 8,
) -> PackedBatch:
    """Pad tokenized samples to a common length -> ``(B, S)`` batch.

    Args:
        samples: dicts with ``input_ids`` and ``labels``.
        pad_token_id: fill value for ``input_ids``. Labels are always filled with
            ``IGNORE_INDEX``, so the pad contributes no gradient.
        max_length: truncate to at most this.
        pad_to_multiple: round the padded length up, so ``torch.compile`` sees
            fewer distinct shapes.

    Returns a :class:`PackedBatch` whose ``seq_info`` is the padded variant, so
    it is a drop-in for the packed collate.
    """
    kept = [s for s in samples if len(s["input_ids"]) > 1]
    if not kept:
        empty = torch.zeros(0, dtype=torch.long)
        return PackedBatch(empty, empty, SeqInfo.padded(0, 0), 0, 0)

    lengths = [len(s["input_ids"]) for s in kept]
    width = max(lengths)
    if max_length is not None:
        width = min(width, max_length)
    if pad_to_multiple > 1:
        width = ((width + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    batch = len(kept)
    tokens = torch.full((batch, width), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, width), IGNORE_INDEX, dtype=torch.long)

    real_tokens = 0
    for i, sample in enumerate(kept):
        ids = sample["input_ids"][:width]
        lab = sample["labels"][:width]
        n = len(ids)
        tokens[i, :n] = torch.tensor(ids, dtype=torch.long)
        # Shift within the row; the final real position predicts nothing.
        if n > 1:
            labels[i, : n - 1] = torch.tensor(lab[1:n], dtype=torch.long)
        real_tokens += n

    return PackedBatch(
        tokens=tokens,
        labels=labels,
        seq_info=SeqInfo.padded(batch, width),
        # Padding included: this counts what the GPU computes on.
        num_tokens=batch * width,
        num_trained=int((labels != IGNORE_INDEX).sum()),
    )


def padding_fraction(batch: PackedBatch) -> float:
    """Share of computed positions that carry no real token."""
    if batch.num_tokens == 0:
        return 0.0
    if batch.seq_info.packed:
        return 0.0
    real = int((batch.tokens != 0).sum())
    return 1.0 - real / batch.num_tokens


def split_padded(batch: PackedBatch, num_chunks: int) -> list[PackedBatch]:
    """Split a padded batch along the batch axis into micro-batches."""
    if num_chunks <= 1 or batch.seq_info.num_seqs <= 1:
        return [batch]
    chunks = []
    per = max(1, batch.seq_info.num_seqs // num_chunks)
    for start in range(0, batch.seq_info.num_seqs, per):
        tokens = batch.tokens[start : start + per]
        labels = batch.labels[start : start + per]
        if tokens.shape[0] == 0:
            continue
        chunks.append(
            PackedBatch(
                tokens=tokens,
                labels=labels,
                seq_info=SeqInfo.padded(
                    tokens.shape[0], tokens.shape[1], tokens.device
                ),
                num_tokens=tokens.numel(),
                num_trained=int((labels != IGNORE_INDEX).sum()),
            )
        )
    return chunks
