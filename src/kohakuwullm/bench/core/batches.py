"""Synthetic packed batches: ragged lengths on ``[lo, hi]``, exact token counts.

See docs/performance/benchmarking.md.
"""

import torch

from kohakuwullm.models.components.seqinfo import SeqInfo

IGNORE_INDEX = -100


def pack_lengths(
    per_micro: int, lo: int, hi: int, generator: torch.Generator
) -> list[int]:
    """Lengths summing to exactly ``per_micro``, drawn from ``[lo, hi]`` via ``generator``."""
    lengths, left = [], per_micro
    while left > 0:
        n = min(int(torch.randint(lo, hi + 1, (1,), generator=generator).item()), left)
        lengths.append(n)
        left -= n
    return lengths


def make_info(per_micro: int, lo: int, hi: int, device, seed: int = 0) -> SeqInfo:
    """One packed microbatch of exactly ``per_micro`` tokens."""
    gen = torch.Generator().manual_seed(seed)
    lengths = pack_lengths(per_micro, lo, hi, gen)
    return SeqInfo.from_lengths(torch.tensor(lengths, dtype=torch.int32), device)


def make_packs(
    budget: int, per_micro: int, lo: int, hi: int, device, seed: int = 0
) -> list[SeqInfo]:
    """One :class:`SeqInfo` per microbatch, together spending ``budget`` tokens."""
    gen = torch.Generator().manual_seed(seed)
    return [
        SeqInfo.from_lengths(
            torch.tensor(pack_lengths(per_micro, lo, hi, gen), dtype=torch.int32),
            device,
        )
        for _ in range(budget // per_micro)
    ]


def make_tokens(infos: list[SeqInfo], vocab: int, device, seed: int = 0):
    """Token ids and next-token targets, every document's last position masked."""
    total = sum(int(i.cu_seqlens[-1]) for i in infos)
    gen = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(0, vocab, (total,), generator=gen).to(device)
    labels = ids.roll(-1)
    offset = 0
    for info in infos:
        ends = info.cu_seqlens[1:].long() + offset - 1
        labels[ends] = IGNORE_INDEX
        offset += int(info.cu_seqlens[-1])
    return ids, labels


def trained_tokens(labels: torch.Tensor) -> int:
    """Positions that contribute a gradient -- the denominator throughput needs."""
    return int((labels != IGNORE_INDEX).sum().item())
