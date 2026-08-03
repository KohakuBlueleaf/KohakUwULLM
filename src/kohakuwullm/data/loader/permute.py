"""Shard-local document order, without materializing a permutation.

A keyed Feistel network is a bijection on ``[0, 2^b)``; cycle-walking restricts
it to ``[0, n)``. Position ``k`` maps to a document in O(1), so a worker draws
its stride of the shuffle without holding the shuffle.
See docs/internals/data.md.
"""

from collections.abc import Iterator

import numpy as np

CHUNK = 1 << 16
ROUNDS = 4
_U64 = np.uint64
_MASK64 = _U64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """The splitmix64 finalizer, elementwise over uint64."""
    x = (x + _U64(0x9E3779B97F4A7C15)) & _MASK64
    x = ((x ^ (x >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)) & _MASK64
    x = ((x ^ (x >> _U64(27))) * _U64(0x94D049BB133111EB)) & _MASK64
    return x ^ (x >> _U64(31))


def round_keys(seed: int, epoch: int) -> list[np.uint64]:
    """One key per Feistel round, derived from ``(seed, epoch)``."""
    state = np.array([(seed & 0xFFFFFFFF) ^ (epoch << 32)], dtype=_U64)
    keys = []
    for _ in range(ROUNDS):
        state = splitmix64(state)
        keys.append(_U64(state[0]))
    return keys


def half_bits(n: int) -> int:
    """Half-width of the smallest even-bit domain that covers ``n``."""
    bits = max(int(n - 1).bit_length(), 2)
    return (bits + 1) // 2


def feistel(pos: np.ndarray, half: int, keys: list[np.uint64]) -> np.ndarray:
    """Map positions through the network; a bijection on ``[0, 2^(2*half))``."""
    mask = _U64((1 << half) - 1)
    shift = _U64(half)
    left = (pos >> shift) & mask
    right = pos & mask
    for key in keys:
        mixed = splitmix64(right ^ key) & mask
        left, right = right, left ^ mixed
    return (left << shift) | right


def permute_positions(
    pos: np.ndarray, n: int, half: int, keys: list[np.uint64]
) -> np.ndarray:
    """Positions -> documents, cycle-walking whatever lands outside ``[0, n)``."""
    out = feistel(pos, half, keys)
    outside = out >= _U64(n)
    while outside.any():
        out[outside] = feistel(out[outside], half, keys)
        outside = out >= _U64(n)
    return out


def held_out(indices: np.ndarray, seed: int, val_frac: float) -> np.ndarray:
    """Boolean mask of the validation side, from the document index alone.

    Epoch-independent by construction, so the split cannot drift between
    epochs. See docs/internals/data.md.
    """
    x = indices.astype(_U64) ^ _U64(seed & 0xFFFFFFFF)
    x = (x * _U64(0x9E3779B97F4A7C15)) & _MASK64
    x ^= x >> _U64(29)
    x = (x * _U64(0xBF58476D1CE4E5B9)) & _MASK64
    x ^= x >> _U64(32)
    return (x % _U64(1_000_000)) < _U64(int(val_frac * 1_000_000))


class ShardIndices:
    """This shard's document indices, generated on demand.

    Iterating yields the shard's stride of the ``(seed, epoch)`` shuffle with the
    other split dropped. ``self[start:]`` resumes after ``start`` yielded
    indices. Nothing proportional to ``n`` is allocated.
    """

    def __init__(
        self,
        n: int,
        shard_id: int,
        num_shards: int,
        seed: int,
        epoch: int,
        val_frac: float = 0.0,
        split: str = "train",
    ) -> None:
        self.n = int(n)
        self.shard_id = int(shard_id)
        self.num_shards = int(num_shards)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.val_frac = float(val_frac)
        self.split = split
        self.half = half_bits(self.n)
        self.keys = round_keys(self.seed, self.epoch)

    def __iter__(self) -> Iterator[int]:
        step = self.num_shards * CHUNK
        for base in range(self.shard_id, self.n, step):
            pos = np.arange(base, min(base + step, self.n), self.num_shards, dtype=_U64)
            docs = permute_positions(pos, self.n, self.half, self.keys)
            if self.val_frac > 0.0:
                mask = held_out(docs, self.seed, self.val_frac)
                docs = docs[mask if self.split == "val" else ~mask]
            yield from docs.tolist()

    def __getitem__(self, window: slice) -> Iterator[int]:
        """Only ``[start:]`` is supported; the cursor counts yielded indices."""
        if not isinstance(window, slice) or window.stop is not None or window.step:
            raise TypeError("ShardIndices supports only a [start:] slice")
        stream = iter(self)
        for _ in range(window.start or 0):
            next(stream, None)
        return stream


def shard_indices(
    n: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    epoch: int,
    val_frac: float = 0.0,
    split: str = "train",
) -> np.ndarray:
    """The materialized form of :class:`ShardIndices`, for callers wanting an array."""
    return np.fromiter(
        ShardIndices(n, shard_id, num_shards, seed, epoch, val_frac, split),
        dtype=np.int64,
    )
