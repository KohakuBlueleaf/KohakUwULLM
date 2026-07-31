"""Key/value cache for padded ``(B, S)`` generation.

Holds one preallocated ``(B, max_length, kv_heads, head_dim)`` key and value
buffer per layer, plus the committed prefix length shared by all layers. The
packed varlen layout is not supported. See docs/concepts/architecture.md.
"""

import torch

from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.models.presets import LMArchConfig


class LayerKVCache:
    """One layer's view of a :class:`KVCache`, as handed to an attention module.

    ``offset`` is the absolute position of the first token of the current step.
    """

    __slots__ = ("cache", "index")

    def __init__(self, cache: "KVCache", index: int) -> None:
        self.cache = cache
        self.index = index

    @property
    def offset(self) -> int:
        return self.cache.length

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Write this step's ``k``/``v`` and return the whole cached prefix."""
        return self.cache.append(self.index, k, v)


class KVCache:
    """Per-layer key/value store for one padded generation.

    Args:
        layers: number of decoder layers.
        batch_size: rows generated together.
        max_length: prompt + generated tokens the buffers are sized for.
        kv_heads: key/value heads per layer.
        head_dim: per-head width.
        device: buffer device; taken from the first appended tensor if ``None``.
        dtype: buffer dtype; taken from the first appended tensor if ``None``.
    """

    def __init__(
        self,
        layers: int,
        batch_size: int,
        max_length: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if layers <= 0 or max_length <= 0:
            raise ValueError(
                f"layers and max_length must be positive, got {layers=} {max_length=}"
            )
        self.layers = layers
        self.batch_size = batch_size
        self.max_length = max_length
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.length = 0
        self.keys: list[torch.Tensor | None] = [None] * layers
        self.values: list[torch.Tensor | None] = [None] * layers

    @classmethod
    def from_config(
        cls,
        config: LMArchConfig,
        batch_size: int,
        max_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KVCache":
        """Size a cache from an architecture spec."""
        return cls(
            layers=config.depth,
            batch_size=batch_size,
            max_length=max_length,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )

    def layer(self, index: int) -> LayerKVCache:
        if not 0 <= index < self.layers:
            raise IndexError(
                f"layer {index} out of range for a {self.layers}-layer cache"
            )
        return LayerKVCache(self, index)

    def _allocate(self, index: int, like: torch.Tensor) -> None:
        shape = (self.batch_size, self.max_length, self.kv_heads, self.head_dim)
        device = self.device or like.device
        dtype = self.dtype or like.dtype
        self.keys[index] = torch.empty(shape, device=device, dtype=dtype)
        self.values[index] = torch.empty(shape, device=device, dtype=dtype)

    def append(self, index: int, k: torch.Tensor, v: torch.Tensor):
        """Store ``k``/``v`` at ``length`` and return the prefix through this step.

        ``k``/``v`` are ``(B, S, kv_heads, head_dim)``. The write does not advance
        ``length``; :meth:`advance` does, once per model forward.
        """
        if k.dim() != 4:
            raise ValueError(
                f"cache takes padded (B, S, kv_heads, head_dim) k/v, got {tuple(k.shape)}"
            )
        batch, steps = k.shape[0], k.shape[1]
        end = self.length + steps
        if end > self.max_length:
            raise ValueError(
                f"KV cache overflow: {self.length} cached + {steps} new exceeds max_length={self.max_length}"
            )
        if batch != self.batch_size:
            raise ValueError(f"cache holds batch {self.batch_size}, got {batch}")
        if k.shape[2:] != (self.kv_heads, self.head_dim):
            raise ValueError(
                f"cache holds (kv_heads, head_dim)=({self.kv_heads}, {self.head_dim}), got {tuple(k.shape[2:])}"
            )
        if self.keys[index] is None:
            self._allocate(index, k)
        keys, values = self.keys[index], self.values[index]
        keys[:, self.length : end] = k
        values[:, self.length : end] = v
        return keys[:, :end], values[:, :end]

    def advance(self, steps: int) -> None:
        """Commit ``steps`` appended positions."""
        if self.length + steps > self.max_length:
            raise ValueError(
                f"KV cache overflow: {self.length} cached + {steps} new exceeds max_length={self.max_length}"
            )
        self.length += steps

    def seq_info(self, tokens: torch.Tensor) -> SeqInfo:
        """Padded :class:`SeqInfo` whose positions continue the cached prefix."""
        if tokens.dim() != 2:
            raise ValueError(
                f"cached generation takes padded (B, S) tokens, got {tuple(tokens.shape)}"
            )
        batch, steps = tokens.shape
        pos = torch.arange(self.length, self.length + steps, device=tokens.device)
        return SeqInfo(
            cu_seqlens=None,
            max_seqlen=self.length + steps,
            position_ids=pos.unsqueeze(0).expand(batch, steps),
            num_seqs=batch,
        )

    def reset(self) -> None:
        """Drop the prefix, keeping the buffers."""
        self.length = 0
