"""Architecture config dataclass + named presets.

``LMArchConfig`` is everything the backbone needs to build itself; its component
fields are *specs* (registry name, dotted path, ``{"name": ..., **kw}`` dict, or
class). See docs/concepts/presets.md for the ladder, docs/concepts/architecture.md for the parts.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LMArchConfig:
    """Full architecture specification for :class:`LMBackbone`."""

    # -- shape --
    vocab_size: int = 65536
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    kv_heads: int | None = 4
    head_dim: int | None = None
    mlp_ratio: float = 4.0
    mlp_hidden: int | None = None
    mlp_multiple_of: int = 128
    max_position: int = 8192

    # -- components --
    norm: Any = "rmsnorm"
    mlp: Any = "swiglu"
    attn: Any = "varlen"
    # Attention backend for sliding-window layers only; None -> use `attn`.
    attn_sliding: Any = None
    posenc: Any = "rope"
    norm_eps: float = 1e-6

    # -- attention detail --
    qk_norm: bool = True
    qk_norm_affine: bool = True
    attn_bias: bool = False
    attn_sink: bool = False
    sliding_window: int | None = None
    global_layer_every: int = 0  # 0 -> every layer is `sliding_window`
    # Explicit per-layer widths, cycled over the depth; overrides the two above.
    window_pattern: list[int | None] | None = None
    rope_theta: float = 10000.0
    rope_scaling: str | None = None
    rope_factor: float = 1.0
    rope_partial: float = 1.0

    # -- MoE (ignored when moe_every == 0) --
    moe_every: int = 0  # 1 -> every layer sparse, 2 -> every other, ...
    moe_first_dense: int = 0  # keep this many leading layers dense
    moe_num_experts: int = 64
    moe_top_k: int = 8
    moe_num_shared: int = 1
    moe_hidden: int | None = None
    moe_ratio: float = 4.0
    moe_router: Any = "topk"
    moe_router_kwargs: dict = field(default_factory=dict)
    moe_mlp: str | None = None  # MLP registry key; None -> "moe"
    moe_mlp_kwargs: dict = field(default_factory=dict)

    # -- head / embedding --
    # Untied by default; every preset's target count absorbs the second matrix.
    tie_embeddings: bool = False
    embedding_scale: bool = False  # multiply embeddings by sqrt(dim), as Gemma does
    logit_soft_cap: float | None = None
    z_loss_weight: float = 0.0

    # -- residual / init --
    post_norm: bool = False
    scale_residual_by_depth: bool = False
    init_std: float = 0.02
    # muP reference width. When set, every hidden matrix is initialized at
    # ``init_std * sqrt(mup_base_dim / fan_in)`` instead of a flat ``init_std``.
    mup_base_dim: int | None = None

    # -- runtime --
    grad_ckpt: bool = False
    # A flag, not a component spec: `swap_mxfp8` runs after `initialize_weights`.
    mxfp8: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.dim % self.heads != 0:
                raise ValueError(
                    f"dim ({self.dim}) not divisible by heads ({self.heads}); "
                    "set head_dim explicitly"
                )
            self.head_dim = self.dim // self.heads
        if self.kv_heads is None:
            self.kv_heads = self.heads
        if self.heads % self.kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads})"
            )

    @property
    def windows(self) -> list[int | None]:
        """Per-layer window width; ``None`` means full causal.

        ``window_pattern`` is cycled over the depth and overrides
        ``sliding_window`` + ``global_layer_every``. The last layer is always
        global.
        """
        if self.window_pattern is not None:
            pattern = list(self.window_pattern)
            out = [pattern[i % len(pattern)] for i in range(self.depth)]
            out[-1] = None  # the layer feeding the head is always global
            return out
        if self.sliding_window is None:
            return [None] * self.depth
        if self.global_layer_every <= 0:
            return [self.sliding_window] * self.depth
        return [
            (
                None
                if (i + 1) % self.global_layer_every == 0 or i == self.depth - 1
                else self.sliding_window
            )
            for i in range(self.depth)
        ]

    @property
    def layer_types(self) -> list[str]:
        """Per-layer ``"full"`` / ``"sliding"``."""
        return ["full" if w is None else "sliding" for w in self.windows]

    @property
    def moe_layers(self) -> list[bool]:
        """Per-layer sparsity flag."""
        if self.moe_every <= 0:
            return [False] * self.depth
        return [
            i >= self.moe_first_dense
            and (i - self.moe_first_dense) % self.moe_every == 0
            for i in range(self.depth)
        ]

    def window_for(self, layer: int) -> int | None:
        return self.windows[layer]

    def attn_for(self, layer: int) -> Any:
        """Attention backend for this layer -- may differ between local and global."""
        if self.attn_sliding is not None and self.layer_types[layer] == "sliding":
            return self.attn_sliding
        return self.attn

    @property
    def residual_scale(self) -> float:
        return self.depth**-0.5 if self.scale_residual_by_depth else 1.0


# RETIRED dense presets, 25M -> 1B, with "wide" / "deep" variants at matched
# parameter count. See docs/concepts/presets.md.
_DENSE: dict[str, dict[str, Any]] = {
    "Nano-25M": dict(dim=384, depth=8, heads=6, kv_heads=2, head_dim=64),
    "Nano-100M": dict(dim=640, depth=14, heads=10, kv_heads=2, head_dim=64),
    # --- 200M ---
    "Nano-200M": dict(dim=896, depth=16, heads=14, kv_heads=2, head_dim=64),
    "Nano-200M-wide": dict(dim=1152, depth=10, heads=18, kv_heads=3, head_dim=64),
    "Nano-200M-deep": dict(dim=704, depth=26, heads=11, kv_heads=1, head_dim=64),
    # --- 500M ---
    "Nano-500M": dict(dim=1280, depth=20, heads=20, kv_heads=4, head_dim=64),
    "Nano-500M-wide": dict(dim=1664, depth=12, heads=26, kv_heads=2, head_dim=64),
    "Nano-500M-deep": dict(dim=1024, depth=32, heads=16, kv_heads=2, head_dim=64),
    # --- 1B ---
    "Nano-1B": dict(dim=1792, depth=26, heads=28, kv_heads=4, head_dim=64),
    "Nano-1B-wide": dict(dim=2304, depth=16, heads=18, kv_heads=3, head_dim=128),
    "Nano-1B-deep": dict(dim=1408, depth=42, heads=22, kv_heads=2, head_dim=64),
}

# RETIRED sparse presets, named <total>-A<active> from measured counts.
# See docs/concepts/presets.md.
_SPARSE: dict[str, dict[str, Any]] = {
    "MoE-1B-A120M": dict(
        dim=640,
        depth=18,
        heads=10,
        kv_heads=2,
        head_dim=64,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=64,
        moe_top_k=3,
        moe_num_shared=1,
        moe_hidden=448,
    ),
    "MoE-1B-A280M": dict(
        dim=1024,
        depth=20,
        heads=16,
        kv_heads=4,
        head_dim=64,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=32,
        moe_top_k=4,
        moe_num_shared=1,
        moe_hidden=512,
    ),
    "MoE-2B-A370M": dict(
        dim=1152,
        depth=24,
        heads=18,
        kv_heads=3,
        head_dim=64,
        moe_every=1,
        moe_first_dense=2,
        moe_num_experts=48,
        moe_top_k=4,
        moe_num_shared=1,
        moe_hidden=512,
    ),
    # base / wide / deep at matched size, so dim-vs-depth is not confounded.
    "MoE-3B-A500M": dict(
        dim=1280,
        depth=28,
        heads=20,
        kv_heads=4,
        head_dim=64,
        moe_every=1,
        moe_first_dense=2,
        moe_num_experts=64,
        moe_top_k=6,
        moe_num_shared=1,
        moe_hidden=448,
    ),
    "MoE-3B-A500M-wide": dict(
        dim=1792,
        depth=16,
        heads=28,
        kv_heads=4,
        head_dim=64,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=48,
        moe_top_k=4,
        moe_num_shared=1,
        moe_hidden=704,
    ),
    "MoE-3B-A500M-deep": dict(
        dim=1024,
        depth=40,
        heads=16,
        kv_heads=2,
        head_dim=64,
        moe_every=1,
        moe_first_dense=2,
        moe_num_experts=80,
        moe_top_k=7,
        moe_num_shared=1,
        moe_hidden=320,
    ),
    # 8.1B total / 1.14B active, ~128 GB of AdamW state: pipeline-only.
    "MoE-8B-A1B": dict(
        dim=1536,
        depth=32,
        heads=24,
        kv_heads=4,
        head_dim=64,
        moe_every=1,
        moe_first_dense=2,
        moe_num_experts=96,
        moe_top_k=8,
        moe_num_shared=1,
        moe_hidden=576,
        tie_embeddings=False,
    ),
}

# The Kohaku ladder: nine rungs, dense and sparse interleaved, superseding
# `Nano-*` / `MoE-*` for new runs. See docs/concepts/presets.md.
#
# `mlp_hidden` is the nearest multiple of 128 to the SwiGLU 8/3 ratio, given
# explicitly rather than left to `resolve_hidden`'s round-up.
_KOHAKU_DENSE: dict[str, dict[str, Any]] = {
    "Kohaku-200M": dict(
        dim=768, depth=17, heads=12, kv_heads=2, head_dim=64, mlp_hidden=2048
    ),
    "Kohaku-500M": dict(
        dim=1280, depth=22, heads=20, kv_heads=4, head_dim=64, mlp_hidden=3456
    ),
    "Kohaku-1B": dict(
        dim=1536, depth=32, heads=24, kv_heads=4, head_dim=64, mlp_hidden=4096
    ),
    "Kohaku-1.5B": dict(
        dim=1792, depth=39, heads=28, kv_heads=4, head_dim=64, mlp_hidden=4736
    ),
}

# Sparse rungs. Two quantities are held exactly constant across all of them:
# sparsity ``kappa = top_k / num_experts = 0.125`` and expert width ``0.5 * dim``.
# Do not change either without reading docs/concepts/presets.md.
_KOHAKU_SPARSE: dict[str, dict[str, Any]] = {
    "Kohaku-MoE-1B": dict(
        dim=768,
        depth=16,
        heads=12,
        kv_heads=2,
        head_dim=64,
        mlp_hidden=2048,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=64,
        moe_top_k=8,
        moe_num_shared=1,
        moe_hidden=384,
    ),
    # The one deviation: `moe_hidden` is 0.571*dim, not 0.500, for MXFP8
    # eligibility. See docs/concepts/presets.md.
    "Kohaku-MoE-2B": dict(
        dim=896,
        depth=21,
        heads=14,
        kv_heads=2,
        head_dim=64,
        mlp_hidden=2432,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=64,
        moe_top_k=8,
        moe_num_shared=1,
        moe_hidden=512,
    ),
    "Kohaku-MoE-3B": dict(
        dim=1024,
        depth=27,
        heads=16,
        kv_heads=2,
        head_dim=64,
        mlp_hidden=2688,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=64,
        moe_top_k=8,
        moe_num_shared=2,
        moe_hidden=512,
    ),
    "Kohaku-MoE-5B": dict(
        dim=1280,
        depth=30,
        heads=20,
        kv_heads=4,
        head_dim=64,
        mlp_hidden=3456,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=64,
        moe_top_k=8,
        moe_num_shared=1,
        moe_hidden=640,
    ),
    # E=128 / top_k=16 where every other sparse rung is 64 / 8: granularity
    # moves, kappa does not. Do not "fix" this back to 64/8; see docs/concepts/presets.md.
    "Kohaku-MoE-8B": dict(
        dim=1536,
        depth=33,
        heads=24,
        kv_heads=4,
        head_dim=64,
        mlp_hidden=4096,
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=128,
        moe_top_k=16,
        moe_num_shared=1,
        moe_hidden=384,  # dim/4; see docs/concepts/presets.md
    ),
}

PRESETS: dict[str, dict[str, Any]] = {
    **_DENSE,
    **_SPARSE,
    **_KOHAKU_DENSE,
    **_KOHAKU_SPARSE,
}

# The rungs in ladder order, for sweeps and for "the next size up".
KOHAKU_LADDER: tuple[str, ...] = (
    "Kohaku-200M",
    "Kohaku-MoE-1B",
    "Kohaku-500M",
    "Kohaku-MoE-2B",
    "Kohaku-1B",
    "Kohaku-MoE-3B",
    "Kohaku-1.5B",
    "Kohaku-MoE-5B",
    "Kohaku-MoE-8B",
)


def get_preset(name: str, **overrides: Any) -> LMArchConfig:
    """Build an :class:`LMArchConfig` from a preset name plus field overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return LMArchConfig(**{**PRESETS[name], **overrides})
