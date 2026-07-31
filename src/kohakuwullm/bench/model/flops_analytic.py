"""Closed-form FLOPs per token, split into matrix and vector work.

One multiply-accumulate is 2 FLOPs; backward is charged 2x forward; gradient
checkpointing charges the block forward twice. The attention term is a lower bound.
See docs/performance/benchmarking.md.
"""

from dataclasses import dataclass

from kohakuwullm.models.components.mlp import resolve_hidden
from kohakuwullm.models.presets import get_preset

MAC = 2.0


@dataclass(frozen=True)
class FlopBudget:
    """Per-token FLOPs of one preset, by kind. All values are FLOPs, not MACs."""

    name: str
    gemm_fwd: float
    gemm_bwd: float
    attn_fwd: float
    attn_bwd: float
    vector_fwd: float
    vector_bwd: float
    pairs_per_token: float

    @property
    def matrix(self) -> float:
        """Everything a tensor core does: weight GEMMs plus the score/AV matmuls."""
        return self.gemm_fwd + self.gemm_bwd + self.attn_fwd + self.attn_bwd

    @property
    def vector(self) -> float:
        return self.vector_fwd + self.vector_bwd

    @property
    def total(self) -> float:
        return self.matrix + self.vector

    @property
    def matrix_share(self) -> float:
        return self.matrix / self.total

    @property
    def gemm_only(self) -> float:
        """Weight GEMMs alone -- what scales with parameters, not with length."""
        return self.gemm_fwd + self.gemm_bwd


def causal_pairs_per_token(len_min: int, len_max: int, window: int | None) -> float:
    """Attended (query, key) pairs per token, for documents uniform on ``[a, b]``.

    Token-weighted, not document-weighted, and summed exactly.
    """
    pairs = tokens = 0
    for length in range(len_min, len_max + 1):
        if window is None or length <= window:
            doc_pairs = length * (length + 1) / 2
        else:
            # Full triangle until the window fills, then a constant band.
            doc_pairs = window * (window + 1) / 2 + (length - window) * window
        pairs += doc_pairs
        tokens += length
    return pairs / tokens


def _layer_kinds(config) -> tuple[int, int]:
    """``(dense_layers, moe_layers)`` under ``moe_every`` / ``moe_first_dense``."""
    if not config.moe_num_experts or not config.moe_every:
        return config.depth, 0
    moe = sum(
        1
        for i in range(config.depth)
        if i >= config.moe_first_dense
        and (i - config.moe_first_dense) % config.moe_every == 0
    )
    return config.depth - moe, moe


def budget(
    name: str,
    len_min: int = 512,
    len_max: int = 4096,
    grad_ckpt: bool = False,
) -> FlopBudget:
    """Per-token FLOP budget for a named preset at a document-length distribution."""
    config = get_preset(name)
    dim, depth = config.dim, config.depth
    q_dim = config.heads * config.head_dim
    kv_dim = config.kv_heads * config.head_dim
    dense_layers, moe_layers = _layer_kinds(config)

    window = config.sliding_window if config.attn_sliding else None
    pairs = causal_pairs_per_token(len_min, len_max, window)

    # -- matrix: weight GEMMs -------------------------------------------------- #
    # Width from the model's own rule; `config.mlp_hidden` is optional and often unset.
    mlp_hidden = resolve_hidden(
        dim, config.mlp_ratio, config.mlp_hidden, config.mlp_multiple_of, glu=True
    )
    attn_proj = MAC * (dim * q_dim + 2 * dim * kv_dim + q_dim * dim)
    dense_mlp = MAC * 3 * dim * mlp_hidden
    # A dense preset leaves the MoE fields None; guard every read of them.
    experts = (config.moe_top_k or 0) + (config.moe_num_shared or 0)
    moe_hidden = (
        resolve_hidden(
            dim, config.moe_ratio, config.moe_hidden, config.mlp_multiple_of, glu=True
        )
        if moe_layers
        else 0
    )
    moe_mlp = (
        MAC * (3 * dim * moe_hidden * experts + dim * config.moe_num_experts)
        if moe_layers
        else 0.0
    )

    gemm_fwd = depth * attn_proj + dense_layers * dense_mlp + moe_layers * moe_mlp
    head = MAC * dim * config.vocab_size
    gemm_fwd += head
    # Checkpointing re-runs the blocks but not the head, which is outside the blocks.
    if grad_ckpt:
        gemm_fwd += gemm_fwd - head
    gemm_bwd = 2.0 * gemm_fwd

    # -- matrix: attention scores and the AV product --------------------------- #
    # `q_dim`, not `kv_dim`: GQA shares keys, but every query head still scores.
    attn_fwd = depth * MAC * 2 * pairs * q_dim
    attn_bwd = 2.0 * attn_fwd

    # -- vector ---------------------------------------------------------------- #
    # RMSNorm: square, sum, scale, weight -- four passes over the row.
    norm = 4.0 * dim
    per_layer = 2 * norm + 2 * dim  # two norms, two residual adds
    if config.qk_norm:
        per_layer += 4.0 * (q_dim + kv_dim)
    # RoPE rotates pairs of elements: two multiplies and one add each, per element.
    per_layer += 3.0 * (q_dim + kv_dim) * config.rope_partial
    # Softmax runs over every attended pair, per head.
    per_layer += 3.0 * pairs * config.heads
    vector_fwd = depth * per_layer
    vector_fwd += dense_layers * 6.0 * mlp_hidden  # SiLU, then two multiplies
    if moe_layers:
        vector_fwd += moe_layers * (
            6.0 * moe_hidden * experts
            + 3.0 * config.moe_num_experts  # router softmax
            + config.moe_top_k * dim  # weighted combine
        )
    vector_fwd += norm  # final norm
    vector_fwd += 5.0 * config.vocab_size  # cross-entropy softmax
    vector_bwd = 2.0 * vector_fwd

    return FlopBudget(
        name=name,
        gemm_fwd=gemm_fwd,
        gemm_bwd=gemm_bwd,
        attn_fwd=attn_fwd,
        attn_bwd=attn_bwd,
        vector_fwd=vector_fwd,
        vector_bwd=vector_bwd,
        pairs_per_token=pairs,
    )
