"""Per-token FLOP accounting -- the numerator behind MFU.

``active - embedding + head``, plus an explicit quadratic attention term. Two
totals: *model* FLOPs and *hardware* FLOPs. See docs/concepts/architecture.md.
"""

from collections import Counter

import torch

from kohakuwullm.models.components.seqinfo import SeqInfo

# One grad-input GEMM and one grad-weight GEMM per forward GEMM.
BACKWARD_MULTIPLIER = 2.0


def document_lengths(seq_info: SeqInfo) -> torch.Tensor:
    """``(N,)`` fp64 lengths; the padded layout reports ``max_seqlen`` per row."""
    if seq_info.packed:
        cu = seq_info.cu_seqlens
        return (cu[1:] - cu[:-1]).to(torch.float64)
    return torch.full(
        (seq_info.num_seqs,),
        float(seq_info.max_seqlen),
        dtype=torch.float64,
        device=seq_info.position_ids.device,
    )


def attended_pairs(lengths: torch.Tensor, window: int | None) -> torch.Tensor:
    """Pairs a causal mask leaves per document: ``L(L+1)/2``, or ``Lw - w(w-1)/2``
    once ``L >= w``."""
    causal = lengths * (lengths + 1) / 2
    if window is None:
        return causal
    w = float(window)
    return torch.where(lengths >= w, lengths * w - w * (w - 1) / 2, causal)


class FlopCounter:
    """FLOPs of one built model, evaluated against real batch shapes.

    Args:
        backbone: the model to charge for, read once at build time.
        grad_ckpt: override the config's setting.
    """

    def __init__(self, backbone, grad_ckpt: bool | None = None) -> None:
        config = backbone.config
        summary = backbone.param_summary()
        # Everything except the vocabulary matrices: blocks, norms, routers.
        self.block_per_token = 2.0 * float(summary["active"] - summary["embedding"])
        self.head_per_token = 2.0 * float(config.dim * config.vocab_size)
        self.q_dim = config.heads * config.head_dim
        self.grad_ckpt = config.grad_ckpt if grad_ckpt is None else grad_ckpt
        # Layers sharing a window share a cost, so charge them in groups.
        self.window_counts = Counter(config.windows)

    def attention_flops(self, lengths: torch.Tensor) -> torch.Tensor:
        """Forward score + AV FLOPs over all layers: ``4 * q_dim`` per pair."""
        # fp64 whatever came in: the pair count is quadratic.
        lengths = lengths.to(torch.float64)
        pairs = lengths.new_zeros(())
        for window, layers in self.window_counts.items():
            pairs = pairs + layers * attended_pairs(lengths, window).sum()
        return 4.0 * self.q_dim * pairs

    def stage_flops(
        self,
        num_tokens: int,
        lengths: torch.Tensor,
        block_share: float = 1.0,
        has_head: bool = True,
    ) -> torch.Tensor:
        """``(2,)`` fp64 model and hardware FLOPs for one pipeline stage.

        Args:
            block_share: fraction of the blocks this stage owns.
            has_head: charge the vocabulary projection to this stage.
        """
        blocks = (
            self.block_per_token * num_tokens + self.attention_flops(lengths)
        ) * block_share
        head = self.head_per_token * num_tokens if has_head else 0.0
        model = (1.0 + BACKWARD_MULTIPLIER) * (blocks + head)
        recompute = blocks if self.grad_ckpt else torch.zeros_like(blocks)
        return torch.stack([model, model + recompute])

    def batch_flops(self, num_tokens: int, lengths: torch.Tensor) -> torch.Tensor:
        """``(2,)`` fp64 model and hardware FLOPs, on ``lengths``' device."""
        return self.stage_flops(num_tokens, lengths)

    def per_token(self, seq_len: int) -> float:
        """Model FLOPs per token at a uniform document length, for the startup log."""
        lengths = torch.tensor([float(seq_len)], dtype=torch.float64)
        return float(self.batch_flops(seq_len, lengths)[0]) / max(seq_len, 1)
