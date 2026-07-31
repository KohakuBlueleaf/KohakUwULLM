"""The decoder block: ``x = x + attn(norm(x))``, then ``x = x + ffn(norm(x))``.

See docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn

from kohakuwullm.registry import ATTENTION, MLP, NORM, build


class DecoderBlock(nn.Module):
    """One transformer decoder layer.

    Args:
        dim: model width.
        heads / kv_heads / head_dim: attention shape.
        norm / mlp / attn: component specs.
        mlp_kwargs / attn_kwargs: extra kwargs forwarded to each.
        sliding_window: window for this layer, or ``None`` for full causal.
        post_norm: add a norm to the output of each residual branch.
        residual_scale: multiply each branch before it joins the residual stream.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int | None = None,
        head_dim: int | None = None,
        *,
        norm="rmsnorm",
        mlp="swiglu",
        attn="varlen",
        mlp_ratio: float = 4.0,
        mlp_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
        sliding_window: int | None = None,
        post_norm: bool = False,
        residual_scale: float = 1.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm_attn = build(norm, NORM, dim=dim, eps=norm_eps)
        self.attn = build(
            attn,
            ATTENTION,
            dim=dim,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
            eps=norm_eps,
            **(attn_kwargs or {}),
        )
        self.norm_mlp = build(norm, NORM, dim=dim, eps=norm_eps)
        self.mlp = build(mlp, MLP, dim=dim, ratio=mlp_ratio, **(mlp_kwargs or {}))

        if post_norm:
            self.post_attn_norm = build(norm, NORM, dim=dim, eps=norm_eps)
            self.post_mlp_norm = build(norm, NORM, dim=dim, eps=norm_eps)
        else:
            self.post_attn_norm = nn.Identity()
            self.post_mlp_norm = nn.Identity()
        self.residual_scale = residual_scale

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        h = self.post_attn_norm(self.attn(self.norm_attn(x), seq_info, posenc, cache))
        x = x + h * self.residual_scale
        h = self.post_mlp_norm(self.mlp(self.norm_mlp(x)))
        return x + h * self.residual_scale
