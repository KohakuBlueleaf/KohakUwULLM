"""The decoder-only LM: embeddings -> blocks -> final norm, plus the head.

A pure ``(tokens, seq_info) -> hidden`` function; the loss lives in
:class:`~kohakuwullm.models.head.LMHead`. See docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from kohakuwullm.models.block import DecoderBlock
from kohakuwullm.models.cache import KVCache
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.models.head import LMHead
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights, swap_mxfp8
from kohakuwullm.models.presets import LMArchConfig
from kohakuwullm.registry import NORM, POSENC, build
from kohakuwullm.utils import count_active_parameters, count_parameters


class LMBackbone(nn.Module):
    """Configurable decoder-only language model.

    Args:
        config: architecture spec (:class:`LMArchConfig`).
        head_kwargs: extra kwargs for :class:`LMHead` (z-loss, label smoothing).
    """

    def __init__(self, config: LMArchConfig, head_kwargs: dict | None = None) -> None:
        super().__init__()
        self.config = config
        self.grad_ckpt = config.grad_ckpt

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.embed_scale = config.dim**0.5 if config.embedding_scale else 1.0

        self.pos_enc = build(
            config.posenc,
            POSENC,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
            factor=config.rope_factor,
            partial_rotary_factor=config.rope_partial,
            original_context=config.max_position,
        )

        self.blocks = nn.ModuleList(
            self.build_block(config, i) for i in range(config.depth)
        )
        self.final_norm = build(config.norm, NORM, dim=config.dim, eps=config.norm_eps)
        self.head = LMHead(
            config.dim,
            config.vocab_size,
            tie_embeddings=config.tie_embeddings,
            embedding=self.embed,
            z_loss_weight=config.z_loss_weight,
            soft_cap=config.logit_soft_cap,
            **(head_kwargs or {}),
        )
        self.moe_blocks = [
            b for b, is_moe in zip(self.blocks, config.moe_layers) if is_moe
        ]
        self.initialize_weights()
        # After `initialize_weights`, never before: the swap copies what the
        # scaled init produced.
        self.mxfp8_projections: tuple[str, ...] = ()
        # Modules, not tensors: this is what `refresh_mxfp8` returns.
        self.mxfp8_modules: tuple[str, ...] = ()
        if config.mxfp8:
            self._swap_to_mxfp8()

    @staticmethod
    def build_block(config: LMArchConfig, index: int) -> DecoderBlock:
        """One decoder block with layer ``index``'s properties.

        The single place a block is constructed, so anything that needs one in
        isolation -- the split autotuner, a probe -- builds what the model does.
        See docs/internals/pipeline.md.
        """
        is_moe = config.moe_layers[index]
        return DecoderBlock(
            config.dim,
            config.heads,
            kv_heads=config.kv_heads,
            head_dim=config.head_dim,
            norm=config.norm,
            mlp=LMBackbone._mlp_spec(config, is_moe),
            attn=config.attn_for(index),
            mlp_ratio=config.moe_ratio if is_moe else config.mlp_ratio,
            mlp_kwargs=LMBackbone._mlp_kwargs(config, is_moe),
            attn_kwargs={
                "qk_norm": config.qk_norm,
                "qk_norm_affine": config.qk_norm_affine,
                "bias": config.attn_bias,
                "sink": config.attn_sink,
            },
            sliding_window=config.window_for(index),
            post_norm=config.post_norm,
            residual_scale=config.residual_scale,
            norm_eps=config.norm_eps,
        )

    @staticmethod
    def _mlp_spec(config: LMArchConfig, is_moe: bool):
        # `moe_mlp` selects the sparse formulation: "moe" or "latent_moe".
        return (config.moe_mlp or "moe") if is_moe else config.mlp

    @staticmethod
    def _mlp_kwargs(config: LMArchConfig, is_moe: bool) -> dict:
        if not is_moe:
            return {"hidden": config.mlp_hidden, "multiple_of": config.mlp_multiple_of}
        return {
            "hidden": config.moe_hidden,
            "multiple_of": config.mlp_multiple_of,
            "num_experts": config.moe_num_experts,
            "top_k": config.moe_top_k,
            "num_shared": config.moe_num_shared,
            "router": config.moe_router,
            **config.moe_router_kwargs,
            **config.moe_mlp_kwargs,
        }

    def initialize_weights(self) -> None:
        """Normal init; output projections scaled by ``1/sqrt(2 * depth)``.

        ``config.mup_base_dim`` adds the width term ``sqrt(base / fan_in)`` to
        every hidden matrix, excluding the embedding. See docs/concepts/architecture.md.
        """
        std = self.config.init_std
        depth_scale = (2 * self.config.depth) ** -0.5
        base = self.config.mup_base_dim

        def width_scale(fan_in: int) -> float:
            return 1.0 if base is None else (base / fan_in) ** 0.5

        def _basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight, std=std * width_scale(module.in_features)
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=std)

        self.apply(_basic)
        for block in self.blocks:
            attn_out = block.attn.o_proj
            nn.init.normal_(
                attn_out.weight,
                std=std * depth_scale * width_scale(attn_out.in_features),
            )
            mlp = block.mlp
            # Structural, not `isinstance(mlp, MoEMLP)`: `latent_moe` exposes the
            # same stacked expert weights without inheriting.
            expert_stack = getattr(mlp, "w_out", None)
            if isinstance(expert_stack, torch.nn.Parameter) and expert_stack.ndim == 3:
                # `_basic` never saw these, and `w_in` keeps MoEMLP's own fan-in
                # init unless muP is on.
                if base is not None:
                    nn.init.normal_(mlp.w_in, std=std * width_scale(mlp.w_in.shape[-1]))
                nn.init.normal_(
                    mlp.w_out, std=std * depth_scale * width_scale(mlp.w_out.shape[-1])
                )
                if mlp.shared is not None:
                    shared_out = mlp.shared.w_out
                    nn.init.normal_(
                        shared_out.weight,
                        std=std * depth_scale * width_scale(shared_out.in_features),
                    )
            else:
                nn.init.normal_(
                    mlp.w_out.weight,
                    std=std * depth_scale * width_scale(mlp.w_out.in_features),
                )

    def _run_block(self, block, x, seq_info, posenc, cache=None):
        if self.grad_ckpt and self.training:
            return checkpoint.checkpoint(
                block, x, seq_info, posenc, cache, use_reentrant=False
            )
        return block(x, seq_info, posenc, cache)

    def forward(
        self,
        tokens: torch.Tensor,
        seq_info: SeqInfo | None = None,
        return_taps: tuple[int, ...] = (),
        cache: KVCache | None = None,
    ):
        """``tokens``: ``(T,)`` packed or ``(B, S)`` padded.

        Returns hidden states of the same leading shape plus a ``dim`` axis, or
        ``(hidden, taps)`` when ``return_taps`` names layer indices.

        ``cache`` makes ``tokens`` the continuation of an already-cached padded
        prefix; it is advanced by ``tokens.shape[1]`` here. See
        docs/concepts/architecture.md.
        """
        if seq_info is None:
            if cache is not None:
                seq_info = cache.seq_info(tokens)
            else:
                seq_info = (
                    SeqInfo.padded(tokens.shape[0], tokens.shape[1], tokens.device)
                    if tokens.dim() == 2
                    else SeqInfo.from_lengths(
                        torch.tensor([tokens.shape[0]]), tokens.device
                    )
                )
        x = self.embed(tokens)
        if self.embed_scale != 1.0:
            x = x * self.embed_scale
        posenc = self.pos_enc.prepare(seq_info.position_ids, x.device, x.dtype)

        tap_set = set(return_taps)
        taps: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = self._run_block(
                block, x, seq_info, posenc, None if cache is None else cache.layer(i)
            )
            if i in tap_set:
                taps[i] = x
        if cache is not None:
            cache.advance(tokens.shape[1])
        x = self.final_norm(x)
        return (x, taps) if tap_set else x

    def loss(
        self,
        tokens,
        labels,
        seq_info=None,
        reduction: str = "sum",
        router_scale: float = 1.0,
    ):
        """Forward + head loss. Returns ``(loss, logs)``.

        ``router_scale`` multiplies the MoE auxiliary terms before they join the
        loss. The terms are per-token means, so a ``reduction="sum"`` caller that
        divides the total by its token count has to pass that count back here.
        See docs/internals/moe-router-loss.md.
        """
        hidden = self.forward(tokens, seq_info)
        loss, logs = self.head.loss(hidden, labels, reduction=reduction)
        router = self.router_losses()
        if router is not None:
            loss = loss + router * router_scale
            logs["router_loss"] = router.detach()
        return loss, logs

    def router_losses(self) -> torch.Tensor | None:
        """Summed MoE auxiliary terms across sparse layers, or ``None``."""
        terms = [t for b in self.moe_blocks if (t := b.mlp.router_losses()) is not None]
        return None if not terms else torch.stack(terms).sum()

    @torch.no_grad()
    def update_router_bias(self) -> dict[str, float]:
        """Advance every MoE router's balancing bias; return load statistics.

        Call once per *optimizer* step, not per micro-batch.
        """
        stats = [
            r for b in self.moe_blocks if (r := b.mlp.router.update_bias()) is not None
        ]
        if not stats:
            return {}
        # One host sync for every layer and every statistic.
        rows = torch.stack(stats).tolist()
        high = [r[0] for r in rows]
        low = [r[1] for r in rows]
        dead = [r[2] for r in rows]
        return {
            "moe/load_imbalance_max": max(high),
            "moe/load_imbalance_mean": sum(high) / len(high),
            "moe/load_starved_min": min(low),
            "moe/dead_experts_max": max(dead),
            "moe/dead_experts_total": sum(dead),
        }

    def _swap_to_mxfp8(self) -> None:
        """Replace the eligible projections with MXFP8 linears, or refuse to.

        Raises on a shape refusal, a declared matmul nothing converted, or a
        matmul-shaped parameter nobody claimed. See docs/concepts/architecture.md.
        """
        report = swap_mxfp8(self)
        if report.blocking:
            raise ValueError(
                f"config.mxfp8 is set but {report.share('fp8'):.1%} of this model's "
                f"per-token matmul reached fp8, so it would train as a bf16/fp8 "
                f"mixture:\n{report.summary()}\n"
                "A contraction axis cannot be padded away -- it is shared with the "
                "activation cast -- so use a preset whose widths the kernels take, or "
                "leave mxfp8 off. Refusing beats a run every log calls fp8."
            )
        self.mxfp8_projections = tuple(report.swapped)
        self.mxfp8_modules = tuple(report.modules)

    @torch.no_grad()
    def refresh_mxfp8(self) -> int:
        """Re-quantize every MXFP8 weight; returns how many modules were refreshed.

        Call once per *optimizer* step, beside :meth:`update_router_bias`. Assert
        the count against ``len(self.mxfp8_modules)``, not
        ``mxfp8_projections``. See docs/concepts/architecture.md.
        """
        if not self.mxfp8_modules:
            return 0
        return refresh_mxfp8_weights(self)

    def param_summary(self) -> dict[str, int]:
        """Total / active / embedding parameter counts, for the startup log."""
        total = count_parameters(self, trainable_only=False)
        embed = self.embed.weight.numel() * (1 if self.config.tie_embeddings else 2)
        return {
            "total": total,
            "active": count_active_parameters(self),
            "embedding": embed,
            "non_embedding": total - embed,
        }
