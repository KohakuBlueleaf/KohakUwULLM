"""DeepSeekMoE-style sparse feed-forward: shared experts + routed experts.

See docs/concepts/architecture.md.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed, expert_sort
from kohakuwullm.kernels.moe.router import fused_router
from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE
from kohakuwullm.kernels.mxfp8.linear import compute_dtype
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts
from kohakuwullm.kernels.mxfp8.moe_unfused import mxfp8_moe_experts_unfused
from kohakuwullm.models.components.mlp import GLUMLP, resolve_hidden
from kohakuwullm.models.mxfp8_protocol import Matmul
from kohakuwullm.registry import MLP, ROUTER, build

_GATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _sqrt_softplus(logits: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4's gate: ``sqrt(softplus(x))``, unbounded above."""
    return F.softplus(logits).sqrt()


_SCORE_FUNCS = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda logits: logits.softmax(-1),
    "sqrtsoftplus": _sqrt_softplus,
}


def expert_counts(flat_idx: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """``(num_buckets,)`` int32 rows per bucket, from a flat ``(T * slots,)`` index.

    The histogram :func:`~kohakuwullm.kernels.moe.moe_dispatch.expert_sort` needs.
    """
    # scatter_add, never bincount: bincount syncs. See docs/concepts/architecture.md.
    counts = torch.zeros(num_buckets, dtype=torch.int32, device=flat_idx.device)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))
    return counts


@ROUTER.register("topk")
class TopKRouter(nn.Module):
    """Token -> expert assignment with aux-loss-free balancing.

    See docs/concepts/architecture.md.

    Args:
        dim: model width.
        num_experts: routed expert count.
        top_k: experts per token.
        score_func: ``"sigmoid"``, ``"softmax"`` or ``"sqrtsoftplus"``.
        norm_topk_prob: renormalize the selected weights to sum to 1.
        routed_scaling_factor: multiply the combined routed output.
        bias_update_rate: step size for the balancing bias; 0 disables it.
        n_groups / topk_groups: node-limited routing -- restrict a token to
            ``topk_groups`` of ``n_groups`` groups before the final top-k.
        aux_loss_weight: classic load-balance auxiliary loss; 0 disables it.
        z_loss_weight: router logit z-loss.
        fused: use the single-launch Triton router where eligible.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 8,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        bias_update_rate: float = 1e-3,
        n_groups: int = 1,
        topk_groups: int = 1,
        aux_loss_weight: float = 0.0,
        z_loss_weight: float = 0.0,
        fused: bool = True,
    ) -> None:
        super().__init__()
        if num_experts % n_groups != 0:
            raise ValueError(
                f"num_experts {num_experts} not divisible by n_groups {n_groups}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        if score_func not in ("sigmoid", "softmax", "sqrtsoftplus"):
            raise ValueError(f"unknown score_func {score_func!r}")
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.bias_update_rate = bias_update_rate
        # The load reduction spans the data-parallel group only; a pipeline stage
        # turns it off. See docs/internals/pipeline.md.
        self.reduce_load = True
        self.load_group = None
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight

        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)
        # Selection-only bias, updated by `update_bias` rather than by gradient.
        if bias_update_rate > 0:
            self.register_buffer("expert_bias", torch.zeros(num_experts))
            self.register_buffer("load_accum", torch.zeros(num_experts))
        else:
            self.expert_bias = None
            self.load_accum = None
        # Populated each forward for the trainer to log / add to the loss.
        self.aux_loss: torch.Tensor | None = None
        self.z_loss: torch.Tensor | None = None

        # Fused-kernel eligibility, resolved once. See docs/concepts/architecture.md.
        self.use_fused = (
            fused
            and score_func in ("sigmoid", "sqrtsoftplus")
            and n_groups == 1
            and num_experts <= 128
        )
        # Whether a forward will populate `aux_loss` / `z_loss`, resolved before
        # any forward runs. See docs/internals/moe-router-loss.md.
        self.emits_loss = aux_loss_weight > 0.0 or z_loss_weight > 0.0

    def _group_limit(self, scores: torch.Tensor) -> torch.Tensor:
        """Mask out all but the ``topk_groups`` strongest expert groups."""
        t = scores.shape[0]
        grouped = scores.view(t, self.n_groups, -1)
        # A group's strength is its top-2 sum (DeepSeek-V3).
        k = min(2, grouped.shape[-1])
        strength = grouped.topk(k, dim=-1).values.sum(-1)
        keep = strength.topk(self.topk_groups, dim=-1).indices
        mask = torch.zeros_like(strength, dtype=torch.bool)
        mask.scatter_(1, keep, True)
        return scores.masked_fill(
            ~mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores), float("-inf")
        )

    def forward(self, x: torch.Tensor):
        """``x`` is ``(T, D)`` flat. Returns ``(topk_idx, topk_weight)``."""
        return self.route(x)[:2]

    def route(self, x: torch.Tensor):
        """As :meth:`forward`, plus this call's per-expert row ``counts``, an int32
        ``(E,)`` tensor :class:`MoEMLP` builds the grouped-GEMM offsets from."""
        # Device is an input property, not a config value: Triton has no CPU
        # backend and a module is built on CPU before it is moved.
        if self.use_fused and x.is_cuda:
            return self._route_fused(x)
        return self._route_eager(x)

    def _route_fused(self, x: torch.Tensor):
        accum = self.load_accum if self.training else None
        topk_idx, topk_weight, counts, aux, z = fused_router(
            x,
            self.weight,
            self.expert_bias,
            self.top_k,
            self.norm_topk_prob,
            self.routed_scaling_factor,
            accum,
            # Gate multiply follows the activation dtype, unlike the eager path.
            x.dtype if x.dtype in _GATE_DTYPES else torch.bfloat16,
            self.score_func,
            self.aux_loss_weight,
            self.z_loss_weight,
        )
        self.z_loss = z
        self.aux_loss = aux
        return topk_idx, topk_weight.to(x.dtype), counts

    def _route_eager(self, x: torch.Tensor):
        logits = F.linear(x.float(), self.weight.float())
        scores = _SCORE_FUNCS[self.score_func](logits)

        select_scores = (
            scores if self.expert_bias is None else scores + self.expert_bias
        )
        if self.n_groups > 1:
            select_scores = self._group_limit(select_scores)

        topk_idx = select_scores.topk(self.top_k, dim=-1).indices
        # Weight by the unbiased score: the bias steers selection only.
        topk_weight = scores.gather(1, topk_idx)
        if self.norm_topk_prob and self.top_k > 1:
            topk_weight = topk_weight / topk_weight.sum(-1, keepdim=True).clamp_min(
                1e-9
            )
        topk_weight = topk_weight * self.routed_scaling_factor

        self.z_loss = (
            self.z_loss_weight * logits.logsumexp(-1).pow(2).mean()
            if self.z_loss_weight > 0
            else None
        )
        counts = expert_counts(topk_idx.reshape(-1), self.num_experts)

        self.aux_loss = (
            self._aux_loss(scores, counts) if self.aux_loss_weight > 0 else None
        )
        if self.training and self.load_accum is not None:
            self.load_accum += counts
        return topk_idx, topk_weight.to(x.dtype), counts

    def _aux_loss(self, scores: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balance loss: ``E * sum_i f_i * P_i``."""
        t = scores.shape[0]
        fraction = counts.float() / (t * self.top_k)
        prob = scores.mean(0)
        return self.aux_loss_weight * self.num_experts * (fraction * prob).sum()

    @torch.no_grad()
    def update_bias(self) -> torch.Tensor | None:
        """Nudge the selection bias toward underloaded experts, reset the counter.

        Call once per optimizer step. Returns ``(max/mean, min/mean, dead)`` on
        device, where ``dead`` counts experts this step routed nothing to. One
        tensor rather than three scalars keeps it to a single host sync when the
        caller reads it. See docs/internals/moe-router-loss.md.
        """
        if self.load_accum is None:
            return None
        # Balance the load over the whole step, not this rank's share of it.
        if self.reduce_load and dist.is_available() and dist.is_initialized():
            dist.all_reduce(
                self.load_accum, op=dist.ReduceOp.SUM, group=self.load_group
            )
        total = self.load_accum.sum()
        if total == 0:
            return None
        mean = self.load_accum.mean()
        # sign(), not the raw error, so the step size stays bounded.
        self.expert_bias += self.bias_update_rate * torch.sign(mean - self.load_accum)
        scale = mean.clamp_min(1e-9)
        stats = torch.stack(
            (
                self.load_accum.max() / scale,
                self.load_accum.min() / scale,
                (self.load_accum == 0).sum().to(scale.dtype),
            )
        )
        self.load_accum.zero_()
        return stats


@MLP.register("moe")
class MoEMLP(nn.Module):
    """Sparse feed-forward: ``shared(x) + sum_k w_k * expert_k(x)``.

    See docs/concepts/architecture.md.

    Args:
        dim: model width.
        ratio / hidden / multiple_of: geometry of **one routed expert**.
        num_experts: routed expert count.
        top_k: experts per token.
        num_shared: always-on expert count (their widths add to the routed one).
        router: router spec (registry name / dict / class).
        router_dim: width the router reads; ``None`` uses ``dim``.
        dense_fallback: use a loop of per-expert GEMMs instead of the grouped
            Triton kernel. Same numerics, far slower; the test reference.
    """

    def __init__(
        self,
        dim: int,
        ratio: float = 4.0,
        hidden: int | None = None,
        multiple_of: int = 128,
        num_experts: int = 64,
        top_k: int = 8,
        num_shared: int = 1,
        bias: bool = False,
        router="topk",
        router_dim: int | None = None,
        dense_fallback: bool = False,
        **router_kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden = resolve_hidden(dim, ratio, hidden, multiple_of, glu=True)
        self.dense_fallback = dense_fallback

        # Routing width is separable from expert width, for latent MoE.
        self.router = build(
            router,
            ROUTER,
            dim=router_dim or dim,
            num_experts=num_experts,
            top_k=top_k,
            **router_kwargs,
        )
        # Dispatch buckets: one more than the experts for a router with a
        # sentinel slot (ReMoE), otherwise exactly num_experts.
        self.num_buckets = getattr(self.router, "num_buckets", num_experts)

        # One stacked tensor per matrix, so the grouped GEMM indexes by expert id.
        self.w_in = nn.Parameter(torch.empty(num_experts, 2 * self.hidden, dim))
        self.w_out = nn.Parameter(torch.empty(num_experts, dim, self.hidden))
        nn.init.normal_(self.w_in, std=dim**-0.5)
        nn.init.normal_(self.w_out, std=(self.hidden * 2) ** -0.5)

        self.shared = (
            GLUMLP(dim, hidden=self.hidden * num_shared, bias=bias, multiple_of=1)
            if num_shared > 0
            else None
        )
        self.num_shared = num_shared
        # The routed path, rebound once by `enable_mxfp8`.
        self._routed = self._routed_eager
        self._packed: MXFP8ExpertWeights | None = None
        # Which MXFP8 expert path `enable_mxfp8` installs: "fused" or "unfused".
        self.mxfp8_expert_path = "fused"
        self._has_sentinel = self.num_buckets != num_experts

    def mxfp8_matmul(self) -> dict[str, Matmul]:
        """The matmul this layer holds as bare parameters, including the router's.

        See ``mxfp8_protocol`` and docs/concepts/architecture.md.
        """
        refusal = self._mxfp8_refusal()
        return {
            "w_in": Matmul(self.w_in[0].numel() * self.top_k, refusal),
            "w_out": Matmul(self.w_out[0].numel() * self.top_k, refusal),
            "router.weight": Matmul(
                self.router.weight.numel(),
                "a router's logit scale is the gate sharpness",
                never=True,
            ),
        }

    def _mxfp8_refusal(self) -> str | None:
        """Why the MXFP8 expert path cannot take this layer, or ``None``.

        See docs/concepts/architecture.md.
        """
        if self.dense_fallback:
            return "dense_fallback=True is the eager test reference, not an fp8 path"
        for name, value in (("dim", self.dim), ("hidden", self.hidden)):
            if value % BLOCK_SCALE:
                return (
                    f"{name}={value} is not a multiple of {BLOCK_SCALE}; it is a "
                    "contraction axis shared with an activation cast"
                )
        return None

    def enable_mxfp8(self) -> None:
        """Send the routed experts through an MXFP8 path, in place, and install
        ``refresh_quantized_weight`` on the instance."""
        match self.mxfp8_expert_path:
            case "fused":
                self._routed = self._routed_mxfp8
            case "unfused":
                self._routed = self._routed_mxfp8_unfused
            case other:
                raise ValueError(
                    f"unknown mxfp8_expert_path {other!r}; expected 'fused' or 'unfused'"
                )
        self.refresh_quantized_weight = self._refresh_mxfp8

    @torch.no_grad()
    def _refresh_mxfp8(self) -> None:
        """Requantize the four fp8 expert copies. Call after **every** optimizer step."""
        if self._packed is None:
            self._packed = MXFP8ExpertWeights(self.w_in, self.w_out)
        else:
            self._packed.refresh(self.w_in, self.w_out)

    def _apply(self, *args, **kwargs):
        """Drop the fp8 expert copies on any device or dtype transform."""
        self._packed = None
        return super()._apply(*args, **kwargs)

    def active_parameters(self) -> int:
        """Parameters one token actually touches (governs FLOPs, not memory)."""
        per_expert = self.w_in[0].numel() + self.w_out[0].numel()
        total = per_expert * self.top_k + self.router.weight.numel()
        if self.shared is not None:
            total += sum(p.numel() for p in self.shared.parameters())
        return total

    def _expert_compute(self, x_sorted, offsets):
        """SwiGLU through the stacked expert matrices, tokens already sorted."""
        if self.dense_fallback:
            outs = []
            off = offsets.tolist()
            for e in range(self.num_experts):
                rows = x_sorted[off[e] : off[e + 1]]
                if rows.shape[0] == 0:
                    outs.append(rows.new_zeros(0, self.dim))
                    continue
                gate, value = (rows @ self.w_in[e].T).chunk(2, dim=-1)
                outs.append((F.silu(gate) * value) @ self.w_out[e].T)
            return torch.cat(outs, dim=0)
        h = grouped_gemm(x_sorted, self.w_in, offsets)
        # Eager `silu * value`, not `swiglu_mul`: this is the bf16 control arm.
        gate, value = h.chunk(2, dim=-1)
        return grouped_gemm(F.silu(gate) * value, self.w_out, offsets)

    def _routed_eager(self, flat, topk_weight, order, token_of, offsets):
        """Gather, expert GEMMs, gate and scatter as three separate launches."""
        # Truncated offsets keep the sentinel bucket's rows out of the grid.
        expert_offsets = offsets[: self.num_experts + 1]
        out_sorted = self._expert_compute(
            flat.index_select(0, token_of).contiguous(), expert_offsets
        )
        return combine_routed(
            out_sorted,
            topk_weight.reshape(-1),
            order,
            token_of,
            flat.shape[0],
            offsets[self.num_experts : self.num_experts + 1],
        )

    def _routed_mxfp8(self, flat, topk_weight, order, token_of, offsets):
        """The same arithmetic as :meth:`_routed_eager`, as four MXFP8 kernels."""
        if self._packed is None:
            self._refresh_mxfp8()
        # Compute dtype comes from autocast; the activation arrives fp32.
        x = flat.to(compute_dtype(flat))
        return mxfp8_moe_experts(
            x,
            self.w_in,
            self.w_out,
            topk_weight.reshape(-1),
            token_of,
            order,
            offsets[: self.num_experts + 1],
            self._packed,
        )

    def _routed_mxfp8_unfused(self, flat, topk_weight, order, token_of, offsets):
        """The same arithmetic again, as six launches over vendor-verified GEMMs."""
        if self._packed is None:
            self._refresh_mxfp8()
        x = flat.to(compute_dtype(flat))
        return mxfp8_moe_experts_unfused(
            x,
            self.w_in,
            self.w_out,
            topk_weight.reshape(-1),
            token_of,
            order,
            offsets[: self.num_experts + 1],
            self._packed,
            # `None` when there is no sentinel: no uncomputed region to zero.
            (
                offsets[self.num_experts : self.num_experts + 1]
                if self._has_sentinel
                else None
            ),
        )

    def forward(self, x: torch.Tensor, route_on: torch.Tensor | None = None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        gate_input = (
            flat if route_on is None else route_on.reshape(-1, route_on.shape[-1])
        )
        topk_idx, topk_weight, counts = self.router.route(gate_input)

        # Group the (token, slot) pairs by expert so each expert owns a
        # contiguous row range; `order` maps sorted position -> flat pair index.
        flat_expert = topk_idx.reshape(-1)
        # Slot width from the tensor, not self.top_k: a ReLU router pads it.
        offsets, order, token_of, _ = expert_sort(
            flat_expert, counts, topk_idx.shape[-1], self.num_buckets
        )
        out = self._routed(flat, topk_weight, order, token_of, offsets)
        if self.shared is not None:
            out = out + self.shared(flat)
        return out.reshape(shape)

    def router_losses(self) -> torch.Tensor | None:
        """Sum of the router's auxiliary terms for this forward, or ``None``."""
        terms = [t for t in (self.router.aux_loss, self.router.z_loss) if t is not None]
        return None if not terms else sum(terms)

    @property
    def emits_router_loss(self) -> bool:
        """Whether :meth:`router_losses` can return a term, known before a forward."""
        return bool(getattr(self.router, "emits_loss", False))
