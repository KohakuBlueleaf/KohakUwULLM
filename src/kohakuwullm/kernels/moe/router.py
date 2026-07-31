"""Fused MoE router: gate GEMM, activation, bias, top-k, load and the aux losses.

One kernel, no ``bincount`` and no autotune. ``E <= 128`` so a token's score row
stays in registers. The balancing bias steers **selection only** -- the returned
weight comes from the unbiased score.

See docs/internals/moe-router-loss.md and docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

# Gate activations the kernel implements, both elementwise. `tl.constexpr`, not a
# bare int: Triton reads a global from a @jit body only as a constexpr instance.
_SIGMOID = tl.constexpr(0)
_SQRT_SOFTPLUS = tl.constexpr(1)
_SCORE_CODES = {"sigmoid": _SIGMOID.value, "sqrtsoftplus": _SQRT_SOFTPLUS.value}

_BLOCK_T = 64
_BLOCK_D = 64


@triton.jit
def _gate(logits, SCORE_FUNC: tl.constexpr):
    """Score from logit. Must agree with ``moe._SCORE_FUNCS`` elementwise."""
    if SCORE_FUNC == _SIGMOID:
        return tl.sigmoid(logits)
    # Overflow-safe softplus: max(z,0) + log1p(exp(-|z|)).
    softplus = tl.maximum(logits, 0.0) + libdevice.log1p(tl.exp(-tl.abs(logits)))
    return tl.sqrt(softplus)


@triton.jit
def _gate_grad(score, SCORE_FUNC: tl.constexpr):
    """``dscore/dlogit`` expressed through the score alone.

    Both activations admit this, which keeps the backward's saved state at
    ``(T, top_k)``.
    """
    if SCORE_FUNC == _SIGMOID:
        return score * (1.0 - score)
    # sigmoid(z) = -expm1(-s^2) for s = sqrt(softplus(z)); expm1, not 1 - exp.
    sigmoid = -libdevice.expm1(-score * score)
    return sigmoid / tl.maximum(2.0 * score, 1e-30)


@triton.jit
def _logsumexp(logits, mask_e):
    """Row log-sum-exp and softmax of a ``(BLOCK_T, BLOCK_E)`` logit tile.

    Lanes past ``E`` are pushed to ``-inf`` so they contribute nothing.
    """
    capped = tl.where(mask_e[None, :], logits, float("-inf"))
    peak = tl.max(capped, axis=1, keep_dims=True)
    weight = tl.exp(capped - peak)
    total = tl.sum(weight, axis=1, keep_dims=True)
    return peak + tl.log(total), weight / total


def _launch_config(block_e: int) -> tuple[int, int]:
    """``(num_warps, num_stages)`` for a score row of ``block_e`` lanes.

    A 128-lane row spills at four warps; narrower rows prefer the deeper pipeline.
    """
    return (8, 3) if block_e >= 128 else (4, 4)


@triton.jit
def _router_fwd(
    x_ptr,
    w_ptr,
    bias_ptr,
    idx_ptr,
    weight_ptr,
    raw_ptr,
    load_ptr,
    counts_ptr,
    logits_ptr,
    stats_ptr,
    T,
    D,
    stride_xt,
    stride_xd,
    stride_we,
    stride_wd,
    scaling,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NORM_PROB: tl.constexpr,
    COUNT_LOAD: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SCORE_FUNC: tl.constexpr,
    NEED_AUX: tl.constexpr,
    NEED_Z: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_e = tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, BLOCK_K)
    mask_t = offs_t < T
    mask_e = offs_e < E

    acc = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
    for d0 in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        xb = tl.load(
            x_ptr + offs_t[:, None] * stride_xt + offs_d[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        )
        wb = tl.load(
            w_ptr + offs_e[None, :] * stride_we + offs_d[:, None] * stride_wd,
            mask=mask_e[None, :] & mask_d[:, None],
            other=0.0,
        )
        acc += tl.dot(xb.to(COMPUTE_DTYPE), wb.to(COMPUTE_DTYPE), out_dtype=tl.float32)

    scores = _gate(acc, SCORE_FUNC)

    if NEED_AUX or NEED_Z:
        # Every lane's logit, which the backward re-reads.
        tl.store(
            logits_ptr + offs_t[:, None] * E + offs_e[None, :],
            acc,
            mask=mask_t[:, None] & mask_e[None, :],
        )
    if NEED_AUX:
        column = tl.sum(tl.where(mask_t[:, None], scores, 0.0), axis=0)
        tl.atomic_add(stats_ptr + offs_e, column, mask=mask_e)
    if NEED_Z:
        # `_prob`, not `_`: the top-k loop below binds `_`, and Triton would then
        # see one name carrying two tile shapes.
        lse, _prob = _logsumexp(acc, mask_e)
        tl.atomic_add(
            stats_ptr + E + tl.arange(0, 1),
            tl.sum(tl.where(mask_t[:, None], lse * lse, 0.0), axis=0),
        )

    select = scores
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0)
        select = select + bias[None, :]
    # Lanes past E must never win an argmax; selected lanes are masked out below.
    select = tl.where(mask_e[None, :], select, float("-inf"))

    idx_acc = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.int32)
    raw_acc = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    chosen = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
    total = tl.zeros((BLOCK_T, 1), dtype=tl.float32)

    for k in range(TOP_K):
        _, best = tl.max(select, axis=1, return_indices=True, keep_dims=True)
        # The unbiased score at the winning lane, as a shuffle.
        picked = tl.gather(scores, best, 1)

        slot = offs_k[None, :] == k
        idx_acc += tl.where(slot, best.to(tl.int32), 0)
        raw_acc += tl.where(slot, picked, 0.0)
        total += picked

        hot = offs_e[None, :] == best
        chosen += tl.where(hot, 1.0, 0.0)
        select = tl.where(hot, float("-inf"), select)

    out = raw_acc
    if NORM_PROB:
        out = out / tl.maximum(total, 1e-9)
    out = out * scaling

    mask_k = offs_k < TOP_K
    store_mask = mask_t[:, None] & mask_k[None, :]
    base = offs_t[:, None] * TOP_K + offs_k[None, :]
    tl.store(idx_ptr + base, idx_acc, mask=store_mask)
    tl.store(weight_ptr + base, out, mask=store_mask)
    tl.store(raw_ptr + base, raw_acc, mask=store_mask)

    hist = tl.sum(tl.where(mask_t[:, None], chosen, 0.0), axis=0)
    tl.atomic_add(counts_ptr + offs_e, hist.to(tl.int32), mask=mask_e)
    if COUNT_LOAD:
        tl.atomic_add(load_ptr + offs_e, hist, mask=mask_e)


@triton.jit
def _router_bwd_logits(
    grad_w_ptr,
    idx_ptr,
    raw_ptr,
    out_ptr,
    logits_ptr,
    counts_ptr,
    gaux_ptr,
    gz_ptr,
    T,
    scaling,
    aux_scale,
    z_scale,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NORM_PROB: tl.constexpr,
    SCORE_FUNC: tl.constexpr,
    NEED_AUX: tl.constexpr,
    NEED_Z: tl.constexpr,
    DENSE: tl.constexpr,
):
    """Gate-logit gradients as a dense ``(T, E)`` tile.

    ``DENSE`` assembles the whole row in registers, including the auxiliary
    terms; otherwise only the ``top_k`` lanes are scattered into a zeroed
    ``out_ptr``. See docs/internals/moe-router-loss.md.
    """
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    mask_t = offs_t < T
    mask = mask_t[:, None] & (offs_k < TOP_K)[None, :]

    base = offs_t[:, None] * TOP_K + offs_k[None, :]
    grad = tl.load(grad_w_ptr + base, mask=mask, other=0.0).to(tl.float32) * scaling
    raw = tl.load(raw_ptr + base, mask=mask, other=0.0)
    idx = tl.load(idx_ptr + base, mask=mask, other=0)

    if NORM_PROB:
        total = tl.maximum(tl.sum(tl.where(mask, raw, 0.0), axis=1), 1e-9)[:, None]
        norm = raw / total
        grad = (
            grad - tl.sum(tl.where(mask, grad * norm, 0.0), axis=1)[:, None]
        ) / total

    grad = grad * _gate_grad(raw, SCORE_FUNC)

    if DENSE:
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < E
        row_mask = mask_t[:, None] & mask_e[None, :]

        dense = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
        # Scatter one slot per iteration, as a broadcast compare.
        for k in range(TOP_K):
            slot = mask & (offs_k[None, :] == k)
            value = tl.sum(tl.where(slot, grad, 0.0), axis=1)[:, None]
            lane = tl.sum(tl.where(slot, idx, 0), axis=1)[:, None]
            dense += tl.where(offs_e[None, :] == lane, value, 0.0)

        logits = tl.load(
            logits_ptr + offs_t[:, None] * E + offs_e[None, :],
            mask=row_mask,
            other=0.0,
        )
        if NEED_AUX:
            counts = tl.load(counts_ptr + offs_e, mask=mask_e, other=0).to(tl.float32)
            scores = _gate(logits, SCORE_FUNC)
            dense += (
                (aux_scale * tl.load(gaux_ptr))
                * counts[None, :]
                * _gate_grad(scores, SCORE_FUNC)
            )
        if NEED_Z:
            lse, prob = _logsumexp(logits, mask_e)
            dense += (z_scale * tl.load(gz_ptr)) * lse * prob

        tl.store(out_ptr + offs_t[:, None] * E + offs_e[None, :], dense, mask=row_mask)
    else:
        tl.store(out_ptr + offs_t[:, None] * E + idx, grad, mask=mask)


class _FusedRouter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        top_k,
        norm_topk_prob,
        scaling,
        load_accum,
        compute_dtype,
        score_func,
        aux_weight,
        z_weight,
    ):
        t, d = x.shape
        e = weight.shape[0]
        block_e = max(16, triton.next_power_of_2(e))
        block_k = max(1, triton.next_power_of_2(top_k))
        warps, stages = _launch_config(block_e)

        idx = torch.empty(t, top_k, dtype=torch.int32, device=x.device)
        out = torch.empty(t, top_k, dtype=torch.float32, device=x.device)
        raw = torch.empty(t, top_k, dtype=torch.float32, device=x.device)
        counts = torch.zeros(e, dtype=torch.int32, device=x.device)
        count = load_accum is not None
        need_aux = aux_weight > 0.0
        need_z = z_weight > 0.0
        need_stats = need_aux or need_z

        # Lanes 0..E-1 sum the scores, lane E sums the squared log-sum-exp.
        stats = (
            torch.zeros(e + 1, dtype=torch.float32, device=x.device)
            if need_stats
            else None
        )
        logits = (
            torch.empty(t, e, dtype=torch.float32, device=x.device)
            if need_stats
            else None
        )

        _router_fwd[(triton.cdiv(t, _BLOCK_T),)](
            x,
            weight,
            bias if bias is not None else out,
            idx,
            out,
            raw,
            load_accum if count else out,
            counts,
            logits if need_stats else out,
            stats if need_stats else out,
            t,
            d,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            scaling,
            E=e,
            BLOCK_E=block_e,
            TOP_K=top_k,
            BLOCK_K=block_k,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_BIAS=bias is not None,
            NORM_PROB=norm_topk_prob and top_k > 1,
            COUNT_LOAD=count,
            COMPUTE_DTYPE=_TL_DTYPE[compute_dtype],
            SCORE_FUNC=score_func,
            NEED_AUX=need_aux,
            NEED_Z=need_z,
            num_warps=warps,
            num_stages=stages,
        )

        aux_scale = aux_weight * e / (t * top_k * t)
        aux = torch.dot(counts.float(), stats[:e]) * aux_scale if need_aux else None
        z = stats[e] * (z_weight / t) if need_z else None

        ctx.save_for_backward(x, weight, idx, raw, logits, counts if need_aux else None)
        ctx.top_k = top_k
        ctx.norm_topk_prob = norm_topk_prob and top_k > 1
        ctx.scaling = scaling
        ctx.block_k = block_k
        ctx.block_e = block_e
        ctx.score_func = score_func
        ctx.aux_scale = aux_scale
        ctx.z_scale = 2.0 * z_weight / t
        return idx, out, counts, aux, z

    @staticmethod
    def backward(ctx, grad_idx, grad_out, grad_counts, grad_aux, grad_z):
        x, weight, idx, raw, logits, counts = ctx.saved_tensors
        t = x.shape[0]
        e = weight.shape[0]
        need_aux = grad_aux is not None
        need_z = grad_z is not None
        dense = need_aux or need_z

        grad_logits = (
            torch.empty(t, e, dtype=torch.float32, device=x.device)
            if dense
            else torch.zeros(t, e, dtype=torch.float32, device=x.device)
        )
        _router_bwd_logits[(triton.cdiv(t, _BLOCK_T),)](
            grad_out.contiguous(),
            idx,
            raw,
            grad_logits,
            logits if dense else grad_logits,
            counts if need_aux else grad_logits,
            grad_aux if need_aux else grad_logits,
            grad_z if need_z else grad_logits,
            t,
            ctx.scaling,
            ctx.aux_scale,
            ctx.z_scale,
            E=e,
            TOP_K=ctx.top_k,
            BLOCK_K=ctx.block_k,
            BLOCK_T=_BLOCK_T,
            BLOCK_E=ctx.block_e,
            NORM_PROB=ctx.norm_topk_prob,
            SCORE_FUNC=ctx.score_func,
            NEED_AUX=need_aux,
            NEED_Z=need_z,
            DENSE=dense,
        )

        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_logits @ weight.float()).to(x.dtype)
        if ctx.needs_input_grad[1]:
            grad_w = (grad_logits.T @ x.float()).to(weight.dtype)
        return grad_x, grad_w, None, None, None, None, None, None, None, None, None


def fused_router(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    top_k: int,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    load_accum: torch.Tensor | None = None,
    compute_dtype: torch.dtype = torch.bfloat16,
    score_func: str = "sigmoid",
    aux_loss_weight: float = 0.0,
    z_loss_weight: float = 0.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Elementwise-gated top-k routing in one launch.

    Args:
        x: ``(T, D)`` tokens.
        weight: ``(E, D)`` gate projection.
        bias: ``(E,)`` selection-only balancing bias, or ``None``.
        top_k: experts per token.
        norm_topk_prob: renormalize the selected weights to sum to 1.
        routed_scaling_factor: scale applied to the returned weights.
        load_accum: ``(E,)`` fp32 accumulator; counts are added in-kernel when
            given, and skipped entirely when ``None`` (eval).
        compute_dtype: dtype the gate multiply runs in; the accumulator is fp32
            regardless. Passed in rather than inferred, so the kernel never reads
            autocast or device state.
        score_func: ``"sigmoid"`` (DeepSeek-V3) or ``"sqrtsoftplus"``
            (DeepSeek-V4). Both are elementwise; a softmax gate stays eager.
        aux_loss_weight: Switch-Transformer load-balance loss coefficient; 0
            skips the term and everything that feeds it.
        z_loss_weight: router logit z-loss coefficient; 0 skips it likewise.

    Returns:
        ``(topk_idx, topk_weight, counts, aux_loss, z_loss)``. ``idx`` int32 and
        ``weight`` fp32 are both ``(T, top_k)``; ``counts`` is the int32 ``(E,)``
        per-expert row count for *this* call, which the dispatch consumes in place
        of ``bincount``. The two losses are fp32 scalars, or ``None`` when their
        weight is zero; each is differentiable with respect to ``x`` and
        ``weight``. See docs/internals/moe-router-loss.md.
    """
    if x.ndim != 2:
        raise ValueError(f"expected (T, D) tokens, got {tuple(x.shape)}")
    if weight.shape[1] != x.shape[1]:
        raise ValueError(f"weight dim {weight.shape[1]} != token dim {x.shape[1]}")
    if top_k > weight.shape[0]:
        raise ValueError(f"top_k {top_k} exceeds expert count {weight.shape[0]}")
    if weight.shape[0] > 128:
        raise ValueError(
            f"fused router holds a score row in registers; E={weight.shape[0]} > 128"
        )
    if score_func not in _SCORE_CODES:
        raise ValueError(
            f"fused router has no {score_func!r} gate; have {sorted(_SCORE_CODES)}"
        )
    return _FusedRouter.apply(
        x,
        weight,
        bias,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
        load_accum,
        compute_dtype,
        _SCORE_CODES[score_func],
        aux_loss_weight,
        z_loss_weight,
    )
