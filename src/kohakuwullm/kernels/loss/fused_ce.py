"""EXPERIMENTAL, FORWARD-ONLY. A documented dead end, imported by nothing.

Fused linear + cross-entropy holding each logit tile in registers, so a tile is
consumed where it is produced and never reaches memory. The backward requires a
power-of-two ``D`` and spills even then, so it raises at the widths this repo
trains; :mod:`kohakuwullm.kernels.loss.chunked_ce` is the production kernel.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100


@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    tgt_ptr,
    lse_ptr,
    tgtlogit_ptr,
    T,
    D: tl.constexpr,
    V,
    stride_xt,
    stride_xd,
    stride_wv,
    stride_wd,
    ignore_index,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    target = tl.load(tgt_ptr + offs_t, mask=mask_t, other=ignore_index)

    running_max = tl.full((BLOCK_T,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_T,), tl.float32)
    target_logit = tl.zeros((BLOCK_T,), tl.float32)

    for v0 in range(0, V, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        acc = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            x = tl.load(
                x_ptr + offs_t[:, None] * stride_xt + offs_d[None, :] * stride_xd,
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd,
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w))

        acc = tl.where(mask_v[None, :], acc, float("-inf"))
        tile_max = tl.max(acc, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        # exp(-inf - -inf) is nan; the first tile has no running mass to rescale.
        scale = tl.where(
            running_max == float("-inf"), 0.0, tl.exp(running_max - new_max)
        )
        running_sum = running_sum * scale + tl.sum(
            tl.where(mask_v[None, :], tl.exp(acc - new_max[:, None]), 0.0), axis=1
        )
        running_max = new_max

        hit = offs_v[None, :] == target[:, None]
        target_logit += tl.sum(tl.where(hit, acc, 0.0), axis=1)

    lse = running_max + tl.log(running_sum)
    tl.store(lse_ptr + offs_t, lse, mask=mask_t)
    tl.store(tgtlogit_ptr + offs_t, target_logit, mask=mask_t)


@triton.jit
def _dx_kernel(
    x_ptr,
    w_ptr,
    tgt_ptr,
    lse_ptr,
    dloss_ptr,
    dx_ptr,
    T,
    D: tl.constexpr,
    V,
    stride_xt,
    stride_xd,
    stride_wv,
    stride_wd,
    ignore_index,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    offs_d = tl.arange(0, D)

    target = tl.load(tgt_ptr + offs_t, mask=mask_t, other=ignore_index)
    lse = tl.load(lse_ptr + offs_t, mask=mask_t, other=0.0)
    g = tl.load(dloss_ptr + offs_t, mask=mask_t, other=0.0)
    g = tl.where(target == ignore_index, 0.0, g)

    dx = tl.zeros((BLOCK_T, D), tl.float32)
    for v0 in range(0, V, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        acc = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_dd = d0 + tl.arange(0, BLOCK_D)
            mask_dd = offs_dd < D
            x = tl.load(
                x_ptr + offs_t[:, None] * stride_xt + offs_dd[None, :] * stride_xd,
                mask=mask_t[:, None] & mask_dd[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_v[:, None] * stride_wv + offs_dd[None, :] * stride_wd,
                mask=mask_v[:, None] & mask_dd[None, :],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w))

        prob = tl.exp(acc - lse[:, None])
        hit = offs_v[None, :] == target[:, None]
        dlogit = tl.where(mask_v[None, :], (prob - tl.where(hit, 1.0, 0.0)), 0.0)
        dlogit = dlogit * g[:, None]

        w_full = tl.load(
            w_ptr + offs_v[:, None] * stride_wv + offs_d[None, :] * stride_wd,
            mask=mask_v[:, None],
            other=0.0,
        )
        dx += tl.dot(dlogit.to(w_full.dtype), w_full)

    tl.store(
        dx_ptr + offs_t[:, None] * D + offs_d[None, :],
        dx,
        mask=mask_t[:, None],
    )


@triton.jit
def _dw_kernel(
    x_ptr,
    w_ptr,
    tgt_ptr,
    lse_ptr,
    dloss_ptr,
    dw_ptr,
    T,
    D: tl.constexpr,
    V,
    stride_xt,
    stride_xd,
    stride_wv,
    stride_wd,
    ignore_index,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_v = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    offs_d = tl.arange(0, D)

    dw = tl.zeros((BLOCK_V, D), tl.float32)
    for t0 in range(0, T, BLOCK_T):
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask_t = offs_t < T
        target = tl.load(tgt_ptr + offs_t, mask=mask_t, other=ignore_index)
        lse = tl.load(lse_ptr + offs_t, mask=mask_t, other=0.0)
        g = tl.load(dloss_ptr + offs_t, mask=mask_t, other=0.0)
        g = tl.where(target == ignore_index, 0.0, g)

        acc = tl.zeros((BLOCK_T, BLOCK_V), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs_dd = d0 + tl.arange(0, BLOCK_D)
            mask_dd = offs_dd < D
            x = tl.load(
                x_ptr + offs_t[:, None] * stride_xt + offs_dd[None, :] * stride_xd,
                mask=mask_t[:, None] & mask_dd[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_v[:, None] * stride_wv + offs_dd[None, :] * stride_wd,
                mask=mask_v[:, None] & mask_dd[None, :],
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(w))

        prob = tl.exp(acc - lse[:, None])
        hit = offs_v[None, :] == target[:, None]
        dlogit = tl.where(mask_t[:, None], (prob - tl.where(hit, 1.0, 0.0)), 0.0)
        dlogit = dlogit * g[:, None]

        x_full = tl.load(
            x_ptr + offs_t[:, None] * stride_xt + offs_d[None, :] * stride_xd,
            mask=mask_t[:, None],
            other=0.0,
        )
        dw += tl.dot(tl.trans(dlogit).to(x_full.dtype), x_full)

    tl.store(
        dw_ptr + offs_v[:, None] * D + offs_d[None, :],
        dw,
        mask=mask_v[:, None],
    )


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target, ignore_index, block_t, block_v, block_d):
        T, D = x.shape
        V = weight.shape[0]
        x = x.contiguous()
        weight = weight.contiguous()
        lse = torch.empty(T, device=x.device, dtype=torch.float32)
        target_logit = torch.empty(T, device=x.device, dtype=torch.float32)

        grid = (triton.cdiv(T, block_t),)
        _fwd_kernel[grid](
            x,
            weight,
            target,
            lse,
            target_logit,
            T,
            D,
            V,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            ignore_index,
            BLOCK_T=block_t,
            BLOCK_V=block_v,
            BLOCK_D=block_d,
            num_warps=8,
        )

        loss = lse - target_logit
        loss = torch.where(target == ignore_index, torch.zeros_like(loss), loss)
        ctx.save_for_backward(x, weight, target, lse)
        ctx.ignore_index = ignore_index
        ctx.blocks = (block_t, block_v, block_d)
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, weight, target, lse = ctx.saved_tensors
        block_t, block_v, block_d = ctx.blocks
        T, D = x.shape
        V = weight.shape[0]
        # Both gradient kernels index the width with `tl.arange(0, D)`, which
        # Triton lowers only for a power-of-two bound.
        if D & (D - 1):
            raise RuntimeError(
                f"fused_ce backward requires a power-of-two width, got D={D}. "
                "This kernel is forward-only at the widths this repo trains; "
                "use kohakuwullm.kernels.loss.chunked_ce for a trainable path."
            )
        dloss = dloss.contiguous().float()

        dx = torch.empty(T, D, device=x.device, dtype=torch.float32)
        dw = torch.empty(V, D, device=x.device, dtype=torch.float32)

        _dx_kernel[(triton.cdiv(T, block_t),)](
            x,
            weight,
            target,
            lse,
            dloss,
            dx,
            T,
            D,
            V,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            ctx.ignore_index,
            BLOCK_T=block_t,
            BLOCK_V=block_v,
            BLOCK_D=block_d,
            num_warps=8,
        )
        _dw_kernel[(triton.cdiv(V, block_v),)](
            x,
            weight,
            target,
            lse,
            dloss,
            dw,
            T,
            D,
            V,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            ctx.ignore_index,
            BLOCK_T=block_t,
            BLOCK_V=block_v,
            BLOCK_D=block_d,
            num_warps=8,
        )
        return dx.to(x.dtype), dw.to(weight.dtype), None, None, None, None, None


def fused_linear_cross_entropy(
    x,
    weight,
    target,
    ignore_index: int = IGNORE_INDEX,
    block_t: int = 64,
    block_v: int = 128,
    block_d: int = 64,
):
    """Per-token cross-entropy of ``x @ weight.T`` against ``target``.

    Forward-only: the backward raises for a non-power-of-two ``D``. Use
    :func:`kohakuwullm.kernels.chunked_linear_cross_entropy` for a gradient.
    Returns an unreduced ``(T,)`` fp32 tensor; reduction is the caller's.
    """
    return _FusedLinearCE.apply(
        x, weight, target, ignore_index, block_t, block_v, block_d
    )
