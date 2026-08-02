"""Varlen MXFP8 flash attention: the autograd node and the public entry point.

Layout matches :func:`kohakuwullm.kernels.attention.flash_attn.triton_varlen_attn`:
``q`` is ``(T, H, D)``, ``k``/``v`` are ``(T, H_kv, D)``, and ``cu_seqlens`` is the
``(N+1,)`` exclusive prefix sum of document lengths. ``QK^T`` runs in block-scaled
e4m3; ``PV`` and both gradient GEMMs stay in the input dtype.

``smooth_k`` subtracts K's per-channel mean before quantizing. Softmax removes the
resulting constant score shift exactly, and both gradients are unchanged by it.

The kernels live in :mod:`kohakuwullm.kernels.mxfp8.attention_fwd` and
:mod:`kohakuwullm.kernels.mxfp8.attention_bwd`.

See docs/internals/mxfp8-attention.md.
"""

import torch
import triton

from kohakuwullm.kernels.attention.flash_attn_bwd import _bwd_preprocess
from kohakuwullm.kernels.mxfp8.attention_bwd import _bwd_dkdv_kernel, _bwd_dq_kernel
from kohakuwullm.kernels.mxfp8.attention_fwd import _fwd_kernel
from kohakuwullm.kernels.mxfp8.attention_quant import (
    BLOCK_SCALE,
    column_mean,
    quantize_heads,
)


class _MXFP8VarlenAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, sm_scale, causal, window, smooth):
        total, heads, head_dim = q.shape
        kv_heads = k.shape[1]
        gqa_group = heads // kv_heads
        num_seqs = cu_seqlens.numel() - 1

        mu = column_mean(k) if smooth else None
        qq, qs = quantize_heads(q)
        kq, ks = quantize_heads(k, mu)

        out = torch.empty_like(q)
        lse = torch.empty(heads, total, device=q.device, dtype=torch.float32)
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(max_seqlen, meta["BLOCK_M"]),
            heads,
            num_seqs,
        )
        _fwd_kernel[grid](
            qq,
            qs,
            kq,
            ks,
            v,
            out,
            lse,
            cu_seqlens,
            sm_scale,
            window if window is not None else 0,
            qq.stride(0),
            qq.stride(1),
            qq.stride(2),
            qs.stride(0),
            qs.stride(1),
            qs.stride(2),
            kq.stride(0),
            kq.stride(1),
            kq.stride(2),
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            BLOCK_SUB=BLOCK_SCALE,
            IS_CAUSAL=causal,
            HAS_WINDOW=window is not None,
        )
        ctx.save_for_backward(q, k, v, qq, qs, kq, ks, out, lse, cu_seqlens, mu)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.window = window
        ctx.max_seqlen = max_seqlen
        ctx.smooth = smooth
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, qq, qs, kq, ks, out, lse, cu_seqlens, mu = ctx.saved_tensors
        dout = dout.contiguous()
        total, heads, head_dim = q.shape
        kv_heads = k.shape[1]
        gqa_group = heads // kv_heads
        num_seqs = cu_seqlens.numel() - 1

        delta = torch.empty_like(lse)
        _bwd_preprocess[(triton.cdiv(total, 128), heads)](
            out,
            dout,
            delta,
            total,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            delta.stride(0),
            delta.stride(1),
            HEAD_DIM=head_dim,
            BLOCK_M=128,
        )

        dq = torch.empty_like(q)
        # fp32 and zeroed: the GQA group atomically accumulates into these.
        dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
        dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
        window = ctx.window if ctx.window is not None else 0

        grid_kv = lambda meta: (  # noqa: E731
            triton.cdiv(ctx.max_seqlen, meta["BLOCK_N"]),
            heads,
            num_seqs,
        )
        _bwd_dkdv_kernel[grid_kv](
            q,
            qq,
            qs,
            kq,
            ks,
            v,
            dout,
            lse,
            delta,
            dk,
            dv,
            cu_seqlens,
            ctx.sm_scale,
            window,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            qq.stride(0),
            qq.stride(1),
            qq.stride(2),
            qs.stride(0),
            qs.stride(1),
            qs.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kq.stride(0),
            kq.stride(1),
            kq.stride(2),
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            lse.stride(0),
            lse.stride(1),
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            BLOCK_SUB=BLOCK_SCALE,
            IS_CAUSAL=ctx.causal,
            HAS_WINDOW=ctx.window is not None,
        )

        grid_q = lambda meta: (  # noqa: E731
            triton.cdiv(ctx.max_seqlen, meta["BLOCK_M"]),
            heads,
            num_seqs,
        )
        _bwd_dq_kernel[grid_q](
            qq,
            qs,
            k,
            kq,
            ks,
            mu if mu is not None else lse,
            v,
            dout,
            lse,
            delta,
            dq,
            cu_seqlens,
            ctx.sm_scale,
            window,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            qq.stride(0),
            qq.stride(1),
            qq.stride(2),
            qs.stride(0),
            qs.stride(1),
            qs.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kq.stride(0),
            kq.stride(1),
            kq.stride(2),
            ks.stride(0),
            ks.stride(1),
            ks.stride(2),
            lse.stride(0),
            lse.stride(1),
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            BLOCK_SUB=BLOCK_SCALE,
            IS_CAUSAL=ctx.causal,
            HAS_WINDOW=ctx.window is not None,
            SMOOTH=ctx.smooth,
        )
        return (
            dq,
            dk.to(k.dtype),
            dv.to(v.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mxfp8_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    sm_scale: float | None = None,
    causal: bool = True,
    window: int | None = None,
    smooth_k: bool = True,
    return_lse: bool = False,
):
    """Varlen MXFP8 attention. ``q``: ``(T, H, D)``, ``k``/``v``: ``(T, H_kv, D)``.

    ``window`` is the *inclusive* left span (``w`` means a query attends to itself
    and the ``w-1`` tokens before it), matching ``VarlenAttention``'s convention.
    ``head_dim`` must be the 32-element scale block times a power of two.
    """
    groups = q.shape[-1] // BLOCK_SCALE
    if q.shape[-1] % BLOCK_SCALE or groups & (groups - 1):
        raise ValueError(
            f"head_dim={q.shape[-1]} must be {BLOCK_SCALE} times a power of two; "
            "the scale-group count reaches tl.arange"
        )
    sm_scale = sm_scale or q.shape[-1] ** -0.5
    out, lse = _MXFP8VarlenAttn.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens,
        max_seqlen,
        sm_scale,
        causal,
        window,
        smooth_k,
    )
    return (out, lse) if return_lse else out
