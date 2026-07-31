"""Hand-written varlen FlashAttention in Triton, tuned for sm_120.

Layout matches ``torch.nn.attention.varlen``: ``q`` is ``(T, H, D)``, ``k``/``v``
are ``(T, H_kv, D)``, and ``cu_seqlens`` is the ``(N+1,)`` exclusive prefix sum of
document lengths. Causal masking anchors to the bottom-right of each document.

The kernels live in :mod:`kohakuwullm.kernels.attention.flash_attn_fwd` and
:mod:`kohakuwullm.kernels.attention.flash_attn_bwd`; what is here is the autograd
node that joins them and the public entry point.

See docs/internals/kernels.md.
"""

import torch
import triton

from kohakuwullm.kernels.attention.flash_attn_bwd import (
    _bwd_dkdv_kernel,
    _bwd_dq_kernel,
    _bwd_preprocess,
)
from kohakuwullm.kernels.attention.flash_attn_fwd import _fwd_kernel


class _TritonVarlenAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, sm_scale, causal, window):
        total, heads, head_dim = q.shape
        kv_heads = k.shape[1]
        gqa_group = heads // kv_heads
        num_seqs = cu_seqlens.numel() - 1

        out = torch.empty_like(q)
        lse = torch.empty(heads, total, device=q.device, dtype=torch.float32)
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(max_seqlen, meta["BLOCK_M"]),
            heads,
            num_seqs,
        )
        _fwd_kernel[grid](
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens,
            sm_scale,
            window if window is not None else 0,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse.stride(0),
            lse.stride(1),
            H=heads,
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            IS_CAUSAL=causal,
            HAS_WINDOW=window is not None,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.window = window
        ctx.max_seqlen = max_seqlen
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
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
            k,
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
            k.stride(0),
            k.stride(1),
            k.stride(2),
            lse.stride(0),
            lse.stride(1),
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            IS_CAUSAL=ctx.causal,
            HAS_WINDOW=ctx.window is not None,
        )

        grid_q = lambda meta: (  # noqa: E731
            triton.cdiv(ctx.max_seqlen, meta["BLOCK_M"]),
            heads,
            num_seqs,
        )
        _bwd_dq_kernel[grid_q](
            q,
            k,
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
            k.stride(0),
            k.stride(1),
            k.stride(2),
            lse.stride(0),
            lse.stride(1),
            GQA_GROUP=gqa_group,
            HEAD_DIM=head_dim,
            IS_CAUSAL=ctx.causal,
            HAS_WINDOW=ctx.window is not None,
        )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def triton_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    sm_scale: float | None = None,
    causal: bool = True,
    window: int | None = None,
    return_lse: bool = False,
):
    """Varlen flash attention. ``q``: ``(T, H, D)``, ``k``/``v``: ``(T, H_kv, D)``.

    ``window`` is the *inclusive* left span (``w`` means a query attends to itself
    and the ``w-1`` tokens before it), matching ``VarlenAttention``'s convention.
    """
    sm_scale = sm_scale or q.shape[-1] ** -0.5
    out, lse = _TritonVarlenAttn.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens,
        max_seqlen,
        sm_scale,
        causal,
        window,
    )
    return (out, lse) if return_lse else out
