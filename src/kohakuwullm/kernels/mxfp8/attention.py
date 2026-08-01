"""MXFP8 flash attention, forward and backward, for sm_120.

`QK^T` runs in block-scaled e4m3; the probabilities feed `PV` in bf16, because
that product's error survives the softmax backward's cancellation. Scales run
along the contraction axis in both cases.

See docs/internals/mxfp8-attention.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8.quantize import BLOCK_SCALE, _quantize_block


@triton.jit
def _fwd_kernel(
    q_ptr,
    qs_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    sm_scale,
    T,
    HEAD: tl.constexpr,
    sq_t,
    sq_d,
    sqs_t,
    sqs_g,
    sk_t,
    sk_d,
    sks_t,
    sks_g,
    sv_t,
    sv_d,
    so_t,
    so_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One BLOCK_M row-block of output; K and V are streamed in BLOCK_N chunks."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    groups: tl.constexpr = HEAD // BLOCK_SUB
    offs_g = tl.arange(0, groups)

    qm = offs_m < T
    q = tl.load(
        q_ptr + offs_m[:, None] * sq_t + offs_d[None, :] * sq_d,
        mask=qm[:, None],
        other=0.0,
    )
    qs = tl.load(
        qs_ptr + offs_m[:, None] * sqs_t + offs_g[None, :] * sqs_g,
        mask=qm[:, None],
        other=0,
    )

    acc = tl.zeros((BLOCK_M, HEAD), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    hi = (pid + 1) * BLOCK_M if CAUSAL else T
    for start in range(0, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        km = offs_n < T
        k = tl.load(
            k_ptr + offs_n[:, None] * sk_t + offs_d[None, :] * sk_d,
            mask=km[:, None],
            other=0.0,
        )
        ks = tl.load(
            ks_ptr + offs_n[:, None] * sks_t + offs_g[None, :] * sks_g,
            mask=km[:, None],
            other=0,
        )
        # Contraction is head_dim, so both scale sets are already on the right axis.
        s = tl.dot_scaled(
            q,
            qs,
            "e4m3",
            tl.trans(k),
            ks,
            "e4m3",
            acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
        )
        s = s * sm_scale
        s = tl.where(km[None, :], s, float("-inf"))
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptr + offs_n[:, None] * sv_t + offs_d[None, :] * sv_d,
            mask=km[:, None],
            other=0.0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        o_ptr + offs_m[:, None] * so_t + offs_d[None, :] * so_d,
        acc.to(o_ptr.dtype.element_ty),
        mask=qm[:, None],
    )
    tl.store(lse_ptr + offs_m, m_i + tl.log(l_i), mask=qm)


@triton.jit
def _col_mean_kernel(
    x_ptr, mu_ptr, T, HEAD: tl.constexpr, sx_t, sx_d, BLOCK_M: tl.constexpr
):
    """Column mean of a 16-bit ``(T, HEAD)`` tensor, accumulated in fp32.

    One program per channel, so the fp32 accumulator stays in registers and no
    widened copy of ``x`` reaches memory. See docs/internals/mxfp8-attention.md.
    """
    d = tl.program_id(0)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start in range(0, T, BLOCK_M):
        offs = start + tl.arange(0, BLOCK_M)
        acc += tl.load(x_ptr + offs * sx_t + d * sx_d, mask=offs < T, other=0.0).to(
            tl.float32
        )
    tl.store(mu_ptr + d, tl.sum(acc, axis=0) / T)


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    mu_ptr,
    T,
    HEAD: tl.constexpr,
    sx_t,
    sx_d,
    sq_t,
    sq_d,
    ss_t,
    ss_g,
    BLOCK_M: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    SMOOTH: tl.constexpr,
):
    """Quantize a ``(T, HEAD)`` tensor with blocks along HEAD.

    ``SMOOTH`` subtracts ``mu`` in fp32 registers, so the residual never reaches
    memory in any dtype.
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    mask = offs_m < T
    x = tl.load(
        x_ptr + offs_m[:, None] * sx_t + offs_d[None, :] * sx_d,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if SMOOTH:
        x = x - tl.load(mu_ptr + offs_d)[None, :]
    q, s = _quantize_block(x, BLOCK_M, HEAD, BLOCK_SUB)
    tl.store(
        q_ptr + offs_m[:, None] * sq_t + offs_d[None, :] * sq_d, q, mask=mask[:, None]
    )
    groups: tl.constexpr = HEAD // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    tl.store(
        s_ptr + offs_m[:, None] * ss_t + offs_g[None, :] * ss_g, s, mask=mask[:, None]
    )


def column_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-channel mean of a ``(T, HEAD)`` tensor as fp32, read at ``x``'s dtype."""
    x = x.contiguous()
    t, head = x.shape
    mu = torch.empty(head, device=x.device, dtype=torch.float32)
    _col_mean_kernel[(head,)](
        x, mu, t, head, x.stride(0), x.stride(1), BLOCK_M=1024, num_warps=4
    )
    return mu


def quantize_rows(x: torch.Tensor, mu: torch.Tensor | None = None):
    """``(T, HEAD)`` -> ``(e4m3, ue8m0)`` with blocks along HEAD.

    ``mu``, when given, is subtracted in fp32 registers before quantizing.
    """
    x = x.contiguous()
    t, head = x.shape
    q = torch.empty(t, head, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(t, head // BLOCK_SCALE, device=x.device, dtype=torch.uint8)
    _quant_rows_kernel[(triton.cdiv(t, 32),)](
        x,
        q,
        s,
        mu if mu is not None else x,
        t,
        head,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        s.stride(0),
        s.stride(1),
        BLOCK_M=32,
        BLOCK_SUB=BLOCK_SCALE,
        SMOOTH=mu is not None,
        num_warps=4,
    )
    return q, s


def mxfp8_attention(
    q,
    k,
    v,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int = 64,
    smooth_k: bool = True,
):
    """Single-head MXFP8 attention over ``(T, HEAD)`` inputs; returns ``(O, lse)``.

    Returns ``(O, lse, mu)``. ``smooth_k`` subtracts K's per-channel mean before
    quantizing; the score shift that causes is constant along a row, so softmax
    removes it exactly and no correction term is needed. ``mu`` is returned for
    the backward pass to reuse. See docs/internals/mxfp8-attention.md.
    """
    t, head = q.shape
    if head % BLOCK_SCALE:
        raise ValueError(f"head_dim={head} must be a multiple of {BLOCK_SCALE}")
    if block_n % BLOCK_SCALE:
        raise ValueError(f"block_n={block_n} must be a multiple of {BLOCK_SCALE}")
    sm_scale = sm_scale if sm_scale is not None else head**-0.5
    mu = column_mean(k) if smooth_k else None
    qq, qs = quantize_rows(q)
    kq, ks = quantize_rows(k, mu)
    o = torch.empty(t, head, device=q.device, dtype=v.dtype)
    lse = torch.empty(t, device=q.device, dtype=torch.float32)
    _fwd_kernel[(triton.cdiv(t, block_m),)](
        qq,
        qs,
        kq,
        ks,
        v,
        o,
        lse,
        sm_scale,
        t,
        head,
        qq.stride(0),
        qq.stride(1),
        qs.stride(0),
        qs.stride(1),
        kq.stride(0),
        kq.stride(1),
        ks.stride(0),
        ks.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_SUB=BLOCK_SCALE,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return o, lse, mu


@triton.jit
def _bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    HEAD: tl.constexpr,
    so_t,
    so_d,
    sdo_t,
    sdo_d,
    BLOCK_M: tl.constexpr,
):
    """``delta = rowsum(dO * O)``, the term softmax backward subtracts."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    mask = offs_m < T
    o = tl.load(
        o_ptr + offs_m[:, None] * so_t + offs_d[None, :] * so_d,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + offs_m[:, None] * sdo_t + offs_d[None, :] * sdo_d,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(delta_ptr + offs_m, tl.sum(o * do, axis=1), mask=mask)


@triton.jit
def _bwd_kernel(
    q_ptr,
    qs_ptr,
    qb_ptr,
    k_ptr,
    ks_ptr,
    kb_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    sm_scale,
    T,
    HEAD: tl.constexpr,
    sq_t,
    sq_d,
    sqs_t,
    sqs_g,
    sk_t,
    sk_d,
    sks_t,
    sks_g,
    sv_t,
    sv_d,
    sdo_t,
    sdo_d,
    sdq_t,
    sdq_d,
    sdk_t,
    sdk_d,
    sdv_t,
    sdv_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One BLOCK_N block of dK and dV; dQ is accumulated atomically."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD)
    groups: tl.constexpr = HEAD // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    nm = offs_n < T

    k = tl.load(
        k_ptr + offs_n[:, None] * sk_t + offs_d[None, :] * sk_d,
        mask=nm[:, None],
        other=0.0,
    )
    ks = tl.load(
        ks_ptr + offs_n[:, None] * sks_t + offs_g[None, :] * sks_g,
        mask=nm[:, None],
        other=0,
    )
    v = tl.load(
        v_ptr + offs_n[:, None] * sv_t + offs_d[None, :] * sv_d,
        mask=nm[:, None],
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, HEAD), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD), dtype=tl.float32)

    lo = pid * BLOCK_N if CAUSAL else 0
    for start in range(lo, T, BLOCK_M):
        offs_m = start + tl.arange(0, BLOCK_M)
        mm = offs_m < T
        q = tl.load(
            q_ptr + offs_m[:, None] * sq_t + offs_d[None, :] * sq_d,
            mask=mm[:, None],
            other=0.0,
        )
        qs = tl.load(
            qs_ptr + offs_m[:, None] * sqs_t + offs_g[None, :] * sqs_g,
            mask=mm[:, None],
            other=0,
        )
        # Recomputed exactly as the forward did, from the same quantized operands.
        s = tl.dot_scaled(
            q,
            qs,
            "e4m3",
            tl.trans(k),
            ks,
            "e4m3",
            acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
        )
        s = s * sm_scale
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        s = tl.where(nm[None, :] & mm[:, None], s, float("-inf"))
        lse = tl.load(lse_ptr + offs_m, mask=mm, other=0.0)
        p = tl.exp(s - lse[:, None])

        do = tl.load(
            do_ptr + offs_m[:, None] * sdo_t + offs_d[None, :] * sdo_d,
            mask=mm[:, None],
            other=0.0,
        )
        qb = tl.load(
            qb_ptr + offs_m[:, None] * sq_t + offs_d[None, :] * sq_d,
            mask=mm[:, None],
            other=0.0,
        )
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        # dP feeds a cancellation below, so it stays in the input dtype.
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        delta = tl.load(delta_ptr + offs_m, mask=mm, other=0.0)
        ds = (p * (dp - delta[:, None]) * sm_scale).to(qb.dtype)

        kb = tl.load(
            kb_ptr + offs_n[:, None] * sk_t + offs_d[None, :] * sk_d,
            mask=nm[:, None],
            other=0.0,
        )
        dk += tl.dot(tl.trans(ds), qb).to(tl.float32)
        dq = tl.dot(ds, kb).to(tl.float32)
        tl.atomic_add(
            dq_ptr + offs_m[:, None] * sdq_t + offs_d[None, :] * sdq_d,
            dq,
            mask=mm[:, None],
            sem="relaxed",
        )

    tl.store(
        dk_ptr + offs_n[:, None] * sdk_t + offs_d[None, :] * sdk_d,
        dk.to(dk_ptr.dtype.element_ty),
        mask=nm[:, None],
    )
    tl.store(
        dv_ptr + offs_n[:, None] * sdv_t + offs_d[None, :] * sdv_d,
        dv.to(dv_ptr.dtype.element_ty),
        mask=nm[:, None],
    )


def mxfp8_attention_backward(
    do, q, k, v, o, lse, mu, causal, sm_scale, block_m=64, block_n=64
):
    """Gradients of `mxfp8_attention`; returns ``(dQ, dK, dV)``."""
    t, head = q.shape
    delta = torch.empty(t, device=q.device, dtype=torch.float32)
    _bwd_preprocess[(triton.cdiv(t, block_m),)](
        o,
        do,
        delta,
        t,
        head,
        o.stride(0),
        o.stride(1),
        do.stride(0),
        do.stride(1),
        BLOCK_M=block_m,
        num_warps=4,
    )

    qq, qs = quantize_rows(q)
    kq, ks = quantize_rows(k, mu)
    kk = k if mu is None else (k.float() - mu[None, :]).to(k.dtype)
    dq = torch.zeros(t, head, device=q.device, dtype=torch.float32)
    dk = torch.empty(t, head, device=q.device, dtype=q.dtype)
    dv = torch.empty(t, head, device=q.device, dtype=q.dtype)
    _bwd_kernel[(triton.cdiv(t, block_n),)](
        qq,
        qs,
        q,
        kq,
        ks,
        kk,
        v,
        do,
        lse,
        delta,
        dq,
        dk,
        dv,
        sm_scale,
        t,
        head,
        q.stride(0),
        q.stride(1),
        qs.stride(0),
        qs.stride(1),
        kk.stride(0),
        kk.stride(1),
        ks.stride(0),
        ks.stride(1),
        v.stride(0),
        v.stride(1),
        do.stride(0),
        do.stride(1),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_SUB=BLOCK_SCALE,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return dq.to(q.dtype), dk, dv


class _MXFP8Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, smooth_k):
        o, lse, mu = mxfp8_attention(
            q, k, v, causal=causal, sm_scale=sm_scale, smooth_k=smooth_k
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.mu = mu
        ctx.causal = causal
        ctx.sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = mxfp8_attention_backward(
            do.contiguous(), q, k, v, o, lse, ctx.mu, ctx.causal, ctx.sm_scale
        )
        return dq, dk, dv, None, None, None


def mxfp8_attn(
    q, k, v, causal: bool = False, sm_scale: float | None = None, smooth_k: bool = True
) -> torch.Tensor:
    """Differentiable single-head MXFP8 attention over ``(T, HEAD)`` inputs."""
    return _MXFP8Attention.apply(q, k, v, causal, sm_scale, smooth_k)
