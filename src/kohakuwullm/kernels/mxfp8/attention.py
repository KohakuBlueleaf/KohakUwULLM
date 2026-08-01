"""MXFP8 flash attention, forward and backward, for sm_120.

`QK^T` runs in block-scaled e4m3; the probabilities feed `PV` in bf16, because
that product's error survives the softmax backward's cancellation. Scales run
along the contraction axis in both cases.

See docs/internals/mxfp8-attention.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.gemm.device import RTX_5090


def _cu2(t, device):
    """A one-document cu_seqlens."""
    return torch.tensor([0, t], device=device, dtype=torch.int32)


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
    cu_ptr,
    sm_scale,
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
    sq_h,
    sqs_h,
    sk_h,
    sks_h,
    sv_h,
    so_h,
    slse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """One BLOCK_M row-block of one document and one head.

    Documents come from ``cu_ptr``, so a block never attends across a boundary.
    See docs/internals/mxfp8-attention.md.
    """
    start_m = tl.program_id(0)
    doc = tl.program_id(1)
    head_id = tl.program_id(2)

    seq_start = tl.load(cu_ptr + doc)
    seq_len = tl.load(cu_ptr + doc + 1) - seq_start
    if start_m * BLOCK_M >= seq_len:
        return

    q_ptr += head_id * sq_h + seq_start * sq_t
    qs_ptr += head_id * sqs_h + seq_start * sqs_t
    k_ptr += head_id * sk_h + seq_start * sk_t
    ks_ptr += head_id * sks_h + seq_start * sks_t
    v_ptr += head_id * sv_h + seq_start * sv_t
    o_ptr += head_id * so_h + seq_start * so_t
    lse_ptr += head_id * slse_h + seq_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    groups: tl.constexpr = HEAD // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    qm = offs_m < seq_len

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

    hi = tl.minimum(seq_len, (start_m + 1) * BLOCK_M) if CAUSAL else seq_len
    lo = 0
    if WINDOW > 0:
        lo = tl.maximum(0, start_m * BLOCK_M - WINDOW + 1)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        km = offs_n < seq_len
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
        keep = km[None, :] & qm[:, None]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        if WINDOW > 0:
            keep = keep & (offs_m[:, None] - offs_n[None, :] < WINDOW)
        s = tl.where(keep, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A fully masked row keeps m_i at -inf; clamp so exp2 sees a finite shift.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(
            (tl.where(m_i == float("-inf"), m_safe, m_i) - m_safe) * 1.4426950408889634
        )
        p = tl.exp2((s - m_safe[:, None]) * 1.4426950408889634)
        p = tl.where(keep, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptr + offs_n[:, None] * sv_t + offs_d[None, :] * sv_d,
            mask=km[:, None],
            other=0.0,
        )
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_safe

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(
        o_ptr + offs_m[:, None] * so_t + offs_d[None, :] * so_d,
        acc.to(o_ptr.dtype.element_ty),
        mask=qm[:, None],
    )
    tl.store(lse_ptr + offs_m, m_i + tl.log(tl.where(l_i == 0.0, 1.0, l_i)), mask=qm)


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
    ROWS_PER_MU,
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
        head = offs_m // ROWS_PER_MU
        x = x - tl.load(mu_ptr + head[:, None] * HEAD + offs_d[None, :])
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


def quantize_rows(
    x: torch.Tensor, mu: torch.Tensor | None = None, rows_per_mu: int | None = None
):
    """``(T, HEAD)`` -> ``(e4m3, ue8m0)`` with blocks along HEAD.

    ``mu`` is subtracted in fp32 registers before quantizing; with
    ``rows_per_mu`` it is a per-group mean, one row of ``mu`` per that many rows.
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
        rows_per_mu if rows_per_mu else t,
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
    block_n: int | None = None,
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
    if block_n is not None and block_n % BLOCK_SCALE:
        raise ValueError(f"block_n={block_n} must be a multiple of {BLOCK_SCALE}")
    sm_scale = sm_scale if sm_scale is not None else head**-0.5
    # A deeper K block needs shared memory proportional to head_dim.
    block_n = block_n if block_n is not None else (128 if head <= 64 else 64)
    stages = 3 if head <= 64 else 2
    mu = column_mean(k) if smooth_k else None
    qq, qs = quantize_rows(q)
    kq, ks = quantize_rows(k, mu)
    o = torch.empty(t, head, device=q.device, dtype=v.dtype)
    lse = torch.empty(t, device=q.device, dtype=torch.float32)
    _fwd_kernel[(triton.cdiv(t, block_m), 1, 1)](
        qq,
        qs,
        kq,
        ks,
        v,
        o,
        lse,
        _cu2(t, qq.device),
        sm_scale,
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
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_SUB=BLOCK_SCALE,
        CAUSAL=causal,
        WINDOW=0,
        num_warps=4,
        num_stages=stages,
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
    """One BLOCK_N block of dK and dV; dQ has its own kernel."""
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


BWD_CANDIDATES = (
    (32, 64, 4, 3),
    (64, 32, 4, 3),
    (64, 64, 4, 3),
    (64, 64, 8, 2),
    (32, 128, 4, 3),
    (64, 128, 4, 2),
    (128, 64, 4, 2),
    (128, 64, 8, 2),
)
_BWD_CACHE: dict[tuple, tuple] = {}


def _legal_bwd(cfg, head, smem_limit):
    """Shared-memory and accumulator budgets for one backward tile."""
    bm, bn, warps, stages = cfg
    smem = (bn * head * 3 + bn * (head // BLOCK_SCALE)) * max(stages - 1, 1)
    return smem <= smem_limit and bm * head / (32 * warps) <= 168


def plan_attn_bwd(t, head, block_m=None, block_n=None, dev=None, time_it=None):
    """Tiles for the dK/dV and dQ kernels, cached per shape.

    Candidates are filtered by the card's budgets; ``time_it(kind, cfg) -> ms``
    picks among them and the winner is cached. Without it the first legal
    candidate is used. See docs/internals/mxfp8-attention.md.
    """
    if block_m or block_n:
        base = (block_m or 64, block_n or 64, 4, 3)
        return base, base
    key = (t, head)
    if key in _BWD_CACHE:
        return _BWD_CACHE[key]
    limit = (dev or RTX_5090).smem_per_cta
    legal = [c for c in BWD_CANDIDATES if _legal_bwd(c, head, limit)]
    if not legal:
        raise ValueError(f"no legal backward tile for T={t} head={head}")
    chosen = []
    for kind in ("dkdv", "dq"):
        best, best_ms = legal[0], float("inf")
        if time_it is not None:
            for cfg in legal:
                try:
                    ms = time_it(kind, cfg)
                except Exception:
                    continue
                if ms < best_ms:
                    best, best_ms = cfg, ms
        chosen.append(best)
    _BWD_CACHE[key] = tuple(chosen)
    return _BWD_CACHE[key]


def clear_bwd_cache() -> None:
    _BWD_CACHE.clear()


def mxfp8_attention_backward(
    do, q, k, v, o, lse, mu, causal, sm_scale, block_m=None, block_n=None
):
    """Gradients of `mxfp8_attention`; returns ``(dQ, dK, dV)``.

    dK/dV and dQ run as separate kernels so each accumulates in registers and
    neither needs atomics. Tiles come from `plan_attn_bwd`.
    """
    t, head = q.shape
    delta = torch.empty(t, device=q.device, dtype=torch.float32)
    _bwd_preprocess[(triton.cdiv(t, 64),)](
        o,
        do,
        delta,
        t,
        head,
        o.stride(0),
        o.stride(1),
        do.stride(0),
        do.stride(1),
        BLOCK_M=64,
        num_warps=4,
    )
    qq, qs = quantize_rows(q)
    kq, ks = quantize_rows(k, mu)
    kk = k if mu is None else (k.float() - mu[None, :]).to(k.dtype)
    dq = torch.empty(t, head, device=q.device, dtype=q.dtype)
    dk = torch.empty(t, head, device=q.device, dtype=q.dtype)
    dv = torch.empty(t, head, device=q.device, dtype=q.dtype)

    def launch_dkdv(cfg):
        bm, bn, w, st = cfg
        _bwd_kernel[(triton.cdiv(t, bn),)](
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
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_SUB=BLOCK_SCALE,
            CAUSAL=causal,
            num_warps=w,
            num_stages=st,
        )

    def launch_dq(cfg):
        bm, bn, w, st = cfg
        _bwd_dq_kernel[(triton.cdiv(t, bm),)](
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
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_SUB=BLOCK_SCALE,
            CAUSAL=causal,
            num_warps=w,
            num_stages=st,
        )

    def time_it(kind, cfg):
        fn = launch_dkdv if kind == "dkdv" else launch_dq
        fn(cfg)
        torch.cuda.synchronize()
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        for _ in range(5):
            fn(cfg)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end)

    dkdv_cfg, dq_cfg = plan_attn_bwd(t, head, block_m, block_n, time_it=time_it)
    launch_dkdv(dkdv_cfg)
    launch_dq(dq_cfg)
    return dq, dk, dv


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


def split_qkv_mx(q_fused, s_fused, dim: int):
    """Split a fused ``(T, 3*dim)`` MXFP8 qkv into three views, scales included.

    ``dim`` must be a multiple of the scale block, so the split lands on group
    boundaries and no re-quantization is needed.
    """
    if dim % BLOCK_SCALE:
        raise ValueError(f"dim={dim} must be a multiple of {BLOCK_SCALE}")
    g = dim // BLOCK_SCALE
    return tuple(
        (q_fused[:, i * dim : (i + 1) * dim], s_fused[:, i * g : (i + 1) * g])
        for i in range(3)
    )


def mxfp8_attention_q(
    qq,
    qs,
    kq,
    ks,
    v,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_n: int | None = None,
):
    """Attention from operands already quantized by the producing kernel.

    Avoids re-reading Q and K to quantize them, which the attention loop would
    otherwise pay once per Q block. See docs/internals/mxfp8-attention.md.
    """
    t, head = qq.shape
    if head % BLOCK_SCALE:
        raise ValueError(f"head_dim={head} must be a multiple of {BLOCK_SCALE}")
    if block_n is not None and block_n % BLOCK_SCALE:
        raise ValueError(f"block_n={block_n} must be a multiple of {BLOCK_SCALE}")
    sm_scale = sm_scale if sm_scale is not None else head**-0.5
    block_n = block_n if block_n is not None else (128 if head <= 64 else 64)
    stages = 3 if head <= 64 else 2
    o = torch.empty(t, head, device=qq.device, dtype=v.dtype)
    lse = torch.empty(t, device=qq.device, dtype=torch.float32)
    _fwd_kernel[(triton.cdiv(t, block_m), 1, 1)](
        qq,
        qs,
        kq,
        ks,
        v,
        o,
        lse,
        _cu2(t, qq.device),
        sm_scale,
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
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_SUB=BLOCK_SCALE,
        CAUSAL=causal,
        WINDOW=0,
        num_warps=4,
        num_stages=stages,
    )
    return o, lse


@triton.jit
def _bwd_dq_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One BLOCK_M block of dQ, accumulated in registers so no atomics are needed."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD)
    groups: tl.constexpr = HEAD // BLOCK_SUB
    offs_g = tl.arange(0, groups)
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
    do = tl.load(
        do_ptr + offs_m[:, None] * sdo_t + offs_d[None, :] * sdo_d,
        mask=mm[:, None],
        other=0.0,
    )
    lse = tl.load(lse_ptr + offs_m, mask=mm, other=0.0)
    delta = tl.load(delta_ptr + offs_m, mask=mm, other=0.0)
    dq = tl.zeros((BLOCK_M, HEAD), dtype=tl.float32)

    hi = (pid + 1) * BLOCK_M if CAUSAL else T
    for start in range(0, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
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
        p = tl.exp(s - lse[:, None])

        v = tl.load(
            v_ptr + offs_n[:, None] * sv_t + offs_d[None, :] * sv_d,
            mask=nm[:, None],
            other=0.0,
        )
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = (p * (dp - delta[:, None]) * sm_scale).to(do.dtype)
        kb = tl.load(
            kb_ptr + offs_n[:, None] * sk_t + offs_d[None, :] * sk_d,
            mask=nm[:, None],
            other=0.0,
        )
        dq += tl.dot(ds, kb).to(tl.float32)

    tl.store(
        dq_ptr + offs_m[:, None] * sdq_t + offs_d[None, :] * sdq_d,
        dq.to(dq_ptr.dtype.element_ty),
        mask=mm[:, None],
    )


def mxfp8_attention_heads(
    q,
    k,
    v,
    causal: bool = True,
    sm_scale=None,
    block_m: int = 64,
    block_n=None,
    smooth_k: bool = True,
    cu_seqlens=None,
    max_seqlen=None,
    window=None,
):
    """Multi-head MXFP8 attention over ``(T, H, D)`` inputs; returns ``(T, H, D)``.

    ``cu_seqlens`` carries document boundaries for a packed batch; attention never
    crosses one. ``window`` bounds each query's history. Heads occupy the grid's
    third axis, which is what fills 170 SMs at real shapes.
    """
    t, heads, head = q.shape
    if head % BLOCK_SCALE:
        raise ValueError(f"head_dim={head} must be a multiple of {BLOCK_SCALE}")
    sm_scale = sm_scale if sm_scale is not None else head**-0.5
    block_n = block_n if block_n is not None else (128 if head <= 64 else 64)
    stages = 3 if head <= 64 else 2
    kv_heads = k.shape[1]
    if kv_heads != heads:
        rep = heads // kv_heads
        k = k.unsqueeze(2).expand(t, kv_heads, rep, head).reshape(t, heads, head)
        v = v.unsqueeze(2).expand(t, kv_heads, rep, head).reshape(t, heads, head)

    qh = q.transpose(0, 1).contiguous()
    kh = k.transpose(0, 1).contiguous()
    vh = v.transpose(0, 1).contiguous()
    # Accumulate the mean in fp32 without ever widening K in memory.
    mu = kh.mean(dim=1, dtype=torch.float32) if smooth_k else None
    qq, qs = quantize_rows(qh.reshape(-1, head))
    kq, ks = quantize_rows(kh.reshape(-1, head), mu, rows_per_mu=t)
    o = torch.empty_like(vh)
    lse = torch.empty(heads, t, device=q.device, dtype=torch.float32)
    if cu_seqlens is None:
        cu_seqlens = _cu2(t, q.device)
    n_docs = cu_seqlens.numel() - 1
    max_seq = max_seqlen or t
    _fwd_kernel[(triton.cdiv(max_seq, block_m), n_docs, heads)](
        qq,
        qs,
        kq,
        ks,
        vh,
        o,
        lse,
        cu_seqlens,
        sm_scale,
        head,
        qq.stride(0),
        qq.stride(1),
        qs.stride(0),
        qs.stride(1),
        kq.stride(0),
        kq.stride(1),
        ks.stride(0),
        ks.stride(1),
        vh.stride(1),
        vh.stride(2),
        o.stride(1),
        o.stride(2),
        t * qq.stride(0),
        t * qs.stride(0),
        t * kq.stride(0),
        t * ks.stride(0),
        vh.stride(0),
        o.stride(0),
        t,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_SUB=BLOCK_SCALE,
        CAUSAL=causal,
        WINDOW=window or 0,
        num_warps=4,
        num_stages=stages,
    )
    return o.transpose(0, 1).contiguous()
