"""Forward pass of the varlen MXFP8 flash attention kernel.

Same layout and grid as :mod:`kohakuwullm.kernels.attention.flash_attn_fwd`. The
score GEMM takes e4m3 operands with ue8m0 block scales along the head dimension;
``PV`` runs in the input dtype.

See docs/internals/mxfp8-attention.md.
"""

import triton
import triton.language as tl

from kohakuwullm.kernels.attention.flash_attn_fwd import NEG_INF


def _fwd_configs():
    # Deeper key blocks than the bf16 kernel: e4m3 operands halve the tile bytes.
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m, n, s, w in (
            (64, 128, 3, 4),
            (128, 128, 3, 8),
            (128, 64, 3, 8),
            (128, 64, 2, 4),
            (64, 64, 3, 4),
            (64, 128, 2, 4),
            (64, 32, 4, 4),
        )
    ]


@triton.autotune(configs=_fwd_configs(), key=["HEAD_DIM", "IS_CAUSAL", "HAS_WINDOW"])
@triton.jit
def _fwd_kernel(
    q_ptr,
    qs_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_ptr,
    sm_scale,
    window,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_qst,
    stride_qsh,
    stride_qsg,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_kst,
    stride_ksh,
    stride_ksg,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lh,
    stride_lt,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program owns a block of queries; grid is (query blocks, heads, documents)."""
    start_m = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)

    seq_start = tl.load(cu_ptr + seq)
    seq_end = tl.load(cu_ptr + seq + 1)
    seq_len = seq_end - seq_start
    if start_m * BLOCK_M >= seq_len:
        return

    kv_head = head // GQA_GROUP
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    groups: tl.constexpr = HEAD_DIM // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    mask_m = offs_m < seq_len

    q = tl.load(
        q_ptr
        + (seq_start + offs_m[:, None]) * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    qs = tl.load(
        qs_ptr
        + (seq_start + offs_m[:, None]) * stride_qst
        + head * stride_qsh
        + offs_g[None, :] * stride_qsg,
        mask=mask_m[:, None],
        other=0,
    )

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], NEG_INF, tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    # Causal: only key blocks at or before this query block can contribute.
    hi = tl.minimum(seq_len, (start_m + 1) * BLOCK_M) if IS_CAUSAL else seq_len
    lo = 0
    if HAS_WINDOW:
        # Skip the key blocks that lie entirely outside the window.
        lo = tl.maximum(0, start_m * BLOCK_M - window + 1)
        lo = (lo // BLOCK_N) * BLOCK_N

    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        k = tl.load(
            k_ptr
            + (seq_start + offs_n[:, None]) * stride_kt
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd,
            mask=mask_n[:, None],
            other=0.0,
        )
        ks = tl.load(
            ks_ptr
            + (seq_start + offs_n[:, None]) * stride_kst
            + kv_head * stride_ksh
            + offs_g[None, :] * stride_ksg,
            mask=mask_n[:, None],
            other=0,
        )
        qk = tl.dot_scaled(
            q,
            qs,
            "e4m3",
            tl.trans(k),
            ks,
            "e4m3",
            acc=tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
        )
        qk = qk * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, NEG_INF)
        if HAS_WINDOW:
            qk = tl.where(offs_m[:, None] - offs_n[None, :] < window, qk, NEG_INF)
        qk = tl.where(mask_n[None, :], qk, NEG_INF)

        # Online softmax: rescale the running accumulator by the change in max.
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptr
            + (seq_start + offs_n[:, None]) * stride_vt
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # A fully-masked row (possible with a window) has l_i == 0; guard the divide.
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), NEG_INF)

    tl.store(
        lse_ptr + head * stride_lh + (seq_start + offs_m) * stride_lt, lse, mask=mask_m
    )
    tl.store(
        out_ptr
        + (seq_start + offs_m[:, None]) * stride_ot
        + head * stride_oh
        + offs_d[None, :] * stride_od,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
