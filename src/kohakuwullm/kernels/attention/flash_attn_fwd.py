"""Forward pass of the varlen flash attention kernel.

One program owns a block of queries and streams the keys it attends to, keeping the
running softmax state in registers so the ``(M, N)`` probability matrix never reaches
memory. The log-sum-exp is written out because both backward kernels recompute ``p``
from it. The leaf module of the three: ``NEG_INF`` lives here because both others
read it.

See docs/internals/kernels.md.
"""

import triton
import triton.language as tl

# Finite sentinel, not -inf: a fully-masked key block would rescale -inf - (-inf).
NEG_INF = tl.constexpr(-1.0e6)


def _fwd_configs():
    # Smaller tiles than a Hopper kernel: sm_120 has ~100 KB of shared memory.
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m, n, s, w in (
            (128, 64, 3, 8),
            (128, 64, 2, 4),
            (64, 64, 3, 4),
            (64, 64, 4, 4),
            (128, 32, 4, 4),
            (64, 32, 4, 4),
            (32, 64, 4, 4),
        )
    ]


@triton.autotune(configs=_fwd_configs(), key=["HEAD_DIM", "IS_CAUSAL", "HAS_WINDOW"])
@triton.jit
def _fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_ptr,
    sm_scale,
    window,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lh,
    stride_lt,
    H: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
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
    mask_m = offs_m < seq_len

    q = tl.load(
        q_ptr
        + (seq_start + offs_m[:, None]) * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
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
        qk = tl.dot(q, k.T) * sm_scale
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
