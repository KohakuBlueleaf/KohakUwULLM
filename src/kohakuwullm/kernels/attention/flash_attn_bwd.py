"""Backward pass of the varlen flash attention kernel: two passes, opposite axes.

``dk``/``dv`` grids over keys and streams queries through the inner loop; ``dq`` is
the mirror image. Both recompute the probability matrix from the forward's saved
``lse``, so nothing ``(M, N)``-shaped is ever stored.

See docs/internals/kernels.md.
"""

import triton
import triton.language as tl

from kohakuwullm.kernels.attention.flash_attn_fwd import NEG_INF


@triton.jit
def _bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    T,
    stride_ot,
    stride_oh,
    stride_od,
    stride_dh,
    stride_dt,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """delta[t] = sum_d out[t, d] * dout[t, d] -- reused by both backward kernels."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    mask = offs_m < T
    o = tl.load(
        o_ptr
        + offs_m[:, None] * stride_ot
        + head * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + offs_m[:, None] * stride_ot
        + head * stride_oh
        + offs_d[None, :] * stride_od,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        delta_ptr + head * stride_dh + offs_m * stride_dt, tl.sum(o * do, 1), mask=mask
    )


# Separate tuning spaces: the two kernels parallelize over opposite axes, so dk/dv
# wants a tall BLOCK_N and a modest BLOCK_M and dq is the mirror image.
def _bwd_dkdv_configs():
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m, n, s, w in (
            (64, 128, 2, 8),
            (64, 128, 3, 8),
            (32, 128, 3, 4),
            (64, 64, 3, 4),
            (64, 64, 2, 4),
            (128, 64, 2, 8),
            (32, 64, 4, 4),
        )
    ]


def _bwd_dq_configs():
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m, n, s, w in (
            (128, 64, 2, 8),
            (128, 64, 3, 8),
            (128, 32, 4, 4),
            (64, 64, 3, 4),
            (64, 64, 4, 4),
            (64, 32, 4, 4),
        )
    ]


@triton.autotune(
    configs=_bwd_dkdv_configs(),
    key=["HEAD_DIM", "IS_CAUSAL", "HAS_WINDOW"],
    # Mandatory with the atomic accumulation below: autotune re-runs the kernel
    # once per candidate config.
    reset_to_zero=["dk_ptr", "dv_ptr"],
)
@triton.jit
def _bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dk_ptr,
    dv_ptr,
    cu_ptr,
    sm_scale,
    window,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_lt,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program owns a block of keys for ONE query head.

    Gridding over query heads rather than kv heads, so the GQA group accumulates
    into shared ``dk``/``dv`` through fp32 atomics rather than in registers.
    """
    start_n = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.program_id(2)
    kv_head = head // GQA_GROUP

    seq_start = tl.load(cu_ptr + seq)
    seq_end = tl.load(cu_ptr + seq + 1)
    seq_len = seq_end - seq_start
    if start_n * BLOCK_N >= seq_len:
        return

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seq_len

    k = tl.load(
        k_ptr
        + (seq_start + offs_n[:, None]) * stride_kt
        + kv_head * stride_kh
        + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + (seq_start + offs_n[:, None]) * stride_kt
        + kv_head * stride_kh
        + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)

    lo = start_n * BLOCK_N if IS_CAUSAL else 0
    lo = (lo // BLOCK_M) * BLOCK_M
    hi = seq_len
    if HAS_WINDOW:
        hi = tl.minimum(seq_len, (start_n + 1) * BLOCK_N + window)

    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        q = tl.load(
            q_ptr
            + (seq_start + offs_m[:, None]) * stride_qt
            + head * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        qk = tl.dot(q, k.T) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, NEG_INF)
        if HAS_WINDOW:
            qk = tl.where(offs_m[:, None] - offs_n[None, :] < window, qk, NEG_INF)

        lse = tl.load(
            lse_ptr + head * stride_lh + (seq_start + offs_m) * stride_lt,
            mask=mask_m,
            other=0.0,
        )
        # p recomputed from the saved lse, so it never touches memory.
        p = tl.exp(qk - lse[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        do = tl.load(
            do_ptr
            + (seq_start + offs_m[:, None]) * stride_qt
            + head * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None],
            other=0.0,
        )
        dv += tl.dot(p.to(do.dtype).T, do)

        delta = tl.load(
            delta_ptr + head * stride_lh + (seq_start + offs_m) * stride_lt,
            mask=mask_m,
            other=0.0,
        )
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * sm_scale
        dk += tl.dot(ds.to(q.dtype).T, q)

    dk_ptrs = (
        dk_ptr
        + (seq_start + offs_n[:, None]) * stride_kt
        + kv_head * stride_kh
        + offs_d[None, :] * stride_kd
    )
    dv_ptrs = (
        dv_ptr
        + (seq_start + offs_n[:, None]) * stride_kt
        + kv_head * stride_kh
        + offs_d[None, :] * stride_kd
    )
    if GQA_GROUP == 1:
        # MHA: this program is the only writer, so a plain store is enough.
        tl.store(dk_ptrs, dk, mask=mask_n[:, None])
        tl.store(dv_ptrs, dv, mask=mask_n[:, None])
    else:
        tl.atomic_add(dk_ptrs, dk, mask=mask_n[:, None])
        tl.atomic_add(dv_ptrs, dv, mask=mask_n[:, None])


@triton.autotune(configs=_bwd_dq_configs(), key=["HEAD_DIM", "IS_CAUSAL", "HAS_WINDOW"])
@triton.jit
def _bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    dq_ptr,
    cu_ptr,
    sm_scale,
    window,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_lt,
    GQA_GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program owns a block of queries and sweeps the keys it attends to."""
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
    do = tl.load(
        do_ptr
        + (seq_start + offs_m[:, None]) * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + head * stride_lh + (seq_start + offs_m) * stride_lt,
        mask=mask_m,
        other=0.0,
    )
    delta = tl.load(
        delta_ptr + head * stride_lh + (seq_start + offs_m) * stride_lt,
        mask=mask_m,
        other=0.0,
    )

    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    hi = tl.minimum(seq_len, (start_m + 1) * BLOCK_M) if IS_CAUSAL else seq_len
    lo = 0
    if HAS_WINDOW:
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
        v = tl.load(
            v_ptr
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
        p = tl.exp(qk - lse[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        dp = tl.dot(do, v.T)
        ds = p * (dp - delta[:, None]) * sm_scale
        dq += tl.dot(ds.to(k.dtype), k)

    tl.store(
        dq_ptr
        + (seq_start + offs_m[:, None]) * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd,
        dq.to(dq_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )
