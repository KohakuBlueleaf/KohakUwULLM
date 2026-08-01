"""MXFP8 matmul: a quantizer, a pre-quantized GEMM, and a fused reference.

Block scales run along K, the contraction axis, and the scale exponent rounds
**up**. Accumulation is fp32 always; the output dtype is the caller's.

See docs/internals/mxfp8.md.
"""

import torch
import triton
import triton.language as tl

# e4m3 max: a block is scaled so its amax lands here rather than at 1.0.
E4M3_MAX = 448.0
BLOCK_SCALE = 32
# Triton reads a module global from a jitted body only as a constexpr instance.
_E4M3_MAX = tl.constexpr(448.0)


def _configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_stages=s,
            num_warps=w,
        )
        # Bounded by sm_120 shared memory, halved by the online quantizer's fp32
        # widening for the amax.
        for bm, bn, bk, s, w in (
            (128, 128, 64, 2, 4),
            (128, 64, 64, 3, 4),
            (64, 128, 64, 3, 4),
            (64, 64, 128, 3, 4),
            (128, 128, 128, 2, 8),
        )
    ]


@triton.jit
def _quantize_block(
    tile, ROWS: tl.constexpr, COLS: tl.constexpr, BLOCK_SUB: tl.constexpr
):
    """One tile -> (e4m3 values, ue8m0 exponents), blocks of ``BLOCK_SUB`` along K.

    Dimensions are passed rather than read from ``tile.shape``, which is not a
    ``constexpr`` at an inlined call site. Returns the exponent as the biased byte
    the MMA consumes.
    """
    groups: tl.constexpr = COLS // BLOCK_SUB
    grouped = tl.reshape(tile.to(tl.float32), (ROWS, groups, BLOCK_SUB))
    amax = tl.max(tl.abs(grouped), axis=2)
    # The scale is a power of two, so ceil(log2(.)) is the whole quantizer.
    exponent = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / _E4M3_MAX))
    exponent = tl.maximum(exponent, -127.0)
    quantized = (grouped / tl.exp2(exponent)[:, :, None]).to(tl.float8e4nv)
    return tl.reshape(quantized, (ROWS, COLS)), (exponent + 127.0).to(tl.uint8)


# Static, and load-bearing: M is the varlen token count, so an autotune key
# containing it re-runs the search on every step.
QUANT_TILE = {"BLOCK_M": 8, "BLOCK_K": 1024}
QUANT_WARPS = 8


@triton.jit
def _quantize_mx_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
):
    """Standalone quantizer: ``(M,K)`` fp16/bf16 -> e4m3 values + ue8m0 scales."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask,
        other=0.0,
    )
    q, s = _quantize_block(x, BLOCK_M, BLOCK_K, BLOCK_SUB)
    tl.store(
        q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, q, mask=mask
    )
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = pid_k * groups + tl.arange(0, groups)
    tl.store(
        s_ptr + offs_m[:, None] * stride_sm + offs_g[None, :] * stride_sk,
        s,
        mask=(offs_m < M)[:, None] & (offs_g < tl.cdiv(K, BLOCK_SUB))[None, :],
    )


def quantize_mx(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M,K)`` -> ``(e4m3 (M,K), ue8m0 (M, K//32))``, blocks along the last axis.

    Bandwidth-bound. ``K`` must be a multiple of 32.
    """
    if x.shape[-1] % BLOCK_SCALE:
        raise ValueError(f"K={x.shape[-1]} must be a multiple of {BLOCK_SCALE}")
    x = x.contiguous()
    m, k = x.shape
    q = torch.empty(m, k, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(m, k // BLOCK_SCALE, device=x.device, dtype=torch.uint8)
    grid = (
        triton.cdiv(m, QUANT_TILE["BLOCK_M"]),
        triton.cdiv(k, QUANT_TILE["BLOCK_K"]),
    )
    _quantize_mx_kernel[grid](
        x,
        q,
        s,
        m,
        k,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        s.stride(0),
        s.stride(1),
        BLOCK_SUB=BLOCK_SCALE,
        num_warps=QUANT_WARPS,
        **QUANT_TILE,
    )
    return q, s


@triton.jit
def _quantize_mx_vendor_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    K,
    col_tiles,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    BLOCK_SUB: tl.constexpr,
):
    """Quantize and write scales straight into cuBLAS's ``SWIZZLE_32_4_4`` layout.

    Fixed 128x128 tiles: a 128-row x 4-scale-column swizzle tile is the unit cuBLAS
    addresses, and one program must cover a whole one.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 128)
    offs_m = pid_m * 128 + rows
    offs_k = pid_k * 128 + cols
    mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask,
        other=0.0,
    )
    q, s = _quantize_block(x, 128, 128, BLOCK_SUB)
    tl.store(
        q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, q, mask=mask
    )
    # One 512-byte tile: 4 contiguous scale columns per row, rows striding by 16
    # within a 32-row group.
    sub = tl.arange(0, 4)
    base = (pid_m * col_tiles + pid_k) * 512
    offs = base + (rows % 32)[:, None] * 16 + (rows // 32)[:, None] * 4 + sub[None, :]
    tl.store(s_ptr + offs, s, mask=(offs_m < M)[:, None])


def quantize_mx_vendor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M,K)`` -> ``(e4m3, e8m0 swizzled)``, the pair ``scaled_mm`` accepts.

    Fuses the layout transform into the cast. ``K`` must be a multiple of 128.
    """
    if x.shape[-1] % (4 * BLOCK_SCALE):
        raise ValueError(f"K must be a multiple of {4 * BLOCK_SCALE} to swizzle")
    # Only a doubly-strided layout needs a copy; the kernel tiles in 2D and takes
    # both strides.
    if x.stride(0) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    m, k = x.shape
    col_tiles = k // (4 * BLOCK_SCALE)
    q = torch.empty(m, k, device=x.device, dtype=torch.float8_e4m3fn)
    # Zeroed: cuBLAS still reads the padding of a partial final row tile.
    s = torch.zeros(
        -(-m // 128) * 128 * col_tiles * 4, device=x.device, dtype=torch.uint8
    )
    _quantize_mx_vendor_kernel[(-(-m // 128), col_tiles)](
        x,
        q,
        s,
        m,
        k,
        col_tiles,
        x.stride(0),
        x.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_SUB=BLOCK_SCALE,
        num_warps=8,
    )
    return q, s.view(torch.float8_e8m0fnu)


def _pq_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_stages=s,
            num_warps=w,
        )
        for bm, bn, bk, s, w in (
            (64, 128, 256, 3, 4),
            (128, 64, 256, 3, 4),
            (128, 128, 64, 3, 4),
            (128, 128, 64, 3, 8),
            (128, 128, 64, 5, 8),
            (128, 128, 128, 3, 8),
            (128, 256, 128, 3, 8),
            (256, 128, 128, 3, 8),
            (128, 128, 128, 4, 8),
            (128, 256, 64, 4, 8),
            (64, 256, 128, 3, 8),
            (128, 128, 256, 3, 8),
        )
    ]


@triton.autotune(configs=_pq_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp8_matmul_pq(
    a_ptr,
    as_ptr,
    b_ptr,
    bs_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_bn,
    stride_bk,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """``C = A @ B.T`` from operands already in e4m3 + ue8m0.

    The grid is flat and rasterized in groups of ``GROUP_M`` row tiles.
    See docs/internals/kernels.md.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group = pid // width
    rows = min(grid_m - group * GROUP_M, GROUP_M)
    pid_m = group * GROUP_M + (pid % rows)
    pid_n = (pid % width) // rows
    # True offsets drive the masks and the store; the wrapped copies drive the
    # pointers, so a tail tile reads in range and is masked off anyway.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_mp = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_np = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    mask_m = offs_m < M
    mask_n = offs_n < N
    scale_cols = tl.cdiv(K, BLOCK_SUB)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        mask_k = k < K
        g = k0 * groups + offs_g
        # The scale needs its own column mask; the paired value load is masked to
        # zero, so an over-read here would be silent.
        mask_g = g < scale_cols
        aq = tl.load(
            a_ptr + offs_mp[:, None] * stride_am + k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
        )
        bq = tl.load(
            b_ptr + offs_np[:, None] * stride_bn + k[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_k[None, :],
        )
        a_scale = tl.load(
            as_ptr + offs_mp[:, None] * stride_asm + g[None, :] * stride_ask,
            mask=mask_m[:, None] & mask_g[None, :],
            other=0,
        )
        b_scale = tl.load(
            bs_ptr + offs_np[:, None] * stride_bsn + g[None, :] * stride_bsk,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0,
        )
        # The weight is stored (N, K) and transposed in registers here.
        acc = tl.dot_scaled(aq, a_scale, "e4m3", tl.trans(bq), b_scale, "e4m3", acc=acc)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def mxfp8_matmul_pq(
    aq: torch.Tensor,
    a_scale: torch.Tensor,
    bq: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``aq @ bq.T`` from pre-quantized operands, fp32 accumulate.

    Needs only ``K % 32 == 0``, unlike the ``K % 128 == 0`` ``quantize_mx_vendor``
    imposes to fill a swizzle tile.
    """
    m, k = aq.shape
    n = bq.shape[0]
    out = torch.empty(m, n, device=aq.device, dtype=out_dtype)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _mxfp8_matmul_pq[grid](
        aq,
        a_scale,
        bq,
        b_scale,
        out,
        m,
        n,
        k,
        aq.stride(0),
        aq.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        bq.stride(0),
        bq.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SUB=BLOCK_SCALE,
        GROUP_M=8,
    )
    return out


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _mxfp8_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
):
    """``C = A @ B.T`` with both operands quantized per K-block inside the loop.

    ``B`` is taken K-minor (``(N, K)``), so both operands block along the same axis.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        mask_k = k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + k[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        aq, a_scale = _quantize_block(a, BLOCK_M, BLOCK_K, BLOCK_SUB)
        bq, b_scale = _quantize_block(b, BLOCK_N, BLOCK_K, BLOCK_SUB)
        # dot_scaled contracts (M,K) against (K,N); the K-minor tile is transposed
        # here rather than on load.
        acc = tl.dot_scaled(aq, a_scale, "e4m3", tl.trans(bq), b_scale, "e4m3", acc=acc)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def mxfp8_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``a @ b.T`` in MXFP8 with fp32 accumulation. The correctness reference.

    Args:
        a: ``(M, K)`` activations in fp16/bf16.
        b: ``(N, K)`` weights in fp16/bf16, K-minor as a linear layer stores them.
        out_dtype: output dtype; defaults to ``a``'s.
    """
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"contraction mismatch: {a.shape} @ {b.shape}.T")
    if a.shape[-1] % BLOCK_SCALE:
        raise ValueError(f"K={a.shape[-1]} must be a multiple of {BLOCK_SCALE}")
    a = a.contiguous()
    b = b.contiguous()
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty(m, n, device=a.device, dtype=out_dtype or a.dtype)
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )
    _mxfp8_matmul[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SUB=BLOCK_SCALE,
    )
    return out
