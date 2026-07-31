"""Grouped MXFP8 GEMM: block-scaled fp8 with a variable row count per expert.

Also holds the WGRAD kernel every routed-expert path shares. The grid is bounded
from host-known values alone, so a whole MoE layer stays CUDA-graph capturable.

See docs/internals/mxfp8.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE

# Fixed, not autotuned; inherited by every forward kernel in this family.
FWD_TILE = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
FWD_STAGES = 3
FWD_WARPS = 4
# WGRAD contracts the token axis, so its tile is (N, K) of the weight and BLOCK_M
# is the row step of the loop.
WGRAD_TILE = {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128}
WGRAD_STAGES = 3
WGRAD_WARPS = 4


@triton.jit
def tile_owner(
    offsets_ptr, pid_t, num_experts, BLOCK_M: tl.constexpr, BLOCK_E: tl.constexpr
):
    """Resolve flat row-tile ``pid_t`` to ``(expert, start, end, local_tile)``.

    ``expert == num_experts`` means ``pid_t`` is past the last real tile, which
    happens because the grid is a bound rather than an exact count. Empty experts
    contribute zero tiles.
    """
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts
    starts = tl.load(offsets_ptr + offs_e, mask=mask_e, other=0).to(tl.int32)
    ends = tl.load(offsets_ptr + 1 + offs_e, mask=mask_e, other=0).to(tl.int32)
    tiles = tl.where(mask_e, tl.cdiv(ends - starts, BLOCK_M), 0)
    # Experts whose tiles all precede this one: the expert id and the tile prefix.
    done = tl.cumsum(tiles, axis=0) <= pid_t

    expert = tl.sum(done.to(tl.int32))
    local = pid_t - tl.sum(tl.where(done, tiles, 0))
    # Clamp the load, not the returned id: the caller needs the unclamped value.
    safe = tl.minimum(expert, num_experts - 1)
    return expert, tl.load(offsets_ptr + safe), tl.load(offsets_ptr + safe + 1), local


@triton.jit
def dequant_mx(q, scale, ROWS: tl.constexpr, COLS: tl.constexpr, SUB: tl.constexpr):
    """``(ROWS, COLS)`` e4m3 values x ``(ROWS, COLS//SUB)`` ue8m0 exponents -> fp32.

    The inverse of ``mxfp8._quantize_block``.
    """
    groups: tl.constexpr = COLS // SUB
    # Broadcast the scale over its block through a reshape; Triton has no interleave.
    grouped = tl.reshape(q.to(tl.float32), (ROWS, groups, SUB))
    return tl.reshape(
        grouped * tl.exp2(scale.to(tl.float32) - 127.0)[:, :, None], (ROWS, COLS)
    )


@triton.jit
def _grouped_mxfp8_fwd(
    xq_ptr,
    xs_ptr,
    index_ptr,
    wq_ptr,
    ws_ptr,
    out_ptr,
    offsets_ptr,
    num_experts,
    N,
    K,
    stride_xm,
    stride_xsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    stride_om,
    GATHER: tl.constexpr,
    K_EXACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """``out[m] = xq[src(m)] @ wq[e(m)].T`` in MXFP8, fp32 accumulate.

    ``GATHER`` reads each row's source index from ``index_ptr`` instead of using
    the sorted position directly.
    """
    expert, start, end, local_m = tile_owner(
        offsets_ptr, tl.program_id(1), num_experts, BLOCK_M, BLOCK_E
    )
    if expert >= num_experts:
        return
    pid_n = tl.program_id(0)

    offs_m = start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < end
    if GATHER:
        rows = tl.load(index_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    else:
        rows = offs_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    mask_n = offs_n < N

    xq_ptrs = xq_ptr + rows[:, None] * stride_xm + offs_k[None, :]
    xs_ptrs = xs_ptr + rows[:, None] * stride_xsm + offs_g[None, :]
    wq_ptrs = (
        wq_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    ws_ptrs = (
        ws_ptr + expert * stride_wse + offs_n[:, None] * stride_wsn + offs_g[None, :]
    )

    scale_cols = tl.cdiv(K, BLOCK_SUB)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < K
        # Scale loads carry their own column mask; K_EXACT compiles it away.
        if K_EXACT:
            mask_g = tl.full((groups,), True, tl.int1)
        else:
            mask_g = k0 * groups + offs_g < scale_cols
        xq = tl.load(xq_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        wq = tl.load(wq_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        xs = tl.load(xs_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0)
        ws = tl.load(ws_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        acc = tl.dot_scaled(xq, xs, "e4m3", tl.trans(wq), ws, "e4m3", acc=acc)
        xq_ptrs += BLOCK_K
        wq_ptrs += BLOCK_K
        xs_ptrs += groups
        ws_ptrs += groups

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def grouped_mxfp8_wgrad_kernel(
    a_ptr,
    bq_ptr,
    bs_ptr,
    index_ptr,
    scale_ptr,
    order_ptr,
    dw_ptr,
    offsets_ptr,
    N,
    K,
    stride_am,
    stride_bm,
    stride_bsm,
    stride_we,
    stride_wn,
    A_GATHER: tl.constexpr,
    B_GATHER: tl.constexpr,
    B_FP8: tl.constexpr,
    MUL_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``dW[e] = A[rows(e)].T @ B[rows(e)]`` -- 16-bit multiply, fp32 sum.

    ``MUL_DTYPE`` is the caller's 16-bit dtype, never a fixed one. ``B_FP8`` selects
    whether ``B`` arrives as an fp8 pair to dequantize or already 16-bit; it is a
    separate question from ``MUL_DTYPE``, and both branches multiply the same way.
    ``A_GATHER``/``B_GATHER`` index the corresponding operand by ``index_ptr``;
    under ``A_GATHER`` the gate weight is applied to ``A``.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    expert = tl.program_id(2)

    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = pid_k * groups + tl.arange(0, groups)
    mask_n = offs_n < N
    mask_k = offs_k < K
    mask_g = offs_g < tl.cdiv(K, BLOCK_SUB)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m in range(start, end, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end
        if A_GATHER:
            a_rows = tl.load(index_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
        else:
            a_rows = offs_m
        if B_GATHER:
            b_rows = tl.load(index_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
        else:
            b_rows = offs_m
        a = tl.load(
            a_ptr + a_rows[:, None] * stride_am + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        if A_GATHER:
            # The combine's contribution to dW: scale the routed row by its gate.
            pair = tl.load(order_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
            a *= tl.load(scale_ptr + pair, mask=mask_m, other=0.0)[:, None]
        bq = tl.load(
            bq_ptr + b_rows[:, None] * stride_bm + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if B_FP8:
            bs = tl.load(
                bs_ptr + b_rows[:, None] * stride_bsm + offs_g[None, :],
                mask=mask_m[:, None] & mask_g[None, :],
                other=0,
            )
            b = dequant_mx(bq, bs, BLOCK_M, BLOCK_K, BLOCK_SUB)
        else:
            b = bq.to(tl.float32)
        acc = tl.dot(
            tl.trans(a.to(MUL_DTYPE)),
            b.to(MUL_DTYPE),
            acc=acc,
            out_dtype=tl.float32,
        )

    tl.store(
        dw_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :],
        acc.to(dw_ptr.dtype.element_ty),
        mask=mask_n[:, None] & mask_k[None, :],
    )


def _block_e(num_experts: int) -> int:
    return max(16, triton.next_power_of_2(num_experts))


def grouped_mxfp8_gemm(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor,
    index: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    rows: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[m] = xq[index[m]] @ wq[e(m)].T`` for expert-sorted rows, in MXFP8.

    The bare primitive. The fused expert path in
    :mod:`kohakuwullm.kernels.mxfp8.experts` differs only in its epilogue and does
    not call this.

    Args:
        xq / x_scale: ``(R, K)`` e4m3 and ``(R, K//32)`` ue8m0, blocks along K.
        wq / w_scale: ``(E, N, K)`` e4m3 and ``(E, N, K//32)`` ue8m0.
        offsets: ``(E+1,)`` int32 exclusive prefix sum of per-expert row counts.
            Never read on the host.
        index: ``(M,)`` int32 source row per sorted position, or ``None`` for the
            identity. With an index, ``xq`` has ``R`` rows and the output ``M``.
        rows: output row count ``M``. Required when ``index`` is ``None`` only if
            it differs from ``xq``'s.
        out: destination, or ``None`` to allocate. Pass a zeroed buffer when a
            sentinel bucket leaves rows outside every tile the grid resolves.
    """
    num_experts, n, k = wq.shape
    if k % BLOCK_SCALE:
        raise ValueError(f"K={k} must be a multiple of {BLOCK_SCALE}")
    if w_scale.shape != (num_experts, n, k // BLOCK_SCALE):
        raise ValueError(f"w_scale {tuple(w_scale.shape)} does not match {wq.shape}")
    if xq.shape[1] != k:
        raise ValueError(f"contraction mismatch: xq {tuple(xq.shape)} vs K={k}")
    m = xq.shape[0] if index is None else index.numel()
    if rows is not None:
        m = rows
    if out is None:
        out = torch.empty(m, n, device=xq.device, dtype=out_dtype)
    elif out.shape != (m, n):
        raise ValueError(f"out {tuple(out.shape)} does not match ({m}, {n})")
    grid = (
        triton.cdiv(n, FWD_TILE["BLOCK_N"]),
        triton.cdiv(m, FWD_TILE["BLOCK_M"]) + num_experts,
    )
    _grouped_mxfp8_fwd[grid](
        xq,
        x_scale,
        index,
        wq,
        w_scale,
        out,
        offsets,
        num_experts,
        n,
        k,
        xq.stride(0),
        x_scale.stride(0),
        wq.stride(0),
        wq.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        GATHER=index is not None,
        K_EXACT=k % FWD_TILE["BLOCK_K"] == 0,
        BLOCK_SUB=BLOCK_SCALE,
        BLOCK_E=_block_e(num_experts),
        num_stages=FWD_STAGES,
        num_warps=FWD_WARPS,
        **FWD_TILE,
    )
    return out


def grouped_mxfp8_reference(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    offsets: torch.Tensor,
    index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert fp64 matmul on dequantized operands. The precision oracle."""
    scale = torch.exp2(x_scale.double() - 127.0).repeat_interleave(BLOCK_SCALE, dim=-1)
    xd = xq.double() * scale
    if index is not None:
        xd = xd.index_select(0, index.long())
    off = offsets.tolist()
    outs = []
    for e in range(wq.shape[0]):
        ws = torch.exp2(w_scale[e].double() - 127.0).repeat_interleave(
            BLOCK_SCALE, dim=-1
        )
        outs.append(xd[off[e] : off[e + 1]] @ (wq[e].double() * ws).T)
    return torch.cat(outs, dim=0)
