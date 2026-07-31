"""Backward of the fused MXFP8 expert path: DGRAD in fp8, and one fused epilogue.

GEMM2's DGRAD also produces the gate-weight gradient, from
``dL/dg_p = <dout[t_p] @ W_out, h_p>``, whose left factor is the DGRAD tile itself.
WGRAD is not here: it is not an fp8 product and has no MoE-specific epilogue, so it
lives in :mod:`kohakuwullm.kernels.mxfp8.grouped`.

See docs/internals/mxfp8.md.
"""

import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8 import _quantize_block
from kohakuwullm.kernels.mxfp8.grouped import dequant_mx, tile_owner

# Neither DGRAD inherits the forward tile, and both take two pipeline stages where
# the forward kernels take three: these are bound by their epilogues, not the MMA.
GEMM2_DGRAD_TILE = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}
GEMM2_DGRAD_STAGES = 2
GEMM2_DGRAD_WARPS = 4
# GEMM1's DGRAD contracts 2H, so its N tile is the model width and BLOCK_N is *not*
# per-half the way the forward's is.
GEMM1_DGRAD_TILE = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
GEMM1_DGRAD_STAGES = 2
GEMM1_DGRAD_WARPS = 4


@triton.jit
def _experts_gemm2_dgrad(
    dq_ptr,
    ds_ptr,
    token_ptr,
    order_ptr,
    gate_ptr,
    wq_ptr,
    ws_ptr,
    hq_ptr,
    hs_ptr,
    pre_ptr,
    dpre_ptr,
    dpq_ptr,
    dps_ptr,
    dgate_ptr,
    offsets_ptr,
    num_experts,
    H,
    N,
    stride_dm,
    stride_dsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    stride_hm,
    stride_hsm,
    stride_pm,
    stride_qm,
    stride_qsm,
    K_EXACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """DGRAD through GEMM2, the combine and swiglu at once.

    The accumulator ``dout[t] @ W_out`` is consumed three ways in registers. ``h``
    is read back from its fp8 copy: the forward multiplied those values.
    """
    expert, start, end, local_m = tile_owner(
        offsets_ptr, tl.program_id(1), num_experts, BLOCK_M, BLOCK_E
    )
    if expert >= num_experts:
        return
    pid_n = tl.program_id(0)

    offs_m = start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < end
    rows = tl.load(token_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    mask_n = offs_n < H

    dq_ptrs = dq_ptr + rows[:, None] * stride_dm + offs_k[None, :]
    ds_ptrs = ds_ptr + rows[:, None] * stride_dsm + offs_g[None, :]
    wq_ptrs = (
        wq_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    ws_ptrs = (
        ws_ptr + expert * stride_wse + offs_n[:, None] * stride_wsn + offs_g[None, :]
    )

    scale_cols = tl.cdiv(N, BLOCK_SUB)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(N, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < N
        # Scale loads carry their own column mask; K_EXACT compiles it away.
        if K_EXACT:
            mask_g = tl.full((groups,), True, tl.int1)
        else:
            mask_g = k0 * groups + offs_g < scale_cols
        dq = tl.load(dq_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        ds = tl.load(ds_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0)
        wq = tl.load(wq_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        ws = tl.load(ws_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        acc = tl.dot_scaled(dq, ds, "e4m3", tl.trans(wq), ws, "e4m3", acc=acc)
        dq_ptrs += BLOCK_K
        ds_ptrs += groups
        wq_ptrs += BLOCK_K
        ws_ptrs += groups

    mask_out = mask_m[:, None] & mask_n[None, :]
    hgroups: tl.constexpr = BLOCK_N // BLOCK_SUB
    offs_hg = pid_n * hgroups + tl.arange(0, hgroups)
    mask_hg = offs_hg < tl.cdiv(H, BLOCK_SUB)
    hq = tl.load(
        hq_ptr + offs_m[:, None] * stride_hm + offs_n[None, :], mask=mask_out, other=0.0
    )
    hs = tl.load(
        hs_ptr + offs_m[:, None] * stride_hsm + offs_hg[None, :],
        mask=mask_m[:, None] & mask_hg[None, :],
        other=0,
    )
    h = dequant_mx(hq, hs, BLOCK_M, BLOCK_N, BLOCK_SUB)

    pair = tl.load(order_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    # fp32 atomic: this dot product over the expert width is split across BLOCK_N
    # tiles, so the sum crosses programs.
    tl.atomic_add(
        dgate_ptr + pair,
        tl.sum(tl.where(mask_out, acc * h, 0.0), axis=1),
        mask=mask_m,
    )

    scale = tl.load(gate_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
    dh = acc * scale[:, None]
    pre_ptrs = pre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
    gate = tl.load(pre_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    value = tl.load(pre_ptrs + H, mask=mask_out, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    # d/dgate [gate * sigmoid(gate)] = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dgate_t = dh * value * sig * (1.0 + gate * (1.0 - sig))
    dvalue_t = dh * gate * sig

    dpre_ptrs = dpre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(dpre_ptrs, dgate_t.to(dpre_ptr.dtype.element_ty), mask=mask_out)
    tl.store(dpre_ptrs + H, dvalue_t.to(dpre_ptr.dtype.element_ty), mask=mask_out)

    # Quantized in this epilogue: GEMM1's DGRAD contracts 2H, the axis this tile
    # already spans, so no scale block straddles two tiles.
    dq_g, ds_g = _quantize_block(dgate_t, BLOCK_M, BLOCK_N, BLOCK_SUB)
    dq_v, ds_v = _quantize_block(dvalue_t, BLOCK_M, BLOCK_N, BLOCK_SUB)
    dpq_ptrs = dpq_ptr + offs_m[:, None] * stride_qm + offs_n[None, :]
    tl.store(dpq_ptrs, dq_g, mask=mask_out)
    tl.store(dpq_ptrs + H, dq_v, mask=mask_out)
    dps_ptrs = dps_ptr + offs_m[:, None] * stride_qsm + offs_hg[None, :]
    mask_s = mask_m[:, None] & mask_hg[None, :]
    tl.store(dps_ptrs, ds_g, mask=mask_s)
    tl.store(dps_ptrs + tl.cdiv(H, BLOCK_SUB), ds_v, mask=mask_s)


@triton.jit
def _experts_gemm1_dgrad(
    dpq_ptr,
    dps_ptr,
    wq_ptr,
    ws_ptr,
    dx_ptr,
    token_ptr,
    offsets_ptr,
    num_experts,
    K,
    TWO_H,
    stride_qm,
    stride_qsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    stride_xm,
    K_EXACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """``dx[token(m)] += dpre[m] @ w_in[e(m)]`` -- DGRAD and the gather's backward.

    A token routed to ``top_k`` experts collects ``top_k`` contributions, added
    straight from the accumulator.
    """
    expert, start, end, local_m = tile_owner(
        offsets_ptr, tl.program_id(1), num_experts, BLOCK_M, BLOCK_E
    )
    if expert >= num_experts:
        return
    pid_n = tl.program_id(0)

    offs_m = start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < end
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    groups: tl.constexpr = BLOCK_K // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    mask_n = offs_n < K

    dq_ptrs = dpq_ptr + offs_m[:, None] * stride_qm + offs_k[None, :]
    ds_ptrs = dps_ptr + offs_m[:, None] * stride_qsm + offs_g[None, :]
    wq_ptrs = (
        wq_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    ws_ptrs = (
        ws_ptr + expert * stride_wse + offs_n[:, None] * stride_wsn + offs_g[None, :]
    )

    scale_cols = tl.cdiv(TWO_H, BLOCK_SUB)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(TWO_H, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < TWO_H
        # Scale loads carry their own column mask; K_EXACT compiles it away.
        if K_EXACT:
            mask_g = tl.full((groups,), True, tl.int1)
        else:
            mask_g = k0 * groups + offs_g < scale_cols
        dq = tl.load(dq_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        ds = tl.load(ds_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0)
        wq = tl.load(wq_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        ws = tl.load(ws_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        acc = tl.dot_scaled(dq, ds, "e4m3", tl.trans(wq), ws, "e4m3", acc=acc)
        dq_ptrs += BLOCK_K
        ds_ptrs += groups
        wq_ptrs += BLOCK_K
        ws_ptrs += groups

    rows = tl.load(token_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    tl.atomic_add(
        dx_ptr + rows[:, None] * stride_xm + offs_n[None, :],
        acc.to(dx_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )
