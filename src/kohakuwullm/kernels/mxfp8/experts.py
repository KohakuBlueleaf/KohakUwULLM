"""The MoE expert path as four MXFP8 kernels instead of seven eager ops.

GEMM1 folds in the gather, SwiGLU and the requantize; GEMM2 folds in the gate
scale and the scatter-add. ``h`` is rounded to the storage dtype before the
activation, so the backward differentiates the function the forward evaluated.

See docs/internals/mxfp8.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, _quantize_block
from kohakuwullm.kernels.mxfp8.grouped import FWD_STAGES, FWD_WARPS, tile_owner

TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

# GEMM1's BLOCK_N is **per half**: one program computes the gate and value columns
# of one hidden index, so its effective N tile is 2*BLOCK_N.
GEMM1_TILE = {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}
GEMM2_TILE = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}


@triton.jit
def _experts_gemm1(
    xq_ptr,
    xs_ptr,
    token_ptr,
    wq_ptr,
    ws_ptr,
    hq_ptr,
    hs_ptr,
    pre_ptr,
    offsets_ptr,
    num_experts,
    H,
    K,
    stride_xm,
    stride_xsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    stride_hm,
    stride_hsm,
    stride_pm,
    SAVE_PRE: tl.constexpr,
    K_EXACT: tl.constexpr,
    PRE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Gather, both halves of ``x @ w_in.T``, swiglu, and requantize -- one pass.

    ``SAVE_PRE`` writes the pre-activation the backward reads; as a constexpr, an
    inference call compiles the store out.
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

    xq_ptrs = xq_ptr + rows[:, None] * stride_xm + offs_k[None, :]
    xs_ptrs = xs_ptr + rows[:, None] * stride_xsm + offs_g[None, :]
    wg_ptrs = (
        wq_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    wv_ptrs = wg_ptrs + H * stride_wn
    sg_ptrs = (
        ws_ptr + expert * stride_wse + offs_n[:, None] * stride_wsn + offs_g[None, :]
    )
    sv_ptrs = sg_ptrs + H * stride_wsn

    scale_cols = tl.cdiv(K, BLOCK_SUB)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < K
        # Scale loads carry their own column mask; K_EXACT compiles it away.
        if K_EXACT:
            mask_g = tl.full((groups,), True, tl.int1)
        else:
            mask_g = k0 * groups + offs_g < scale_cols
        xq = tl.load(xq_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        xs = tl.load(xs_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0)
        wg = tl.load(wg_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        sg = tl.load(sg_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        wv = tl.load(wv_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        sv = tl.load(sv_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        acc_g = tl.dot_scaled(xq, xs, "e4m3", tl.trans(wg), sg, "e4m3", acc=acc_g)
        acc_v = tl.dot_scaled(xq, xs, "e4m3", tl.trans(wv), sv, "e4m3", acc=acc_v)
        xq_ptrs += BLOCK_K
        xs_ptrs += groups
        wg_ptrs += BLOCK_K
        wv_ptrs += BLOCK_K
        sg_ptrs += groups
        sv_ptrs += groups

    mask_out = mask_m[:, None] & mask_n[None, :]
    # Rounded to storage before the activation. `PRE_DTYPE`, not `pre_ptr.dtype`,
    # because inference passes None.
    gate = acc_g.to(PRE_DTYPE).to(tl.float32)
    value = acc_v.to(PRE_DTYPE).to(tl.float32)
    if SAVE_PRE:
        pre_ptrs = pre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
        tl.store(pre_ptrs, gate.to(PRE_DTYPE), mask=mask_out)
        tl.store(pre_ptrs + H, value.to(PRE_DTYPE), mask=mask_out)

    hq, hs = _quantize_block(
        gate * tl.sigmoid(gate) * value, BLOCK_M, BLOCK_N, BLOCK_SUB
    )
    tl.store(hq_ptr + offs_m[:, None] * stride_hm + offs_n[None, :], hq, mask=mask_out)
    hgroups: tl.constexpr = BLOCK_N // BLOCK_SUB
    offs_hg = pid_n * hgroups + tl.arange(0, hgroups)
    tl.store(
        hs_ptr + offs_m[:, None] * stride_hsm + offs_hg[None, :],
        hs,
        mask=mask_m[:, None] & (offs_hg < tl.cdiv(H, BLOCK_SUB))[None, :],
    )


@triton.jit
def _experts_gemm2(
    hq_ptr,
    hs_ptr,
    wq_ptr,
    ws_ptr,
    out_ptr,
    token_ptr,
    order_ptr,
    gate_ptr,
    offsets_ptr,
    num_experts,
    N,
    H,
    stride_hm,
    stride_hsm,
    stride_we,
    stride_wn,
    stride_wse,
    stride_wsn,
    stride_om,
    K_EXACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """``out[token(m)] += gate(m) * (h[m] @ w_out[e(m)].T)`` -- GEMM and combine.

    No ``valid_rows`` guard, unlike the standalone combine: ``offsets`` is truncated
    to the real experts, so a sentinel bucket's rows are never visited.
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
    mask_n = offs_n < N

    hq_ptrs = hq_ptr + offs_m[:, None] * stride_hm + offs_k[None, :]
    hs_ptrs = hs_ptr + offs_m[:, None] * stride_hsm + offs_g[None, :]
    wq_ptrs = (
        wq_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    )
    ws_ptrs = (
        ws_ptr + expert * stride_wse + offs_n[:, None] * stride_wsn + offs_g[None, :]
    )

    scale_cols = tl.cdiv(H, BLOCK_SUB)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(H, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < H
        # Scale loads carry their own column mask; K_EXACT compiles it away.
        if K_EXACT:
            mask_g = tl.full((groups,), True, tl.int1)
        else:
            mask_g = k0 * groups + offs_g < scale_cols
        hq = tl.load(hq_ptrs, mask=mask_m[:, None] & mask_k[None, :])
        hs = tl.load(hs_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0)
        wq = tl.load(wq_ptrs, mask=mask_n[:, None] & mask_k[None, :])
        ws = tl.load(ws_ptrs, mask=mask_n[:, None] & mask_g[None, :], other=0)
        acc = tl.dot_scaled(hq, hs, "e4m3", tl.trans(wq), ws, "e4m3", acc=acc)
        hq_ptrs += BLOCK_K
        hs_ptrs += groups
        wq_ptrs += BLOCK_K
        ws_ptrs += groups

    rows = tl.load(token_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    pair = tl.load(order_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    scale = tl.load(gate_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
    tl.atomic_add(
        out_ptr + rows[:, None] * stride_om + offs_n[None, :],
        (acc * scale[:, None]).to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def launch_gemm1(
    xq,
    xs,
    token_of,
    wq,
    ws,
    hq,
    hs,
    pre,
    offsets,
    num_experts,
    hidden,
    dim,
    block_e,
    dtype,
):
    grid = (
        triton.cdiv(hidden, GEMM1_TILE["BLOCK_N"]),
        triton.cdiv(token_of.numel(), GEMM1_TILE["BLOCK_M"]) + num_experts,
    )
    _experts_gemm1[grid](
        xq,
        xs,
        token_of,
        wq,
        ws,
        hq,
        hs,
        pre,
        offsets,
        num_experts,
        hidden,
        dim,
        xq.stride(0),
        xs.stride(0),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        hq.stride(0),
        hs.stride(0),
        1 if pre is None else pre.stride(0),
        SAVE_PRE=pre is not None,
        K_EXACT=dim % GEMM1_TILE["BLOCK_K"] == 0,
        PRE_DTYPE=TL_DTYPE[dtype],
        BLOCK_SUB=BLOCK_SCALE,
        BLOCK_E=block_e,
        num_stages=FWD_STAGES,
        num_warps=FWD_WARPS,
        **GEMM1_TILE,
    )


def launch_gemm2(
    hq,
    hs,
    wq,
    ws,
    out,
    token_of,
    order,
    gate,
    offsets,
    num_experts,
    dim,
    hidden,
    block_e,
):
    grid = (
        triton.cdiv(dim, GEMM2_TILE["BLOCK_N"]),
        triton.cdiv(token_of.numel(), GEMM2_TILE["BLOCK_M"]) + num_experts,
    )
    _experts_gemm2[grid](
        hq,
        hs,
        wq,
        ws,
        out,
        token_of,
        order,
        gate,
        offsets,
        num_experts,
        dim,
        hidden,
        hq.stride(0),
        hs.stride(0),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        out.stride(0),
        K_EXACT=hidden % GEMM2_TILE["BLOCK_K"] == 0,
        BLOCK_SUB=BLOCK_SCALE,
        BLOCK_E=block_e,
        num_stages=FWD_STAGES,
        num_warps=FWD_WARPS,
        **GEMM2_TILE,
    )
