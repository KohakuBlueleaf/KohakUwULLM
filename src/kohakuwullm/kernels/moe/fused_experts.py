"""The routed-expert path in fp16/bf16 as fused kernels instead of seven eager ops.

GEMM1 folds in the gather, both halves of the input projection, SwiGLU and the
rounding to storage; GEMM2 folds in the gate scale and the scatter-add. Tiles come
from :mod:`kohakuwullm.kernels.moe.tune`, never from a constant here.

See docs/internals/fused-moe-16bit.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8.grouped import tile_owner

TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


@triton.jit
def _fused_gemm1(
    x_ptr,
    token_ptr,
    w_ptr,
    h_ptr,
    pre_ptr,
    offsets_ptr,
    num_experts,
    H,
    K,
    stride_xm,
    stride_we,
    stride_wn,
    stride_hm,
    stride_pm,
    SAVE_PRE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Gather, both halves of ``x @ w_in.T``, swiglu and the store -- one pass.

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
    mask_n = offs_n < H

    x_ptrs = x_ptr + rows[:, None] * stride_xm + offs_k[None, :]
    wg_ptrs = w_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]
    wv_ptrs = wg_ptrs + H * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wg = tl.load(wg_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        wv = tl.load(wv_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc_g = tl.dot(a, tl.trans(wg), acc=acc_g)
        acc_v = tl.dot(a, tl.trans(wv), acc=acc_v)
        x_ptrs += BLOCK_K
        wg_ptrs += BLOCK_K
        wv_ptrs += BLOCK_K

    mask_out = mask_m[:, None] & mask_n[None, :]
    # Rounded to storage before the activation. See docs/internals/fused-moe-16bit.md.
    gate = acc_g.to(OUT_DTYPE).to(tl.float32)
    value = acc_v.to(OUT_DTYPE).to(tl.float32)
    if SAVE_PRE:
        pre_ptrs = pre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
        tl.store(pre_ptrs, gate.to(OUT_DTYPE), mask=mask_out)
        tl.store(pre_ptrs + H, value.to(OUT_DTYPE), mask=mask_out)

    h = gate * tl.sigmoid(gate) * value
    tl.store(
        h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :],
        h.to(OUT_DTYPE),
        mask=mask_out,
    )


@triton.jit
def _fused_gemm2(
    h_ptr,
    w_ptr,
    out_ptr,
    token_ptr,
    order_ptr,
    gate_ptr,
    offsets_ptr,
    num_experts,
    N,
    H,
    stride_hm,
    stride_we,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """``out[token(m)] += gate(m) * (h[m] @ w_out[e(m)].T)`` -- GEMM and combine.

    No ``valid_rows`` guard: ``offsets`` is truncated to the real experts, so a
    sentinel bucket's rows are never visited.
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
    mask_n = offs_n < N

    h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_k[None, :]
    w_ptrs = w_ptr + expert * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(H, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < H
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(h, tl.trans(w), acc=acc)
        h_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    rows = tl.load(token_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    pair = tl.load(order_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    scale = tl.load(gate_ptr + pair, mask=mask_m, other=0.0).to(tl.float32)
    tl.atomic_add(
        out_ptr + rows[:, None] * stride_om + offs_n[None, :],
        (acc * scale[:, None]).to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _fused_gemm2_dgrad(
    dout_ptr,
    token_ptr,
    order_ptr,
    gate_ptr,
    w_ptr,
    pre_ptr,
    dpre_ptr,
    dgate_ptr,
    offsets_ptr,
    num_experts,
    H,
    N,
    stride_dm,
    stride_we,
    stride_wn,
    stride_pm,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """DGRAD through GEMM2, the combine and swiglu at once.

    The accumulator ``dout[t] @ W_out`` is consumed three ways in registers, and
    ``h`` is rebuilt from the pre-activation under the forward's rounding.
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
    mask_n = offs_n < H

    d_ptrs = dout_ptr + rows[:, None] * stride_dm + offs_k[None, :]
    # w_out is (E, N, H) and the contraction runs over N, so the tile is loaded
    # transposed rather than transposed after the load.
    w_ptrs = w_ptr + expert * stride_we + offs_k[:, None] * stride_wn + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(N, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < N
        d = tl.load(d_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(d, w, acc=acc)
        d_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wn

    mask_out = mask_m[:, None] & mask_n[None, :]
    pre_ptrs = pre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
    gate_pre = tl.load(pre_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    value = tl.load(pre_ptrs + H, mask=mask_out, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate_pre)
    h = (gate_pre * sig * value).to(OUT_DTYPE).to(tl.float32)

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
    # d/dgate [gate * sigmoid(gate)] = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dgate_t = dh * value * sig * (1.0 + gate_pre * (1.0 - sig))
    dvalue_t = dh * gate_pre * sig

    dpre_ptrs = dpre_ptr + offs_m[:, None] * stride_pm + offs_n[None, :]
    tl.store(dpre_ptrs, dgate_t.to(OUT_DTYPE), mask=mask_out)
    tl.store(dpre_ptrs + H, dvalue_t.to(OUT_DTYPE), mask=mask_out)


@triton.jit
def _fused_gemm1_dgrad(
    dpre_ptr,
    w_ptr,
    dx_ptr,
    token_ptr,
    offsets_ptr,
    num_experts,
    K,
    TWO_H,
    stride_qm,
    stride_we,
    stride_wn,
    stride_xm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    mask_n = offs_n < K

    d_ptrs = dpre_ptr + offs_m[:, None] * stride_qm + offs_k[None, :]
    w_ptrs = w_ptr + expert * stride_we + offs_k[:, None] * stride_wn + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(TWO_H, BLOCK_K)):
        mask_k = k0 * BLOCK_K + offs_k < TWO_H
        d = tl.load(d_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(d, w, acc=acc)
        d_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wn

    rows = tl.load(token_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    tl.atomic_add(
        dx_ptr + rows[:, None] * stride_xm + offs_n[None, :],
        acc.to(dx_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def launch_gemm1(x, token_of, w_in, h, pre, offsets, num_experts, block_e, plan):
    """Gather, both halves of ``x @ w_in.T``, swiglu; writes ``h`` and maybe ``pre``."""
    hidden, dim = w_in.shape[1] // 2, w_in.shape[2]
    grid = (
        triton.cdiv(hidden, plan.bn),
        triton.cdiv(token_of.numel(), plan.bm) + num_experts,
    )
    _fused_gemm1[grid](
        x,
        token_of,
        w_in,
        h,
        pre,
        offsets,
        num_experts,
        hidden,
        dim,
        x.stride(0),
        w_in.stride(0),
        w_in.stride(1),
        h.stride(0),
        1 if pre is None else pre.stride(0),
        SAVE_PRE=pre is not None,
        OUT_DTYPE=TL_DTYPE[x.dtype],
        BLOCK_E=block_e,
        num_stages=plan.stages,
        num_warps=plan.warps,
        **plan.tile,
    )


def launch_gemm2(
    h, w_out, out, token_of, order, gate, offsets, num_experts, block_e, plan
):
    """``out[token(m)] += gate(m) * (h[m] @ w_out[e(m)].T)``."""
    dim, hidden = w_out.shape[1], w_out.shape[2]
    grid = (
        triton.cdiv(dim, plan.bn),
        triton.cdiv(token_of.numel(), plan.bm) + num_experts,
    )
    _fused_gemm2[grid](
        h,
        w_out,
        out,
        token_of,
        order,
        gate,
        offsets,
        num_experts,
        dim,
        hidden,
        h.stride(0),
        w_out.stride(0),
        w_out.stride(1),
        out.stride(0),
        BLOCK_E=block_e,
        num_stages=plan.stages,
        num_warps=plan.warps,
        **plan.tile,
    )


def launch_gemm2_dgrad(
    dout,
    token_of,
    order,
    gate,
    w_out,
    pre,
    dpre,
    dgate,
    offsets,
    num_experts,
    block_e,
    plan,
):
    """DGRAD through GEMM2, the combine and swiglu; writes ``dpre`` and ``dgate``."""
    dim, hidden = w_out.shape[1], w_out.shape[2]
    grid = (
        triton.cdiv(hidden, plan.bn),
        triton.cdiv(token_of.numel(), plan.bm) + num_experts,
    )
    _fused_gemm2_dgrad[grid](
        dout,
        token_of,
        order,
        gate,
        w_out,
        pre,
        dpre,
        dgate,
        offsets,
        num_experts,
        hidden,
        dim,
        dout.stride(0),
        w_out.stride(0),
        w_out.stride(1),
        pre.stride(0),
        OUT_DTYPE=TL_DTYPE[dout.dtype],
        BLOCK_E=block_e,
        num_stages=plan.stages,
        num_warps=plan.warps,
        **plan.tile,
    )


def launch_gemm1_dgrad(dpre, w_in, dx, token_of, offsets, num_experts, block_e, plan):
    """``dx[token(m)] += dpre[m] @ w_in[e(m)]``."""
    two_h, dim = w_in.shape[1], w_in.shape[2]
    grid = (
        triton.cdiv(dim, plan.bn),
        triton.cdiv(token_of.numel(), plan.bm) + num_experts,
    )
    _fused_gemm1_dgrad[grid](
        dpre,
        w_in,
        dx,
        token_of,
        offsets,
        num_experts,
        dim,
        two_h,
        dpre.stride(0),
        w_in.stride(0),
        w_in.stride(1),
        dx.stride(0),
        BLOCK_E=block_e,
        num_stages=plan.stages,
        num_warps=plan.warps,
        **plan.tile,
    )
