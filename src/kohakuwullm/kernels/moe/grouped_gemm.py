"""Grouped GEMM for MoE expert compute.

One launch computes ``out[i] = x[i] @ W[expert_of(i)].T`` for every token ``i``,
given tokens already sorted by expert. The backward reuses the same primitive.
The grid is sized entirely from host-known values, so the layer stays
CUDA-graph capturable.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl

_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


@triton.jit
def _tile_owner(
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
    start = tl.load(offsets_ptr + safe)
    end = tl.load(offsets_ptr + safe + 1)
    return expert, start, end, local


def _fwd_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
            num_stages=s,
            num_warps=w,
        )
        for bm, bn, bk, s, w in (
            (64, 64, 32, 4, 4),
            (128, 64, 32, 4, 4),
            (64, 128, 32, 4, 4),
            (128, 128, 32, 3, 8),
            (64, 256, 32, 3, 8),
            (128, 64, 64, 3, 4),
            (64, 64, 64, 4, 4),
        )
    ]


@triton.autotune(configs=_fwd_configs(), key=["N", "K", "ACC_FP16", "COMPUTE_DTYPE"])
@triton.jit
def _grouped_gemm_fwd(
    x_ptr,
    w_ptr,
    out_ptr,
    offsets_ptr,
    N,
    K,
    num_experts,
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    ACC_FP16: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """out[m, :] = x[m, :] @ W[e(m)].T -- W is (E, N, K), row-major per expert.

    ``ACC_FP16`` selects an fp16 accumulator, safe here only because an expert's
    ``K`` is the model width and the reduction is short.
    """
    expert, start, end, local_m = _tile_owner(
        offsets_ptr, tl.program_id(1), num_experts, BLOCK_M, BLOCK_E
    )
    if expert >= num_experts:
        return
    pid_n = tl.program_id(0)

    offs_m = start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < end
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = (
        w_ptr
        + expert * stride_we
        + offs_n[:, None] * stride_wn
        + offs_k[None, :] * stride_wk
    )

    if ACC_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] + k * BLOCK_K < K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k, other=0.0)
        b = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k, other=0.0)
        if ACC_FP16:
            acc = tl.dot(a, b.T, acc=acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(
                a.to(COMPUTE_DTYPE),
                b.T.to(COMPUTE_DTYPE),
                acc=acc,
                out_dtype=tl.float32,
            )
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(configs=_fwd_configs(), key=["N", "K", "COMPUTE_DTYPE"])
@triton.jit
def _grouped_gemm_dw(
    x_ptr,
    dout_ptr,
    dw_ptr,
    offsets_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_dm,
    stride_dn,
    stride_we,
    stride_wn,
    stride_wk,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """dW[e] = dout[rows(e)].T @ x[rows(e)] -- reduction runs over the token axis."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    expert = tl.program_id(2)

    start = tl.load(offsets_ptr + expert)
    end = tl.load(offsets_ptr + expert + 1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m in range(start, end, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end
        d = tl.load(
            dout_ptr + offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        a = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(d.T.to(COMPUTE_DTYPE), a.to(COMPUTE_DTYPE), out_dtype=tl.float32)

    dw_ptrs = (
        dw_ptr
        + expert * stride_we
        + offs_n[:, None] * stride_wn
        + offs_k[None, :] * stride_wk
    )
    tl.store(
        dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :]
    )


@triton.autotune(configs=_fwd_configs(), key=["N", "K", "COMPUTE_DTYPE"])
@triton.jit
def _grouped_gemm_dx(
    dout_ptr,
    w_ptr,
    dx_ptr,
    offsets_ptr,
    N,
    K,
    num_experts,
    stride_dm,
    stride_dn,
    stride_we,
    stride_wn,
    stride_wk,
    stride_xm,
    stride_xk,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """dx[m, :] = dout[m, :] @ W[e(m)] -- same grouping, no transpose on W."""
    expert, start, end, local_m = _tile_owner(
        offsets_ptr, tl.program_id(1), num_experts, BLOCK_M, BLOCK_E
    )
    if expert >= num_experts:
        return
    pid_k = tl.program_id(0)

    offs_m = start + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < end
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        mask_n = offs_n + n * BLOCK_N < N
        d = tl.load(
            dout_ptr
            + offs_m[:, None] * stride_dm
            + (offs_n[None, :] + n * BLOCK_N) * stride_dn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr
            + expert * stride_we
            + (offs_n[:, None] + n * BLOCK_N) * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.dot(d.to(COMPUTE_DTYPE), w.to(COMPUTE_DTYPE), out_dtype=tl.float32)

    tl.store(
        dx_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        acc.to(dx_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_k[None, :],
    )


class _GroupedGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, offsets, acc_fp16, compute_dtype):
        # x: (M, K) sorted by expert; weight: (E, N, K); offsets: (E+1,) int32
        m, k = x.shape
        num_experts, n, _ = weight.shape
        block_e = max(16, triton.next_power_of_2(num_experts))
        out = torch.empty(m, n, device=x.device, dtype=x.dtype)
        # N first, then the flat row-tile axis: cdiv(m, BLOCK_M) + E bounds
        # sum_e cdiv(count_e, BLOCK_M), so no device value enters the launch.
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(n, meta["BLOCK_N"]),
            triton.cdiv(m, meta["BLOCK_M"]) + num_experts,
        )
        _grouped_gemm_fwd[grid](
            x,
            weight,
            out,
            offsets,
            n,
            k,
            num_experts,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            out.stride(0),
            out.stride(1),
            ACC_FP16=acc_fp16,
            COMPUTE_DTYPE=_TL_DTYPE[compute_dtype],
            BLOCK_E=block_e,
        )
        ctx.save_for_backward(x, weight, offsets)
        ctx.compute_dtype = compute_dtype
        ctx.block_e = block_e
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, offsets = ctx.saved_tensors
        dout = dout.contiguous()
        num_experts, n, k = weight.shape
        m = x.shape[0]

        dx = torch.empty_like(x)
        grid_dx = lambda meta: (  # noqa: E731
            triton.cdiv(k, meta["BLOCK_K"]),
            triton.cdiv(m, meta["BLOCK_M"]) + num_experts,
        )
        _grouped_gemm_dx[grid_dx](
            dout,
            weight,
            dx,
            offsets,
            n,
            k,
            num_experts,
            dout.stride(0),
            dout.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            dx.stride(0),
            dx.stride(1),
            COMPUTE_DTYPE=_TL_DTYPE[ctx.compute_dtype],
            BLOCK_E=ctx.block_e,
        )

        dw = torch.zeros_like(weight, dtype=torch.float32)
        grid_dw = lambda meta: (  # noqa: E731
            triton.cdiv(n, meta["BLOCK_N"]),
            triton.cdiv(k, meta["BLOCK_K"]),
            num_experts,
        )
        _grouped_gemm_dw[grid_dw](
            x,
            dout,
            dw,
            offsets,
            n,
            k,
            x.stride(0),
            x.stride(1),
            dout.stride(0),
            dout.stride(1),
            dw.stride(0),
            dw.stride(1),
            dw.stride(2),
            COMPUTE_DTYPE=_TL_DTYPE[ctx.compute_dtype],
        )
        return dx, dw.to(weight.dtype), None, None, None


def grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    acc_fp16: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``out[i] = x[i] @ weight[e(i)].T`` for expert-sorted ``x``.

    Args:
        x: ``(M, K)`` tokens, sorted so each expert owns a contiguous row range.
        weight: ``(E, N, K)`` expert matrices.
        offsets: ``(E+1,)`` int32 exclusive prefix sum of per-expert row counts.
            The only routing input, and it is never read on the host: the grid is
            bounded from ``M`` and ``E`` alone so the call can be graph-captured.
        compute_dtype: dtype the multiply runs in; the accumulator stays fp32.
            ``None`` keeps a 16-bit input as it is and lifts fp32 to bf16.
            ``torch.float32`` forces the vector cores and is a measurement
            baseline, not a training setting.
        acc_fp16: accumulate the forward in fp16 (fp16 inputs only). The backward
            stays fp32-accumulate either way.
    """
    if compute_dtype is None:
        compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
        if compute_dtype is torch.float32:
            compute_dtype = torch.bfloat16
    if acc_fp16 and x.dtype is not torch.float16:
        raise ValueError(f"acc_fp16 requires fp16 inputs, got {x.dtype}")
    # The row-tile axis is grid dimension 1, capped at 65535; the smallest BLOCK_M
    # in the autotune set decides the worst case.
    worst_case_tiles = triton.cdiv(x.shape[0], 64) + weight.shape[0]
    if worst_case_tiles > 65535:
        raise ValueError(
            f"{x.shape[0]} rows over {weight.shape[0]} experts needs "
            f"{worst_case_tiles} row tiles, past the 65535 grid limit"
        )
    return _GroupedGEMM.apply(x, weight, offsets, acc_fp16, compute_dtype)


def grouped_gemm_reference(
    x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Loop-of-GEMMs reference. Slow; used by tests to pin the Triton kernel."""
    outs = []
    off = offsets.tolist()
    for e in range(weight.shape[0]):
        rows = x[off[e] : off[e + 1]]
        outs.append(rows @ weight[e].T)
    return torch.cat(outs, dim=0)
