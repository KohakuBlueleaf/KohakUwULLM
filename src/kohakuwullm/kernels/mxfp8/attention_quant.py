"""Quantize a ``(T, H, D)`` attention operand to e4m3 with ue8m0 blocks along D.

Scale blocks run along the head dimension, which is the contraction axis of
``QK^T``. ``mu`` is subtracted in fp32 registers, so no widened or shifted copy of
the operand reaches memory.

See docs/internals/mxfp8-attention.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8.quantize import BLOCK_SCALE, _quantize_block


@triton.jit
def _quant_heads_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    mu_ptr,
    T,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_st,
    stride_sh,
    stride_sg,
    HEAD_DIM: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    SMOOTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """One program quantizes BLOCK_M tokens of one head."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    head = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_DIM)
    mask = offs_m < T

    x = tl.load(
        x_ptr
        + offs_m[:, None] * stride_xt
        + head * stride_xh
        + offs_d[None, :] * stride_xd,
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if SMOOTH:
        x = x - tl.load(mu_ptr + head * HEAD_DIM + offs_d)[None, :]

    q, s = _quantize_block(x, BLOCK_M, HEAD_DIM, BLOCK_SUB)
    tl.store(
        q_ptr
        + offs_m[:, None] * stride_qt
        + head * stride_qh
        + offs_d[None, :] * stride_qd,
        q,
        mask=mask[:, None],
    )
    groups: tl.constexpr = HEAD_DIM // BLOCK_SUB
    offs_g = tl.arange(0, groups)
    tl.store(
        s_ptr
        + offs_m[:, None] * stride_st
        + head * stride_sh
        + offs_g[None, :] * stride_sg,
        s,
        mask=mask[:, None],
    )


def column_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-head, per-channel mean of a ``(T, H, D)`` operand as ``(H, D)`` fp32."""
    return x.mean(dim=0, dtype=torch.float32)


def quantize_heads(x: torch.Tensor, mu: torch.Tensor | None = None, block_m: int = 32):
    """``(T, H, D)`` -> ``(e4m3 (T, H, D), ue8m0 (T, H, D // 32))``.

    ``mu`` is an ``(H, D)`` fp32 mean subtracted before quantizing.
    """
    t, heads, head_dim = x.shape
    if head_dim % BLOCK_SCALE:
        raise ValueError(f"head_dim={head_dim} must be a multiple of {BLOCK_SCALE}")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        t, heads, head_dim // BLOCK_SCALE, device=x.device, dtype=torch.uint8
    )
    _quant_heads_kernel[(triton.cdiv(t, block_m), heads)](
        x,
        q,
        s,
        mu if mu is not None else x,
        t,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        s.stride(0),
        s.stride(1),
        s.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_SUB=BLOCK_SCALE,
        SMOOTH=mu is not None,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return q, s
