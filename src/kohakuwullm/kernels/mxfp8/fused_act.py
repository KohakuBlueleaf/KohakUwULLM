"""Activation epilogues that emit MXFP8 directly, so the consumer needs no cast.

The bf16 intermediate never reaches memory and the consumer GEMM's prologue does
no work. See docs/internals/mxfp8-fusion-and-attention.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8.quantize import BLOCK_SCALE, _quantize_block

ACT_TILE = {"BLOCK_M": 8, "BLOCK_K": 1024}
ACT_WARPS = 8


@triton.jit
def _swiglu_mx_kernel(
    g_ptr,
    u_ptr,
    q_ptr,
    s_ptr,
    M,
    K,
    stride_gm,
    stride_gk,
    stride_um,
    stride_uk,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
):
    """``silu(gate) * up`` quantized to e4m3 with ue8m0 blocks along K."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]

    g = tl.load(
        g_ptr + offs_m[:, None] * stride_gm + offs_k[None, :] * stride_gk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    u = tl.load(
        u_ptr + offs_m[:, None] * stride_um + offs_k[None, :] * stride_uk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    act = (g * tl.sigmoid(g) * u).to(tl.float32)

    q, s = _quantize_block(act, BLOCK_M, BLOCK_K, BLOCK_SUB)
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


def swiglu_mx(gate: torch.Tensor, up: torch.Tensor):
    """``silu(gate) * up`` as ``(e4m3 (M,K), ue8m0 (M, K//32))``; K a multiple of 32."""
    if gate.shape != up.shape:
        raise ValueError(f"gate {tuple(gate.shape)} != up {tuple(up.shape)}")
    if gate.shape[-1] % BLOCK_SCALE:
        raise ValueError(f"K={gate.shape[-1]} must be a multiple of {BLOCK_SCALE}")
    gate, up = gate.contiguous(), up.contiguous()
    shape = gate.shape
    g2, u2 = gate.reshape(-1, shape[-1]), up.reshape(-1, shape[-1])
    m, k = g2.shape
    q = torch.empty(m, k, device=gate.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(m, k // BLOCK_SCALE, device=gate.device, dtype=torch.uint8)
    grid = (triton.cdiv(m, ACT_TILE["BLOCK_M"]), triton.cdiv(k, ACT_TILE["BLOCK_K"]))
    _swiglu_mx_kernel[grid](
        g2,
        u2,
        q,
        s,
        m,
        k,
        g2.stride(0),
        g2.stride(1),
        u2.stride(0),
        u2.stride(1),
        q.stride(0),
        q.stride(1),
        s.stride(0),
        s.stride(1),
        BLOCK_SUB=BLOCK_SCALE,
        num_warps=ACT_WARPS,
        **ACT_TILE,
    )
    return q, s
