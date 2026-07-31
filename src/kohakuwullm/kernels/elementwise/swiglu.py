"""Fused SwiGLU elementwise product.

``silu(gate) * value`` as one Triton kernel, saving only ``gate`` and ``value``
and recomputing the activation in the backward. The inputs are read through their
own strides and never contiguified; the gradients are written contiguously.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w)
        for bm, bn in ((16, 128), (32, 128), (16, 256), (64, 64), (8, 512))
        for w in (4, 8)
    ]


# ``cols`` alone, never ``rows``: ``rows`` is the varlen token count, so a key
# containing it re-runs the search on every step.
AUTOTUNE_KEY = ["cols"]


def _as_2d(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``(2-D view, row stride)`` over the last axis, without copying if possible.

    The leading axes collapse into one row axis when they are contiguous *relative
    to each other*, which holds for a chunk of a contiguous tensor.
    """
    if x.stride(-1) != 1:
        x = x.contiguous()
    if x.ndim == 1:
        return x.reshape(1, -1), x.shape[0]
    shape, stride = x.shape, x.stride()
    for axis in range(x.ndim - 2):
        if stride[axis] != shape[axis + 1] * stride[axis + 1]:
            x = x.contiguous()
            break
    cols = x.shape[-1]
    rows = x.numel() // cols
    # Higher rank collapses to (rows, cols) keeping the innermost row stride.
    return (x if x.ndim == 2 else x.reshape(rows, cols)), x.stride(-2)


@triton.autotune(configs=_configs(), key=AUTOTUNE_KEY)
@triton.jit
def _swiglu_fwd(
    gate_ptr,
    value_ptr,
    out_ptr,
    rows,
    cols,
    stride_gm,
    stride_vm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < rows)[:, None] & (offs_n < cols)[None, :]

    gate = tl.load(
        gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.load(
        value_ptr + offs_m[:, None] * stride_vm + offs_n[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    out = gate * tl.sigmoid(gate) * value
    tl.store(
        out_ptr + offs_m[:, None] * cols + offs_n[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(configs=_configs(), key=AUTOTUNE_KEY)
@triton.jit
def _swiglu_bwd(
    grad_ptr,
    gate_ptr,
    value_ptr,
    dgate_ptr,
    dvalue_ptr,
    rows,
    cols,
    stride_gm,
    stride_vm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m < rows)[:, None] & (offs_n < cols)[None, :]
    flat = offs_m[:, None] * cols + offs_n[None, :]

    grad = tl.load(grad_ptr + flat, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.load(
        value_ptr + offs_m[:, None] * stride_vm + offs_n[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu = gate * sig
    # d/dgate [gate * sigmoid(gate)] = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    tl.store(
        dgate_ptr + flat,
        (grad * value * dsilu).to(dgate_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        dvalue_ptr + flat, (grad * silu).to(dvalue_ptr.dtype.element_ty), mask=mask
    )


def _grid(rows, cols):
    return lambda meta: (  # noqa: E731
        triton.cdiv(rows, meta["BLOCK_M"]),
        triton.cdiv(cols, meta["BLOCK_N"]),
    )


class _SwiGLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        gate2d, stride_gm = _as_2d(gate)
        value2d, stride_vm = _as_2d(value)
        rows, cols = gate2d.shape
        out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
        _swiglu_fwd[_grid(rows, cols)](
            gate2d, value2d, out, rows, cols, stride_gm, stride_vm
        )
        ctx.save_for_backward(gate2d, value2d)
        ctx.strides = (stride_gm, stride_vm)
        ctx.shape = gate.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate2d, value2d = ctx.saved_tensors
        stride_gm, stride_vm = ctx.strides
        rows, cols = gate2d.shape
        dgate = torch.empty(ctx.shape, dtype=gate2d.dtype, device=gate2d.device)
        dvalue = torch.empty_like(dgate)
        _swiglu_bwd[_grid(rows, cols)](
            grad_out.contiguous(),
            gate2d,
            value2d,
            dgate,
            dvalue,
            rows,
            cols,
            stride_gm,
            stride_vm,
        )
        return dgate, dvalue


def swiglu_mul(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * value``, fused. Falls back to eager on CPU."""
    if gate.shape != value.shape:
        raise ValueError(f"shape mismatch: gate {gate.shape} value {value.shape}")
    if not gate.is_cuda:
        return torch.nn.functional.silu(gate) * value
    return _SwiGLUMul.apply(gate, value)
