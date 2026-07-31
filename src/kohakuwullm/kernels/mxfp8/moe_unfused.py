"""The MoE expert path in MXFP8, composed out of verified pieces instead of fused.

Six launches with the same signature and arithmetic as
:func:`~kohakuwullm.kernels.mxfp8.moe.mxfp8_moe_experts`, so the two are
exchangeable in an A/B and share one ``MXFP8ExpertWeights``. Every GEMM is
:func:`~kohakuwullm.kernels.mxfp8.grouped.grouped_mxfp8_gemm`; WGRAD's B operand
stays 16-bit.

See docs/internals/mxfp8.md.
"""

import torch

from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed
from kohakuwullm.kernels.mxfp8 import quantize_mx
from kohakuwullm.kernels.mxfp8.grouped import grouped_mxfp8_gemm
from kohakuwullm.kernels.mxfp8.moe import (
    MXFP8ExpertWeights,
    _launch_wgrad,
    _validate_expert_call,
)


class _GroupedExpertMatmul(torch.autograd.Function):
    """``out[m] = x[index[m]] @ w[e(m)].T`` over expert-sorted rows, in MXFP8.

    The only autograd node this path adds. ``w`` is passed but never read: the
    forward consumes the quantized copies, and the parameter is here so autograd
    routes ``dW`` back to the master tensor.
    """

    @staticmethod
    def forward(ctx, x, w, wq, ws, wtq, wts, offsets, index, sentinel):
        del w
        xq, xs = quantize_mx(x)
        rows = xq.shape[0] if index is None else index.numel()
        out = grouped_mxfp8_gemm(
            xq,
            xs,
            wq,
            ws,
            offsets,
            index=index,
            out_dtype=x.dtype,
            out=_maybe_zeros(sentinel, rows, wq.shape[1], x),
        )
        # `ctx.needs_input_grad`, not `torch.is_grad_enabled()`: autograd runs
        # `forward` under `no_grad`, so the latter is always False here.
        if any(ctx.needs_input_grad):
            # `x` in 16 bits, not `xq`/`xs`: WGRAD multiplies this operand.
            ctx.save_for_backward(x, wtq, wts, offsets, index)
            ctx.shape = wq.shape
            ctx.sentinel = sentinel
        return out

    @staticmethod
    def backward(ctx, dout):
        x, wtq, wts, offsets, index = ctx.saved_tensors
        num_experts, n, k = ctx.shape
        dout = dout.contiguous()

        dx = None
        if ctx.needs_input_grad[0]:
            dq, ds = quantize_mx(dout)
            # `wtq`/`wts` are the K-minor transpose: DGRAD contracts N.
            dx_sorted = grouped_mxfp8_gemm(
                dq,
                ds,
                wtq,
                wts,
                offsets,
                out_dtype=dout.dtype,
                out=_maybe_zeros(ctx.sentinel, dout.shape[0], k, dout),
            )
            dx = (
                dx_sorted
                if index is None
                else torch.zeros_like(x).index_add_(0, index.long(), dx_sorted)
            )

        dw = None
        if ctx.needs_input_grad[1]:
            # The caller's dtype: each `(expert, n, k)` tile has exactly one writing
            # program, so the fp32 reduction happens in registers.
            dw = torch.empty(num_experts, n, k, device=dout.device, dtype=dout.dtype)
            # Placeholder for the gate and order pointers, read only under `A_GATHER`.
            stand_in = offsets if index is None else index
            _launch_wgrad(
                dout,
                x,
                None,
                stand_in,
                stand_in,
                stand_in,
                dw,
                offsets,
                n,
                k,
                a_gather=False,
                b_gather=index is not None,
            )
        return dx, dw, None, None, None, None, None, None, None


def _maybe_zeros(
    sentinel: bool, rows: int, cols: int, like: torch.Tensor
) -> torch.Tensor | None:
    """A zeroed ``(rows, cols)`` buffer when a sentinel bucket exists, else ``None``.

    A sentinel bucket's rows lie outside every tile the grid resolves, so they are
    never written and a later ``index_add_`` would spread their garbage.
    """
    if not sentinel:
        return None
    return torch.zeros(rows, cols, device=like.device, dtype=like.dtype)


def mxfp8_moe_experts_unfused(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    gate: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    packed: MXFP8ExpertWeights,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """The routed-expert path in MXFP8 as six launches instead of one fused graph.

    Args:
        x: ``(T, dim)`` activations, one row per token, **not** gathered.
        w_in: ``(E, 2H, dim)`` master weights. Passed for the gradient only; the
            forward reads ``packed``.
        w_out: ``(E, dim, H)`` master weights, likewise.
        gate: ``(T*top_k,)`` gate weights indexed by pair index.
        token_of: ``(M,)`` int32, sorted position -> token index.
        order: ``(M,)`` int32, sorted position -> pair index.
        offsets: ``(E+1,)`` int32 exclusive prefix sum of per-expert row counts,
            **truncated to the real experts**. Never read on the host.
        packed: the four quantized copies, refreshed after each optimizer step.
        valid_rows: one-element device tensor bounding the sorted positions that a
            real expert owns, or ``None`` when every row does. Passing it turns on
            the zero-filled GEMM buffers -- see :func:`_maybe_zeros`.
    """
    _validate_expert_call(x, w_in, w_out, token_of, order)
    sentinel = valid_rows is not None
    pre = _GroupedExpertMatmul.apply(
        x, w_in, *packed.in_fwd, *packed.in_dgrad, offsets, token_of, sentinel
    )
    # `swiglu_mul`, not `F.silu(a) * b`: it reads the chunks through their strides.
    gate_h, value = pre.chunk(2, dim=-1)
    h = swiglu_mul(gate_h, value)
    out_sorted = _GroupedExpertMatmul.apply(
        h, w_out, *packed.out_fwd, *packed.out_dgrad, offsets, None, sentinel
    )
    return combine_routed(out_sorted, gate, order, token_of, x.shape[0], valid_rows)
