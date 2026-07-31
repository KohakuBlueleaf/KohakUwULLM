"""Autograd wiring for the fused MXFP8 expert path, and the weights it reads.

The kernels live in :mod:`kohakuwullm.kernels.mxfp8.experts`; what is here is the
graph they form and the four quantized weight copies they consume -- each expert
matrix blocked along K for FPROP and along its transpose for DGRAD.

See docs/internals/mxfp8.md.
"""

import torch
import triton

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, quantize_mx
from kohakuwullm.kernels.mxfp8.experts import TL_DTYPE, launch_gemm1, launch_gemm2
from kohakuwullm.kernels.mxfp8.experts_bwd import (
    GEMM1_DGRAD_STAGES,
    GEMM1_DGRAD_TILE,
    GEMM1_DGRAD_WARPS,
    GEMM2_DGRAD_STAGES,
    GEMM2_DGRAD_TILE,
    GEMM2_DGRAD_WARPS,
    _experts_gemm1_dgrad,
    _experts_gemm2_dgrad,
)
from kohakuwullm.kernels.mxfp8.grouped import (
    WGRAD_STAGES,
    WGRAD_TILE,
    WGRAD_WARPS,
    _block_e,
    grouped_mxfp8_wgrad_kernel,
)


def _quantize_stacked(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``(E, N, K)`` row by row, keeping the expert axis. Blocks along K."""
    e, n, k = w.shape
    q, s = quantize_mx(w.reshape(e * n, k))
    return q.view(e, n, k), s.view(e, n, k // BLOCK_SCALE)


class MXFP8ExpertWeights:
    """The four fp8 copies of one MoE layer's expert matrices.

    Not an ``nn.Module``: it owns no parameters, only buffers derived from them, so
    they stay out of ``state_dict``.
    """

    __slots__ = ("in_fwd", "in_dgrad", "out_fwd", "out_dgrad")

    def __init__(self, w_in: torch.Tensor, w_out: torch.Tensor) -> None:
        self.refresh(w_in, w_out)

    @torch.no_grad()
    def refresh(self, w_in: torch.Tensor, w_out: torch.Tensor) -> None:
        """Rebuild every copy from the 16-bit masters. Call after each optimizer step."""
        self.in_fwd = _quantize_stacked(w_in)
        self.out_fwd = _quantize_stacked(w_out)
        # The DGRAD copies are the transposes: `dx` contracts the 2H axis of `w_in`
        # and `dh` the model axis of `w_out`.
        self.in_dgrad = _quantize_stacked(w_in.transpose(1, 2).contiguous())
        self.out_dgrad = _quantize_stacked(w_out.transpose(1, 2).contiguous())


def _launch_wgrad(a, bq, bs, index, gate, order, dw, offsets, n, k, a_gather, b_gather):
    """``bs is None`` means ``bq`` is already 16-bit and carries no MX scale plane.

    ``bq`` then stands in for the scale pointer, which ``B_FP8=False`` compiles out.
    """
    grid = (
        triton.cdiv(n, WGRAD_TILE["BLOCK_N"]),
        triton.cdiv(k, WGRAD_TILE["BLOCK_K"]),
        dw.shape[0],
    )
    grouped_mxfp8_wgrad_kernel[grid](
        a,
        bq,
        bq if bs is None else bs,
        index,
        gate,
        order,
        dw,
        offsets,
        n,
        k,
        a.stride(0),
        bq.stride(0),
        1 if bs is None else bs.stride(0),
        dw.stride(0),
        dw.stride(1),
        A_GATHER=a_gather,
        B_GATHER=b_gather,
        B_FP8=bs is not None,
        # From `a`, not `dw`: this is the multiply's operand precision.
        MUL_DTYPE=TL_DTYPE[a.dtype],
        BLOCK_SUB=BLOCK_SCALE,
        num_stages=WGRAD_STAGES,
        num_warps=WGRAD_WARPS,
        **WGRAD_TILE,
    )


def _launch_gemm2_dgrad(ctx_shapes, tensors, offsets, block_e):
    hidden, dim, num_experts, m = ctx_shapes
    dq, ds, token_of, order, gate, wq, ws, hq, hs, pre, dpre, dpq, dps, dgate = tensors
    grid = (
        triton.cdiv(hidden, GEMM2_DGRAD_TILE["BLOCK_N"]),
        triton.cdiv(m, GEMM2_DGRAD_TILE["BLOCK_M"]) + num_experts,
    )
    _experts_gemm2_dgrad[grid](
        dq,
        ds,
        token_of,
        order,
        gate,
        wq,
        ws,
        hq,
        hs,
        pre,
        dpre,
        dpq,
        dps,
        dgate,
        offsets,
        num_experts,
        hidden,
        dim,
        dq.stride(0),
        ds.stride(0),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        hq.stride(0),
        hs.stride(0),
        pre.stride(0),
        dpq.stride(0),
        dps.stride(0),
        K_EXACT=dim % GEMM2_DGRAD_TILE["BLOCK_K"] == 0,
        BLOCK_SUB=BLOCK_SCALE,
        BLOCK_E=block_e,
        num_stages=GEMM2_DGRAD_STAGES,
        num_warps=GEMM2_DGRAD_WARPS,
        **GEMM2_DGRAD_TILE,
    )


def _launch_gemm1_dgrad(dpq, dps, wq, ws, dx, token_of, offsets, shapes, block_e):
    hidden, dim, num_experts, m = shapes
    grid = (
        triton.cdiv(dim, GEMM1_DGRAD_TILE["BLOCK_N"]),
        triton.cdiv(m, GEMM1_DGRAD_TILE["BLOCK_M"]) + num_experts,
    )
    _experts_gemm1_dgrad[grid](
        dpq,
        dps,
        wq,
        ws,
        dx,
        token_of,
        offsets,
        num_experts,
        dim,
        2 * hidden,
        dpq.stride(0),
        dps.stride(0),
        wq.stride(0),
        wq.stride(1),
        ws.stride(0),
        ws.stride(1),
        dx.stride(0),
        K_EXACT=(2 * hidden) % GEMM1_DGRAD_TILE["BLOCK_K"] == 0,
        BLOCK_SUB=BLOCK_SCALE,
        BLOCK_E=block_e,
        num_stages=GEMM1_DGRAD_STAGES,
        num_warps=GEMM1_DGRAD_WARPS,
        **GEMM1_DGRAD_TILE,
    )


class _MXFP8Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_in, w_out, gate, token_of, order, offsets, packed, hidden):
        del w_in, w_out  # present so autograd routes dW to the master parameters
        rows, dim = x.shape
        num_experts = packed.in_fwd[0].shape[0]
        block_e = _block_e(num_experts)
        m = token_of.numel()
        # `ctx.needs_input_grad`, not `torch.is_grad_enabled()`: autograd runs
        # `forward` under `no_grad`, so the latter is always False here.
        needs_grad = any(ctx.needs_input_grad)

        xq, xs = quantize_mx(x)
        hq = torch.empty(m, hidden, device=x.device, dtype=torch.float8_e4m3fn)
        hs = torch.empty(m, hidden // BLOCK_SCALE, device=x.device, dtype=torch.uint8)
        # `empty` for the rows a sentinel bucket owns: `offsets` is truncated to the
        # real experts, so no tile the grid resolves covers them.
        pre = (
            torch.empty(m, 2 * hidden, device=x.device, dtype=x.dtype)
            if needs_grad
            else None
        )
        launch_gemm1(
            xq,
            xs,
            token_of,
            *packed.in_fwd,
            hq,
            hs,
            pre,
            offsets,
            num_experts,
            hidden,
            dim,
            block_e,
            x.dtype,
        )
        out = torch.zeros(rows, dim, device=x.device, dtype=x.dtype)
        launch_gemm2(
            hq,
            hs,
            *packed.out_fwd,
            out,
            token_of,
            order,
            gate,
            offsets,
            num_experts,
            dim,
            hidden,
            block_e,
        )

        if needs_grad:
            # `x` in 16 bits, not `xq`/`xs`: `dw_in`'s WGRAD multiplies this operand.
            ctx.save_for_backward(x, hq, hs, pre, gate, token_of, order, offsets)
            ctx.packed = packed
            ctx.shapes = (hidden, dim, num_experts, m)
            ctx.block_e = block_e
            ctx.rows = rows
        return out

    @staticmethod
    def backward(ctx, dout):
        x, hq, hs, pre, gate, token_of, order, offsets = ctx.saved_tensors
        packed = ctx.packed
        hidden, dim, num_experts, m = ctx.shapes
        dout = dout.contiguous()
        dq, ds = quantize_mx(dout)

        dpre = torch.empty(m, 2 * hidden, device=dout.device, dtype=dout.dtype)
        dpq = torch.empty(m, 2 * hidden, device=dout.device, dtype=torch.float8_e4m3fn)
        dps = torch.empty(
            m, 2 * hidden // BLOCK_SCALE, device=dout.device, dtype=torch.uint8
        )
        # fp32 and zeroed: this reduction crosses programs through an atomic.
        dgate = torch.zeros(gate.numel(), device=dout.device, dtype=torch.float32)
        _launch_gemm2_dgrad(
            ctx.shapes,
            (
                dq,
                ds,
                token_of,
                order,
                gate,
                *packed.out_dgrad,
                hq,
                hs,
                pre,
                dpre,
                dpq,
                dps,
                dgate,
            ),
            offsets,
            ctx.block_e,
        )

        dx = torch.zeros(ctx.rows, dim, device=dout.device, dtype=dout.dtype)
        _launch_gemm1_dgrad(
            dpq, dps, *packed.in_dgrad, dx, token_of, offsets, ctx.shapes, ctx.block_e
        )

        # Caller's dtype, not fp32: each element has exactly one writing program, so
        # the fp32 reduction happens in registers and the epilogue rounds.
        dw_out = torch.empty(
            num_experts, dim, hidden, device=dout.device, dtype=dout.dtype
        )
        # Rebuilt from the stored pre-activation rather than read back as fp8.
        pre_gate, pre_value = pre.chunk(2, dim=-1)
        h16 = torch.nn.functional.silu(pre_gate) * pre_value
        _launch_wgrad(
            dout,
            h16,
            None,
            token_of,
            gate,
            order,
            dw_out,
            offsets,
            dim,
            hidden,
            a_gather=True,
            b_gather=False,
        )
        dw_in = torch.empty(
            num_experts, 2 * hidden, dim, device=dout.device, dtype=dout.dtype
        )
        # `x` in 16 bits with `bs=None`: the operand is the activation, not the fp8
        # copy GEMM1 consumed.
        _launch_wgrad(
            dpre,
            x,
            None,
            token_of,
            gate,
            order,
            dw_in,
            offsets,
            2 * hidden,
            dim,
            a_gather=False,
            b_gather=True,
        )
        # `dgate` is cast on the host: its epilogue is a cross-program `atomic_add`.
        return (
            dx,
            dw_in,
            dw_out,
            dgate.to(gate.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def _validate_expert_call(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
) -> int:
    """Shared contract of the fused and unfused expert entry points; returns ``hidden``.

    ``x`` must be unit-stride on its last axis and 16-bit; both are refused rather
    than repaired.
    """
    hidden = w_out.shape[2]
    if w_in.shape[1] != 2 * hidden:
        raise ValueError(f"w_in {tuple(w_in.shape)} is not 2x w_out's hidden {hidden}")
    if w_in.shape[2] != x.shape[1] or w_out.shape[1] != x.shape[1]:
        raise ValueError(f"width mismatch: x {tuple(x.shape)} vs {tuple(w_in.shape)}")
    for name, value in (("dim", x.shape[1]), ("hidden", hidden)):
        if value % BLOCK_SCALE:
            raise ValueError(f"{name}={value} must be a multiple of {BLOCK_SCALE}")
    if token_of.numel() != order.numel():
        raise ValueError(f"token_of {token_of.numel()} != order {order.numel()}")
    if x.stride(-1) != 1:
        raise ValueError(
            f"x must be contiguous along its last axis, got strides {x.stride()}; "
            "WGRAD indexes the contraction axis without a stride"
        )
    if x.dtype not in TL_DTYPE:
        raise ValueError(
            f"x.dtype={x.dtype} is not one of {tuple(TL_DTYPE)}; the compute dtype is "
            "the caller's to choose -- consult autocast, as `MoEMLP._routed_mxfp8` does"
        )
    return hidden


def mxfp8_moe_experts(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    gate: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    packed: MXFP8ExpertWeights,
) -> torch.Tensor:
    """The whole routed-expert path in MXFP8: gather, SwiGLU MLP, gate, combine.

    Returns the combined ``(T, dim)`` output directly; there is no intermediate
    ``(T*top_k, ...)`` tensor.

    Args:
        x: ``(T, dim)`` activations, one row per token, **not** gathered.
        w_in: ``(E, 2H, dim)`` master weights. Passed for the gradient only; the
            forward reads ``packed``.
        w_out: ``(E, dim, H)`` master weights, likewise.
        gate: ``(T*top_k,)`` gate weights indexed by pair index.
        token_of: ``(M,)`` int32, sorted position -> token index.
        order: ``(M,)`` int32, sorted position -> pair index.
        offsets: ``(E+1,)`` int32 exclusive prefix sum of per-expert row counts,
            **truncated to the real experts**, so a sentinel bucket's rows fall
            outside every tile the grid resolves. Never read on the host.
        packed: the four quantized copies, refreshed after each optimizer step.
    """
    hidden = _validate_expert_call(x, w_in, w_out, token_of, order)
    return _MXFP8Experts.apply(
        x, w_in, w_out, gate, token_of, order, offsets, packed, hidden
    )


def mxfp8_moe_experts_reference(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    gate: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Autograd-composed equivalent in the input dtype, with no quantization.

    The structural reference: routing, gather, SwiGLU and combine, with gradients
    from autograd rather than from any kernel here.
    """
    rows = x.index_select(0, token_of.long())
    off = offsets.tolist()
    outs = []
    for e in range(w_in.shape[0]):
        block = rows[off[e] : off[e + 1]]
        gate_h, value = (block @ w_in[e].T).chunk(2, dim=-1)
        outs.append((torch.nn.functional.silu(gate_h) * value) @ w_out[e].T)
    sorted_out = torch.cat(outs, dim=0)
    real = sorted_out.shape[0]
    scaled = sorted_out * gate.index_select(0, order[:real].long()).reshape(-1, 1)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    return out.index_add(0, token_of[:real].long(), scaled)
