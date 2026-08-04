"""Autograd wiring for the fused 16-bit expert path.

The kernels live in :mod:`kohakuwullm.kernels.moe.fused_experts` and their tiles
in :mod:`kohakuwullm.kernels.moe.tune`; what is here is the graph they form.

See docs/internals/fused-moe-16bit.md.
"""

import torch
import triton

from kohakuwullm.kernels.moe.fused_experts import (
    TL_DTYPE,
    launch_gemm1,
    launch_gemm1_dgrad,
    launch_gemm2,
    launch_gemm2_dgrad,
)
from kohakuwullm.kernels.moe.tune import tuned_plans
from kohakuwullm.kernels.mxfp8.grouped import (
    WGRAD_STAGES,
    WGRAD_TILE,
    WGRAD_WARPS,
    grouped_mxfp8_wgrad_kernel,
)


def _launch_wgrad(a, b, index, gate, order, dw, offsets, n, k, a_gather, b_gather):
    """``dW[e] = A[rows(e)].T @ B[rows(e)]`` -- the 16-bit arm of the shared kernel.

    ``bs`` is unused under ``B_FP8=False``, so ``b`` stands in for the scale pointer.
    """
    grid = (
        triton.cdiv(n, WGRAD_TILE["BLOCK_N"]),
        triton.cdiv(k, WGRAD_TILE["BLOCK_K"]),
        dw.shape[0],
    )
    grouped_mxfp8_wgrad_kernel[grid](
        a,
        b,
        b,
        index,
        gate,
        order,
        dw,
        offsets,
        n,
        k,
        a.stride(0),
        b.stride(0),
        1,
        dw.stride(0),
        dw.stride(1),
        A_GATHER=a_gather,
        B_GATHER=b_gather,
        B_FP8=False,
        MUL_DTYPE=TL_DTYPE[a.dtype],
        BLOCK_SUB=WGRAD_TILE["BLOCK_K"],
        num_stages=WGRAD_STAGES,
        num_warps=WGRAD_WARPS,
        **WGRAD_TILE,
    )


class _FusedExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_in, w_out, gate, token_of, order, offsets, hidden, plans):
        rows, dim = x.shape
        num_experts = w_in.shape[0]
        block_e = max(16, triton.next_power_of_2(num_experts))
        m = token_of.numel()
        # `ctx.needs_input_grad`, not `torch.is_grad_enabled()`: autograd runs
        # `forward` under `no_grad`, so the latter is always False here.
        needs_grad = any(ctx.needs_input_grad)

        h = torch.empty(m, hidden, device=x.device, dtype=x.dtype)
        # `empty` for the rows a sentinel bucket owns: `offsets` is truncated to the
        # real experts, so no tile the grid resolves covers them.
        pre = (
            torch.empty(m, 2 * hidden, device=x.device, dtype=x.dtype)
            if needs_grad
            else None
        )
        launch_gemm1(
            x, token_of, w_in, h, pre, offsets, num_experts, block_e, plans.gemm1
        )
        out = torch.zeros(rows, dim, device=x.device, dtype=x.dtype)
        launch_gemm2(
            h,
            w_out,
            out,
            token_of,
            order,
            gate,
            offsets,
            num_experts,
            block_e,
            plans.gemm2,
        )

        if needs_grad:
            ctx.save_for_backward(
                x, h, pre, gate, token_of, order, offsets, w_in, w_out
            )
            ctx.shapes = (hidden, dim, num_experts, m)
            ctx.block_e = block_e
            ctx.plans = plans
            ctx.rows = rows
        return out

    @staticmethod
    def backward(ctx, dout):
        x, h, pre, gate, token_of, order, offsets, w_in, w_out = ctx.saved_tensors
        hidden, dim, num_experts, m = ctx.shapes
        block_e, plans = ctx.block_e, ctx.plans
        dout = dout.contiguous()

        dpre = torch.empty(m, 2 * hidden, device=dout.device, dtype=dout.dtype)
        # fp32 and zeroed: this reduction crosses programs through an atomic.
        dgate = torch.zeros(gate.numel(), device=dout.device, dtype=torch.float32)
        launch_gemm2_dgrad(
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
            plans.gemm2_dgrad,
        )

        dx = torch.zeros(ctx.rows, dim, device=dout.device, dtype=dout.dtype)
        launch_gemm1_dgrad(
            dpre, w_in, dx, token_of, offsets, num_experts, block_e, plans.gemm1_dgrad
        )

        # Caller's dtype, not fp32: each element has exactly one writing program, so
        # the fp32 reduction happens in registers and the epilogue rounds.
        dw_out = torch.empty(
            num_experts, dim, hidden, device=dout.device, dtype=dout.dtype
        )
        _launch_wgrad(
            dout,
            h,
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
        # No gate under `A_GATHER=False`: `dpre` already carries it.
        _launch_wgrad(
            dpre,
            x,
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
        return dx, dw_in, dw_out, dgate.to(gate.dtype), None, None, None, None, None


def fused_moe_experts(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    gate: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """The whole routed-expert path in fp16/bf16: gather, SwiGLU MLP, gate, combine.

    Returns the combined ``(T, dim)`` output directly; there is no intermediate
    ``(T*top_k, dim)`` tensor. Tiles are resolved once per routing shape by
    :func:`kohakuwullm.kernels.moe.tune.tuned_plans` and cached.

    Args:
        x: ``(T, dim)`` activations, one row per token, **not** gathered. Must be
            unit-stride on its last axis and 16-bit; both are refused, not repaired.
        w_in: ``(E, 2H, dim)`` expert input projections, gate half first.
        w_out: ``(E, dim, H)`` expert output projections.
        gate: ``(T*top_k,)`` gate weights indexed by pair index.
        token_of: ``(M,)`` int32, sorted position -> token index.
        order: ``(M,)`` int32, sorted position -> pair index.
        offsets: ``(E+1,)`` int32 exclusive prefix sum of per-expert row counts,
            **truncated to the real experts**, so a sentinel bucket's rows fall
            outside every tile the grid resolves. Never read on the host.
    """
    hidden = w_out.shape[2]
    if w_in.shape[1] != 2 * hidden:
        raise ValueError(f"w_in {tuple(w_in.shape)} is not 2x w_out's hidden {hidden}")
    if w_in.shape[2] != x.shape[1] or w_out.shape[1] != x.shape[1]:
        raise ValueError(f"width mismatch: x {tuple(x.shape)} vs {tuple(w_in.shape)}")
    if token_of.numel() != order.numel():
        raise ValueError(f"token_of {token_of.numel()} != order {order.numel()}")
    if x.stride(-1) != 1:
        raise ValueError(
            f"x must be contiguous along its last axis, got strides {x.stride()}; "
            "WGRAD indexes the contraction axis without a stride"
        )
    if x.dtype not in TL_DTYPE:
        raise ValueError(f"x.dtype={x.dtype} is not one of {tuple(TL_DTYPE)}")
    if w_in.dtype != x.dtype or w_out.dtype != x.dtype or gate.dtype != x.dtype:
        raise ValueError(
            f"every operand must share x.dtype={x.dtype}; got w_in {w_in.dtype}, "
            f"w_out {w_out.dtype}, gate {gate.dtype}"
        )

    num_experts = w_in.shape[0]
    plans = tuned_plans(token_of.numel(), x.shape[1], hidden, num_experts, x.dtype)
    # The row-tile axis is grid dimension 1, capped at 65535.
    worst = triton.cdiv(token_of.numel(), plans.min_block_m) + num_experts
    if worst > 65535:
        raise ValueError(
            f"{token_of.numel()} rows over {num_experts} experts needs {worst} row "
            f"tiles at BLOCK_M={plans.min_block_m}, past the 65535 grid limit"
        )
    return _FusedExperts.apply(
        x, w_in, w_out, gate, token_of, order, offsets, hidden, plans
    )


def fused_moe_experts_reference(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    gate: torch.Tensor,
    token_of: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Autograd-composed equivalent: routing, gather, SwiGLU and combine."""
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
