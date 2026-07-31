"""Fused RMSNorm (forward + backward) in Triton.

One row per program. The forward caches ``rstd`` for the backward and accumulates
the weight gradient into per-program partials rather than a global atomic. It can
also emit its output as MXFP8, blocked along the row -- which is the K axis of the
projection that consumes it.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, _quantize_block

MX_BLOCK = tl.constexpr(BLOCK_SCALE)


@triton.jit
def _rmsnorm_fwd(
    x_ptr,
    w_ptr,
    out_ptr,
    rstd_ptr,
    q_ptr,
    s_ptr,
    stride,
    q_stride,
    s_stride,
    n_cols,
    eps,
    HAS_WEIGHT: tl.constexpr,
    WRITE_OUT: tl.constexpr,
    QUANT: tl.constexpr,
    SWIZZLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    out_ptr += row * stride
    q_ptr += row * q_stride
    if SWIZZLE:
        # cuBLAS addresses MX scales as 128-row x 4-scale-column tiles of 512 B;
        # one program owns one row, so the row contribution is a scalar.
        s_row = (row % 32) * 16 + ((row % 128) // 32) * 4
        s_ptr += (row // 128) * s_stride
    else:
        s_row = 0
        s_ptr += row * s_stride

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc) / n_cols + eps)
    tl.store(rstd_ptr + row, rstd)

    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * w
        if WRITE_OUT:
            tl.store(out_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)
        if QUANT:
            # Every 32-element block is contained in this column tile.
            q, s = _quantize_block(tl.reshape(y, (1, BLOCK)), 1, BLOCK, MX_BLOCK)
            tl.store(q_ptr + cols, tl.reshape(q, (BLOCK,)), mask=mask)
            groups: tl.constexpr = BLOCK // MX_BLOCK
            g = off // MX_BLOCK + tl.arange(0, groups)
            if SWIZZLE:
                offs = (g // 4) * 512 + s_row + (g % 4)
            else:
                offs = g
            tl.store(
                s_ptr + offs,
                tl.reshape(s, (groups,)),
                mask=g < (n_cols + MX_BLOCK - 1) // MX_BLOCK,
            )


@triton.jit
def _rmsnorm_bwd(
    x_ptr,
    w_ptr,
    dout_ptr,
    dx_ptr,
    dw_part_ptr,
    rstd_ptr,
    stride,
    n_cols,
    n_rows,
    dw_stride,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    if HAS_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        w = tl.full([BLOCK], 1.0, tl.float32)

    for row in range(pid, n_rows, n_programs):
        xr = x_ptr + row * stride
        dor = dout_ptr + row * stride
        dxr = dx_ptr + row * stride
        x = tl.load(xr + cols, mask=mask, other=0.0).to(tl.float32)
        dout = tl.load(dor + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        xhat = x * rstd
        dw_acc += dout * xhat
        # d/dx [x * rstd * w] with rstd = (mean(x^2) + eps)^-1/2
        dy = dout * w
        dx = (dy - xhat * tl.sum(dy * xhat, axis=0) / n_cols) * rstd
        tl.store(dxr + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)

    if HAS_WEIGHT:
        tl.store(dw_part_ptr + pid * dw_stride + cols, dw_acc, mask=mask)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, quant=False, write_out=True, swizzle=False):
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2d.shape
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        block = min(triton.next_power_of_2(n_cols), 16384)
        if quant and (n_cols % BLOCK_SCALE or block % BLOCK_SCALE):
            raise ValueError(f"quant needs n_cols and BLOCK % {BLOCK_SCALE} == 0")
        if swizzle and n_cols % (4 * BLOCK_SCALE):
            raise ValueError(f"swizzle needs n_cols % {4 * BLOCK_SCALE} == 0")
        out = torch.empty_like(x2d) if write_out else x2d.new_empty(0)
        if quant:
            q = torch.empty(
                n_rows, n_cols, dtype=torch.float8_e4m3fn, device=x2d.device
            )
            if swizzle:
                col_tiles = n_cols // (4 * BLOCK_SCALE)
                # Zeroed: cuBLAS still reads the padding of a partial row tile.
                s = torch.zeros(
                    -(-n_rows // 128) * 128 * col_tiles * 4,
                    dtype=torch.uint8,
                    device=x2d.device,
                )
                s_stride = col_tiles * 512
            else:
                s = torch.empty(
                    n_rows, n_cols // BLOCK_SCALE, dtype=torch.uint8, device=x2d.device
                )
                s_stride = s.stride(0)
        else:
            q = s = x2d.new_empty(0)
            s_stride = 0

        num_warps = max(4, min(16, block // 512))
        _rmsnorm_fwd[(n_rows,)](
            x2d,
            weight,
            out,
            rstd,
            q,
            s,
            x2d.stride(0),
            q.stride(0) if quant else 0,
            s_stride,
            n_cols,
            eps,
            HAS_WEIGHT=weight is not None,
            WRITE_OUT=write_out,
            QUANT=quant,
            SWIZZLE=swizzle,
            BLOCK=block,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.eps = eps
        ctx.block = block
        ctx.num_warps = num_warps
        if quant:
            # The fp8 pair carries no gradient; dgrad reaches the norm through `out`.
            ctx.mark_non_differentiable(q, s)
            return (out.reshape(shape) if write_out else out), q, s
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dout, *_unused_fp8_grads):
        x2d, weight, rstd = ctx.saved_tensors
        shape = dout.shape
        dout2d = dout.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2d.shape
        dx = torch.empty_like(x2d)

        # One partial-sum row per program, reduced once at the end.
        n_programs = min(n_rows, 512)
        has_w = weight is not None
        dw_part = (
            torch.zeros(n_programs, n_cols, dtype=torch.float32, device=x2d.device)
            if has_w
            else x2d.new_empty(0)
        )
        _rmsnorm_bwd[(n_programs,)](
            x2d,
            weight,
            dout2d,
            dx,
            dw_part,
            rstd,
            x2d.stride(0),
            n_cols,
            n_rows,
            n_cols if has_w else 0,
            HAS_WEIGHT=has_w,
            BLOCK=ctx.block,
            num_warps=ctx.num_warps,
        )
        dw = dw_part.sum(0).to(weight.dtype) if has_w else None
        return dx.reshape(shape), dw, None, None, None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6):
    """Fused RMSNorm. Falls back to ATen on CPU or for a >16k feature dim."""
    if not x.is_cuda or x.shape[-1] > 16384:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
    return _RMSNorm.apply(x, weight, eps)


def rms_norm_mx(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6):
    """Fused RMSNorm returning ``(out, e4m3, ue8m0)``, the fp8 pair K-blocked.

    No ATen fallback: there is no bf16-only form of this return shape.
    """
    return _RMSNorm.apply(x, weight, eps, True, True)


def rms_norm_mx_vendor(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6):
    """RMSNorm emitting scales already in cuBLAS's ``SWIZZLE_32_4_4`` layout.

    The variant ``torch._scaled_mm`` consumes directly. Returned scales are
    **flat**, not ``(rows, K//32)``: the layout zero-pads to whole 128-row tiles,
    so the padded extent is not the logical shape.
    """
    out, q, s = _RMSNorm.apply(x, weight, eps, True, True, True)
    return out, q, s.view(torch.float8_e8m0fnu)
