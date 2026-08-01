"""Vendor MXFP8 ABI: the scale swizzle cuBLAS wants, and a thin call wrapper.

One definition of the layout contract ``torch.nn.functional.scaled_mm`` imposes,
shared by the precision test and the benchmark.

See docs/internals/mxfp8.md.
"""

import torch
import torch.nn.functional as F

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE

# cuBLAS reads MX scale factors as 128-row x 4-scale-column tiles, each tile
# further interleaved 32x4x4.
SCALE_ROW_TILE = 128
SCALE_COL_TILE = 4


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def swizzle_mx_scales(scales: torch.Tensor) -> torch.Tensor:
    """``(rows, K//32)`` ue8m0 bytes -> the flat ``SWIZZLE_32_4_4`` layout.

    Zero-pads to whole 128x4 tiles, so the result is flat: the padded extent is
    not the logical shape.
    """
    rows, cols = scales.shape
    row_tiles = _ceil_div(rows, SCALE_ROW_TILE)
    col_tiles = _ceil_div(cols, SCALE_COL_TILE)
    padded_rows = row_tiles * SCALE_ROW_TILE
    padded_cols = col_tiles * SCALE_COL_TILE
    if (rows, cols) != (padded_rows, padded_cols):
        padded = scales.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols] = scales
        scales = padded
    tiles = scales.view(row_tiles, SCALE_ROW_TILE, col_tiles, SCALE_COL_TILE)
    tiles = tiles.permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def expected_swizzled_numel(rows: int, k: int) -> int:
    """What ``scaled_mm`` checks the swizzled scale count against."""
    row_tiles = _ceil_div(rows, SCALE_ROW_TILE) * SCALE_ROW_TILE
    col_tiles = _ceil_div(_ceil_div(k, BLOCK_SCALE), SCALE_COL_TILE) * SCALE_COL_TILE
    return row_tiles * col_tiles


def vendor_mxfp8_matmul_swizzled(
    aq: torch.Tensor,
    a_swizzled: torch.Tensor,
    bq: torch.Tensor,
    b_swizzled: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``aq @ bq.T`` from operands whose scales are **already** swizzled.

    The primitive a benchmark must time: swizzling inside the timed call charges
    the GEMM for two full layout copies.
    """
    scaling = torch._C._ScalingType.BlockWise1x32
    swizzle = torch._C._SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        aq,
        bq.T,
        a_swizzled,
        scaling,
        b_swizzled,
        scaling,
        swizzle,
        swizzle,
        output_dtype=out_dtype,
    )


def as_vendor_scales(scales: torch.Tensor) -> torch.Tensor:
    """``(rows, K//32)`` uint8 -> the swizzled e8m0 tensor ``scaled_mm`` wants."""
    return swizzle_mx_scales(scales).view(torch.float8_e8m0fnu)


def vendor_mxfp8_matmul(
    aq: torch.Tensor,
    a_scale: torch.Tensor,
    bq: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convenience wrapper: swizzle the natural-layout scales, then multiply.

    For correctness-checking callers; timing callers want
    :func:`vendor_mxfp8_matmul_swizzled` with the swizzle hoisted out of the loop.
    """
    return vendor_mxfp8_matmul_swizzled(
        aq,
        as_vendor_scales(a_scale),
        bq,
        as_vendor_scales(b_scale),
        out_dtype,
    )


@torch.library.custom_op("kohakuwullm::mxfp8_mm_swizzled", mutates_args=())
def mxfp8_mm_swizzled(
    aq: torch.Tensor,
    a_scale: torch.Tensor,
    bq: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """`vendor_mxfp8_matmul_swizzled` as an opaque op, so Inductor can compile past it.

    Inductor's `_scaled_mm_v2` lowering rejects non-trivial swizzles, which is the
    layout this path requires. See docs/internals/mxfp8-difficulty.md.
    """
    return vendor_mxfp8_matmul_swizzled(aq, a_scale, bq, b_scale, out_dtype)


@mxfp8_mm_swizzled.register_fake
def _mxfp8_mm_swizzled_fake(aq, a_scale, bq, b_scale, out_dtype=torch.bfloat16):
    return aq.new_empty((aq.shape[0], bq.shape[0]), dtype=out_dtype)
