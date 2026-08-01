"""The swizzled MXFP8 matmul must survive torch.compile as an opaque op."""

import pytest
import torch

from kohakuwullm.kernels.mxfp8.interop import (
    as_vendor_scales,
    vendor_mxfp8_matmul_swizzled,
)
from kohakuwullm.kernels.mxfp8.quantize import quantize_mx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_custom_op_compiles_and_matches_eager():
    """fullgraph compile works, and the result is bitwise the eager one."""
    torch.manual_seed(0)
    m, n, k = 512, 256, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    aq, asc = quantize_mx(a)
    bq, bsc = quantize_mx(b)
    asw, bsw = as_vendor_scales(asc), as_vendor_scales(bsc)

    eager = torch.ops.kohakuwullm.mxfp8_mm_swizzled(aq, asw, bq, bsw)
    assert torch.equal(
        eager, vendor_mxfp8_matmul_swizzled(aq, asw, bq, bsw, torch.bfloat16)
    )

    fn = torch.compile(
        lambda *t: torch.ops.kohakuwullm.mxfp8_mm_swizzled(*t), fullgraph=True
    )
    assert torch.equal(fn(aq, asw, bq, bsw), eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_raw_scaled_mm_still_breaks_compile():
    """Pins why the wrapper exists: the raw swizzled path does not lower."""
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    aq, asc = quantize_mx(a)
    bq, bsc = quantize_mx(b)
    asw, bsw = as_vendor_scales(asc), as_vendor_scales(bsc)
    fn = torch.compile(
        lambda *t: vendor_mxfp8_matmul_swizzled(*t, torch.bfloat16), fullgraph=True
    )
    with pytest.raises(Exception, match="swizzle"):
        fn(aq, asw, bq, bsw)
