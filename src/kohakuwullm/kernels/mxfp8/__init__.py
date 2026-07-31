"""MXFP8: e4m3 values with a UE8M0 power-of-two scale per 32 elements.

* :mod:`quantize` -- quantizer and standalone matmul.
* :mod:`grouped` -- grouped GEMM and WGRAD, shared by every routed-expert path.
* :mod:`linear` -- the ``nn.Linear`` replacement, on vendor ``torch._scaled_mm``.
* :mod:`experts`, :mod:`experts_bwd` -- fused expert kernels and their epilogues.
* :mod:`moe`, :mod:`moe_unfused` -- the two routed-expert paths built on those.
* :mod:`interop` -- conversion between the natural and cuBLAS scale layouts.

The names below are re-exported so ``from kohakuwullm.kernels.mxfp8 import
quantize_mx`` keeps working; ``_quantize_block`` is private but re-exported because
``rmsnorm`` fuses it into its own epilogue.

See docs/internals/mxfp8.md.
"""

from kohakuwullm.kernels.mxfp8.quantize import (
    BLOCK_SCALE,
    E4M3_MAX,
    _quantize_block,
    mxfp8_matmul,
    mxfp8_matmul_pq,
    quantize_mx,
    quantize_mx_vendor,
)

__all__ = [
    "BLOCK_SCALE",
    "E4M3_MAX",
    "_quantize_block",
    "mxfp8_matmul",
    "mxfp8_matmul_pq",
    "quantize_mx",
    "quantize_mx_vendor",
]
