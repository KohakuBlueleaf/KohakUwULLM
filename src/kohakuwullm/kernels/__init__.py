"""Hand-written Triton kernels.

See docs/internals/kernels.md, and docs/internals/mxfp8.md for the block-scaled fp8 subpackage.
"""

from kohakuwullm.kernels.attention.flash_attn import triton_varlen_attn
from kohakuwullm.kernels.attention.rope import apply_rope
from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm
from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.kernels.loss.chunked_ce import chunked_linear_cross_entropy
from kohakuwullm.kernels.loss.zloss import logsumexp_square
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm, grouped_gemm_reference
from kohakuwullm.kernels.optim.stochastic_round import (
    stochastic_round_,
    stochastic_round_reference,
    stochastic_round_update_,
)
from kohakuwullm.kernels.optim.stochastic_round_grouped import GroupedWriteback

__all__ = [
    "GroupedWriteback",
    "chunked_linear_cross_entropy",
    "grouped_gemm",
    "grouped_gemm_reference",
    "rms_norm",
    "apply_rope",
    "stochastic_round_",
    "stochastic_round_reference",
    "stochastic_round_update_",
    "triton_varlen_attn",
    "swiglu_mul",
    "logsumexp_square",
]
