"""MXFP8 linear: fp8 FPROP and DGRAD, bf16 WGRAD.

``in_features`` must be a multiple of 128; ``out_features`` is zero-padded to one,
which is exact. Compute and output dtype follow autocast, never the weight.

See docs/internals/mxfp8.md.
"""

import torch
import torch.nn.functional as F

from kohakuwullm.kernels.mxfp8 import quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.interop import mxfp8_mm_swizzled

# The alignment `quantize_mx_vendor` needs to emit SWIZZLE_32_4_4 scales.
VENDOR_K_ALIGN = 128


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype ``nn.Linear`` would compute in for this input.

    Autocast when it is on, the input's own dtype otherwise -- not the weight's.
    """
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


def _unpadded(tensor: torch.Tensor) -> torch.Tensor:
    """The no-padding case, resolved at build time so no call site tests for it."""
    return tensor


class _PadLastAxis:
    """Zero-pad the last axis by a fixed width.

    A class rather than a closure so the module stays picklable.
    """

    __slots__ = ("width",)

    def __init__(self, width: int) -> None:
        self.width = width

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, self.width))


class _MXFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, wq_f, ws_f, wq_d, ws_d, pad_k):
        compute = compute_dtype(x)
        shape = x.shape
        # Cast before saving, so WGRAD's two operands cannot disagree.
        x2d = x.reshape(-1, shape[-1]).to(compute)
        xq, xs = quantize_mx_vendor(x2d)
        # `out_features` is FPROP's N, not its K, so no padding here.
        out = mxfp8_mm_swizzled(xq, xs, wq_f, ws_f, compute)
        ctx.save_for_backward(x2d, wq_d, ws_d)
        ctx.pad_k = pad_k
        # `dx` comes back in the dtype of `x` as the caller passed it.
        ctx.x_dtype = x.dtype
        return out.reshape(*shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, dout):
        x2d, wq_d, ws_d = ctx.saved_tensors
        shape = dout.shape
        d2d = dout.reshape(-1, shape[-1]).to(x2d.dtype).contiguous()
        dq, ds = quantize_mx_vendor(ctx.pad_k(d2d))
        dx = mxfp8_mm_swizzled(dq, ds, wq_d, ws_d, ctx.x_dtype)
        # 16-bit, and unpadded: the padded axis is this product's N.
        dw = d2d.t() @ x2d
        return dx.reshape(*shape[:-1], dx.shape[-1]), dw, None, None, None, None, None


class MXFP8Linear(torch.nn.Module):
    """``nn.Linear`` (no bias) with fp8 FPROP/DGRAD and a bf16 WGRAD.

    ``refresh_quantized_weight`` must be called after every optimizer step.
    Compute and output dtype follow autocast, so ``dw`` comes back in the compute
    dtype rather than the parameter's and autograd casts it during accumulation.
    """

    def __init__(self, in_features: int, out_features: int, dtype=torch.bfloat16):
        super().__init__()
        # Only `in_features` is a hard requirement; `out_features` is zero-padded.
        if in_features % VENDOR_K_ALIGN:
            raise ValueError(
                f"in_features={in_features} must be a multiple of {VENDOR_K_ALIGN}; "
                "it is FPROP's contraction axis and is shared with the activation "
                "cast, so it cannot be padded without padding every input"
            )
        self.in_features = in_features
        self.out_features = out_features
        padded = -(-out_features // VENDOR_K_ALIGN) * VENDOR_K_ALIGN
        self.padded_out_features = padded
        self._pad_k = (
            _unpadded if padded == out_features else _PadLastAxis(padded - out_features)
        )
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, std=in_features**-0.5)
        self._cache: tuple[torch.Tensor, ...] | None = None

    @torch.no_grad()
    def refresh_quantized_weight(self) -> None:
        wq_f, ws_f = quantize_mx_vendor(self.weight)
        # No `.contiguous()`: the quantizer takes both strides.
        wq_d, ws_d = quantize_mx_vendor(self._pad_k(self.weight.t()))
        self._cache = (wq_f, ws_f, wq_d, ws_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cache is None:
            self.refresh_quantized_weight()
        return _MXFP8Linear.apply(x, self.weight, *self._cache, self._pad_k)

    def _apply(self, *args, **kwargs):
        """Drop the quantized cache on any device or dtype transform.

        The cache is a plain attribute, never a registered buffer. ``forward``
        rebuilds it on the next call.
        """
        self._cache = None
        return super()._apply(*args, **kwargs)
