"""Stochastic rounding on the way into a 16-bit parameter.

Rounds to one of the two neighbouring representable values with probability equal
to the distance to the other. bf16 targets only.

See docs/internals/kernels.md.
"""

import functools
import typing

import torch
import triton
import triton.language as tl

# Per target: discarded fp32 mantissa bits, minimum normal exponent, subnormal gap.
_FORMAT: dict[torch.dtype, tuple[int, int, bool]] = {
    torch.bfloat16: (16, -126, False),
}
_FP32_MANTISSA = 23
_SIGN_BIT = -(2**31)
# Draw field width for the below-smallest-subnormal branch.
_TAIL_ONE = 1 << 30
_TAIL_MASK = _TAIL_ONE - 1

# constexpr twins for the jitted bodies.
_TL_MANTISSA = tl.constexpr(_FP32_MANTISSA)
_TL_SIGN_BIT = tl.constexpr(_SIGN_BIT)
_TL_TAIL_ONE = tl.constexpr(_TAIL_ONE)
_TL_TAIL_ONE_F = tl.constexpr(float(_TAIL_ONE))
_TL_TAIL_MASK = tl.constexpr(_TAIL_MASK)


class _Target(typing.NamedTuple):
    """Everything the kernels need to know about a rounding target."""

    k_normal: int
    min_exp: int
    tiny_bits: int
    tail_scale: float
    subnormal_gap: bool


@triton.jit
def _sr_round(
    x,
    draw,
    K_NORMAL: tl.constexpr,
    MIN_EXP: tl.constexpr,
    TINY_BITS: tl.constexpr,
    TAIL_SCALE: tl.constexpr,
    SUBNORMAL_GAP: tl.constexpr,
):
    """fp32 block -> fp32 block exact in the target format.

    Returns fp32, not the target dtype, so the caller's ``.to()`` cannot round again.
    """
    bits = x.to(tl.int32, bitcast=True)
    if SUBNORMAL_GAP:
        # Discarded-bit count, widening across the target's subnormal range.
        exponent = ((bits >> _TL_MANTISSA) & 0xFF) - 127
        k = K_NORMAL + tl.maximum(MIN_EXP - exponent, 0)
        # Clamp the shift into range.
        k_safe = tl.minimum(k, _TL_MANTISSA)
        carried = ((bits + (draw & ((1 << k_safe) - 1))) & -(1 << k_safe)).to(
            tl.float32, bitcast=True
        )
        # Below the smallest subnormal: round to 0 or +-TINY, probability |x| / TINY.
        p = tl.minimum(tl.abs(x) * TAIL_SCALE, _TL_TAIL_ONE_F).to(tl.int32)
        up = (draw & _TL_TAIL_MASK) >= _TL_TAIL_ONE - p
        # Sign carried as a bit.
        tail = (tl.where(up, TINY_BITS, 0) | (bits & _TL_SIGN_BIT)).to(
            tl.float32, bitcast=True
        )
        return tl.where(k <= _TL_MANTISSA, carried, tail)
    return ((bits + (draw & ((1 << K_NORMAL) - 1))) & -(1 << K_NORMAL)).to(
        tl.float32, bitcast=True
    )


_BLOCK = 1024
_WARPS = 4


@triton.jit
def _sr_cast_kernel(
    src_ptr,
    dst_ptr,
    n,
    seed,
    rng_offset,
    K_NORMAL: tl.constexpr,
    MIN_EXP: tl.constexpr,
    TINY_BITS: tl.constexpr,
    TAIL_SCALE: tl.constexpr,
    SUBNORMAL_GAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src_ptr + offs, mask=mask, other=0.0)
    # Bitcast: tl.randint returns uint32 and the truncation mask is negative.
    draw = tl.randint(seed, offs.to(tl.int64) + rng_offset).to(tl.int32, bitcast=True)
    y = _sr_round(x, draw, K_NORMAL, MIN_EXP, TINY_BITS, TAIL_SCALE, SUBNORMAL_GAP)
    tl.store(dst_ptr + offs, y.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sr_update_kernel(
    param_ptr,
    update_ptr,
    n,
    keep,
    alpha,
    seed,
    rng_offset,
    K_NORMAL: tl.constexpr,
    MIN_EXP: tl.constexpr,
    TINY_BITS: tl.constexpr,
    TAIL_SCALE: tl.constexpr,
    SUBNORMAL_GAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    w = tl.load(param_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Widened to fp32 before the scale.
    u = tl.load(update_ptr + offs, mask=mask, other=0.0).to(tl.float32) * alpha
    # Bitcast: tl.randint returns uint32 and the truncation mask is negative.
    draw = tl.randint(seed, offs.to(tl.int64) + rng_offset).to(tl.int32, bitcast=True)
    y = _sr_round(
        w * keep + u, draw, K_NORMAL, MIN_EXP, TINY_BITS, TAIL_SCALE, SUBNORMAL_GAP
    )
    tl.store(param_ptr + offs, y.to(param_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _format_of(dtype: torch.dtype) -> _Target:
    try:
        k_normal, min_exp, gap = _FORMAT[dtype]
    except KeyError:
        raise ValueError(f"stochastic rounding targets bf16, got {dtype}") from None
    tiny = 2.0 ** (min_exp - (_FP32_MANTISSA - k_normal))
    tiny_bits = int(torch.tensor(tiny, dtype=torch.float32).view(torch.int32).item())
    # Unreachable without a subnormal gap.
    tail_scale = _TAIL_ONE / tiny if gap else 0.0
    return _Target(k_normal, min_exp, tiny_bits, tail_scale, gap)


def _check(low: torch.Tensor, other: torch.Tensor, seed: int, rng_offset: int) -> int:
    """Shape, layout and RNG arguments. ``other``'s dtype is deliberately unchecked."""
    if low.shape != other.shape:
        raise ValueError(f"shape mismatch: {tuple(low.shape)} vs {tuple(other.shape)}")
    if not low.is_contiguous() or not other.is_contiguous():
        raise ValueError("stochastic rounding needs contiguous tensors")
    n = low.numel()
    if rng_offset < 0:
        raise ValueError(f"rng_offset must be non-negative, got {rng_offset}")
    # Per-tensor addressing is 32-bit; the RNG counter is int64 and is not the limit.
    if n >= 2**31:
        raise ValueError(f"numel {n} exceeds the 32-bit addressing limit")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return n


def stochastic_round_(
    dst: torch.Tensor, src: torch.Tensor, seed: int, *, rng_offset: int = 0
) -> torch.Tensor:
    """Write ``src`` (fp32) into ``dst`` (bf16), rounding stochastically.

    ``seed`` must advance every step and must be identical across data-parallel
    replicas; ``rng_offset`` separates the draws of several tensors rounded
    under one seed.
    """
    if src.dtype is not torch.float32:
        raise ValueError(f"the fp32 side must be float32, got {src.dtype}")
    n = _check(dst, src, seed, rng_offset)
    fmt = _format_of(dst.dtype)
    if n:
        _sr_cast_kernel[(triton.cdiv(n, _BLOCK),)](
            src,
            dst,
            n,
            seed,
            rng_offset,
            K_NORMAL=fmt.k_normal,
            MIN_EXP=fmt.min_exp,
            TINY_BITS=fmt.tiny_bits,
            TAIL_SCALE=fmt.tail_scale,
            SUBNORMAL_GAP=fmt.subnormal_gap,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    return dst


def stochastic_round_update_(
    param: torch.Tensor,
    update: torch.Tensor,
    seed: int,
    *,
    decay: float = 0.0,
    alpha: float = 1.0,
    rng_offset: int = 0,
) -> torch.Tensor:
    """``param = SR(param * (1 - decay) + alpha * update)`` in one pass.

    Args:
        param: bf16, rounded in place.
        update: fp32 or 16-bit float; promoted to fp32 on load.
        seed: must advance every step and match across data-parallel replicas.
        decay: the ``lr * weight_decay`` product.
        alpha: scale applied to ``update``.
        rng_offset: separates the draws of several tensors rounded under one seed.
    """
    if update.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"update must be fp32 or 16-bit float, got {update.dtype}")
    n = _check(param, update, seed, rng_offset)
    fmt = _format_of(param.dtype)
    if n:
        _sr_update_kernel[(triton.cdiv(n, _BLOCK),)](
            param,
            update,
            n,
            1.0 - decay,
            alpha,
            seed,
            rng_offset,
            K_NORMAL=fmt.k_normal,
            MIN_EXP=fmt.min_exp,
            TINY_BITS=fmt.tiny_bits,
            TAIL_SCALE=fmt.tail_scale,
            SUBNORMAL_GAP=fmt.subnormal_gap,
            BLOCK=_BLOCK,
            num_warps=_WARPS,
        )
    return param


def stochastic_round_reference(
    dst: torch.Tensor,
    src: torch.Tensor,
    draw: torch.Tensor,
) -> torch.Tensor:
    """The same construction in torch, driven by an explicit ``draw`` tensor.

    ``draw`` is int32 and supplied by the caller rather than sampled here.
    """
    fmt = _format_of(dst.dtype)
    bits = src.view(torch.int32)
    if not fmt.subnormal_gap:
        k = torch.full_like(bits, fmt.k_normal)
    else:
        exponent = ((bits >> _FP32_MANTISSA) & 0xFF) - 127
        k = fmt.k_normal + (fmt.min_exp - exponent).clamp_min(0)
    k_safe = k.clamp_max(_FP32_MANTISSA)
    one = torch.ones_like(k_safe)
    carried = ((bits + (draw & ((one << k_safe) - 1))) & -(one << k_safe)).view(
        torch.float32
    )
    if not fmt.subnormal_gap:
        return dst.copy_(carried)
    p = (src.abs() * fmt.tail_scale).clamp_max(_TAIL_ONE).to(torch.int32)
    up = (draw & _TAIL_MASK) >= _TAIL_ONE - p
    magnitude = torch.where(
        up, torch.full_like(bits, fmt.tiny_bits), torch.zeros_like(bits)
    )
    tail = (magnitude | (bits & _SIGN_BIT)).view(torch.float32)
    return dst.copy_(torch.where(k <= _FP32_MANTISSA, carried, tail))
