"""Rotary position embedding: ``triton``, ``compiled`` and ``eager``, and the
selector over them.

Every implementation takes ``(x, cos, sin, rotary_dim)`` with the *half* cos/sin
table, returns a new tensor, and passes channels past ``rotary_dim`` through
untouched. ``resolve_rope(name)`` returns one; ``DEFAULT_ROPE_IMPL`` is the default.

See docs/internals/kernels.md.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_R": r}, num_warps=w, num_stages=s)
        for r in (16, 32, 64, 128)
        for w in (2, 4, 8)
        for s in (1, 2)
    ]


@triton.autotune(configs=_configs(), key=["half", "head_dim", "n_heads"])
@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_xr,
    stride_xd,
    stride_or,
    stride_od,
    stride_ct,
    stride_cd,
    n_rows,
    n_heads,
    half,
    head_dim,
    BACKWARD: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Rotate a tile of ``BLOCK_R`` (token, head) rows. Grid is 1-D over rows."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_ok = rows < n_rows
    d = tl.arange(0, BLOCK_D)
    keep = row_ok[:, None] & (d < half)[None, :]

    # cos/sin are per token; `n_heads` consecutive rows share one token.
    tok = rows // n_heads

    x_lo = x_ptr + rows[:, None] * stride_xr + d[None, :] * stride_xd
    x1 = tl.load(x_lo, mask=keep, other=0.0).to(tl.float32)
    x2 = tl.load(x_lo + half * stride_xd, mask=keep, other=0.0).to(tl.float32)

    table = tok[:, None] * stride_ct + d[None, :] * stride_cd
    cos = tl.load(cos_ptr + table, mask=keep, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + table, mask=keep, other=0.0).to(tl.float32)
    if BACKWARD:
        sin = -sin

    o_lo = out_ptr + rows[:, None] * stride_or + d[None, :] * stride_od
    tl.store(o_lo, (x1 * cos - x2 * sin).to(out_ptr.dtype.element_ty), mask=keep)
    tl.store(
        o_lo + half * stride_od,
        (x2 * cos + x1 * sin).to(out_ptr.dtype.element_ty),
        mask=keep,
    )

    if HAS_TAIL:
        t = 2 * half + tl.arange(0, BLOCK_T)
        tail_ok = row_ok[:, None] & (t < head_dim)[None, :]
        val = tl.load(
            x_ptr + rows[:, None] * stride_xr + t[None, :] * stride_xd,
            mask=tail_ok,
            other=0.0,
        )
        tl.store(
            out_ptr + rows[:, None] * stride_or + t[None, :] * stride_od,
            val,
            mask=tail_ok,
        )


def _launch(x2d, cos2d, sin2d, out, rotary_dim, backward):
    """Run the kernel over ``x2d`` of shape ``(tokens, heads, head_dim)``."""
    tokens, heads, head_dim = x2d.shape
    half = rotary_dim // 2
    n_rows = tokens * heads
    tail = head_dim - 2 * half

    def grid(meta):
        return (triton.cdiv(n_rows, meta["BLOCK_R"]),)

    _rope_kernel[grid](
        x2d,
        cos2d,
        sin2d,
        out,
        x2d.stride(1),
        x2d.stride(2),
        out.stride(1),
        out.stride(2),
        cos2d.stride(0),
        cos2d.stride(1),
        n_rows,
        heads,
        half,
        head_dim,
        BACKWARD=backward,
        HAS_TAIL=tail > 0,
        BLOCK_D=triton.next_power_of_2(half),
        BLOCK_T=triton.next_power_of_2(max(tail, 1)),
    )


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, rotary_dim):
        shape = x.shape
        x2d = x.reshape(-1, shape[-2], shape[-1])
        cos2d = cos.reshape(-1, cos.shape[-1])
        sin2d = sin.reshape(-1, sin.shape[-1])
        out = torch.empty_like(x2d)
        _launch(x2d, cos2d, sin2d, out, rotary_dim, False)
        ctx.save_for_backward(cos2d, sin2d)
        ctx.rotary_dim = rotary_dim
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_out):
        cos2d, sin2d = ctx.saved_tensors
        shape = ctx.shape
        g2d = grad_out.reshape(-1, shape[-2], shape[-1]).contiguous()
        dx = torch.empty_like(g2d)
        _launch(g2d, cos2d, sin2d, dx, ctx.rotary_dim, True)
        return dx.reshape(shape), None, None, None


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int | None = None
) -> torch.Tensor:
    """Rotate ``x`` of shape ``(..., H, D)`` by the per-token angles in cos/sin.

    ``cos``/``sin`` are ``(..., rotary_dim // 2)`` -- the *half* table, not the
    doubled one a ``rotate_half`` formulation uses. Non-CUDA input takes the eager
    path.
    """
    rotary_dim = rotary_dim or x.shape[-1]
    if not x.is_cuda:
        return _reference_rope(x, cos, sin, rotary_dim)
    return _RoPE.apply(x.contiguous(), cos.contiguous(), sin.contiguous(), rotary_dim)


def _reference_rope(x, cos, sin, rotary_dim):
    """Eager rotation; the CPU fallback and the test oracle."""
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([out, x_pass], dim=-1) if x_pass.numel() else out


def apply_rope_eager(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int | None = None
) -> torch.Tensor:
    """:func:`apply_rope`'s contract, run eagerly on every device."""
    return _reference_rope(x, cos, sin, rotary_dim or x.shape[-1])


# torch.compile arguments for the `compiled` implementation. See docs/internals/kernels.md.
ROPE_COMPILE_OPTIONS = {"dynamic": None, "fullgraph": True, "mode": "default"}

_COMPILED_REFERENCE: Callable[..., torch.Tensor] | None = None


def compiled_rope() -> Callable[..., torch.Tensor]:
    """``torch.compile(_reference_rope, **ROPE_COMPILE_OPTIONS)``, built once."""
    global _COMPILED_REFERENCE
    if _COMPILED_REFERENCE is None:
        _COMPILED_REFERENCE = torch.compile(_reference_rope, **ROPE_COMPILE_OPTIONS)
    return _COMPILED_REFERENCE


def apply_rope_compiled(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int | None = None
) -> torch.Tensor:
    """:func:`apply_rope`'s contract, through Inductor. Non-CUDA input stays eager."""
    rotary_dim = rotary_dim or x.shape[-1]
    if not x.is_cuda:
        return _reference_rope(x, cos, sin, rotary_dim)
    return compiled_rope()(x, cos, sin, rotary_dim)


ROPE_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "triton": apply_rope,
    "compiled": apply_rope_compiled,
    "eager": apply_rope_eager,
}

DEFAULT_ROPE_IMPL = "triton"


def resolve_rope(impl: str = DEFAULT_ROPE_IMPL) -> Callable[..., torch.Tensor]:
    """The rotation callable named ``impl``; raises ``ValueError`` on an unknown name."""
    if impl not in ROPE_IMPLS:
        raise ValueError(f"unknown rope impl {impl!r}; known: {sorted(ROPE_IMPLS)}")
    return ROPE_IMPLS[impl]
