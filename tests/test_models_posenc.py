"""Position encodings: which rotation implementation the carrier selects.

``triton``, ``compiled`` and ``eager`` are peer implementations of one contract,
and a carrier resolves one of them by name at construction. These tests pin the
*selection* -- the default, that every position encoding takes the same one, and
that an unknown name raises -- and then the arithmetic all three must agree on.

See docs/internals/kernels.md for what separates the three.
"""

import pytest
import torch

from kohakuwullm.bench.core.timing import ulp_error
from kohakuwullm.kernels.attention.rope import (
    ROPE_IMPLS,
    _reference_rope,
    compiled_rope,
    resolve_rope,
)
from kohakuwullm.models.components.ndrope import GGRoPE, NDRoPE
from kohakuwullm.models.components.posenc import RoPE, RotaryCache

IMPLS = sorted(ROPE_IMPLS)
DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]

# Every position encoding that builds a RotaryCache, at one head width.
POSENCS = {
    "rope": lambda **kw: RoPE(head_dim=8, **kw),
    "ndrope": lambda **kw: NDRoPE(head_dim=8, n_dims=1, **kw),
    "ggrope": lambda **kw: GGRoPE(head_dim=8, n_dims=1, **kw),
}

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)


def _prepare(module, tokens: int = 8):
    positions = torch.arange(tokens)
    return module.prepare(positions, torch.device("cpu"), torch.float32)


def _rotate_half_oracle(x, cos, sin, rotary_dim):
    """The doubled-table formulation, independent of the kernel module."""
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    out = x_rot * cos + torch.cat([-x2, x1], dim=-1) * sin
    return torch.cat([out, x_pass], dim=-1) if x_pass.numel() else out


def test_rotary_cache_defaults_to_the_fused_kernel():
    """Constructed without an explicit choice, the carrier takes the Triton kernel."""
    cache = RotaryCache(torch.ones(4, 8), torch.zeros(4, 8), rotary_dim=8)
    assert cache.impl == "triton"
    assert cache.rotate is resolve_rope("triton")


@pytest.mark.parametrize("posenc", POSENCS)
def test_prepare_propagates_the_default(posenc):
    """Every ``prepare`` builds its carrier on the same default.

    Parametrized over all three position encodings that return a ``RotaryCache``,
    because each constructs one itself: a default fixed in one and overridden in
    another is silent divergence, one class further out.
    """
    cache = _prepare(POSENCS[posenc]())
    assert cache.impl == "triton"
    assert cache.rotate is resolve_rope("triton")


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("posenc", POSENCS)
def test_the_selection_reaches_the_carrier(posenc, impl):
    """A configured name survives ``__init__`` into the carrier that ``prepare`` builds.

    Over all three again: a knob honoured on one position encoding and dropped by
    another's ``**_unused`` is the same divergence as a diverging default, except
    it only shows up in a config nobody has written yet.
    """
    cache = _prepare(POSENCS[posenc](impl=impl))
    assert cache.impl == impl
    assert cache.rotate is ROPE_IMPLS[impl]


@pytest.mark.parametrize("impl", ["fused", "Triton", "", None, True])
def test_an_unknown_implementation_raises(impl):
    """No silent fallback: a name nobody implements is a build-time error.

    At the carrier and at every module that configures one, since a module is
    where a config typo arrives and the carrier is one forward later.
    """
    with pytest.raises(ValueError, match="unknown rope impl"):
        RotaryCache(torch.ones(4, 8), torch.zeros(4, 8), rotary_dim=8, impl=impl)
    for posenc, build in POSENCS.items():
        with pytest.raises(ValueError, match="unknown rope impl"):
            build(impl=impl)


def test_the_compiled_callable_is_built_once():
    """Compilation is lazy and its result is cached at module level."""
    assert compiled_rope() is compiled_rope()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("partial", [1.0, 0.5], ids=["full", "partial"])
def test_every_implementation_rotates_identically(impl, partial):
    """All three agree with a ``rotate_half`` oracle, at full and partial rotary.

    On CPU every implementation is the eager reference, so what this pins is
    ``RotaryCache.apply``: that it hands each one the half table and the channel
    count they document, for a partial rotary as well as a whole head.
    """
    torch.manual_seed(0)
    rope = RoPE(head_dim=8, partial_rotary_factor=partial, impl=impl)
    q = torch.randn(6, 2, 8)
    k = torch.randn(6, 2, 8)
    cache = _prepare(rope, tokens=6)

    for got, x in zip(cache.apply(q, k), (q, k)):
        want = _rotate_half_oracle(x, cache.cos, cache.sin, rope.rotary_dim)
        assert got.shape == x.shape
        torch.testing.assert_close(got, want)


@requires_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_every_implementation_matches_an_fp64_reference(dtype):
    """Forward and backward, in both low-precision dtypes, against fp64.

    ``rms`` scaling for the shared bound: an output is ``x1*cos - x2*sin``, so a
    near-zero element is cancellation between larger terms, not a small true
    value. ``triton`` and ``compiled`` reduce in fp32 and round once, at the
    store, which the *elementwise* half-ULP bound is what pins -- eager rounds
    each product and the difference and cannot meet it.
    """
    torch.manual_seed(0)
    tokens, heads, head_dim = 512, 8, 64
    x = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)
    x.requires_grad_(True)
    angles = torch.randn(tokens, head_dim // 2, device="cuda")
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    grad = torch.randn(tokens, heads, head_dim, device="cuda", dtype=dtype)

    x64 = x.detach().double().requires_grad_(True)
    ref = _reference_rope(x64, cos.double(), sin.double(), head_dim)
    ref.backward(grad.double())

    for impl in IMPLS:
        x.grad = None
        out = resolve_rope(impl)(x, cos, sin, head_dim)
        out.backward(grad)
        assert ulp_error(out, ref, dtype, mode="rms") <= 8.0, impl
        assert ulp_error(x.grad, x64.grad, dtype, mode="rms") <= 8.0, impl
        if impl != "eager":
            assert ulp_error(out, ref, dtype, mode="elementwise") <= 1.0, impl
            assert ulp_error(x.grad, x64.grad, dtype, mode="elementwise") <= 1.0, impl
