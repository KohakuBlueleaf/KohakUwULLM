"""Correctness checks for the stochastic-rounding kernel.

Four of these are the point of the file, and every one of them fails on an
implementation that merely *looks* stochastic:

* :func:`test_nearest_rounding_loses_the_update_stochastic_keeps` -- the control.
  Without it an SR test passes on a round-to-nearest implementation.
* :func:`test_a_frozen_seed_freezes_most_of_the_weights` -- a seed that does not
  advance leaves the ensemble mean *correct* while most coordinates never move at
  all. Only the never-moved fraction sees it.
* :func:`test_too_few_random_bits_biases_by_the_published_bound` -- pins the
  ``N == D`` rule against the bound it comes from, and shows the suite can detect
  a wrong draw width.

The probability reference is exhaustive, not sampled, wherever the draw space
allows: every draw is enumerated and the up-fraction must match
``(x - lo) / (hi - lo)`` in fp64 with *zero* error, where ``lo`` and ``hi`` come
from a search over every finite value of the target format. A statistical test
would pass on a construction that is off by one draw in 8192.
"""

import pytest
import torch

from kohakuwullm.bench.core.timing import ulp_error
from kohakuwullm.kernels.optim.stochastic_round import (
    stochastic_round_,
    stochastic_round_reference,
    stochastic_round_update_,
)
from kohakuwullm.kernels.optim.stochastic_round_grouped import GroupedWriteback

# bf16 only: fp16's draw width varies with the exponent and the repo does not train
# in it, so `_format_of` rejects it rather than carrying a second derivation.
DTYPES = [torch.bfloat16]
# Discarded fp32 mantissa bits in the target's normal range. bf16 is a bit prefix of
# fp32, so this is constant -- which is the property that makes the construction exact.
WIDTH = {torch.bfloat16: 16}
# Draws enumerated exhaustively. Wider than either normal-range width, so the
# low-k patterns of a subnormal target are still each hit an equal number of
# times and the up-fraction stays exact.
ENUM_BITS = 16
INT32 = (-(2**31), 2**31 - 1)


def _grid(dtype: torch.dtype) -> torch.Tensor:
    """Every finite non-negative value of a 16-bit format, ascending, as fp32."""
    codes = torch.arange(0, 1 << 15, dtype=torch.int32).to(torch.int16)
    vals = codes.view(dtype).float()
    return torch.sort(vals[torch.isfinite(vals)])[0]


def _neighbours(x: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """(toward-zero, away-from-zero) neighbours of ``x`` in ``dtype``, signed.

    The grid is extended with infinity, because above the largest finite value
    that *is* the away-from-zero neighbour and rounding to it is legal SR.
    """
    grid = torch.cat([_grid(dtype), torch.tensor([float("inf")])])
    idx = (torch.searchsorted(grid, x.abs(), right=True) - 1).clamp(0, grid.numel() - 2)
    sign = torch.where(x < 0, -1.0, 1.0)
    return sign * grid[idx], sign * grid[idx + 1]


def _round(src: torch.Tensor, draw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty(src.shape, dtype=dtype)
    stochastic_round_reference(out, src, draw)
    return out.float()


def _rand_int32(shape, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(*INT32, shape, dtype=torch.int32, generator=gen)


def _probes(dtype: torch.dtype, count: int = 40) -> torch.Tensor:
    """Interior points spread across every binade the format has, both signs.

    The fractions are deliberately not dyadic. With 1/8 and 7/8 the discarded
    field is a multiple of 256, and a draw that samples only the *top* 8 bits of
    that field then produces the correct up-fraction by coincidence -- an
    implementation using too few random bits passes.
    """
    grid = _grid(dtype)
    idx = torch.linspace(1, grid.numel() - 2, count).long()
    lo, hi = grid[idx], grid[idx + 1]
    pos = torch.cat([lo + f * (hi - lo) for f in (0.1234, 0.5003, 0.8766)])
    pos = pos[pos > 0]
    return torch.cat([pos, -pos])


# -- the construction ---------------------------------------------------- #


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_draw_lands_on_a_neighbour_with_the_exact_probability(dtype):
    """Enumerate the whole draw space against the fp64 rational probability.

    ``lo``/``hi`` come from a search over the format's finite values and the
    target probability from ``(x - lo) / (hi - lo)``, so this checks the bit
    trick against the *definition* of stochastic rounding rather than against
    another copy of the trick.
    """
    x = _probes(dtype)
    lo, hi = _neighbours(x, dtype)
    assert torch.isfinite(hi).all(), "probes must not reach the overflow boundary"

    ups = torch.zeros(x.numel(), dtype=torch.float64)
    for chunk in torch.arange(0, 1 << ENUM_BITS, dtype=torch.int32).split(4096):
        got = _round(
            x.expand(chunk.numel(), -1), chunk[:, None].expand(-1, x.numel()), dtype
        )
        assert torch.isin(got, torch.stack([lo, hi])).all()
        ups += (got == hi).sum(0).double()

    want = ((x.double() - lo.double()) / (hi.double() - lo.double())).clamp(0, 1)
    assert torch.equal(ups / (1 << ENUM_BITS), want)
    assert (want > 0).all() and (want < 1).all(), "probes must be non-representable"


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_and_maximal_draws_bracket_the_neighbours(dtype):
    """Every binade, two draws: the cheap sweep the exhaustive test cannot afford."""
    grid = _grid(dtype)[:-1]
    pos = torch.cat(
        [grid] + [grid[:-1] + f * (grid[1:] - grid[:-1]) for f in (0.01, 0.99)]
    )
    pos = pos[pos > 0]
    x = torch.cat([pos, -pos])
    lo, hi = _neighbours(x, dtype)
    zero = torch.zeros(x.numel(), dtype=torch.int32)
    assert torch.equal(_round(x, zero, dtype), lo), "a zero draw rounds toward zero"
    # -1 is the maximal draw for both paths: all bits set saturates the k-bit
    # carry field and the tail compare's field alike, where (1 << 23) - 1
    # saturates only the first.
    maximal = torch.full((x.numel(),), -1, dtype=torch.int32)
    representable = _round(x, zero, dtype) == x
    assert torch.equal(_round(x, maximal, dtype), torch.where(representable, lo, hi))


@pytest.mark.parametrize("dtype", DTYPES)
def test_grid_values_never_move_under_any_draw(dtype):
    """A representable input has no discarded bits, so no draw may carry.

    Catches an off-by-one in either the draw range or the truncation mask: with a
    draw in ``[0, 2^k]`` or a mask one bit too wide, exact values drift.
    """
    grid = _grid(dtype)[:-1]
    x = torch.cat([grid, -grid])
    for draw in (0, 1, (1 << WIDTH[dtype]) - 1, (1 << 23) - 1, -1):
        got = _round(x, torch.full((x.numel(),), draw, dtype=torch.int32), dtype)
        assert torch.equal(got, x), f"draw {draw} moved a representable value"


@pytest.mark.parametrize("dtype", DTYPES)
def test_infinities_and_canonical_nan_survive(dtype):
    """The writeback must not launder a dead run into finite numbers."""
    x = torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float32)
    for draw in (0, -1):
        got = _round(x, torch.full((3,), draw, dtype=torch.int32), dtype)
        assert got[0] == float("inf") and got[1] == float("-inf")
        assert got[2].isnan()

    # Above the largest finite value, infinity is the away-from-zero neighbour,
    # so overflow is a legal SR outcome rather than a bug -- and RTN overflows on
    # the same inputs.
    top = _grid(dtype)[-1]
    mid = (top + 0.5 * (top - _grid(dtype)[-2])).reshape(1)
    assert _round(mid, torch.zeros(1, dtype=torch.int32), dtype).item() == top
    assert _round(mid, torch.full((1,), -1, dtype=torch.int32), dtype).isinf()


@pytest.mark.parametrize("dtype", DTYPES)
def test_statistical_expectation_matches_over_the_whole_exponent_range(dtype):
    """Sampled check across every binade, where enumeration is too wide."""
    reps = 1 << 14
    base = torch.tensor([0.3, -1.7, 2e-3, -4e-5, 7e-7, 3e-8], dtype=torch.float32)
    src = base.repeat(reps).contiguous()
    got = _round(src, _rand_int32(src.shape, 5), dtype).double().reshape(reps, -1)
    lo, hi = _neighbours(base, dtype)
    # Signed neighbours, so hi - lo is negative below zero; the standard error
    # and the variance bound both need the magnitude.
    frac = ((base.double() - lo.double()) / (hi.double() - lo.double())).clamp(0, 1)
    ulp = (hi - lo).double().abs()

    err = (got.mean(0) - base.double()).abs()
    sem = (frac * (1 - frac)).sqrt() * ulp / reps**0.5
    assert (err <= 5 * sem + 1e-6 * ulp).all(), (err, sem)
    # The variance must match the Bernoulli bound, not merely be small: a rule
    # that always rounds to the nearer neighbour also has a tiny mean error.
    var = got.var(0)
    assert torch.allclose(var, frac * (1 - frac) * ulp**2, rtol=0.1, atol=1e-45)


# -- the negative cases -------------------------------------------------- #


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale", [1.0, 2.0**-20])
def test_nearest_rounding_loses_the_update_stochastic_keeps(dtype, scale):
    """The control: a sub-half-ulp update is discarded by RTN and kept by SR.

    ``scale=2^-20`` puts the fp16 weight in its subnormal range, where a
    fixed-width implementation of this kernel is wrong. bf16 shares fp32's
    exponent and has no subnormal range worth testing at that scale.
    """
    if dtype is torch.bfloat16 and scale != 1.0:
        pytest.skip("bf16 is still normal at 2^-20")
    n, steps = 1 << 14, 200
    start = torch.tensor(scale, dtype=dtype).float()
    lo, hi = _neighbours(start.reshape(1), dtype)
    ulp = (hi - lo).item()
    update = 0.05 * ulp

    nearest = start.expand(n).clone()
    for _ in range(steps):
        nearest = (nearest + update).to(dtype).float()
    assert torch.equal(nearest, start.expand(n)), "RTN must lose every update"

    stochastic = start.expand(n).clone()
    for step in range(steps):
        stochastic = _round(stochastic + update, _rand_int32((n,), step), dtype)
    moved = stochastic.double().mean().item() - start.item()
    want = steps * update
    # Three sigma of a random walk of `steps` Bernoulli jumps of one ulp, plus a
    # 2% allowance so the bound does not depend on the seed.
    sem = 3.0 * ulp * (steps * 0.05) ** 0.5 / n**0.5
    assert abs(moved - want) < sem + 0.02 * want, (moved, want, sem)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_frozen_seed_freezes_most_of_the_weights(dtype):
    """A non-advancing seed keeps the mean right and stops most coordinates dead.

    This is why the seed is an argument. With a constant dither each element's
    rounding decision repeats forever, so it either never moves or moves on every
    step; the ensemble mean stays close to the truth, which is exactly what makes
    it dangerous. The never-moved fraction is the only thing that sees it, and it
    is the same statistic the low-precision-freeze literature reports.
    """
    n, steps = 1 << 14, 100
    start = torch.ones(n)
    lo, hi = _neighbours(torch.ones(1), dtype)
    ulp = (hi - lo).item()
    update = 0.25 * ulp

    frozen_draw = _rand_int32((n,), 3)
    frozen = start.clone()
    for _ in range(steps):
        frozen = _round(frozen + update, frozen_draw, dtype)

    advancing = start.clone()
    for step in range(steps):
        advancing = _round(advancing + update, _rand_int32((n,), 100 + step), dtype)

    assert (frozen == 1.0).float().mean().item() > 0.5
    assert (advancing == 1.0).float().mean().item() < 1e-3
    # The trap: both means are within a fraction of an ulp of the truth, so a
    # mean-only test is blind to the freeze.
    for got in (frozen, advancing):
        assert abs(got.double().mean().item() - (1.0 + steps * update)) < 0.5 * ulp


def test_too_few_random_bits_biases_by_the_published_bound():
    """N < D biases toward zero by exactly ``(2^-D - 2^-N)/2`` ulp.

    arXiv:2504.20634 §III-E, evaluated as the exact discrete sum over all
    ``2^D`` inputs between one pair of bf16 neighbours rather than sampled, so
    the comparison to the bound is not a tolerance. ``N == D`` is the only width
    that lands on zero, which is the rule the module is built on.
    """
    d = WIDTH[torch.bfloat16]
    lo = torch.tensor(1.0)
    bits = lo.view(torch.int32) + torch.arange(0, 1 << d, dtype=torch.int32)
    x = bits.view(torch.float32)
    ulp = 2.0**-7

    for n_bits in (2, 3, 4, 8):
        ups = torch.zeros(x.numel(), dtype=torch.float64)
        for draw in range(1 << n_bits):
            # SRFF with N bits places them at the top of the discarded field;
            # placing them at the bottom instead is the separate failure that
            # test_a_fixed_width_shift_degenerates_to_nearest_for_fp16_subnormals
            # pins.
            shifted = (bits + (draw << (d - n_bits))) & -(1 << d)
            ups += (shifted.view(torch.float32) > lo).double()
        bias = (ups / (1 << n_bits) - (x.double() - 1.0) / ulp).mean().item()
        assert bias == pytest.approx((2.0**-d - 2.0**-n_bits) / 2, abs=1e-12)

    # The module's own width, by the same measure.
    exact = _round(x, torch.zeros(x.numel(), dtype=torch.int32), torch.bfloat16)
    assert torch.equal(exact, torch.full_like(exact, 1.0))


def test_stochastic_round_rejects_bad_arguments():
    fp32 = torch.zeros(4)
    bf16 = torch.zeros(4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="targets bf16"):
        stochastic_round_(fp32, fp32, 0)
    with pytest.raises(ValueError, match="targets bf16"):
        stochastic_round_update_(fp32, fp32, 0)
    with pytest.raises(ValueError, match="shape mismatch"):
        stochastic_round_(torch.zeros(5, dtype=torch.bfloat16), fp32, 0)
    with pytest.raises(ValueError, match="contiguous"):
        stochastic_round_(
            torch.zeros(4, 2, dtype=torch.bfloat16).T, torch.zeros(2, 4), 0
        )
    with pytest.raises(ValueError, match="must be float32"):
        stochastic_round_(bf16, torch.zeros(4, dtype=torch.float64), 0)

    # The two entry points take *different* second operands, and the asymmetry is
    # load-bearing rather than an oversight. `stochastic_round_` copies an fp32 master,
    # so 16-bit there is a caller error. `stochastic_round_update_` takes Muon's
    # orthogonalized factor, which is `ns_dtype` (bf16) -- demanding fp32 of it made the
    # guard reject the only production caller the update path has, and
    # `MuonW(rounding="stochastic")` raised on every step until this was split.
    with pytest.raises(ValueError, match="must be float32"):
        stochastic_round_(bf16, bf16.clone(), 0)
    with pytest.raises(ValueError, match="fp32 or 16-bit"):
        stochastic_round_update_(bf16, torch.zeros(4, dtype=torch.float64), 0)

    # A strided *update*, which is what `orthogonalize` hands back for a tall parameter
    # under the default `cubic5`. The kernel walks both operands with one flat offset, so
    # this must raise rather than silently read the wrong elements.
    strided = torch.zeros(4, 2, dtype=torch.bfloat16).T
    assert strided.shape == (2, 4) and not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        stochastic_round_update_(torch.zeros(2, 4, dtype=torch.bfloat16), strided, 0)
    with pytest.raises(ValueError, match="rng_offset must be non-negative"):
        stochastic_round_(bf16, fp32, 0, rng_offset=-1)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        stochastic_round_(bf16, fp32, -1)


# -- the kernel ---------------------------------------------------------- #


def _draws(n: int, seed: int, offset: int, device) -> torch.Tensor:
    """The exact Philox stream the kernel consumes, dumped from Triton.

    Handing the kernel and the reference the same bits turns a distributional
    comparison into an equality one, leaving only the RNG itself to statistics.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _dump(out_ptr, n, seed, rng_offset, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        draw = tl.randint(seed, offs + rng_offset).to(tl.int32, bitcast=True)
        tl.store(out_ptr + offs, draw, mask=offs < n)

    out = torch.empty(n, dtype=torch.int32, device=device)
    _dump[(triton.cdiv(n, 1024),)](out, n, seed, offset, BLOCK=1024)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
def test_kernel_matches_the_reference_bit_for_bit(dtype):
    torch.manual_seed(0)
    n = 4099  # not a multiple of BLOCK, so the tail mask is exercised
    src = torch.randn(n, device="cuda") * torch.exp2(
        torch.randint(-30, 8, (n,), device="cuda").float()
    )
    dst = torch.empty(n, dtype=dtype, device="cuda")
    stochastic_round_(dst, src, seed=7, rng_offset=1234)

    ref = torch.empty_like(dst)
    stochastic_round_reference(ref, src, _draws(n, 7, 1234, "cuda"))
    assert torch.equal(dst.view(torch.int16), ref.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
def test_kernel_expectation_matches_fp64_within_the_standard_error(dtype):
    """``mode="rms"``: averaging many draws is a reduction, so an element whose
    true value sits near zero inherits its error from the binade's ulp, not from
    its own magnitude."""
    reps = 4096
    base = torch.tensor([0.3, 1.7, -0.02, 5e-4, -3e-5, 1e-6])
    src = base.cuda().repeat(reps).contiguous()
    dst = torch.empty_like(src, dtype=dtype)
    stochastic_round_(dst, src, seed=99)
    got = dst.double().reshape(reps, -1).mean(0)

    lo, hi = _neighbours(base, dtype)
    # `.abs()`: the neighbours are signed and hi is the further of the two from
    # zero, so hi - lo runs negative below zero and the bound would invert.
    ulp = (hi - lo).double().abs().cuda()
    ref = base.double().cuda()
    assert (got - ref).abs().le(6 * 0.5 * ulp / reps**0.5).all(), (got - ref, ulp)
    assert ulp_error(got, ref, dtype, mode="rms") < 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("alpha", [1.0, -0.02])
def test_fused_update_matches_the_unfused_writeback(dtype, alpha):
    """Decay, scale and add all happen in fp32, before a single rounding.

    ``alpha`` is low-precision-hostile on purpose: Muon's writeback scales a 16-bit
    orthogonalized update by ``-lr``, and doing that multiply in the update's own dtype
    would round the step before it reached the parameter -- a loss no comparison against
    an fp32 reference would show, because fp32 has the mantissa to hide it.
    """
    torch.manual_seed(0)
    n = 3000
    param = (torch.randn(n, device="cuda") * 0.02).to(dtype)
    update = (torch.randn(n, device="cuda") * 1e-6).to(dtype)
    decay = 1e-3

    got = param.clone()
    stochastic_round_update_(
        got, update, seed=5, decay=decay, alpha=alpha, rng_offset=64
    )

    want = torch.empty_like(param)
    stochastic_round_reference(
        want,
        param.float() * (1.0 - decay) + alpha * update.float(),
        _draws(n, 5, 64, "cuda"),
    )
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
def test_kernel_stream_depends_on_seed_and_offset(dtype):
    """Identical seed and offset reproduce; either one changed must not."""
    torch.manual_seed(0)
    src = (torch.rand(8192, device="cuda") + 1.0) * 1.0009765
    out = [torch.empty_like(src, dtype=dtype) for _ in range(4)]
    stochastic_round_(out[0], src, seed=1)
    stochastic_round_(out[1], src, seed=1)
    stochastic_round_(out[2], src, seed=2)
    stochastic_round_(out[3], src, seed=1, rng_offset=1)
    assert torch.equal(out[0], out[1])
    assert not torch.equal(out[0], out[2])
    assert not torch.equal(out[0], out[3])


# -- the grouped writeback ----------------------------------------------- #

# Sizes chosen so the chunk table is exercised rather than merely built: one
# tensor smaller than a block, one exactly a block, one straddling by a single
# element, and one spanning many blocks. A uniform list would pass on a kernel
# that mishandled every partial chunk.
GROUP_SHAPES = [(896,), (4096,), (4097,), (129, 512), (1,), (64, 63)]


def _group_tensors(dtype, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    params = [
        (torch.randn(s, generator=gen, device=device) * 0.02).to(dtype)
        for s in GROUP_SHAPES
    ]
    updates = [
        torch.randn(s, generator=gen, device=device) * 1e-6 for s in GROUP_SHAPES
    ]
    return params, updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
def test_grouped_matches_the_per_parameter_loop(dtype):
    """One launch must be bit-identical to the loop it replaces.

    The grouped kernel gives element ``i`` of tensor ``t`` the RNG offset
    ``cu_numel[t] + i``, which is what the per-parameter call produces at
    ``rng_offset=cu_numel[t]``. Equality, not a distribution comparison -- a
    grouped kernel that mixed up its chunk offsets would still round correctly on
    average and be completely wrong per element.
    """
    params, updates = _group_tensors(dtype)
    loop = [p.clone() for p in params]
    group = GroupedWriteback(params, updates)
    offsets = group.rng_offsets()
    assert offsets[0] == 0
    assert len(offsets) == len(params)

    group(seed=11, decay=1e-4)
    for param, update, offset in zip(loop, updates, offsets):
        stochastic_round_update_(param, update, 11, decay=1e-4, rng_offset=offset)
    for got, want in zip(params, loop):
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_grouped_covers_every_element_including_partial_chunks():
    """Every element must be written exactly once -- none skipped, none doubled.

    Driven with a decay of 1 and a zero update, so the correct result is exactly
    zero everywhere: any element the chunk table misses keeps its old value and
    any element covered twice is still zero, so this catches the skip. The
    complementary double-write case is caught by the equality test above, which a
    doubled element would fail because it would be rounded twice.
    """
    params, updates = _group_tensors(torch.bfloat16)
    for update in updates:
        update.zero_()
    group = GroupedWriteback(params, updates)
    assert group.numel == sum(p.numel() for p in params)
    group(seed=3, decay=1.0)
    for param in params:
        assert torch.equal(param, torch.zeros_like(param)), param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_grouped_chunk_table_matches_the_shapes():
    """The descriptor count and the per-tensor lengths must follow from the sizes."""
    params, updates = _group_tensors(torch.bfloat16)
    group = GroupedWriteback(params, updates, block=1024)
    expected = sum(-(-p.numel() // 1024) for p in params)
    assert group.n_chunks == expected
    for index, param in enumerate(params):
        covered = group.chunk_len[group.chunk_tensor == index].sum().item()
        assert covered == param.numel()
    assert int(group.chunk_len.max()) <= 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_grouped_rejects_bad_groups():
    params, updates = _group_tensors(torch.bfloat16)
    with pytest.raises(ValueError, match="mixed parameter dtypes"):
        GroupedWriteback([params[0], params[1].half()], updates[:2])
    with pytest.raises(ValueError, match="must be float32"):
        GroupedWriteback(params[:1], [updates[0].bfloat16()])
    with pytest.raises(ValueError, match="shape mismatch"):
        GroupedWriteback(params[:1], [updates[1]])
    with pytest.raises(ValueError, match="at least one parameter"):
        GroupedWriteback([], [])
    with pytest.raises(ValueError, match="params against"):
        GroupedWriteback(params, updates[:2])
    with pytest.raises(ValueError, match="contiguous"):
        wide = torch.zeros(8, 4, dtype=torch.bfloat16, device="cuda").T
        GroupedWriteback([wide], [torch.zeros(4, 8, device="cuda")])
    with pytest.raises(ValueError, match="seed must be non-negative"):
        GroupedWriteback(params, updates)(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
def test_grouped_rebuild_follows_reallocated_tensors():
    """The table holds raw pointers, so a reallocation must be re-read."""
    params, updates = _group_tensors(torch.bfloat16)
    group = GroupedWriteback(params, updates)
    moved = [torch.zeros_like(p) for p in params]
    fresh = [u.clone() for u in updates]
    group.rebuild(moved, fresh)
    group(seed=5, decay=0.0)
    # The originals must be untouched and the new buffers must have moved.
    assert all(
        torch.equal(p, q) for p, q in zip(params, _group_tensors(torch.bfloat16)[0])
    )
    assert any((m != 0).any() for m in moved)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", DTYPES)
def test_rng_counter_is_64_bit(dtype):
    """An offset past 2^31 must be a distinct stream, not a wrapped one.

    The whole-model RNG offset is the global element index, and the ladder's 3B
    and larger rungs pass 2^31 -- an int32 counter would silently hand element
    ``i`` and element ``i + 2^31`` the same dither. Pinned against the specific
    wrap rather than against "large offsets work": ``2**31`` and ``0`` are exactly
    the pair a 32-bit counter would collide.
    """
    torch.manual_seed(0)
    src = ((torch.rand(4096, device="cuda") + 1.0) * 1.0009765).contiguous()
    base = torch.empty_like(src, dtype=dtype)
    wrapped = torch.empty_like(src, dtype=dtype)
    far = torch.empty_like(src, dtype=dtype)
    stochastic_round_(base, src, seed=1, rng_offset=0)
    stochastic_round_(wrapped, src, seed=1, rng_offset=2**31)
    stochastic_round_(far, src, seed=1, rng_offset=2**40)
    assert not torch.equal(base, wrapped), "2^31 collided with 0: counter is 32-bit"
    assert not torch.equal(base, far)
    assert not torch.equal(wrapped, far)
