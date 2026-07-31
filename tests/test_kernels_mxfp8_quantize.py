"""The MXFP8 format itself: quantization, the dense GEMM, the swizzled layout.

Everything here is upstream of any model code. If these fail, every other mxfp8
test is measuring a broken format rather than a broken kernel.

Two of these are *negative controls* rather than accuracy checks. The
scale-column over-read is invisible to any ULP check by construction -- the
paired value load is masked to zero on the final K iteration, and a garbage
exponent times zero is zero -- so the only way to know the mask works is to
execute a copy of the kernel with the mask removed and require that it fails.
The round-to-nearest variant is the same shape of argument in the other
direction: a source rewrite that matched nothing would leave the two arms
bit-identical and report that the rounding choice does not matter.
"""

import pytest
import torch
import triton
from mxfp8_unmasked import load_unmasked_pq

from kohakuwullm.bench.core.timing import rel_error, ulp_error
from kohakuwullm.bench.vendor.mxfp8_rounding import load_round_to_nearest
from kohakuwullm.bench.vendor.vendor_moe import scale_views
from kohakuwullm.kernels.mxfp8 import (
    BLOCK_SCALE,
    E4M3_MAX,
    mxfp8_matmul,
    mxfp8_matmul_pq,
    quantize_mx,
    quantize_mx_vendor,
)
from kohakuwullm.kernels.mxfp8.interop import (
    as_vendor_scales,
    expected_swizzled_numel,
    vendor_mxfp8_matmul,
    vendor_mxfp8_matmul_swizzled,
)
from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear

# The tile constant and the autotune configs are internals of the quantizer, so
# they come from the module that owns them rather than the package facade -- the
# facade re-exports the public surface, and pulling privates through it is what
# broke this import the last time the package was reorganised.
from kohakuwullm.kernels.mxfp8.quantize import (
    QUANT_TILE,
    _pq_configs,
    _quantize_mx_kernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


# e4m3 carries 3 mantissa bits, so its worst relative half-spacing is 2^-4 and
# every element of a block is bounded by that fraction of the block's amax. This
# is the right yardstick for MXFP8: judging it in ULP of the *input* dtype
# reports fp16 as 8x worse than bf16 purely because fp16's eps is 8x smaller,
# when the quantizer discards far more precision than either format holds and
# the measured accuracy is identical.
E4M3_BLOCK_ULP = 2.0**-4


def _dequantize_mx(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """e4m3 values x ue8m0 block exponents -> fp32, for round-trip checks."""
    scale = torch.exp2(scales.float() - 127.0)
    return q.float() * scale.repeat_interleave(BLOCK_SCALE, dim=1)


def _block_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-32-block amax, broadcast back over the block for elementwise bounds."""
    blocks = x.float().abs().reshape(x.shape[0], -1, BLOCK_SCALE)
    return blocks.amax(-1).repeat_interleave(BLOCK_SCALE, dim=1)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_scale_rounds_up_never_to_nearest(dtype):
    """The block scale exponent must be ceil(log2(amax/448)), never round().

    Round-to-nearest picks a scale one binade too small whenever the amax sits in
    the lower half of a binade, which maps the amax above 448 and clips it. NVIDIA
    measured an 843M model diverging at 300B tokens on exactly this.

    The amax multiplier is 1.4 and the choice is load-bearing. ``round`` only goes
    the wrong way while amax/448 is below sqrt(2), so a multiplier at or above
    1.4143 would let both roundings agree and pin nothing. A multiplier just
    *above* 1.0 also fails to discriminate: at 1.01 both paths dequantize to
    exactly 448.0 and the test passes either way.
    """
    torch.manual_seed(0)
    amax = E4M3_MAX * 1.4
    x = torch.full((4, BLOCK_SCALE), amax / 8.0, device="cuda", dtype=dtype)
    x[:, 5] = amax

    q, scales = quantize_mx(x)
    # Scoped to this construction: "nothing equals 448" is NOT a general
    # invariant even under round-up, because a legitimate value anywhere in
    # (432, 464) rounds to exactly 448 in e4m3. Here round-up puts the amax at
    # 313.6 -> 320, far from the top of the format, so equality means clipping.
    assert q.float().abs().max().item() < E4M3_MAX

    deq = _dequantize_mx(q, scales)
    recovered = deq.abs().max().item()
    assert abs(recovered - amax) <= E4M3_BLOCK_ULP * amax, (
        f"amax {amax} came back as {recovered}: the scale clipped it, which is "
        "what round-to-nearest does and round-up does not"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_blocks_are_32_wide_along_k_only(dtype):
    """A block's amax must come from its own 32 elements along K, nothing else.

    A quantizer that reduced amax over the whole tile, or that reshaped K into
    strided rather than contiguous groups, still returns plausible numbers on
    random input. It only fails when one block holds values a reducing bug would
    swallow: here block 0 is ~1e3 and block 1 is ~1e-3, so a shared scale drives
    every element of block 1 to zero.
    """
    torch.manual_seed(0)
    rows = 8
    x = torch.empty(rows, 2 * BLOCK_SCALE, device="cuda", dtype=dtype)
    x[:, :BLOCK_SCALE] = 1.0e3
    x[:, BLOCK_SCALE:] = 1.0e-3

    q, scales = quantize_mx(x)
    assert scales.shape == (rows, 2)
    # Each block picked its own exponent, ~20 binades apart.
    assert (scales[:, 0].int() - scales[:, 1].int()).min().item() >= 15

    deq = _dequantize_mx(q, scales)
    tiny = deq[:, BLOCK_SCALE:]
    assert (tiny != 0).all(), "the small block underflowed: amax leaked across K"
    assert rel_error(tiny, x[:, BLOCK_SCALE:].float()) < E4M3_BLOCK_ULP

    # Contiguous, not strided: with a (ROWS, BLOCK_SUB, groups) reshape the two
    # blocks would interleave and each would inherit the other's magnitude.
    assert rel_error(deq[:, :BLOCK_SCALE], x[:, :BLOCK_SCALE].float()) < E4M3_BLOCK_ULP


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_quantize_round_trips_within_e4m3(dtype):
    """Dequantizing by the returned scales recovers the input to e4m3 precision.

    The bound is relative to each block's amax, not to each element: an element
    far below its block's amax lands on an e4m3 subnormal or on zero, so its own
    relative error reaches 1.0 while its absolute error stays tiny. Measured
    worst case here is 0.0588 of the block amax against the 0.0625 bound.
    """
    torch.manual_seed(0)
    x = torch.randn(128, 256, device="cuda", dtype=dtype)
    # Spread across binades so blocks disagree about their exponents.
    x = x * torch.exp2(torch.randint(-6, 6, x.shape, device="cuda").to(dtype))

    q, scales = quantize_mx(x)
    assert q.shape == x.shape and q.dtype == torch.float8_e4m3fn
    assert scales.shape == (x.shape[0], x.shape[1] // BLOCK_SCALE)
    assert scales.dtype == torch.uint8

    deq = _dequantize_mx(q, scales)
    assert ((deq - x.float()).abs() <= E4M3_BLOCK_ULP * _block_amax(x)).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_quantize_mx_costs_one_launch_at_any_shape(dtype):
    """A row count this call has never seen must not make the quantizer search.

    ``M`` is the varlen token count, so in a packed stream every step is a new
    shape. Under ``@triton.autotune(key=["M", "K"])`` each one re-ran the eight-config
    search, and each search is ~840 ms of device time -- almost all of it
    ``do_bench``'s 256 MB L2 flush. Against ``cache8192``'s 174 distinct token counts
    in 400 steps that was 365 ms/step, which was the whole of the MoE fp8 regression.

    Counting launches rather than asserting the decorator is absent: any mechanism
    that benchmarks per shape fails this, and the launch count is the thing that
    actually hurt.
    """
    # One call at a shape *not* used below, so the JIT compile is already paid and
    # what the profiler sees is dispatch alone.
    quantize_mx(torch.randn(64, BLOCK_SCALE, device="cuda", dtype=dtype))
    torch.cuda.synchronize()

    for rows in (65, 66, 67):
        x = torch.randn(rows, 2 * BLOCK_SCALE, device="cuda", dtype=dtype)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            quantize_mx(x)
        torch.cuda.synchronize()
        launches = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launches) == 1, (
            f"M={rows} issued {len(launches)} device ops, not 1: the quantizer is "
            f"searching on a shape it has not seen. {launches[:6]}"
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_quantize_mx_result_does_not_depend_on_the_tile(dtype):
    """The tile is a throughput knob and never a numerics one.

    What makes that true is that an MX block is 32 wide along K and a tile spans
    whole blocks, so no reduction ever crosses a tile edge -- which is also why
    replacing the autotune with a fixed tile could not move a single bit of any
    training run. A ``BLOCK_K`` that is not a multiple of ``BLOCK_SCALE`` would break
    it, so that is asserted directly rather than left to the tilings sampled here.
    """
    assert QUANT_TILE["BLOCK_K"] % BLOCK_SCALE == 0

    torch.manual_seed(0)
    # 16 MX blocks wide and a row count no tile divides, so the three tilings below
    # disagree about where every boundary falls: 4 column tiles, 2, and 1-with-masking,
    # each over a partial final row tile. At one block wide they would all reduce to a
    # single masked tile and the test would pin nothing.
    rows, cols = 200, 16 * BLOCK_SCALE
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    # Per-row binade spread, so a tiling that leaked amax across rows would show up.
    x = x * torch.exp2(torch.randint(-8, 8, (rows, 1), device="cuda").to(dtype))
    expect_q, expect_s = quantize_mx(x)

    for block_m, block_k, warps in ((8, 1024, 8), (32, 128, 4), (128, 256, 8)):
        q = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.empty(rows, cols // BLOCK_SCALE, device="cuda", dtype=torch.uint8)
        _quantize_mx_kernel[(triton.cdiv(rows, block_m), triton.cdiv(cols, block_k))](
            x,
            q,
            s,
            rows,
            cols,
            x.stride(0),
            x.stride(1),
            q.stride(0),
            q.stride(1),
            s.stride(0),
            s.stride(1),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_SUB=BLOCK_SCALE,
            num_warps=warps,
        )
        # Bit-identical, not close: e4m3 through `torch.equal` needs a uint8 view,
        # because comparing float8 tensors goes through a promotion that hides a
        # one-ULP difference the whole point of this test is to catch.
        assert torch.equal(
            q.view(torch.uint8), expect_q.view(torch.uint8)
        ), f"tile {block_m}x{block_k} changed the quantized values"
        assert torch.equal(s, expect_s), f"tile {block_m}x{block_k} changed the scales"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", [(256, 128, 512), (128, 256, 64), (192, 96, 256)])
def test_mxfp8_matmul_matches_fp64_and_the_vendor(dtype, shape):
    """All three MXFP8 paths agree, and agree with fp64 to e4m3 precision.

    The fp64 reference is what pins accuracy, but it cannot catch a wrong scale
    *layout*: two implementations with mismatched layouts are both
    "MXFP8-accurate" against fp64 and still disagree with each other. So the
    pre-quantized path is pinned to the fused one and to ``F.scaled_mm``, both of
    which are bit-identical to it on every shape measured -- quantization is the
    same arithmetic in all three, only its placement differs.
    """
    torch.manual_seed(0)
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=dtype) * 0.05
    # fp64 on the operands only; an fp64 (m,n) accumulator at bench sizes has
    # OOMed this repo before, so the shapes here stay small deliberately.
    ref = a.double() @ b.double().T

    fused = mxfp8_matmul(a, b)
    aq, a_scale = quantize_mx(a)
    bq, b_scale = quantize_mx(b)
    pq = mxfp8_matmul_pq(aq, a_scale, bq, b_scale, dtype)

    assert ulp_error(fused, ref, torch.float8_e4m3fn, mode="rms") <= 2.5
    assert ulp_error(pq, ref, torch.float8_e4m3fn, mode="rms") <= 2.5
    assert ulp_error(pq, fused.double(), torch.float8_e4m3fn, mode="rms") == 0.0

    vendor = vendor_mxfp8_matmul(aq, a_scale, bq, b_scale, dtype)
    assert ulp_error(pq, vendor.double(), torch.float8_e4m3fn, mode="rms") == 0.0


def test_mxfp8_rejects_k_not_a_multiple_of_the_block():
    """A K that straddles a block cannot be scaled, so it must not be accepted."""
    a = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be a multiple of 32"):
        quantize_mx(a)
    with pytest.raises(ValueError, match="must be a multiple of 32"):
        mxfp8_matmul(a, b)
    with pytest.raises(ValueError, match="contraction mismatch"):
        mxfp8_matmul(
            torch.randn(64, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(32, 32, device="cuda", dtype=torch.bfloat16),
        )


def _pq_scale_columns_touched(k: int) -> int:
    """Widest scale-column extent the pre-quantized K loop can reach for this K.

    Taken over **every** autotune config, not the winning one: the tuning sweep
    runs them all, so a probe sized for the chosen BLOCK_K would let a deeper
    config read past the padding it owns and turn a regression test into the
    out-of-bounds read it is testing for.
    """
    return max(
        -(-k // config.kwargs["BLOCK_K"]) * (config.kwargs["BLOCK_K"] // BLOCK_SCALE)
        for config in _pq_configs()
    )


@pytest.mark.parametrize("k", [160, 96, 288], ids=["160", "96", "288"])
def test_mxfp8_pq_never_reads_past_the_scale_width(k):
    """An MX scale column beyond the logical width must never be loaded.

    The dense twin of ``test_grouped_mxfp8_never_reads_past_the_scale_width``, and
    the same defect: ``tl.cdiv`` rounds the K loop up, so a K that is not a
    multiple of ``BLOCK_K`` gives a final iteration whose scale column is past the
    end of the scale tensor, while the paired *value* load is masked to zero there.
    That is what made it invisible -- a garbage exponent times zero is zero, so the
    kernel stayed correct to a few ULP while reading memory it did not own.

    No shipped preset is exposed, because every model K here is a multiple of 64;
    what is exposed is any autotuned ``BLOCK_K`` that does not divide K.

    The probe pads the scale tensor and passes a narrower **view**, so the row
    stride still steps over the padding but the read stays inside our own
    allocation. ``0xFF`` is NaN in e8m0 and ``nan * 0`` is ``nan``, the one way the
    read becomes observable at all.
    """
    m, n = 128, 128
    groups = k // BLOCK_SCALE
    padded = _pq_scale_columns_touched(k)
    assert padded > groups, f"k={k} divides every BLOCK_K; this case proves nothing"
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    aq, a_scale = quantize_mx(a)
    bq, b_scale = quantize_mx(b)
    as_pad = torch.full((m, padded), 0xFF, dtype=torch.uint8, device="cuda")
    as_pad[:, :groups] = a_scale
    bs_pad = torch.full((n, padded), 0xFF, dtype=torch.uint8, device="cuda")
    bs_pad[:, :groups] = b_scale

    got = mxfp8_matmul_pq(aq, as_pad[:, :groups], bq, bs_pad[:, :groups])
    assert torch.isfinite(got).all(), "a scale column past the logical width was read"
    assert ulp_error(got, a.double() @ b.double().T, torch.float8_e4m3fn, "rms") <= 2.5
    # The padding must not have perturbed the answer either: same operands, same
    # kernel, contiguous scales. A mask that zeroed a *live* scale column would
    # still be finite and still be wrong.
    assert torch.equal(got, mxfp8_matmul_pq(aq, a_scale, bq, b_scale))

    poisoned = load_unmasked_pq().mxfp8_matmul_pq(
        aq, as_pad[:, :groups], bq, bs_pad[:, :groups]
    )
    assert not torch.isfinite(poisoned).all(), (
        "the unmasked kernel returned finite output, so this probe never reaches "
        "the over-read and the assertions above prove nothing"
    )


def test_mxfp8_round_to_nearest_variant_clips_what_round_up_keeps():
    """The rewritten quantizer must actually round to nearest, not to +inf.

    The whole point of a round-up-vs-round-to-nearest training A/B is that the
    two arms differ. A source rewrite that matched nothing would leave them
    bit-identical, and the experiment would then report that the rounding choice
    does not matter -- a false negative with no symptom.

    So this pins the *direction*, on the construction from
    ``test_mxfp8_scale_rounds_up_never_to_nearest``: at ``amax = 1.4 * 448`` the
    nearest exponent is one binade too small, which maps the amax above the
    format's max and clips it. Round-up recovers the amax; round-to-nearest must
    not.
    """
    rtn = load_round_to_nearest()
    torch.manual_seed(0)
    amax = E4M3_MAX * 1.4
    x = torch.full((4, 128), amax / 8.0, device="cuda", dtype=torch.bfloat16)
    x[:, 5] = amax

    up_q, up_s = quantize_mx(x)
    rtn_q, rtn_s = rtn.quantizer.quantize_mx(x)
    assert not torch.equal(
        up_s, rtn_s
    ), "the rewrite produced the shipped rounding; the variant is void"
    # One binade smaller, and clipped against 448 as a direct consequence.
    assert (rtn_s.int() == up_s.int() - 1).all()
    assert rtn_q.float().abs().max().item() == E4M3_MAX
    assert up_q.float().abs().max().item() < E4M3_MAX

    # The clip is what costs accuracy: measured 2.0% relative error under
    # round-up against 28.6% under round-to-nearest on this block.
    assert rel_error(_dequantize_mx(up_q, up_s), x.float()) < 0.05
    assert rel_error(_dequantize_mx(rtn_q, rtn_s), x.float()) > 0.20

    # The swizzled path shares the quantizer, so it must have moved too.
    assert not torch.equal(
        quantize_mx_vendor(x)[1].view(torch.uint8),
        rtn.quantizer.quantize_mx_vendor(x)[1].view(torch.uint8),
    )

    # The linear copy has to consume the rewritten quantizer, not the shipped one.
    layer_up = MXFP8Linear(256, 256).cuda()
    layer_rtn = rtn.linear.MXFP8Linear(256, 256).cuda()
    with torch.no_grad():
        layer_rtn.weight.copy_(layer_up.weight)
    layer_rtn.refresh_quantized_weight()
    probe = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    probe[:, 3] *= 400.0  # a per-block amax high in its binade, where the two differ
    assert not torch.equal(layer_up(probe), layer_rtn(probe))

    # The A/B instantiates these classes directly -- the ``_Bf16Input`` wrapper it used
    # to go through is retired, because the module consults autocast itself. What has to
    # hold is that the rewritten copy **inherits that contract**: if it did not, the
    # round-to-nearest arm would differ from round-up in dtype as well as in rounding,
    # and the loss gap could not be attributed to the rounding the arm exists to test.
    with torch.autocast("cuda", torch.bfloat16):
        auto_up = layer_up(probe.float())
        auto_rtn = layer_rtn(probe.float())
    assert auto_up.dtype == auto_rtn.dtype == torch.bfloat16
    assert not torch.equal(auto_up, auto_rtn)
    # Autocast changes the dtype contract and nothing else about the arm.
    assert torch.equal(auto_rtn, layer_rtn(probe))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_swizzle_matches_reference_swizzle(dtype):
    """The in-kernel scale swizzle must equal the reference permutation exactly.

    A wrong permutation is not a precision bug: it feeds cuBLAS scales belonging
    to other rows, which stays finite and plausible-looking. Measured at 0.43
    relative error on a kernel that differed from this only in the offset, so the
    assertion is bit-equality rather than a tolerance.
    """
    torch.manual_seed(0)
    for m, k in ((256, 512), (129, 256), (4096, 1280)):
        x = torch.randn(m, k, device="cuda", dtype=dtype)
        q_nat, s_nat = quantize_mx(x)
        q_swz, s_swz = quantize_mx_vendor(x)
        assert torch.equal(q_nat, q_swz)
        assert s_swz.numel() == expected_swizzled_numel(m, k)
        assert torch.equal(
            s_swz.view(torch.uint8), as_vendor_scales(s_nat).view(torch.uint8)
        )

    # End to end: the vendor GEMM fed by the fused swizzle must agree with our
    # Triton kernel fed by the natural layout, since both consume the same values.
    a = torch.randn(512, 512, device="cuda", dtype=dtype)
    b = torch.randn(256, 512, device="cuda", dtype=dtype)
    ref = a.float() @ b.float().T
    aq, asc = quantize_mx_vendor(a)
    bq, bsc = quantize_mx_vendor(b)
    got = vendor_mxfp8_matmul_swizzled(aq, asc, bq, bsc)
    assert (got.float() - ref).norm() / ref.norm() < 0.06


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_vendor_scale_views_slice_one_swizzled_buffer(dtype):
    """A 128-aligned row slice of a ``quantize_mx_vendor`` scale buffer is a valid operand.

    This is what makes a per-expert loop of vendor GEMMs cost **two** quantize launches
    instead of ``2 + E``: ``quantize_mx_vendor`` writes swizzle tile
    ``(row_tile, col_tile)`` at ``(row_tile*col_tiles + col_tile)*512``, so a span of
    whole row-tiles is contiguous and an expert starting on a 128-aligned row can take
    its scales as a view. The count need not be aligned, only the start.

    Bit-equality, not closeness, and against an **independently derived** operand --
    the expert's rows quantized alone in natural layout and swizzled. Both consume the
    same e4m3 values and the same ue8m0 exponents through the same cuBLAS call, so
    anything but equality is a layout bug, and a layout bug is exactly what an fp64
    reference cannot see.

    The unaligned case is pinned because it is silent, not loud: ``scaled_mm`` checks
    the scale *element count*, and an unaligned start produces a correctly sized view of
    the wrong rows. The GEMM then runs and returns MXFP8-plausible numbers scaled by a
    neighbouring expert's exponents.
    """
    k, n = 768, 256
    torch.manual_seed(0)
    # Ragged counts on aligned starts: the property under test is about starts, and
    # equal counts would not distinguish the two.
    starts, counts = [0, 128, 384, 640], [100, 256, 200, 77]
    total = 896

    x = torch.randn(total, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype) * k**-0.5
    xq, x_swizzled = quantize_mx_vendor(x)
    wq, w_swizzled = quantize_mx_vendor(w)
    wq_nat, ws_nat = quantize_mx(w)

    views = scale_views(x_swizzled, starts, counts, k)
    for expert, (start, count) in enumerate(zip(starts, counts)):
        got = vendor_mxfp8_matmul_swizzled(
            xq[start : start + count], views[expert], wq, w_swizzled, dtype
        )
        xq_alone, xs_alone = quantize_mx(x[start : start + count].contiguous())
        ref = vendor_mxfp8_matmul(xq_alone, xs_alone, wq_nat, ws_nat, dtype)
        assert torch.equal(got, ref), f"expert {expert} at row {start} differs"

    with pytest.raises(ValueError, match="not 128-aligned"):
        scale_views(x_swizzled, [64], [128], k)


# --- layout invariance of the vendor-swizzled quantizer ---------------------
#
# ``quantize_mx_vendor`` used to call ``x.contiguous()`` unconditionally. The
# kernel tiles in 2D and takes both strides, so it coalesces along whichever axis
# is unit-stride and a strided read is free -- 0.398 ms against 0.389 ms
# contiguous on a 149553x896 fp32 read. The copy was therefore pure cost: 1.22 ms
# on ``weight.t()``, 8% of a 96-module MXFP8 weight refresh, plus one launch per
# module.
#
# Removing it is only valid if the output is *bit-identical*. The quantized
# weight feeds every DGRAD in the model, so a quantizer that rounded differently
# by layout would make the fp8 backward depend on how its input happened to be
# stored. Hence byte comparisons rather than tolerances.

SHAPES = ((896, 896), (128, 896), (4864, 896), (896, 2432))


def raw(t: torch.Tensor) -> torch.Tensor:
    """fp8 dtypes have no ``equal`` promotion, so compare the underlying bytes."""
    return t.view(torch.uint8)


@pytest.mark.parametrize("out_features,in_features", SHAPES)
def test_quantize_is_bit_identical_read_strided_or_copied(out_features, in_features):
    torch.manual_seed(0)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
    transposed = weight.t()
    k = transposed.shape[-1]
    if k % (BLOCK_SCALE * 4):
        pytest.skip(f"K={k} is not swizzle-legal")
    assert not transposed.is_contiguous(), "fixture must exercise the strided path"

    strided_q, strided_s = quantize_mx_vendor(transposed)
    copied_q, copied_s = quantize_mx_vendor(transposed.contiguous())
    assert torch.equal(raw(strided_q), raw(copied_q))
    assert torch.equal(raw(strided_s), raw(copied_s))


def test_a_doubly_strided_input_is_still_copied_and_correct():
    """The one layout with no unit-stride axis: the guard must fall back to copying.

    A column slice of a larger matrix strides by the parent's row length on one
    axis and by the slice step on the other, so neither is 1 and the kernel has
    nothing to coalesce along. It must still produce what the contiguous path
    produces.
    """
    torch.manual_seed(0)
    base = torch.randn(512, 4096, device="cuda", dtype=torch.float32)
    view = base[:, ::2]
    assert view.stride(0) != 1 and view.stride(1) != 1

    q, s = quantize_mx_vendor(view)
    ref_q, ref_s = quantize_mx_vendor(view.contiguous())
    assert torch.equal(raw(q), raw(ref_q))
    assert torch.equal(raw(s), raw(ref_s))
