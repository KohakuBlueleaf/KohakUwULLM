"""``MXFP8Linear``: the drop-in projection, and the three ways it can lie.

The asymmetry pinned here is the module's whole contract. FPROP and DGRAD are
fp8 and inherit e4m3's per-GEMM error; WGRAD is 16-bit and must stay near-exact,
because a weight-gradient error integrates into optimizer state as systematic
bias instead of being resampled every micro-batch.

The other two failures are quieter. The quantized weight is a *cache* derived
from the 16-bit master, so a loop that never refreshes it trains on
initialization-time weights with no symptom in the loss. And the output dtype
must follow autocast rather than the stored weight, or an fp16 arm silently
measures bf16's mantissa.
"""

import pytest
import torch

from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.kernels.mxfp8.linear import VENDOR_K_ALIGN, MXFP8Linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_linear_gradients_track_bf16(dtype):
    """dx and dw follow a bf16 nn.Linear to within MXFP8's own precision.

    Loose on ``dx`` and tight on ``dw`` on purpose, and the asymmetry is the
    contract: FPROP and DGRAD are fp8, so they inherit e4m3's ~3.8% per-GEMM
    relative error, while WGRAD is bf16 and must stay near-exact. A ``dw`` that
    drifted like ``dx`` would mean the fp8 path had leaked into the one product
    whose error integrates into optimizer state.
    """
    torch.manual_seed(0)
    tokens, din, dout = 512, 256, 384
    ref = torch.nn.Linear(din, dout, bias=False, device="cuda", dtype=dtype)
    fp8 = MXFP8Linear(din, dout, dtype=dtype).cuda()
    with torch.no_grad():
        fp8.weight.copy_(ref.weight)
    fp8.refresh_quantized_weight()

    x = torch.randn(tokens, din, device="cuda", dtype=dtype)
    xr = x.clone().requires_grad_()
    xq = x.clone().requires_grad_()
    grad = torch.randn(tokens, dout, device="cuda", dtype=dtype)

    ref(xr).backward(grad)
    fp8(xq).backward(grad)

    assert rel_error(xq.grad, xr.grad) < 0.10
    # WGRAD is bf16; only the accumulation order differs from aten's.
    assert rel_error(fp8.weight.grad, ref.weight.grad) < 0.02


def test_mxfp8_linear_stale_cache_is_detectable():
    """A weight update without ``refresh_quantized_weight`` must be observable.

    The module caches two fp8 layouts, so a forgotten refresh trains against the
    weights from the previous optimizer step and the loss curve still looks
    plausible. This pins that the staleness is detectable rather than silent --
    it is the single most likely way an A/B against bf16 produces a fake result.
    """
    torch.manual_seed(0)
    layer = MXFP8Linear(256, 256).cuda()
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    before = layer(x).clone()

    with torch.no_grad():
        layer.weight.mul_(2.0)
    stale = layer(x)
    assert torch.equal(stale, before), "cache did not go stale; this test is void"

    layer.refresh_quantized_weight()
    fresh = layer(x)
    assert not torch.equal(fresh, before)
    # Doubling the weight doubles the product, so the refresh took effect in the
    # right direction rather than merely perturbing the output.
    assert rel_error(fresh, before * 2.0) < 0.10


@pytest.mark.parametrize("out_features", [192, 64, 320], ids=["192", "64", "320"])
def test_mxfp8_linear_pads_out_features_exactly(out_features):
    """Zero-padding DGRAD's contraction axis must be bit-identical, not merely close.

    ``out_features`` is only DGRAD's K, so it is padded rather than rejected -- which
    is what lets MoE-2B-A370M and Nano-200M-wide (both kv_out=192) use this module
    without reshaping their GQA ratios to suit a kernel. The claim is checked against
    an already-aligned layer whose extra columns are genuinely zero: same arithmetic,
    no padding logic, so bitwise agreement means the zeros contributed nothing.

    A tolerance would not do. The failure this guards is a scale computed over a block
    straddling real and padded columns, which shifts every value in that block by a
    power of two -- small in norm, and exactly what a loose bound absorbs.
    """
    in_features = 256
    padded = -(-out_features // VENDOR_K_ALIGN) * VENDOR_K_ALIGN
    assert padded > out_features, f"out_features={out_features} proves nothing"
    torch.manual_seed(5)

    narrow = MXFP8Linear(in_features, out_features).cuda()
    wide = MXFP8Linear(in_features, padded).cuda()
    with torch.no_grad():
        wide.weight.zero_()
        wide.weight[:out_features] = narrow.weight
    narrow.refresh_quantized_weight()
    wide.refresh_quantized_weight()

    x = torch.randn(64, in_features, device="cuda", dtype=torch.bfloat16)
    xn, xw = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_n, out_w = narrow(xn), wide(xw)
    assert out_n.shape[-1] == out_features, "the forward must return the logical width"
    assert torch.equal(out_n, out_w[:, :out_features]), "padding changed the forward"

    dout = torch.randn_like(out_w)
    dout[:, out_features:] = 0.0
    out_n.backward(dout[:, :out_features])
    out_w.backward(dout)

    assert torch.isfinite(xn.grad).all(), "an all-zero padded MX block produced NaN"
    assert torch.equal(xn.grad, xw.grad), "padding changed dx"
    assert narrow.weight.grad.shape == (out_features, in_features)
    assert torch.equal(narrow.weight.grad, wide.weight.grad[:out_features])


def test_mxfp8_linear_padded_block_stays_finite_at_the_format_edge():
    """The NaN x 0 trap, pushed as hard as e4m3 allows.

    The padded columns form entirely-zero MX blocks whose scale is whatever the
    quantizer emits for ``amax == 0``. Were that an inf or a NaN the product would be
    NaN rather than zero, and invisible to any norm-based check. Values just under
    e4m3's 448 maximise the exponent gap between the real and padded blocks, which is
    the arrangement most likely to expose a degenerate scale -- so this is the worst
    case rather than a typical one.
    """
    layer = MXFP8Linear(256, 192).cuda()
    with torch.no_grad():
        layer.weight.fill_(400.0)
    layer.refresh_quantized_weight()
    x = torch.full((32, 256), 400.0, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_()
    out = layer(x)
    out.backward(torch.full_like(out, 400.0))
    assert torch.isfinite(out).all(), "forward went non-finite"
    assert torch.isfinite(x.grad).all(), "dx went non-finite: scale * 0 became NaN * 0"
    assert torch.isfinite(layer.weight.grad).all()


def test_mxfp8_linear_rejects_in_features_but_not_out_features():
    """The asymmetry is the point, so it is pinned in both directions.

    A future simplification restoring the symmetric check would pass every other test
    in this file -- the presets that need the padding are not the ones the rest of the
    suite exercises.
    """
    with pytest.raises(ValueError, match="in_features"):
        MXFP8Linear(192, 256)
    layer = MXFP8Linear(256, 192)
    assert (layer.out_features, layer.padded_out_features) == (192, 256)
    assert layer.weight.shape == (192, 256), "the parameter must stay logical"


@pytest.mark.parametrize(
    "weight_dtype",
    [torch.float32, torch.bfloat16, torch.float16],
    ids=["w-fp32", "w-bf16", "w-fp16"],
)
@pytest.mark.parametrize(
    "autocast_dtype",
    [torch.bfloat16, torch.float16, None],
    ids=["ac-bf16", "ac-fp16", "ac-off"],
)
def test_mxfp8_linear_output_follows_autocast_not_the_weight(
    weight_dtype, autocast_dtype
):
    """Under autocast the output dtype is autocast's, whatever the weight holds.

    **``w-fp32`` is the case that matters, and its absence is the whole lesson of this
    fix.** An earlier attempt keyed the output off ``weight.dtype`` and passed nine
    isolated checks, because every one of them constructed the layer with bf16 or fp16 --
    the dtypes the author expected. A real ``LMBackbone`` holds **fp32 masters** and
    relies on autocast for compute precision, so weight-dtype resolved to fp32, a
    norm-fed projection handed attention an fp32 ``v``, ``varlen_attn`` refused it, and
    attention dropped silently to SDPA with a quadratic ``(T, T)`` mask. A green suite
    that covers the wrong configuration space is worse than a red one, so the axis the
    production path actually uses is parametrized here rather than assumed.

    Hardcoding bf16 instead fails the other diagonal: it passes ``ac-bf16`` and silently
    downcasts ``ac-fp16``, which matters because fp16 measures 8x more accurate than
    bf16 at the same speed on this card. Only consulting autocast satisfies both.

    ``ac-off`` pins the eager half of the contract, where the input's own dtype governs
    -- otherwise a fix could satisfy every autocast case by ignoring its input entirely.
    """
    layer = MXFP8Linear(256, 256, dtype=weight_dtype).cuda()
    if autocast_dtype is None:
        x = torch.randn(64, 256, device="cuda", dtype=weight_dtype, requires_grad=True)
        out = layer(x)
        expected = weight_dtype
    else:
        # fp32 in, because that is what a norm hands a projection under mixed precision.
        x = torch.randn(64, 256, device="cuda", dtype=torch.float32, requires_grad=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            out = layer(x)
        expected = autocast_dtype
    assert out.dtype == expected, (
        f"weight={weight_dtype} autocast={autocast_dtype} gave {out.dtype}, "
        f"expected {expected}: the output must follow autocast, not the weight"
    )

    out.backward(torch.randn_like(out))
    assert x.grad.dtype == x.dtype, "dx must return in the caller's dtype"
    assert layer.weight.grad.dtype == weight_dtype, "autograd must land dw on the param"


def test_mxfp8_linear_weight_grad_needs_no_fp32_buffer():
    """WGRAD writes the parameter dtype directly -- no fp32 round trip to fold.

    The MoE expert path had one, and removing it was worth 0.94 ms per layer at the 8B
    preset. This records that the dense path never had the same defect, so nobody goes
    looking for the same win twice: ``d2d.t() @ x2d`` is a single cuBLAS call that
    accumulates in fp32 internally and stores the operand dtype. Even a hypothetical
    fp32 buffer here would be 12-49 MiB across the Nano ladder -- inside L2, where the
    round trip is nearly free. The MoE win existed only because its ``dw`` carries an
    expert axis.
    """
    layer = MXFP8Linear(256, 256).cuda()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    layer(x).backward(torch.randn(64, 256, device="cuda", dtype=torch.bfloat16))
    assert layer.weight.grad.dtype == layer.weight.dtype
