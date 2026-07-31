"""The fused MXFP8 expert path: GEMM1+SwiGLU, then GEMM2+gate+combine.

Four kernels, two of which only run in the backward, against the fp64 oracle in
``mxfp8_oracle.py``.

``dout_scale`` is parametrized over unit *and* training scale (2^-20), and that
is not decoration: every unit-scale case passed on the kernel that was flushing
1.26e-7 gradients to zero through a hard-coded ``tl.float16`` multiply. A
gradient test run only at unit scale tests a regime training never visits.

The masking negative controls are in ``test_kernels_mxfp8_experts_masking.py``,
the separate-GEMM reference path in ``test_kernels_mxfp8_experts_unfused.py``.
"""

import pytest
import torch
from mxfp8_oracle import experts_oracle, routing

from kohakuwullm.bench.core.timing import rel_error, ulp_error
from kohakuwullm.kernels.mxfp8 import quantize_mx
from kohakuwullm.kernels.mxfp8.moe import (
    MXFP8ExpertWeights,
    _launch_wgrad,
    mxfp8_moe_experts,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("skew", [False, True], ids=["balanced", "skewed"])
@pytest.mark.parametrize(
    "geometry",
    [(128, 64), (160, 96)],
    ids=["divides-block-k", "indivisible-block-k"],
)
@pytest.mark.parametrize(
    "dout_scale", [1.0, 2.0**-20], ids=["unit-gradient", "training-gradient"]
)
def test_mxfp8_experts_matches_fp64_algorithm(dtype, skew, geometry, dout_scale):
    """Forward and all four gradients within a few ULP of the fp64 algorithm.

    ``top_k=1`` on purpose: the forward output and ``dx`` both land through 16-bit
    atomics, so at ``top_k>1`` their error is dominated by that accumulation and no
    tolerance on them can pin the GEMMs. The atomic depth gets its own test.

    ``dout_scale`` is a parameter because **the same tolerances must hold at both
    magnitudes**. Every kernel in this path is either MX-block-scaled -- an 8-bit
    exponent per 32 values -- or an fp32 accumulator, so the composition is exactly
    scale-invariant and 2^-20 is not a looser case, it is the same case. Running only
    at ``randn`` is how the WGRAD's fixed ``tl.float16`` multiply survived: a real
    weight-gradient operand measures 1.26e-7 RMS at MoE-1B-A280M, below fp16's 6.1e-5
    smallest normal, and it cost 0.21 nats over 400 steps. The sibling kernel test
    pins the WGRAD alone; this pins the whole composed path, which is what a *new*
    expert implementation would be checked against.
    """
    if dtype is torch.float16 and dout_scale != 1.0:
        # Not a kernel property: an fp16 *container* cannot hold these gradients. `dpre`
        # and `dx` are allocated in the caller's dtype, and storing a 1.26e-7 tensor in
        # fp16 costs 13.8% with 18.9% of entries flushed to zero, against bf16's
        # 0.17% and none. No multiply precision reaches that, so asserting it here
        # would pin the format rather than the kernels.
        pytest.skip("fp16 storage is subnormal at this magnitude; see moe_fp8_diag")
    experts, tokens = 8, 96
    # The second geometry has every contraction length indivisible by BLOCK_K=64:
    # GEMM2 contracts hidden=96, its DGRAD contracts dim=160. Both are multiples of
    # 32, so the MX blocking is legal and only the loop rounding differs.
    dim, hidden = geometry
    offsets, order, token_of, gate = routing(tokens, 1, experts, dtype, skew=skew)
    torch.manual_seed(1)
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype, requires_grad=True)
    w_in = (
        torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    ).requires_grad_()
    w_out = (
        torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    ).requires_grad_()
    gate = gate.requires_grad_()
    packed = MXFP8ExpertWeights(w_in.detach(), w_out.detach())

    out = mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)
    # A power of two, so the operand a kernel reads differs from the unit case only in
    # its exponent and any change in the ULP figure is the kernel's.
    dout = (torch.randn_like(out).float() * dout_scale).to(dtype)
    got = (out,) + torch.autograd.grad(out, [x, w_in, w_out, gate], dout)
    ref = experts_oracle(
        x.detach(),
        w_in.detach(),
        w_out.detach(),
        gate.detach(),
        token_of,
        order,
        offsets,
        hidden,
        dout,
    )
    # WGRAD gets the looser bound: it is the only product whose reduction runs over
    # the token axis, so its accumulation order differs from the reference's most.
    limits = {"out": 8.0, "dx": 8.0, "dw_in": 14.0, "dw_out": 14.0, "dgate": 8.0}
    for name, g, r in zip(limits, got, ref):
        assert ulp_error(g, r, dtype, "rms") < limits[name], (
            f"{name}: {ulp_error(g, r, dtype, 'rms'):.1f} ULP, "
            f"rel {rel_error(g, r):.2e}"
        )

    # Which operand WGRAD contracts, asserted by discrimination rather than by tolerance.
    # A bound alone cannot pin this: the two oracles differ by roughly one fp8 mantissa,
    # so a 14-ULP limit admits both and reverting the operand would stay green. Being
    # *nearer* the 16-bit oracle than the fp8 one is the property that cannot be
    # satisfied by accident, and it is the one that carries the 0.21 nats.
    fp8_operand = experts_oracle(
        x.detach(),
        w_in.detach(),
        w_out.detach(),
        gate.detach(),
        token_of,
        order,
        offsets,
        hidden,
        dout,
        wgrad_fp8=True,
    )
    for idx, name in ((2, "dw_in"), (3, "dw_out")):
        near = rel_error(got[idx], ref[idx])
        far = rel_error(got[idx], fp8_operand[idx])
        assert near < far, (
            f"{name} is nearer the fp8-operand oracle ({far:.2e}) than the 16-bit one "
            f"({near:.2e}); WGRAD is contracting the quantized copy again"
        )
        # The two oracles must actually differ, or the comparison above proves nothing.
        assert rel_error(ref[idx], fp8_operand[idx]) > 1e-3, (
            f"{name}: the two oracles agree to "
            f"{rel_error(ref[idx], fp8_operand[idx]):.2e}; this check is void"
        )


def test_mxfp8_experts_swiglu_halves_are_not_interchangeable():
    """``silu(gate) * value``, in that order, with ``gate`` the first H rows of w_in.

    GEMM1 loads the two halves of ``w_in`` as separate weight tiles and applies the
    activation to one of them. Swapping which is which leaves every shape, every
    norm and every scale identical, so the fp64 oracle is the only thing that would
    catch it -- and only because this asserts the two are actually distinguishable.
    A symmetric test construction would make that check vacuous.
    """
    experts, dim, hidden, tokens = 4, 128, 64, 64
    offsets, order, token_of, gate = routing(tokens, 1, experts, torch.bfloat16)
    torch.manual_seed(2)
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=torch.bfloat16)
    w_in *= dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=torch.bfloat16)
    w_out *= hidden**-0.5
    swapped = torch.cat([w_in[:, hidden:], w_in[:, :hidden]], dim=1).contiguous()

    args = (gate, token_of, order, offsets)
    straight = mxfp8_moe_experts(x, w_in, w_out, *args, MXFP8ExpertWeights(w_in, w_out))
    other = mxfp8_moe_experts(
        x, swapped, w_out, *args, MXFP8ExpertWeights(swapped, w_out)
    )
    assert (
        rel_error(other, straight) > 0.1
    ), "swapping the SwiGLU halves changed nothing; this test cannot fail"


def test_mxfp8_experts_never_reads_sentinel_bucket_rows():
    """Rows past ``offsets[E]`` must not be computed, not merely be scaled to zero.

    A ReLU router parks its inactive slots in a bucket with no expert matrix, and
    those rows hold whatever the allocator left. Multiplying them by their zero gate
    weight looks equivalent and is not: ``nan * 0`` is ``nan``, so one uninitialised
    row would poison its token's output. The check is that a **NaN gate weight** on
    every sentinel pair leaves the output finite and bit-identical.
    """
    experts, dim, hidden, tokens, top_k = 6, 128, 64, 64, 2
    sentinel = 24
    offsets, order, token_of, gate = routing(
        tokens, top_k, experts, torch.bfloat16, sentinel=sentinel
    )
    torch.manual_seed(3)
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=torch.bfloat16)
    w_in *= dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=torch.bfloat16)
    w_out *= hidden**-0.5
    packed = MXFP8ExpertWeights(w_in, w_out)
    real = offsets[: experts + 1]
    assert int(offsets[experts + 1] - offsets[experts]) == sentinel, "no sentinel rows"

    clean = mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, real, packed)
    poisoned = gate.clone()
    poisoned[order[int(offsets[experts]) :].long()] = float("nan")
    dirty = mxfp8_moe_experts(x, w_in, w_out, poisoned, token_of, order, real, packed)
    assert torch.isfinite(
        dirty
    ).all(), "a sentinel row's gate weight reached the output"
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_experts_gate_gradient_is_not_double_scaled(dtype):
    """``dL/dg`` comes from the *unscaled* DGRAD tile, so it must not carry ``g``.

    Fusing the combine into GEMM2's epilogue destroys ``out_sorted``, which the gate
    gradient would otherwise need, and it is recovered from
    ``<dout @ W_out, h> == <dout, h @ W_out.T>``. Taking the already-scaled tile
    instead is a one-token edit that multiplies the gradient by the gate weight a
    second time -- finite, plausible, and wrong. Gate weights here sit near 4, so a
    double scale is a 4x error rather than something a tolerance might absorb.
    """
    experts, dim, hidden, tokens = 4, 128, 64, 64
    offsets, order, token_of, _ = routing(tokens, 1, experts, dtype)
    torch.manual_seed(4)
    gate = torch.full((tokens,), 4.0, device="cuda", dtype=dtype).requires_grad_()
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    packed = MXFP8ExpertWeights(w_in, w_out)

    out = mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)
    dout = torch.randn_like(out)
    (dgate,) = torch.autograd.grad(out, [gate], dout)
    # The output is linear in the gate weight, so <dout, out>/g is the exact sum of
    # every pair's gate gradient -- independent of the kernels, and 4x off if the
    # tile were scaled twice.
    expected = (dout.double() * out.double()).sum() / 4.0
    assert rel_error(dgate.double().sum(), expected) < 0.05


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "k", [160, 128], ids=["indivisible-block-k", "divides-block-k"]
)
def test_mxfp8_wgrad_epilogue_rounding_equals_an_fp32_round_trip(dtype, k):
    """Rounding in the WGRAD epilogue must equal rounding after an fp32 round trip.

    The backward allocates ``dw_in`` / ``dw_out`` in the caller's dtype and lets the
    kernel's ``acc.to(dw_ptr.dtype)`` store round, rather than filling an fp32 buffer
    and casting on the host. At the 8B preset that is 324 MiB of traffic per layer
    against 1620, and two launches fewer -- but only worth taking if it changes
    nothing, so **bit** equality is the assertion rather than a tolerance.

    It holds because the grid gives every ``(expert, n-tile, k-tile)`` exactly one
    program and loops the token axis *inside* it, so the fp32 sum is complete in
    registers before either version rounds. Splitting that loop across programs
    later -- a split-K WGRAD, say -- would make the in-kernel store round partial
    sums, and this is the test that would catch it.
    """
    experts, n, rows = 4, 128, 40
    total = rows * experts
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.full((experts,), rows).cumsum(0).to(torch.int32).cuda()
    torch.manual_seed(9)
    a = torch.randn(total, n, device="cuda", dtype=dtype)
    bq, bs = quantize_mx(torch.randn(total, k, device="cuda", dtype=dtype))
    # Unused with both gather flags off, but the signature still wants them.
    ignored = torch.arange(total, dtype=torch.int32, device="cuda")

    narrow = torch.empty(experts, n, k, device="cuda", dtype=dtype)
    wide = torch.empty(experts, n, k, device="cuda", dtype=torch.float32)
    for dw in (narrow, wide):
        _launch_wgrad(
            a,
            bq,
            bs,
            ignored,
            torch.ones(total, device="cuda", dtype=dtype),
            ignored,
            dw,
            offsets,
            n,
            k,
            a_gather=False,
            b_gather=False,
        )
    assert torch.equal(
        narrow, wide.to(dtype)
    ), "in-epilogue rounding differs from fp32-then-cast; did the grid gain a K split?"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_wgrad_accuracy_is_invariant_to_the_gradient_scale(dtype):
    """The same product at 1e-6 must be as accurate as at 1.0, relative to its scale.

    A GEMM whose operands and accumulator all carry an 8-bit exponent is exactly
    scale-invariant, so this is a *slope* test rather than a tolerance: it compares
    the kernel against itself at two magnitudes and needs no number chosen by hand.

    That is what a fixed tolerance missed. This kernel's other tests feed ``randn``
    at O(1) -- where its ``a.to(tl.float16)`` costs nothing, and the docstring's
    "2.07e-04 against bf16's 1.66e-03" is a real measurement of an input training
    never produces -- while a real weight gradient is 1e-6 or below. fp16's smallest
    *normal* is 6.1e-5, so at that magnitude the cast lands in the subnormal range
    and throws away most of the mantissa. The fp64 oracle could not see it either:
    it mirrored the cast, which is the "reference derived the same way" trap.

    The reference here is built from the operands **as stored**, so the assertion is
    that the kernel loses nothing its input dtype has not already lost. In fp16 that
    makes the test near-vacuous by construction -- an fp16 caller's tensor is already
    subnormal at 1e-6 -- and that is the honest statement of the defect's reach.
    """
    experts, n, k, rows = 4, 128, 128, 64
    total = rows * experts
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.full((experts,), rows).cumsum(0).to(torch.int32).cuda()
    torch.manual_seed(11)
    base_a = torch.randn(total, n, device="cuda")
    b = torch.randn(total, k, device="cuda").to(dtype)
    ignored = torch.arange(total, dtype=torch.int32, device="cuda")

    errors = []
    for scale in (1.0, 2.0**-20):
        a = (base_a * scale).to(dtype)
        dw = torch.empty(experts, n, k, device="cuda", dtype=dtype)
        _launch_wgrad(
            a,
            b,
            None,
            ignored,
            ignored,
            ignored,
            dw,
            offsets,
            n,
            k,
            a_gather=False,
            b_gather=False,
        )
        ref = torch.stack(
            [
                a[e * rows : (e + 1) * rows].double().T
                @ b[e * rows : (e + 1) * rows].double()
                for e in range(experts)
            ]
        )
        errors.append(ulp_error(dw, ref, dtype, "rms"))
    assert errors[1] <= 2.0 * errors[0] + 1.0, (
        f"WGRAD is {errors[1] / max(errors[0], 1e-9):.1f}x less accurate at 2^-20 than "
        f"at 1.0 ({errors[1]:.2f} vs {errors[0]:.2f} ULP); the multiply is narrowing "
        "the operand to a format whose exponent range the gradient has left"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_experts_weight_grads_arrive_in_the_caller_dtype(dtype):
    """No fp32 weight-gradient buffer survives into the returned gradients.

    Pins the allocation, not just the values: autograd would happily accept an fp32
    gradient for a bf16 parameter here and the optimizer would cast it later, so the
    saving is only real if the buffer itself is narrow.
    """
    experts, dim, hidden, tokens = 6, 160, 96, 96
    offsets, order, token_of, gate = routing(tokens, 2, experts, dtype)
    torch.manual_seed(9)
    w_in = (
        torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    ).requires_grad_()
    w_out = (
        torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    ).requires_grad_()
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype, requires_grad=True)
    gate = gate.requires_grad_()
    packed = MXFP8ExpertWeights(w_in.detach(), w_out.detach())

    out = mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)
    grads = torch.autograd.grad(out, [x, w_in, w_out, gate], torch.randn_like(out))
    for name, grad, target in zip(
        ("dx", "dw_in", "dw_out", "dgate"), grads, (x, w_in, w_out, gate)
    ):
        assert grad.dtype == target.dtype, f"{name} came back as {grad.dtype}"


def test_mxfp8_experts_is_cuda_graph_capturable():
    """No host sync anywhere in the fused path, which is what a capture proves.

    The row count per expert is a device value, and reading it to size a grid costs
    an ``.item()`` -- a full host stall per MoE layer and an outright capture
    failure. Every kernel here bounds its own grid from ``M`` and ``E`` instead, and
    a successful capture is the only test of that which cannot be satisfied by
    accident.
    """
    experts, dim, hidden, tokens = 8, 128, 64, 128
    offsets, order, token_of, gate = routing(tokens, 2, experts, torch.bfloat16)
    torch.manual_seed(5)
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=torch.bfloat16)
    w_in *= dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=torch.bfloat16)
    w_out *= hidden**-0.5
    packed = MXFP8ExpertWeights(w_in, w_out)
    call = lambda: mxfp8_moe_experts(  # noqa: E731
        x, w_in, w_out, gate, token_of, order, offsets, packed
    )

    eager = call()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


def test_mxfp8_experts_stale_weight_cache_is_detectable():
    """A weight update without ``refresh`` must be observable, not silent.

    ``MXFP8ExpertWeights`` holds four derived copies, so a forgotten refresh trains
    against the previous step's weights and the loss curve still looks plausible.
    This is the most likely way an A/B against bf16 produces a fake result, so the
    staleness is pinned as *detectable* rather than assumed away.
    """
    experts, dim, hidden, tokens = 4, 128, 64, 64
    offsets, order, token_of, gate = routing(tokens, 1, experts, torch.bfloat16)
    torch.manual_seed(6)
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=torch.bfloat16)
    w_in *= dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=torch.bfloat16)
    w_out *= hidden**-0.5
    packed = MXFP8ExpertWeights(w_in, w_out)
    args = (gate, token_of, order, offsets, packed)

    before = mxfp8_moe_experts(x, w_in, w_out, *args).clone()
    with torch.no_grad():
        w_out.mul_(2.0)
    assert torch.equal(
        mxfp8_moe_experts(x, w_in, w_out, *args), before
    ), "cache did not go stale; this test is void"

    packed.refresh(w_in, w_out)
    after = mxfp8_moe_experts(x, w_in, w_out, *args)
    assert not torch.equal(after, before)
    # Doubling w_out doubles the output, so the refresh moved in the right
    # direction rather than merely perturbing something.
    assert rel_error(after, before * 2.0) < 0.05
