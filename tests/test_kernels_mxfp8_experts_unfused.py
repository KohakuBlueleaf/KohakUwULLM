"""The unfused MXFP8 expert path: the same algorithm as separate GEMMs.

It exists as the thing the fused path is allowed to be checked against, so its
own verification cannot lean on the fused path at all. Hence two independent
bars: bit equality against a per-expert loop of ``torch._scaled_mm``, and the
fp64 oracle in ``mxfp8_oracle.py``.
"""

import pytest
import torch
from mxfp8_oracle import mx_roundtrip, routing

from kohakuwullm.bench.core.timing import rel_error, ulp_error
from kohakuwullm.kernels.mxfp8 import quantize_mx
from kohakuwullm.kernels.mxfp8.grouped import grouped_mxfp8_gemm
from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights
from kohakuwullm.kernels.mxfp8.moe_unfused import mxfp8_moe_experts_unfused

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


def _unfused_oracle(x, w_in, w_out, gate, token_of, order, offsets, hidden, dout):
    """fp64 re-derivation of the **unfused** path, kernel for kernel.

    A separate oracle from :func:`experts_oracle` rather than a flag on it, because
    the two paths differ at four rounding points and a shared oracle would have to
    branch at each: the unfused ``h`` is rounded to the storage dtype by
    ``swiglu_mul`` *before* it is quantized; ``dgate`` comes from the combine's own
    ``<dout, out_sorted>`` instead of the fused identity ``<dout @ W_out, h>``; the
    gate scale is applied to the row gradient *before* it is quantized rather than
    inside a DGRAD epilogue; and WGRAD contracts the ``h`` the forward stored rather
    than one rebuilt from the pre-activation.

    Nothing here mirrors a narrowing the kernels perform *inside* a product. Operands
    are rounded because that is what the kernel reads; every multiply is fp64.
    """
    experts, two_h, dim = w_in.shape
    dtype = x.dtype
    off = offsets.tolist()
    tok, prs = token_of.long(), order.long()
    xd = mx_roundtrip(x)
    xg = xd.index_select(0, tok)
    xg16 = x.double().index_select(0, tok)

    out = torch.zeros(x.shape[0], dim, dtype=torch.float64, device=x.device)
    dx = torch.zeros_like(out)
    dgate = torch.zeros(gate.numel(), dtype=torch.float64, device=x.device)
    dw_in = torch.zeros(experts, two_h, dim, dtype=torch.float64, device=x.device)
    dw_out = torch.zeros(experts, dim, hidden, dtype=torch.float64, device=x.device)
    narrow = lambda t: t.to(dtype).double()  # noqa: E731

    for e in range(experts):
        lo, hi = off[e], off[e + 1]
        if hi <= lo:
            continue
        w_f = mx_roundtrip(w_in[e])
        w_d = mx_roundtrip(w_in[e].t().contiguous())
        o_f = mx_roundtrip(w_out[e])
        o_d = mx_roundtrip(w_out[e].t().contiguous())
        rows_t, pairs = tok[lo:hi], prs[lo:hi]
        weight = gate.double()[pairs]

        pre = narrow(xg[lo:hi] @ w_f.T)
        gate_h, value = pre[:, :hidden], pre[:, hidden:]
        sig = torch.sigmoid(gate_h)
        # One rounding, not two: `swiglu_mul` widens both operands to fp32, forms the
        # whole product there and rounds once on the store.
        h16 = narrow(gate_h * sig * value)
        out_sorted = narrow(mx_roundtrip(h16) @ o_f.T)
        out.index_add_(0, rows_t, narrow(out_sorted * weight[:, None]))

        # `combine_routed`'s backward: the row gradient is scaled and rounded to the
        # storage dtype, and only then quantized -- so unlike the fused path the gate
        # is inside the fp8 operand rather than applied to the accumulator after it.
        grad_rows = narrow(dout.double()[rows_t] * weight[:, None])
        dgate.index_add_(0, pairs, (dout.double()[rows_t] * out_sorted).sum(1))

        dh = narrow(mx_roundtrip(grad_rows) @ o_d.T)
        dpre = narrow(
            torch.cat(
                [dh * value * sig * (1.0 + gate_h * (1.0 - sig)), dh * gate_h * sig],
                dim=1,
            )
        )
        dx.index_add_(0, rows_t, narrow(mx_roundtrip(dpre) @ w_d.T))
        dw_out[e] = grad_rows.T @ h16
        dw_in[e] = dpre.T @ xg16[lo:hi]
    return out, dx, dw_in, dw_out, dgate


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "geometry", [(128, 64), (160, 96)], ids=["divides-block-k", "indivisible-block-k"]
)
def test_mxfp8_unfused_experts_gemms_match_the_vendor(dtype, geometry):
    """Every GEMM the unfused path issues, bit-equal to a loop of vendor GEMMs.

    The whole argument for this path is that it is built out of one primitive that a
    vendor kernel confirms, so the confirmation has to cover the *layouts it is
    called with* and not one representative shape.
    ``test_grouped_mxfp8_matches_fp64_and_the_vendor`` pins the primitive; what is
    new here is the two things only this path exercises: the **gather** on GEMM1's A
    operand, and the two **transposed** weight copies DGRAD reads out of
    ``MXFP8ExpertWeights``. A transposed copy is exactly where a scale layout would
    drift, and it is exactly what an fp64 oracle cannot see -- a consistently wrong
    scale is still MXFP8-accurate against a reference derived the same way.

    Bit equality, not a tolerance: both sides consume the same e4m3 values and the
    same ue8m0 exponents through the same MMA, so any difference is a layout bug.
    """
    experts, tokens, top_k = 6, 96, 2
    dim, hidden = geometry
    offsets, order, token_of, _ = routing(tokens, top_k, experts, dtype, skew=True)
    torch.manual_seed(3)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    packed = MXFP8ExpertWeights(w_in, w_out)
    rows = token_of.numel()

    # The four (operand, weight copy, index) triples the path issues, in order:
    # FPROP1 gathers, FPROP2 and both DGRADs read rows already in sorted order.
    cases = {
        "fprop_w_in": (torch.randn(tokens, dim), packed.in_fwd, token_of),
        "fprop_w_out": (torch.randn(rows, hidden), packed.out_fwd, None),
        "dgrad_w_out": (torch.randn(rows, dim), packed.out_dgrad, None),
        "dgrad_w_in": (torch.randn(rows, 2 * hidden), packed.in_dgrad, None),
    }
    off = offsets.tolist()
    for name, (a, (wq, ws), index) in cases.items():
        aq, asc = quantize_mx(a.to(dtype).cuda())
        got = grouped_mxfp8_gemm(aq, asc, wq, ws, offsets, index=index, out_dtype=dtype)
        for e in range(experts):
            if off[e + 1] <= off[e]:
                continue
            span = slice(off[e], off[e + 1])
            take = (
                torch.arange(rows, device="cuda")[span]
                if index is None
                else index[span]
            )
            vendor = vendor_mxfp8_matmul(
                aq.index_select(0, take.long()).contiguous(),
                asc.index_select(0, take.long()).contiguous(),
                wq[e],
                ws[e],
                dtype,
            )
            assert torch.equal(got[span], vendor), f"{name}: expert {e} differs"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("skew", [False, True], ids=["balanced", "skewed"])
@pytest.mark.parametrize(
    "geometry", [(128, 64), (160, 96)], ids=["divides-block-k", "indivisible-block-k"]
)
@pytest.mark.parametrize(
    "dout_scale", [1.0, 2.0**-20], ids=["unit-gradient", "training-gradient"]
)
def test_mxfp8_unfused_experts_matches_fp64_algorithm(
    dtype, skew, geometry, dout_scale
):
    """The composed forward and all four gradients, within a few ULP of fp64.

    ``top_k=1`` for the same reason the fused test uses it: the forward output and
    ``dx`` both land through an accumulation whose depth is ``top_k``, so at
    ``top_k>1`` their error is dominated by it and no tolerance on them pins a GEMM.

    ``dout_scale`` carries **the same tolerances**, not looser ones: every kernel this
    path composes is either MX-block-scaled or an fp32 accumulator, so the composition
    is exactly scale-invariant and 2^-20 is the same case at a different exponent.
    The point of running it is that this path was written to replace one whose defect
    was invisible at ``randn`` scale -- a real weight-gradient operand measures 1.26e-7
    RMS at MoE-1B-A280M -- so a new implementation that is only checked at O(1) has not
    been checked where the old one broke.
    """
    if dtype is torch.float16 and dout_scale != 1.0:
        # The fp16 *container*, not any kernel: `swiglu_mul` stores `dpre` in the
        # caller's dtype, and a 1.26e-7 tensor there costs 13.8% with 18.9% of entries
        # flushed to zero against bf16's 0.17% and none. Asserting it would pin the
        # format. Composing the path differently does not reach it -- the same skip
        # the fused test takes, for the same reason.
        pytest.skip("fp16 storage is subnormal at this magnitude; see moe_fp8_diag")
    experts, tokens = 8, 96
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

    out = mxfp8_moe_experts_unfused(
        x, w_in, w_out, gate, token_of, order, offsets, packed
    )
    # A power of two, so the operand a kernel reads differs from the unit case only in
    # its exponent and any change in the ULP figure is the kernel's.
    dout = (torch.randn_like(out).float() * dout_scale).to(dtype)
    got = (out,) + torch.autograd.grad(out, [x, w_in, w_out, gate], dout)
    ref = _unfused_oracle(
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
    limits = {"out": 8.0, "dx": 8.0, "dw_in": 14.0, "dw_out": 14.0, "dgate": 8.0}
    for name, g, r in zip(limits, got, ref):
        assert ulp_error(g, r, dtype, "rms") < limits[name], (
            f"{name}: {ulp_error(g, r, dtype, 'rms'):.1f} ULP, "
            f"rel {rel_error(g, r):.2e}"
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_mxfp8_unfused_experts_ignores_a_sentinel_bucket(dtype):
    """Rows a sentinel bucket owns must reach neither the output nor any gradient.

    They are routed but never computed -- ``offsets`` is truncated to the real
    experts, so no tile the grid resolves covers them -- and every buffer they sit in
    comes from ``torch.empty``.

    ``isfinite`` is **not** the assertion, and the first version of this test used it
    and passed with the guard removed: uninitialised memory is usually a plausible
    finite number, so a NaN check tests the allocator's mood. The parking is arranged
    so that the first ``parked // top_k`` tokens have *every* slot in the sentinel
    bucket, which makes their output and their input gradient exactly zero by
    construction. Zero is a value garbage cannot imitate, and no atomic ordering can
    perturb it, because nothing real adds to those rows at all.
    """
    experts, tokens, top_k, dim, hidden = 6, 64, 2, 128, 64
    # A multiple of top_k, and taken from the front, because `routing` parks the
    # first `parked` *pairs* and `token_of` is `pair // top_k` -- so this is exactly
    # the condition that tokens 0..parked/top_k-1 are routed nowhere.
    parked = 8
    offsets, order, token_of, gate = routing(
        tokens, top_k, experts, dtype, sentinel=parked
    )
    torch.manual_seed(5)
    w_in = torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=dtype) * dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device="cuda", dtype=dtype) * hidden**-0.5
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    packed = MXFP8ExpertWeights(w_in, w_out)
    real = int(offsets[experts])

    def run(valid_rows):
        xg = x.clone().requires_grad_()
        wi = w_in.clone().requires_grad_()
        wo = w_out.clone().requires_grad_()
        g = gate.clone().requires_grad_()
        out = mxfp8_moe_experts_unfused(
            xg,
            wi,
            wo,
            g,
            token_of,
            order,
            offsets[: experts + 1],
            packed,
            valid_rows,
        )
        return (out,) + torch.autograd.grad(out, [xg, wi, wo, g], torch.ones_like(out))

    out, dx, dw_in, dw_out, dgate = run(offsets[experts : experts + 1])
    orphans = parked // top_k
    assert (out[:orphans] == 0).all(), "the combine scattered an uncomputed row"
    # This is the one `_maybe_zeros` exists for: DGRAD writes `(M, dim)` and the
    # gather's backward scatters it, so an `empty` tail lands on a real token.
    assert (dx[:orphans] == 0).all(), "DGRAD's scatter spread an uncomputed row"
    assert (dgate[order[real:].long()] == 0).all(), "a parked pair got a gate gradient"
    for name, tensor in (("dw_in", dw_in), ("dw_out", dw_out)):
        assert torch.isfinite(tensor).all(), f"{name} took an uncomputed row"
