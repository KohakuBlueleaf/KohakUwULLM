"""``grouped_mxfp8_gemm``: one launch over ragged per-expert row blocks.

The bar is **bit equality against vendor cuBLAS**, not a tolerance. This is the
one verification layer in the mxfp8 stack that shares no assumption with the
thing it checks, which is why it is the layer that cleared the GEMMs cleanly
while fp64 oracles were still passing a kernel that flushed real gradients to
zero.

The ragged cases are the point: empty and single-row experts, and contraction
lengths both divisible and indivisible by the tile.
"""

import pytest
import torch
from mxfp8_unmasked import unmasked_expert_module

from kohakuwullm.bench.core.timing import ulp_error
from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, quantize_mx
from kohakuwullm.kernels.mxfp8.grouped import (
    FWD_TILE,
    grouped_mxfp8_gemm,
    grouped_mxfp8_reference,
)
from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", [(384, 256, 128), (256, 128, 192)])
def test_grouped_mxfp8_matches_fp64_and_the_vendor(dtype, shape):
    """The bare grouped kernel against fp64, and against a loop of vendor GEMMs.

    Both references, because they catch different things. fp64 catches arithmetic
    but would pass a kernel whose scale layout drifted, since a consistently wrong
    scale is still "MXFP8-accurate" against a reference derived the same way. A
    per-expert loop of verified ``scaled_mm`` calls catches exactly that, and is
    sound here even though no *grouped* vendor kernel exists -- CUTLASS 3.9.2's
    sm_120 grouped block-scaled example segfaults in its own host verification, so
    it is not usable as an oracle.
    """
    rows_per_expert, n, k = shape
    experts = 5
    torch.manual_seed(0)
    counts = torch.tensor([rows_per_expert, 1, 0, rows_per_expert // 3, 7])
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = counts.cumsum(0).to(torch.int32).cuda()
    total = int(counts.sum())

    x = torch.randn(total, k, device="cuda", dtype=dtype)
    w = torch.randn(experts, n, k, device="cuda", dtype=dtype) * k**-0.5
    xq, xs = quantize_mx(x)
    wq, ws = quantize_mx(w.reshape(experts * n, k))
    wq = wq.view(experts, n, k)
    ws = ws.view(experts, n, k // BLOCK_SCALE)

    got = grouped_mxfp8_gemm(xq, xs, wq, ws, offsets, out_dtype=dtype)
    ref = grouped_mxfp8_reference(xq, xs, wq, ws, offsets.cpu())
    # e4m3 ULP, not the output dtype's: the error floor here is set by the fp8
    # operands, and judging a 3-mantissa-bit product against fp16's 11 bits reports
    # 12 ULP for a kernel that is as exact as the format allows. Same convention as
    # `test_mxfp8_matmul_matches_fp64_and_the_vendor`.
    assert ulp_error(got, ref, torch.float8_e4m3fn, "rms") <= 2.5

    off = offsets.tolist()
    for e in range(experts):
        if off[e + 1] <= off[e]:
            continue
        vendor = vendor_mxfp8_matmul(
            xq[off[e] : off[e + 1]], xs[off[e] : off[e + 1]], wq[e], ws[e], dtype
        )
        # Bit-equal, not close: both consume the same e4m3 values and the same
        # ue8m0 exponents through the same MMA, so any difference is a layout bug.
        assert torch.equal(got[off[e] : off[e + 1]], vendor), f"expert {e} differs"


@pytest.mark.parametrize(
    "counts",
    [(64, 0, 0, 0), (1, 1, 1, 1), (0, 0, 0, 130), (17, 3, 200, 1)],
    ids=["one-expert", "single-rows", "last-only", "ragged"],
)
def test_grouped_mxfp8_grid_bound_covers_every_split(counts):
    """``cdiv(M, BLOCK_M) + E`` must cover every distribution of rows over experts.

    The same invariant ``grouped_gemm`` needs, re-pinned here because this kernel
    computes the bound itself from a different BLOCK_M. An uncovered tile is not an
    error: it is a block of rows still holding whatever ``torch.empty`` left, so the
    assertion is exact equality against the reference rather than a norm, which
    would average the bad rows away.
    """
    experts, n, k = len(counts), 128, 64
    torch.manual_seed(0)
    count = torch.tensor(counts)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = count.cumsum(0).to(torch.int32).cuda()
    total = int(count.sum())

    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    xq, xs = quantize_mx(x)
    wq, ws = quantize_mx(w.reshape(experts * n, k))
    got = grouped_mxfp8_gemm(
        xq, xs, wq.view(experts, n, k), ws.view(experts, n, k // BLOCK_SCALE), offsets
    )
    ref = grouped_mxfp8_reference(
        xq,
        xs,
        wq.view(experts, n, k),
        ws.view(experts, n, k // BLOCK_SCALE),
        offsets.cpu(),
    )
    assert torch.equal(
        got, ref.to(got.dtype)
    ), f"{(got != ref.to(got.dtype)).any(-1).sum()} of {total} rows differ"


@pytest.mark.parametrize("k", [160, 96, 288], ids=["160", "96", "288"])
def test_grouped_mxfp8_never_reads_past_the_scale_width(k):
    """An MX scale column beyond the logical width must never be loaded.

    ``tl.cdiv`` rounds the K loop up, so a contraction length that is not a multiple
    of ``BLOCK_K`` gives a final iteration whose scale column is past the end of the
    scale tensor. The paired *value* load is masked to zero there, which is exactly
    what made the read invisible: a garbage exponent times zero is zero, so the
    kernel stayed correct to 1 ULP while reading memory it did not own. On a shared
    card it took the device down with an Xid 43 and killed a training run.

    The probe pads the scale tensor and passes a narrower **view**, so the row stride
    still steps over the padding but the read stays in bounds and under our control.
    ``0xFF`` is NaN in e8m0 and ``nan * 0`` is ``nan``, which is the one way the read
    becomes observable.

    The unmasked twin at the end is what makes the ``isfinite`` above evidence. It
    used to be a claim in this docstring that the test had been verified against the
    unfixed kernel; a probe that stopped reaching the over-read would then pass while
    proving nothing, which is the failure mode both sibling tests execute a control
    for. ``BLOCK_K`` is read from the kernel for the same reason -- a probe sized
    against a hard-coded tile stops covering the loop the day the tile changes.
    """
    experts, n, rows = 4, 128, 40
    groups = k // BLOCK_SCALE
    # However many scale columns the rounded-up loop will touch.
    block_k = FWD_TILE["BLOCK_K"]
    padded = -(-k // block_k) * (block_k // BLOCK_SCALE)
    assert padded > groups, f"k={k} divides BLOCK_K; this case proves nothing"
    torch.manual_seed(0)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = torch.full((experts,), rows).cumsum(0).to(torch.int32).cuda()
    total = rows * experts

    x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    xq, xs = quantize_mx(x)
    wq, ws = quantize_mx(w.reshape(experts * n, k))
    xs_pad = torch.full((total, padded), 0xFF, dtype=torch.uint8, device="cuda")
    xs_pad[:, :groups] = xs
    ws_pad = torch.full((experts * n, padded), 0xFF, dtype=torch.uint8, device="cuda")
    ws_pad[:, :groups] = ws

    got = grouped_mxfp8_gemm(
        xq,
        xs_pad[:, :groups],
        wq.view(experts, n, k),
        ws_pad[:, :groups].view(experts, n, groups),
        offsets,
    )
    assert torch.isfinite(got).all(), "a scale column past the logical width was read"
    ref = grouped_mxfp8_reference(
        xq, xs, wq.view(experts, n, k), ws.view(experts, n, groups), offsets.cpu()
    )
    assert ulp_error(got, ref, torch.float8_e4m3fn, "rms") <= 2.5

    poisoned = unmasked_expert_module("grouped").grouped_mxfp8_gemm(
        xq,
        xs_pad[:, :groups],
        wq.view(experts, n, k),
        ws_pad[:, :groups].view(experts, n, groups),
        offsets,
    )
    assert not torch.isfinite(poisoned).all(), (
        "the unmasked kernel returned finite output, so this probe never reaches "
        "the over-read and the assertions above prove nothing"
    )
