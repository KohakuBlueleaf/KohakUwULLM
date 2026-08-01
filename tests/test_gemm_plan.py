"""Planner and Stream-K GEMM: legality of the plan, accuracy of the kernel."""

import pytest
import torch

from kohakuwullm.kernels.gemm import (
    RTX_5090,
    Device,
    StreamKGemm,
    TunedGemm,
    plan,
    plan_topk,
)
from kohakuwullm.kernels.gemm.tune import _CACHE, clear, key

SHAPES = [
    (4096, 4096, 4096),
    (2048, 1024, 3072),
    (6000, 3000, 5000),
    (1024, 8192, 512),
    (333, 517, 1024),
]


def test_plan_is_legal_and_deterministic():
    """A plan fits the card's budgets and does not depend on when it was made."""
    for m, n, k in SHAPES:
        p = plan(m, n, k, RTX_5090)
        assert p == plan(m, n, k, RTX_5090)
        smem = (p.bm + p.bn) * p.bk * 2 * (p.stages - 1)
        assert smem <= RTX_5090.smem_per_cta
        assert p.bm * p.bn / (32 * p.warps) <= 168
        assert 1 <= p.cta_per_sm <= 8
        assert p.ctas <= RTX_5090.sms * p.cta_per_sm
        assert 0 <= p.sk_ctas <= p.ctas
        assert p.tiles == -(-m // p.bm) * -(-n // p.bn)


def test_plan_requires_measured_fields():
    """A card whose ceilings were never measured is rejected, not guessed at."""
    bare = Device(
        name="unknown",
        sms=1,
        regs_per_sm=1,
        smem_per_cta=1,
        smem_per_sm=1,
        max_threads_per_sm=1,
        l2_bytes=1,
        mma_peak=float("nan"),
        dram_bw=1.0,
        l2_bw=1.0,
    )
    with pytest.raises(ValueError, match="mma_peak"):
        plan(64, 64, 64, bare)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_streamk_matches_fp64(shape, dtype):
    """The kernel is as accurate as the output dtype allows, on every shape."""
    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    ref = a.double() @ b.double()
    g = StreamKGemm(m, n, k, RTX_5090, a.element_size())
    out = g(a, b)
    rel = (out.double() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.02, f"{shape} {dtype}: rel {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_streamk_scratch_does_not_carry_over():
    """Repeated calls stay accurate, so the fixup scratch is cleared each time.

    The atomic reduction makes the result nondeterministic, so this pins accuracy
    across calls rather than bitwise equality.
    """
    m, n, k = 4096, 4096, 4096
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    ref = a.double() @ b.double()
    scale = ref.abs().max().item()
    g = StreamKGemm(m, n, k, RTX_5090)
    for _ in range(4):
        rel = (g(a, b).double() - ref).abs().max().item() / scale
        assert rel < 0.02, f"drifted to rel {rel}"


def test_plan_topk_is_ranked():
    """The shortlist is ordered by prediction and never repeats a candidate."""
    cands = plan_topk(4096, 4096, 4096, RTX_5090, 2, 6)
    assert len(cands) == 6
    preds = [c.predicted_tflops for c in cands]
    assert preds == sorted(preds, reverse=True)
    assert len({(c.tile, c.warps, c.stages) for c in cands}) == len(cands)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_tuned_gemm_is_cached_and_correct():
    """Tuning runs once per shape and its winner is still numerically right."""
    clear()
    m, n, k = 2048, 1024, 3072
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    g = TunedGemm(m, n, k, RTX_5090, shortlist=3)
    assert key(m, n, k, 2) in _CACHE
    again = TunedGemm(m, n, k, RTX_5090, shortlist=3)
    assert again.plan == g.plan
    ref = a.double() @ b.double()
    rel = (g(a, b).double() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < 0.02
    clear()
