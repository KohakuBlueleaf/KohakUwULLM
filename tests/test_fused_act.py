"""Fused activation epilogues that emit MXFP8."""

import pytest
import torch

from kohakuwullm.kernels.mxfp8.fused_act import swiglu_mx
from kohakuwullm.kernels.mxfp8.quantize import BLOCK_SCALE, quantize_mx


def _dequant(q, s, k):
    exp = (s.int() - 127).float()
    return (
        q.float().reshape(q.shape[0], -1, BLOCK_SCALE) * torch.exp2(exp)[:, :, None]
    ).reshape(-1, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(512, 1024), (4096, 8192), (37, 64)])
def test_swiglu_mx_is_no_worse_than_unfused(shape):
    """Fusing keeps the intermediate in fp32, so it cannot be less accurate."""
    m, k = shape
    torch.manual_seed(0)
    g = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = (g.double() * torch.sigmoid(g.double())) * u.double()
    scale = ref.abs().max().item()

    qf, sf = swiglu_mx(g, u)
    fused = (_dequant(qf, sf, k).double() - ref).abs().max().item() / scale

    unf = torch.nn.functional.silu(g.float()) * u.float()
    qu, su = quantize_mx(unf.to(torch.bfloat16))
    plain = (_dequant(qu, su, k).double() - ref).abs().max().item() / scale

    assert fused < 0.05, f"{shape}: fused rel {fused}"
    assert fused <= plain * 1.05, f"{shape}: fused {fused} worse than unfused {plain}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_swiglu_mx_rejects_bad_shapes():
    """Mismatched operands and a K that blocks cannot tile are caller errors."""
    g = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="!="):
        swiglu_mx(g, torch.randn(8, 32, device="cuda", dtype=torch.bfloat16))
    bad = torch.randn(8, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of"):
        swiglu_mx(bad, bad)
