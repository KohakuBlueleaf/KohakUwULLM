"""MXFP8 flash attention forward: correctness and the error it costs."""

import pytest
import torch

from kohakuwullm.kernels.mxfp8.attention import mxfp8_attention, quantize_rows


def _reference(q, k, v, causal):
    """fp64 attention over the exact bf16 inputs."""
    s = (q.double() @ k.double().T) * (q.shape[-1] ** -0.5)
    if causal:
        n = q.shape[0]
        s = s.masked_fill(
            torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), 1),
            float("-inf"),
        )
    return torch.softmax(s, dim=-1) @ v.double()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(512, 64), (2048, 128)])
def test_forward_error_is_bounded(shape, causal):
    """e4m3 on Q and K costs error, but a bounded amount of it."""
    t, head = shape
    torch.manual_seed(0)
    q = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    ref = _reference(q, k, v, causal)
    out, lse, _ = mxfp8_attention(q, k, v, causal=causal)

    scale = ref.abs().max().item()
    rel = (out.double() - ref).abs().max().item() / scale
    rms = ((out.double() - ref) ** 2).mean().sqrt().item() / (
        ref**2
    ).mean().sqrt().item()
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    # e4m3 carries 3 mantissa bits, so a per-32 block quantization of both GEMM
    # operands lands well inside a tenth; bf16 SDPA is about 0.003 for contrast.
    assert rel < 0.10, f"{shape} causal={causal}: relmax {rel}"
    assert rms < 0.06, f"{shape} causal={causal}: rms {rms}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("causal", [False, True])
def test_smoothing_k_reduces_error_and_is_exact_under_softmax(causal):
    """Subtracting K's channel mean shifts each score row by a constant, so it
    cannot change the reference, and it must lower the quantization error."""
    t, head = 1024, 64
    torch.manual_seed(0)
    q = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    k = (
        torch.randn(t, head, device="cuda") + 3.0 * torch.randn(1, head, device="cuda")
    ).to(torch.bfloat16)
    v = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    ref = _reference(q, k, v, causal)
    scale = ref.abs().max().item()

    plain, _, _ = mxfp8_attention(q, k, v, causal=causal, smooth_k=False)
    smooth, _, mu = mxfp8_attention(q, k, v, causal=causal, smooth_k=True)
    e_plain = (plain.double() - ref).abs().max().item() / scale
    e_smooth = (smooth.double() - ref).abs().max().item() / scale
    assert e_smooth < e_plain, f"causal={causal}: {e_smooth} not better than {e_plain}"

    # Exact statement of the cancellation, in fp64: shifting every key by a
    # constant channel vector leaves softmax(Q K^T) unchanged.
    shift = k.double().mean(0, keepdim=True)
    s_plain = (q.double() @ k.double().T) * (head**-0.5)
    s_shift = (q.double() @ (k.double() - shift).T) * (head**-0.5)
    if causal:
        upper = torch.triu(torch.ones(t, t, device=q.device, dtype=torch.bool), 1)
        s_plain = s_plain.masked_fill(upper, float("-inf"))
        s_shift = s_shift.masked_fill(upper, float("-inf"))
    assert torch.allclose(
        torch.softmax(s_plain, -1),
        torch.softmax(s_shift, -1),
        rtol=1e-9,
        atol=1e-12,
    )
    assert mu is not None and mu.shape == (head,) and mu.dtype is torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_causal_mask_actually_masks():
    """A change beyond the diagonal must not move an earlier row's output."""
    t, head = 256, 64
    torch.manual_seed(0)
    q = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    base, _, _ = mxfp8_attention(q, k, v, causal=True)
    v2 = v.clone()
    v2[t // 2 :] += 10.0
    moved, _, _ = mxfp8_attention(q, k, v2, causal=True)
    assert torch.equal(base[: t // 2], moved[: t // 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_quantize_rows_blocks_along_head_dim():
    """Scales must run along the contraction axis, one per 32 head elements."""
    t, head = 128, 128
    x = torch.randn(t, head, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_rows(x)
    assert q.shape == (t, head) and s.shape == (t, head // 32)
    deq = q.float().reshape(t, -1, 32) * torch.exp2((s.int() - 127).float())[:, :, None]
    rel = (
        deq.reshape(t, head).double() - x.double()
    ).abs().max().item() / x.abs().max().item()
    assert rel < 0.10, rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_rejects_unsupported_shapes():
    """head_dim and block_n must both hold whole scale groups."""
    q = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        mxfp8_attention(q, q, q)
    ok = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block_n"):
        mxfp8_attention(ok, ok, ok, block_n=48)
