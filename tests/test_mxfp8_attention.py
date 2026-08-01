"""MXFP8 varlen flash attention: correctness, gradients, and the error it costs.

The negative tests are the ones that matter. ``test_output_is_differentiable``
exists because the kernel was once called outside an autograd node: training ran,
loss fell, and every gradient below ``o_proj`` was silently zero.
"""

import pytest
import torch

from kohakuwullm import SeqInfo
from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.kernels.mxfp8.attention import mxfp8_varlen_attn
from kohakuwullm.kernels.mxfp8.attention_quant import column_mean, quantize_heads
from kohakuwullm.models.components.attention import MXFP8Attention, VarlenAttention

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cu(lengths, device):
    ends = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)
    return torch.cat([torch.zeros(1, device=device, dtype=torch.int32), ends]).int()


def _reference(q, k, v, cu, causal=True, window=None):
    """fp64 attention, one document at a time, over the exact inputs given."""
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float64)
    rep = q.shape[1] // k.shape[1]
    for i in range(len(cu) - 1):
        a, b = int(cu[i]), int(cu[i + 1])
        qq = q[a:b].double().transpose(0, 1)
        kk = k[a:b].double().repeat_interleave(rep, 1).transpose(0, 1)
        vv = v[a:b].double().repeat_interleave(rep, 1).transpose(0, 1)
        s = qq @ kk.transpose(-1, -2) * (q.shape[-1] ** -0.5)
        idx = torch.arange(b - a, device=q.device)
        mask = (
            idx[:, None] < idx[None, :]
            if causal
            else torch.zeros_like(s[0], dtype=torch.bool)
        )
        if window is not None:
            mask = mask | (idx[:, None] - idx[None, :] >= window)
        out[a:b] = (
            torch.softmax(s.masked_fill(mask, float("-inf")), -1) @ vv
        ).transpose(0, 1)
    return out


@cuda_only
def test_output_is_differentiable():
    """The entry point must build an autograd node, not just launch a kernel."""
    torch.manual_seed(0)
    q = torch.randn(128, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(128, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(128, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = mxfp8_varlen_attn(q, k, v, _cu([128], "cuda"), 128)
    assert out.grad_fn is not None
    out.sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None, f"{name} received no gradient"
        assert t.grad.abs().sum() > 0, f"{name} received an all-zero gradient"


@cuda_only
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lengths", [[1024], [700, 1300, 512, 1536]])
def test_forward_and_gradients_are_bounded(lengths, causal):
    """e4m3 on Q and K costs error in the output and in all three gradients."""
    torch.manual_seed(0)
    heads, kv_heads, head_dim = 8, 2, 64
    total = sum(lengths)
    cu = _cu(lengths, "cuda")
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, v = (t.requires_grad_() for t in (q, k, v))

    out = mxfp8_varlen_attn(q, k, v, cu, max(lengths), causal=causal)
    ref = _reference(q, k, v, cu, causal)
    assert torch.isfinite(out).all()
    # L2, not max-abs: non-causal attention averages over every key, so its peak
    # output is a cancellation and a max-relative metric reads 2x high there.
    # e4m3 on both score operands costs about 4%; bf16 varlen is 0.002.
    assert rel_error(out, ref) < 0.05, f"forward {rel_error(out, ref)}"

    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), g)
    fp64 = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref_grads = torch.autograd.grad(_reference(*fp64, cu, causal), fp64, g.double())
    for name, got, want in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert rel_error(got, want) < 0.06, f"{name} {rel_error(got, want)}"


@cuda_only
def test_packed_attention_does_not_cross_documents():
    """Perturbing document 0 must not change document 1's output, bit for bit."""
    torch.manual_seed(0)
    lengths = [96, 160]
    cu = _cu(lengths, "cuda")
    q = torch.randn(256, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(256, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(256, 2, 64, device="cuda", dtype=torch.bfloat16)

    base = mxfp8_varlen_attn(q, k, v, cu, 160)
    v2 = v.clone()
    v2[:96] += 10.0
    moved = mxfp8_varlen_attn(q, k, v2, cu, 160)
    assert torch.equal(base[96:], moved[96:]), "document 1 changed when document 0 did"
    assert not torch.equal(base[:96], moved[:96])


@cuda_only
def test_sliding_window_limits_receptive_field():
    """A key outside the window must not reach the query, bit for bit."""
    torch.manual_seed(0)
    cu = _cu([256], "cuda")
    q = torch.randn(256, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(256, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(256, 2, 64, device="cuda", dtype=torch.bfloat16)

    base = mxfp8_varlen_attn(q, k, v, cu, 256, window=32)
    v2 = v.clone()
    v2[0] += 10.0
    moved = mxfp8_varlen_attn(q, k, v2, cu, 256, window=32)
    assert torch.equal(base[-1], moved[-1])
    assert not torch.equal(base[0], moved[0])


@cuda_only
@pytest.mark.parametrize("causal", [False, True])
def test_smoothing_k_reduces_error_and_is_exact_under_softmax(causal):
    """Subtracting K's channel mean shifts each score row by a constant, so it
    cannot change the reference, and it must lower the quantization error."""
    torch.manual_seed(0)
    total, heads, head_dim = 1024, 4, 64
    cu = _cu([total], "cuda")
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = (
        torch.randn(total, heads, head_dim, device="cuda")
        + 3.0 * torch.randn(1, heads, head_dim, device="cuda")
    ).to(torch.bfloat16)
    v = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    ref = _reference(q, k, v, cu, causal)

    plain = mxfp8_varlen_attn(q, k, v, cu, total, causal=causal, smooth_k=False)
    smooth = mxfp8_varlen_attn(q, k, v, cu, total, causal=causal, smooth_k=True)
    e_plain = rel_error(plain, ref)
    e_smooth = rel_error(smooth, ref)
    assert e_smooth < e_plain, f"causal={causal}: {e_smooth} not better than {e_plain}"

    # Exact statement of the cancellation, in fp64: shifting every key by a
    # constant channel vector leaves softmax(Q K^T) unchanged.
    shift = k.double().mean(0, keepdim=True)
    assert torch.allclose(
        _reference(q, k.double(), v, cu, causal),
        _reference(q, k.double() - shift, v, cu, causal),
        rtol=1e-9,
        atol=1e-12,
    )
    mu = column_mean(k)
    assert mu.shape == (heads, head_dim) and mu.dtype is torch.float32


@cuda_only
def test_quantize_heads_blocks_along_head_dim():
    """Scales must run along the contraction axis, one per 32 head elements."""
    x = torch.randn(128, 3, 64, device="cuda", dtype=torch.bfloat16)
    q, s = quantize_heads(x)
    assert q.shape == (128, 3, 64) and s.shape == (128, 3, 2)
    deq = (
        q.float().reshape(128, 3, 2, 32)
        * torch.exp2((s.int() - 127).float())[..., None]
    )
    rel = (
        deq.reshape(x.shape).double() - x.double()
    ).abs().max().item() / x.abs().max()
    assert rel < 0.10, rel


@cuda_only
def test_module_matches_varlen_and_reaches_the_input():
    """Through the module contract: close to bf16 varlen, and x gets a gradient."""
    torch.manual_seed(0)
    dim, heads, kv_heads, head_dim = 256, 8, 2, 32
    lengths = torch.tensor([37, 128, 64, 5], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    x = torch.randn(int(lengths.sum()), dim, device="cuda", dtype=torch.bfloat16)

    base = VarlenAttention(dim, heads, kv_heads=kv_heads, head_dim=head_dim)
    base = base.cuda().to(torch.bfloat16)
    mx = MXFP8Attention(dim, heads, kv_heads=kv_heads, head_dim=head_dim)
    mx = mx.cuda().to(torch.bfloat16)
    mx.load_state_dict(base.state_dict())

    x = x.requires_grad_()
    out = mx(x, info)
    ref = base(x.detach(), info)
    rel = (out.detach().float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < 0.05, rel
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


@cuda_only
def test_rejects_unsupported_head_dim():
    """head_dim must hold whole scale groups."""
    q = torch.randn(64, 2, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        mxfp8_varlen_attn(q, q, q, _cu([64], "cuda"), 64)
