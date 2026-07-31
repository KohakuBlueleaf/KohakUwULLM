"""Attention: the backends must agree, and none of them may cross a document.

The negative tests are the ones that matter. A backend that forgets the document
boundary still trains -- just worse, for reasons the loss curve will not explain
-- so ``test_packed_attention_does_not_cross_documents`` perturbs document 0 and
requires document 1's output to be **bit**-identical rather than close.
"""

import pytest
import torch

from kohakuwullm import SeqInfo
from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.models.components.attention import (
    FlexAttention,
    SDPAAttention,
    VarlenAttention,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------- attention


@cuda_only
def test_attention_backends_agree():
    """varlen / sdpa / flex must produce the same output on the same input."""
    torch.manual_seed(0)
    dim, heads, kv_heads, head_dim = 256, 8, 2, 32
    lengths = torch.tensor([37, 128, 64, 5], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    x = torch.randn(int(lengths.sum()), dim, device="cuda", dtype=torch.bfloat16)

    base = VarlenAttention(dim, heads, kv_heads=kv_heads, head_dim=head_dim)
    base = base.cuda().to(torch.bfloat16)
    outs = {}
    for name, cls in (
        ("varlen", VarlenAttention),
        ("sdpa", SDPAAttention),
        ("flex", FlexAttention),
    ):
        module = cls(dim, heads, kv_heads=kv_heads, head_dim=head_dim)
        module = module.cuda().to(torch.bfloat16)
        module.load_state_dict(base.state_dict())
        outs[name] = module(x, info).float()

    assert rel_error(outs["sdpa"], outs["varlen"]) < 2e-2
    assert rel_error(outs["flex"], outs["varlen"]) < 2e-2


@cuda_only
def test_packed_attention_does_not_cross_documents():
    """Perturbing document 0 must not change document 1's output.

    This is the invariant that separates a packed batch from a concatenated
    one. A backend that forgets the boundary still trains -- slightly worse, for
    reasons nothing in the loss curve explains -- so it is pinned explicitly.
    """
    torch.manual_seed(0)
    dim, heads = 128, 4
    lengths = torch.tensor([32, 48], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    module = (
        VarlenAttention(dim, heads, kv_heads=2, head_dim=32).cuda().to(torch.bfloat16)
    )

    x = torch.randn(80, dim, device="cuda", dtype=torch.bfloat16)
    out_a = module(x, info)

    x2 = x.clone()
    x2[:32] = torch.randn(32, dim, device="cuda", dtype=torch.bfloat16)
    out_b = module(x2, info)

    assert torch.equal(out_a[32:], out_b[32:]), "document 1 changed when document 0 did"
    assert not torch.equal(out_a[:32], out_b[:32])


@cuda_only
def test_sdpa_packed_matches_varlen_document_masking():
    """The SDPA fallback must apply the block-diagonal mask, not plain causal."""
    torch.manual_seed(0)
    dim, heads = 128, 4
    lengths = torch.tensor([24, 40], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    varlen = (
        VarlenAttention(dim, heads, kv_heads=2, head_dim=32).cuda().to(torch.bfloat16)
    )
    sdpa = SDPAAttention(dim, heads, kv_heads=2, head_dim=32).cuda().to(torch.bfloat16)
    sdpa.load_state_dict(varlen.state_dict())

    x = torch.randn(64, dim, device="cuda", dtype=torch.bfloat16)
    assert rel_error(sdpa(x, info).float(), varlen(x, info).float()) < 2e-2


@cuda_only
def test_sliding_window_limits_receptive_field():
    torch.manual_seed(0)
    dim, heads, window = 128, 4, 16
    lengths = torch.tensor([64], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    module = (
        VarlenAttention(dim, heads, kv_heads=2, head_dim=32, sliding_window=window)
        .cuda()
        .to(torch.bfloat16)
    )

    x = torch.randn(64, dim, device="cuda", dtype=torch.bfloat16)
    out_a = module(x, info)
    x2 = x.clone()
    x2[0] = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    out_b = module(x2, info)
    # Token 0 is outside the window of the last token.
    assert torch.equal(out_a[-1], out_b[-1])


@cuda_only
def test_attention_sink_changes_output_and_stays_finite():
    torch.manual_seed(0)
    dim, heads = 128, 4
    info = SeqInfo.from_lengths(torch.tensor([32], dtype=torch.int32), "cuda")
    x = torch.randn(32, dim, device="cuda", dtype=torch.bfloat16)

    plain = (
        VarlenAttention(dim, heads, kv_heads=2, head_dim=32).cuda().to(torch.bfloat16)
    )
    sinky = (
        VarlenAttention(dim, heads, kv_heads=2, head_dim=32, sink=True)
        .cuda()
        .to(torch.bfloat16)
    )
    sinky.load_state_dict(
        {**plain.state_dict(), "sink": torch.zeros(heads).cuda()}, strict=False
    )

    out = sinky(x, info)
    assert torch.isfinite(out.float()).all()
    # A zero sink logit still absorbs probability mass, so it must differ.
    assert not torch.equal(out, plain(x, info))
