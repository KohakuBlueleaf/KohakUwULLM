"""Negative control: no expert kernel may read an MX scale column it does not own.

Separate from the accuracy tests because it is a different kind of test. The
defect is invisible to every accuracy check by construction -- ``tl.cdiv`` rounds
the K loop up, the paired value load is masked to zero on the final iteration,
and a garbage exponent times zero is zero -- so the kernel stays correct to 1 ULP
while reading memory it does not own.

``compute-sanitizer`` does not see it either: the overrun is one or two bytes
inside PyTorch's 512-byte-rounded allocation block, so the access is genuinely
legal. Only poisoned padding (0xFF is NaN in e8m0, and ``nan * 0`` is ``nan``)
makes it observable, and only a copy of the kernel with every mask removed
proves the probe reaches it at all.
"""

import pytest
import torch
from mxfp8_oracle import routing
from mxfp8_unmasked import load_unmasked_experts

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, experts, experts_bwd
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)


def _poison_expert_scales(packed, dim: int, hidden: int) -> int:
    """Repack every weight-scale tensor as a narrow view of a ``0xFF``-padded one.

    Returns how many of the four were actually widened. The padding is sized per
    tensor from the ``BLOCK_K`` of the kernel that reads it, so each view is exactly
    as wide as its logical scale width and the row stride steps over poison.
    """
    # (attribute, contraction length, the reading kernel's BLOCK_K).
    plan = (
        ("in_fwd", dim, experts.GEMM1_TILE["BLOCK_K"]),
        ("out_fwd", hidden, experts.GEMM2_TILE["BLOCK_K"]),
        ("out_dgrad", dim, experts_bwd.GEMM2_DGRAD_TILE["BLOCK_K"]),
        ("in_dgrad", 2 * hidden, experts_bwd.GEMM1_DGRAD_TILE["BLOCK_K"]),
    )
    widened = 0
    for attr, contraction, block_k in plan:
        values, scales = getattr(packed, attr)
        groups = contraction // BLOCK_SCALE
        touched = -(-contraction // block_k) * (block_k // BLOCK_SCALE)
        if touched <= groups:
            continue
        padded = torch.full(
            scales.shape[:-1] + (touched,),
            0xFF,
            dtype=torch.uint8,
            device=scales.device,
        )
        padded[..., :groups] = scales
        setattr(packed, attr, (values, padded[..., :groups]))
        widened += 1
    return widened


@pytest.mark.parametrize(
    "geometry", [(160, 96), (224, 160)], ids=["dim160-h96", "dim224-h160"]
)
def test_mxfp8_experts_never_read_past_the_scale_width(geometry):
    """No expert kernel may load an MX scale column beyond the logical width.

    Same defect as :func:`test_grouped_mxfp8_never_reads_past_the_scale_width`, in the
    four fused kernels, where it was **not** pinned: the only coverage was an ULP
    check at an indivisible geometry, and this read is invisible to any ULP check by
    construction. ``tl.cdiv`` rounds the K loop up, the paired value load is masked to
    zero on the final iteration, and a garbage exponent times zero is zero. So the
    kernel stays correct to 1 ULP while reading memory it does not own.

    ``compute-sanitizer --tool memcheck`` does not see it either, which is why this
    is a poisoned-padding probe and not a sanitizer run: the overrun is one or two
    bytes and lands inside PyTorch's 512-byte-rounded allocation block, so the access
    is genuinely legal. Only ``0xFF`` -- NaN in e8m0, and ``nan * 0`` is ``nan`` --
    makes it observable.

    The backward is included because two of the four kernels only run there, and the
    gradients are what a training run would carry the NaN into.
    """
    experts, tokens = 6, 64
    dim, hidden = geometry
    offsets, order, token_of, gate = routing(tokens, 2, experts, torch.bfloat16)
    torch.manual_seed(11)
    x = torch.randn(
        tokens, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    w_in = (
        torch.randn(experts, 2 * hidden, dim, device="cuda", dtype=torch.bfloat16)
        * dim**-0.5
    ).requires_grad_()
    w_out = (
        torch.randn(experts, dim, hidden, device="cuda", dtype=torch.bfloat16)
        * hidden**-0.5
    ).requires_grad_()
    gate = gate.requires_grad_()

    packed = MXFP8ExpertWeights(w_in.detach(), w_out.detach())
    widened = _poison_expert_scales(packed, dim, hidden)
    # `in_dgrad` contracts 2*hidden, which is a multiple of 64 whenever hidden is a
    # multiple of the 32-wide MX block -- so GEMM1's DGRAD can never round its loop
    # up and has nothing to over-read. Three of four is the most this geometry can
    # poison, and a test that expected four would be unsatisfiable.
    assert widened == 3, f"{widened} of 4 scale tensors widened; the probe is vacuous"

    out = mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)
    grads = torch.autograd.grad(out, [x, w_in, w_out, gate], torch.randn_like(out))
    for name, tensor in zip(("out", "dx", "dw_in", "dw_out", "dgate"), (out, *grads)):
        assert torch.isfinite(
            tensor
        ).all(), f"{name} read a scale column past the logical width"

    # The control: the same probe against the same code with the masks taken out. An
    # assertion that the fixed kernel is finite proves nothing on its own -- the
    # padding could be unreachable, or the poison could be multiplied by a zero that
    # is not a NaN-maker -- so the unfixed twin has to be shown to fail it.
    #
    # The **forward output** is what the control asserts on, not the gradients. The
    # forward runs only GEMM1 and GEMM2, so a NaN there is attributable to the two
    # loop-rounding masks this probe poisons for; a gradient would also carry WGRAD,
    # whose `mask_g` guards an out-of-range grid tile rather than a rounded-up loop
    # and would let the control pass for the wrong reason.
    unmasked = load_unmasked_experts()
    packed_bad = unmasked.MXFP8ExpertWeights(w_in.detach(), w_out.detach())
    _poison_expert_scales(packed_bad, dim, hidden)
    with torch.no_grad():
        bad = unmasked.mxfp8_moe_experts(
            x, w_in, w_out, gate, token_of, order, offsets, packed_bad
        )
    assert not torch.isfinite(
        bad
    ).all(), "the unmasked forward stayed finite; this test cannot fail"


def test_mxfp8_experts_rejects_mismatched_expert_geometry():
    torch.manual_seed(7)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    w_in = torch.randn(4, 96, 128, device="cuda", dtype=torch.bfloat16)
    w_out = torch.randn(4, 128, 64, device="cuda", dtype=torch.bfloat16)
    offsets, order, token_of, gate = routing(32, 1, 4, torch.bfloat16)
    packed = MXFP8ExpertWeights(w_in, torch.randn_like(w_out))
    with pytest.raises(ValueError, match="not 2x"):
        mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)
    odd = torch.randn(4, 96, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of"):
        mxfp8_moe_experts(
            x,
            odd,
            torch.randn(4, 128, 48, device="cuda", dtype=torch.bfloat16),
            gate,
            token_of,
            order,
            offsets,
            packed,
        )
    # fp32 is the dtype that actually arrives, and on a 16-bit model too: a
    # normalization op is on autocast's fp32 list, so a caller that does not consult
    # autocast hands this a norm's fp32 output whatever its weights are. Without the
    # check it surfaced as a bare KeyError from inside a launcher, which sends the
    # reader to the tile constants rather than to the missing cast.
    good_out = torch.randn(4, 128, 96, device="cuda", dtype=torch.bfloat16)
    good_in = torch.randn(4, 192, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="consult autocast"):
        mxfp8_moe_experts(
            x.float(),
            good_in,
            good_out,
            gate,
            token_of,
            order,
            offsets,
            MXFP8ExpertWeights(good_in, good_out),
        )
