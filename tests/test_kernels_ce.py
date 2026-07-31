"""The two kernels between the hidden states and the loss.

Both exist for the same reason and are tested for the same reason: at vocab
65536 a materialized logit tensor decides the batch size, so both walk the
vocabulary in tiles and neither may ever hold ``(tokens, vocab)``. A tiled
reduction is also exactly where a low-precision accumulation goes wrong quietly,
which is why the reference here is fp64 and the tolerance is in ULP.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from kohakuwullm.bench.core.timing import rel_error, ulp_error
from kohakuwullm.kernels.loss.chunked_ce import chunked_linear_cross_entropy
from kohakuwullm.kernels.loss.zloss import logsumexp_square
from kohakuwullm.models.head import LMHead

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_zloss_matches_materialized(dtype):
    """Chunked z-loss equals the materialized version, forward and backward."""
    torch.manual_seed(0)
    n, dim, vocab = 512, 256, 4096
    x = (torch.randn(n, dim, device="cuda", dtype=dtype) * 0.5).requires_grad_()
    w = (torch.randn(vocab, dim, device="cuda", dtype=dtype) * 0.02).requires_grad_()
    labels = torch.randint(0, vocab, (n,), device="cuda")
    labels[::5] = -100

    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    got = logsumexp_square(x, w, labels, -100)
    lse = torch.logsumexp(F.linear(x_ref, w_ref).float(), dim=-1)
    ref = torch.where(labels != -100, lse, torch.zeros_like(lse)).pow(2)

    assert rel_error(got, ref) < 1e-3
    assert got[labels == -100].abs().max() == 0

    grad = torch.randn(n, device="cuda")
    got.backward(grad)
    ref.backward(grad)
    assert rel_error(x.grad, x_ref.grad) < 5e-2
    assert rel_error(w.grad, w_ref.grad) < 5e-2


def _ce_fp64_reference(x, w, target, dloss, ignore_index=-100):
    """Loss, dX and dW in fp64 for the exact bf16/fp16 inputs given."""
    xd = x.detach().double().requires_grad_()
    wd = w.detach().double().requires_grad_()
    logits = xd @ wd.t()
    picked = logits.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
    loss = torch.logsumexp(logits, dim=-1) - picked
    loss = torch.where(target == ignore_index, torch.zeros_like(loss), loss)
    loss.backward(dloss.double())
    return loss.detach(), xd.grad, wd.grad


# Tile geometries chosen so every branch is exercised: whole-batch, ragged token
# chunks, ragged vocabulary blocks, and a partially filled logit cache.
#
# ``vocab_block=300`` is not decorative. It is the only entry where a non-final
# vocabulary block is not a multiple of the epilogue's power-of-two load width,
# so its padding lanes carry *in-range* global indices. A forward that forgets to
# mask them picks up ``-inf`` from a lane belonging to a later block whenever a
# target lands there, and every power-of-two geometry hides it.
CE_PLANS = [
    (None, None, 0.0),
    (64, None, 0.0),
    (128, 256, 0.0),
    (128, 256, 0.5),
    (517, 1031, 1.0),
    (128, 300, 0.0),
    (256, 300, 0.5),
]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("chunk,vocab_block,retain", CE_PLANS)
def test_chunked_ce_matches_fp64(dtype, chunk, vocab_block, retain):
    """Tiling must not change the answer, only the memory it takes to get it."""
    torch.manual_seed(0)
    n, dim, vocab = 517, 128, 1031
    x = (torch.randn(n, dim, device="cuda", dtype=dtype) * 0.5).requires_grad_()
    w = (torch.randn(vocab, dim, device="cuda", dtype=dtype) * 0.05).requires_grad_()
    target = torch.randint(0, vocab, (n,), device="cuda")
    target[::7] = -100
    dloss = torch.randn(n, device="cuda")

    ref_loss, ref_dx, ref_dw = _ce_fp64_reference(x, w, target, dloss)
    loss = chunked_linear_cross_entropy(
        x, w, target, chunk=chunk, vocab_block=vocab_block, retain=retain
    )
    loss.backward(dloss)

    assert loss.dtype is torch.float32
    assert ulp_error(loss, ref_loss, dtype, mode="rms") <= 1.0
    # dX and dW sum over the whole vocabulary / batch, so they inherit the
    # rounding of the bf16 dlogits tile the GEMM consumes; the materializing
    # torch path measures the same error at this shape.
    assert ulp_error(x.grad, ref_dx, dtype, mode="rms") <= 12.0
    assert ulp_error(w.grad, ref_dw, dtype, mode="rms") <= 12.0
    assert loss[target == -100].abs().max() == 0


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_chunked_ce_ignore_index_contributes_nothing(dtype):
    """An ignored token is *exactly* zero everywhere, not merely small.

    The negative case that matters: a target of -100 is a valid-looking column
    index once it wraps, so a kernel that forgets to mask it still produces a
    plausible loss curve while training on the wrong column.
    """
    torch.manual_seed(0)
    n, dim, vocab = 96, 64, 300
    x = (torch.randn(n, dim, device="cuda", dtype=dtype) * 0.5).requires_grad_()
    w = (torch.randn(vocab, dim, device="cuda", dtype=dtype) * 0.05).requires_grad_()
    target = torch.full((n,), -100, device="cuda", dtype=torch.long)

    loss = chunked_linear_cross_entropy(x, w, target, chunk=32, vocab_block=128)
    loss.backward(torch.randn(n, device="cuda"))
    assert loss.abs().max() == 0
    assert x.grad.abs().max() == 0
    assert w.grad.abs().max() == 0

    # Half ignored: the kept half must equal what it would be on its own.
    target = torch.randint(0, vocab, (n,), device="cuda")
    masked = target.clone()
    masked[n // 2 :] = -100
    full = chunked_linear_cross_entropy(x.detach(), w.detach(), target, chunk=32)
    part = chunked_linear_cross_entropy(x.detach(), w.detach(), masked, chunk=32)
    assert torch.equal(full[: n // 2], part[: n // 2])
    assert part[n // 2 :].abs().max() == 0


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_chunked_ce_retention_is_bit_identical(dtype):
    """Caching a logit tile must equal recomputing it, bit for bit.

    Recompute is the same GEMM on the same inputs, so anything other than
    equality means a cached tile was read at the wrong offset or overwritten by
    a later one -- a bug that a tolerance-based check would wave through.
    """
    torch.manual_seed(0)
    n, dim, vocab = 260, 64, 700
    x0 = torch.randn(n, dim, device="cuda", dtype=dtype) * 0.5
    w0 = torch.randn(vocab, dim, device="cuda", dtype=dtype) * 0.05
    target = torch.randint(0, vocab, (n,), device="cuda")
    target[::5] = -100
    dloss = torch.randn(n, device="cuda")

    grads = []
    for retain in (0.0, 0.5, 1.0):
        x = x0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        loss = chunked_linear_cross_entropy(
            x, w, target, chunk=64, vocab_block=256, retain=retain
        )
        loss.backward(dloss)
        grads.append((loss.detach(), x.grad, w.grad))
    for loss, dx, dw in grads[1:]:
        assert torch.equal(loss, grads[0][0])
        assert torch.equal(dx, grads[0][1])
        assert torch.equal(dw, grads[0][2])

    # Replaying the graph must raise, not silently reuse consumed tiles.
    x = x0.clone().requires_grad_()
    loss = chunked_linear_cross_entropy(x, w0, target, chunk=64, retain=1.0)
    loss.backward(dloss, retain_graph=True)
    with pytest.raises(RuntimeError, match="consumed in place"):
        loss.backward(dloss)
    x = x0.clone().requires_grad_()
    loss = chunked_linear_cross_entropy(x, w0, target, chunk=64, retain=0.0)
    loss.backward(dloss, retain_graph=True)
    loss.backward(dloss)


def test_chunked_ce_grads_are_independent():
    """Requesting one gradient must not silently skip or corrupt the other."""
    torch.manual_seed(0)
    n, dim, vocab = 130, 64, 300
    x0 = torch.randn(n, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    w0 = torch.randn(vocab, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    target = torch.randint(0, vocab, (n,), device="cuda")
    dloss = torch.randn(n, device="cuda")

    both_x = x0.clone().requires_grad_()
    both_w = w0.clone().requires_grad_()
    chunked_linear_cross_entropy(both_x, both_w, target, chunk=64).backward(dloss)

    only_x = x0.clone().requires_grad_()
    chunked_linear_cross_entropy(only_x, w0, target, chunk=64).backward(dloss)
    only_w = w0.clone().requires_grad_()
    chunked_linear_cross_entropy(x0, only_w, target, chunk=64).backward(dloss)

    assert torch.equal(only_x.grad, both_x.grad)
    assert torch.equal(only_w.grad, both_w.grad)


@pytest.mark.parametrize("kernel", ["chunked_ce", "torch"])
def test_the_head_actually_learns_through_the_loss_kernel(kernel):
    """Both loss kernels must drive the head's weight, not merely report a number.

    The fp64 checks above pin ``dW`` for one call. They cannot see a head whose
    gradient never reaches the optimizer -- tying, a detached projection or a
    ``dW`` of the right magnitude but the wrong sign all leave them green. So
    this memorizes a fixed batch and demands the loss collapse.
    """
    torch.manual_seed(0)
    tokens, dim, vocab = 256, 64, 512
    head = LMHead(dim, vocab, tie_embeddings=False, kernel=kernel).cuda()
    hidden = torch.randn(tokens, dim, device="cuda") * 0.5
    labels = torch.randint(0, vocab, (tokens,), device="cuda")
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)

    losses = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        total, _ = head.loss(hidden, labels, reduction="sum")
        loss = total / tokens
        loss.backward()
        assert head.weight.grad is not None, "no gradient reached the projection"
        assert torch.isfinite(head.weight.grad).all()
        opt.step()
        losses.append(float(loss.detach()))

    start, end = losses[0], losses[-1]
    # Random init over `vocab` classes starts at ln(vocab); memorizing a fixed
    # batch of that size must get most of the way to zero.
    assert start == pytest.approx(math.log(vocab), rel=0.15), start
    assert end < 0.2 * start, f"{kernel}: loss {start:.3f} -> {end:.3f}, not learning"
    assert head.weight.grad.abs().max() > 0
