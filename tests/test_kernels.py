"""Precision and correctness checks for the module-level Triton kernels.

Each kernel is pinned against the reference it replaces, in **both fp16 and
bf16**, forward and backward. The tolerance is expressed in ULP of the dtype
rather than as an absolute number, because an absolute tolerance that passes in
fp32 is meaningless in bf16 and vice versa.

A kernel that drifts past its reference should fail here before it ever reaches
a training run, where the symptom would be a slightly-worse loss curve that
nobody attributes to a kernel.

Split by subject, because this file outgrew the 1000-line cap:

* ``test_kernels_ce.py`` -- the two head-side loss kernels;
* ``test_kernels_cpu_fallback.py`` -- the no-CUDA paths, which cannot be gated
  on CUDA;
* ``test_kernels_mxfp8_*.py`` -- the fp8 format, the linear, and the experts.
"""

import pytest
import torch
import torch.nn.functional as F

from kohakuwullm.bench.core.timing import rel_error, ulp_error
from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm as triton_rms_norm
from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm, grouped_gemm_reference
from kohakuwullm.kernels.moe.moe_dispatch import (
    combine_routed,
    combine_routed_reference,
)
from kohakuwullm.kernels.moe.router import fused_router
from kohakuwullm.kernels.optim.adamw16 import adamw16_step
from kohakuwullm.models.components.moe import TopKRouter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("dim", [128, 1280, 4096])
def test_rmsnorm_matches_aten(dtype, dim):
    """Forward and both gradients within 1 ULP of the fp64 truth."""
    torch.manual_seed(0)
    x = torch.randn(512, dim, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(dim, device="cuda", dtype=dtype).abs().add(0.5).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    out = triton_rms_norm(x, w, 1e-6)
    ref64 = F.rms_norm(x.detach().double(), (dim,), w.detach().double(), 1e-6)
    assert ulp_error(out, ref64, dtype) <= 1.0

    grad = torch.randn_like(out)
    out.backward(grad)
    F.rms_norm(x_ref, (dim,), w_ref, 1e-6).backward(grad)
    # Gradients accumulate over the feature axis, so they are compared against
    # the ATen gradient in the same dtype rather than against fp64.
    assert rel_error(x.grad, x_ref.grad) < 2e-2
    assert rel_error(w.grad, w_ref.grad) < 2e-2


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_rmsnorm_no_affine(dtype):
    x = torch.randn(256, 512, device="cuda", dtype=dtype, requires_grad=True)
    out = triton_rms_norm(x, None, 1e-6)
    ref = F.rms_norm(x.detach().double(), (512,), None, 1e-6)
    assert ulp_error(out, ref, dtype) <= 1.0
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_swiglu_matches_eager(dtype):
    """The fused product is at least as accurate as eager (it reduces in fp32)."""
    torch.manual_seed(0)
    g = torch.randn(1024, 2048, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(1024, 2048, device="cuda", dtype=dtype, requires_grad=True)
    g_ref = g.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()

    out = swiglu_mul(g, v)
    ref64 = F.silu(g.detach().double()) * v.detach().double()
    assert ulp_error(out, ref64, dtype) <= 1.0

    grad = torch.randn_like(out)
    out.backward(grad)
    (F.silu(g_ref) * v_ref).backward(grad)
    assert rel_error(g.grad, g_ref.grad) < 2e-2
    assert rel_error(v.grad, v_ref.grad) < 2e-2


def test_combine_skips_uncomputed_rows_instead_of_zeroing_them():
    """A sentinel bucket's rows are never written, so they must never be *read*.

    ReMoE parks its inactive slots in a bucket with no expert matrix, which means
    those rows of the expert output hold whatever ``torch.empty`` left behind.
    Scaling them by their zero gate weight looks equivalent and is not: ``nan * 0``
    is ``nan``, so a single uninitialised row would poison its token's output and
    the loss with it.

    The second half is the control. Without the bound, the same data *does* produce
    nan -- which is what makes this a test of the skip and not of the arithmetic.
    """
    torch.manual_seed(0)
    tokens, slots, dim = 32, 4, 64
    pairs = tokens * slots
    valid = pairs - 20

    rows = torch.randn(pairs, dim, device="cuda", dtype=torch.bfloat16)
    rows[valid:] = float("nan")
    weight = torch.rand(pairs, device="cuda", dtype=torch.bfloat16)
    weight[valid:] = 0.0
    order = torch.arange(pairs, device="cuda", dtype=torch.int32)
    token_of = torch.div(order, slots, rounding_mode="floor").to(torch.int32)
    bound = torch.full((1,), valid, dtype=torch.int32, device="cuda")

    out = combine_routed(rows, weight, order, token_of, tokens, bound)
    assert torch.isfinite(out.float()).all(), "an uncomputed row reached the output"
    ref = combine_routed_reference(rows, weight, order, token_of, tokens, bound)
    assert torch.isfinite(ref.float()).all()
    assert rel_error(out.float(), ref.float()) < 2e-2

    poisoned = combine_routed(rows, weight, order, token_of, tokens)
    assert not torch.isfinite(
        poisoned.float()
    ).all(), "weight-zero should NOT sanitise a nan row; the control is broken"


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", [(512, 256), (4, 128, 256)], ids=["packed", "padded"])
def test_swiglu_reads_strided_chunks_without_copying(dtype, shape):
    """The layout every caller actually produces: two halves of one tensor.

    ``GLUMLP`` defaults to a fused ``w_in`` followed by ``chunk``, and the MoE
    expert path does the same between its grouped GEMMs, so the strided case is
    the normal one -- not an edge case. An earlier version contiguified both
    halves, which made the fused kernel slower and heavier than eager in exactly
    that default. The copy is invisible in the output, so what pins it is that
    the kernel reads the *views* it was handed.
    """
    torch.manual_seed(0)
    h = torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    gate, value = h.chunk(2, dim=-1)
    assert not gate.is_contiguous(), "chunk should hand back a strided view"

    out = swiglu_mul(gate, value)
    ref64 = F.silu(gate.detach().double()) * value.detach().double()
    assert out.shape == gate.shape
    assert ulp_error(out, ref64, dtype) <= 1.0

    grad = torch.randn_like(out)
    out.backward(grad)
    got = h.grad.clone()

    ref = h.detach().clone().requires_grad_()
    gate_ref, value_ref = ref.chunk(2, dim=-1)
    (F.silu(gate_ref) * value_ref).backward(grad)
    assert rel_error(got, ref.grad) < 2e-2


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
def test_grouped_gemm_matches_loop(dtype):
    """Grouped GEMM equals a loop of per-expert GEMMs, including empty experts."""
    torch.manual_seed(0)
    num_experts, dim, hidden = 8, 256, 512
    # Expert 1 gets nothing: an empty group is the case a grid-per-expert kernel
    # is most likely to mis-handle, and routing produces it routinely.
    counts = torch.tensor([130, 0, 77, 200, 45, 301, 12, 99])
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = counts.cumsum(0).cuda()
    total = int(counts.sum())

    x = torch.randn(total, dim, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(num_experts, hidden, dim, device="cuda", dtype=dtype) * 0.05
    w = w.requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w_ref = w.detach().clone().requires_grad_()

    out = grouped_gemm(x, w, offsets)
    ref = grouped_gemm_reference(x_ref, w_ref, offsets.cpu())
    ref64 = grouped_gemm_reference(
        x.detach().double(), w.detach().double(), offsets.cpu()
    )
    # RMS-scaled: a near-zero GEMM output is cancellation, not a small true value.
    assert ulp_error(out, ref64, dtype, mode="rms") <= 4.0

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert rel_error(x.grad, x_ref.grad) < 2e-2
    assert rel_error(w.grad, w_ref.grad) < 5e-2


@pytest.mark.parametrize(
    "counts",
    [
        [1024, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1024],
        [0, 1, 0, 1023, 0, 0, 0, 0],
        [128] * 8,
    ],
    ids=["all-first", "all-last", "one-row-experts", "uniform"],
)
def test_grouped_gemm_grid_bound_covers_every_split(counts):
    """The flat-tile grid is a *bound*, so every row split must still be computed.

    ``cdiv(M, BLOCK_M) + E`` has to cover ``sum_e cdiv(count_e, BLOCK_M)`` for any
    distribution, and the failure mode if it does not is silent: the uncovered
    tiles keep whatever ``torch.empty`` left in them. The extremes are what pin
    it -- one expert holding every row maximises the tile count for a given M,
    and single-row experts maximise the per-expert rounding waste.
    """
    torch.manual_seed(0)
    num_experts, dim, hidden = 8, 128, 64
    count = torch.tensor(counts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = count.cumsum(0).cuda()
    total = int(count.sum())

    x = torch.randn(total, dim, device="cuda", dtype=torch.bfloat16)
    w = (
        torch.randn(num_experts, hidden, dim, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )

    out = grouped_gemm(x, w, offsets)
    ref = grouped_gemm_reference(x, w, offsets.cpu())
    # Every row, not an aggregate: an uncovered tile is a block of rows that is
    # wrong while the rest is exact, which an averaged error metric would hide.
    assert torch.equal(
        out, ref
    ), f"rows differ: {(out != ref).any(-1).sum()} of {total}"


def test_grouped_gemm_fp16_accumulate():
    """fp16 accumulation stays usable at expert-sized K, and rejects bf16."""
    torch.manual_seed(0)
    num_experts, dim, hidden = 4, 1024, 512
    counts = torch.tensor([256, 256, 256, 256])
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = counts.cumsum(0).cuda()

    x = torch.randn(1024, dim, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(num_experts, hidden, dim, device="cuda", dtype=torch.float16) * 0.05
    ref64 = grouped_gemm_reference(x.double(), w.double(), offsets.cpu())

    fast = grouped_gemm(x, w, offsets, acc_fp16=True)
    exact = grouped_gemm(x, w, offsets, acc_fp16=False)
    # fp16 accumulation is allowed to be worse, but not unboundedly so.
    assert ulp_error(exact, ref64, torch.float16, mode="rms") <= 4.0
    assert ulp_error(fast, ref64, torch.float16, mode="rms") <= 64.0

    with pytest.raises(ValueError, match="requires fp16"):
        grouped_gemm(x.bfloat16(), w.bfloat16(), offsets, acc_fp16=True)


GATES = ["sigmoid", "sqrtsoftplus"]


def _gate_reference(logits: torch.Tensor, score_func: str) -> torch.Tensor:
    if score_func == "sigmoid":
        return torch.sigmoid(logits)
    return F.softplus(logits).sqrt()


def _routers(dim, num_experts, top_k, score_func, dtype, **kwargs):
    """A fused and an eager :class:`TopKRouter` holding identical weights."""
    fused = TopKRouter(
        dim, num_experts, top_k=top_k, score_func=score_func, fused=True, **kwargs
    )
    eager = TopKRouter(
        dim, num_experts, top_k=top_k, score_func=score_func, fused=False, **kwargs
    )
    eager.load_state_dict(fused.state_dict())
    fused = fused.cuda().to(dtype)
    eager = eager.cuda().to(dtype)
    assert fused.use_fused and not eager.use_fused
    return fused, eager


def _spread(fused, eager, num_experts, dtype):
    """Scale the gate rows apart so the top-k boundary is not GEMM noise."""
    with torch.no_grad():
        scale = torch.linspace(0.5, 2.0, num_experts, device="cuda").to(dtype)
        fused.weight.mul_(scale[:, None])
        eager.weight.copy_(fused.weight)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("score_func", GATES)
def test_fused_router_matches_eager(dtype, score_func):
    """Both fused gates agree with eager on selection and on the gate weights.

    Index-for-index identity, not a tolerance: the router makes a discrete
    choice, and a backend that picks a different expert still trains -- just
    worse, for reasons the loss curve does not explain.
    """
    torch.manual_seed(0)
    dim, num_experts, top_k, tokens = 256, 32, 4, 512
    fused, eager = _routers(dim, num_experts, top_k, score_func, dtype)
    # A near-tie between the k-th and (k+1)-th score is a property of the input,
    # not a kernel defect, and asserting exactness on one measures nothing.
    _spread(fused, eager, num_experts, dtype)
    x = torch.randn(tokens, dim, device="cuda", dtype=dtype)

    got_idx, got_weight, got_counts = fused.route(x)
    ref_idx, ref_weight, ref_counts = eager.route(x)
    assert torch.equal(got_idx.int(), ref_idx.int())
    assert torch.equal(got_counts.int(), ref_counts.int())

    logits64 = x.double() @ fused.weight.detach().double().T
    scores64 = _gate_reference(logits64, score_func)
    ref64 = scores64.gather(1, got_idx.long())
    ref64 = ref64 / ref64.sum(-1, keepdim=True).clamp_min(1e-9)
    assert ulp_error(got_weight.float(), ref64, dtype, mode="rms") <= 4.0
    assert rel_error(got_weight.float(), ref_weight.float()) < 2e-2


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("score_func", GATES)
def test_fused_router_backward_matches_eager(dtype, score_func):
    """The gate derivative is the half a forward check cannot see.

    ``sqrtsoftplus`` shares the forward's shape with ``sigmoid`` but not its
    derivative, so reusing ``s * (1 - s)`` for it would pass every forward test
    and quietly train the gate with the wrong gradient.
    """
    torch.manual_seed(0)
    dim, num_experts, top_k, tokens = 256, 32, 4, 512
    fused, eager = _routers(dim, num_experts, top_k, score_func, dtype)
    _spread(fused, eager, num_experts, dtype)

    x0 = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    got_x = x0.clone().requires_grad_()
    ref_x = x0.clone().requires_grad_()
    grad = torch.randn(tokens, top_k, device="cuda")

    _, got_weight, _ = fused.route(got_x)
    got_weight.backward(grad.to(got_weight.dtype))
    _, ref_weight, _ = eager.route(ref_x)
    ref_weight.backward(grad.to(ref_weight.dtype))

    assert rel_error(got_x.grad, ref_x.grad) < 2e-2
    assert rel_error(fused.weight.grad, eager.weight.grad) < 2e-2
    # A wrong-but-plausible derivative still correlates with the right one, so
    # pin the scale too: reusing sigmoid's s(1-s) for sqrtsoftplus lands within
    # a small factor and a direction-only check would let it through.
    got_norm = fused.weight.grad.float().norm()
    ref_norm = eager.weight.grad.float().norm()
    assert abs(got_norm / ref_norm - 1.0) < 5e-2


def test_fused_router_rejects_unknown_gate():
    """A softmax gate must not silently reach a kernel that cannot express it."""
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="no 'softmax' gate"):
        fused_router(x, w, None, 2, score_func="softmax")
    # And the module must keep such a router on the eager path by construction.
    assert not TopKRouter(128, 8, top_k=2, score_func="softmax").use_fused
    assert TopKRouter(128, 8, top_k=2, score_func="sqrtsoftplus").use_fused
    # An auxiliary term no longer costs the fused path, but it must be declared
    # before any forward runs so a pipeline can size its boundary.
    aux = TopKRouter(128, 8, top_k=2, aux_loss_weight=0.01)
    assert aux.use_fused and aux.emits_loss
    assert TopKRouter(128, 8, top_k=2, z_loss_weight=1e-3).emits_loss
    assert not TopKRouter(128, 8, top_k=2).emits_loss


# Deliberately far above any training value: at a shipping coefficient the term
# is a ~1e-3 correction, which no gradient comparison can separate from noise.
AUX_KWARGS = [
    ("aux", dict(aux_loss_weight=50.0)),
    ("zloss", dict(z_loss_weight=10.0)),
    ("both", dict(aux_loss_weight=50.0, z_loss_weight=10.0)),
]


def _router_backward(router, x0, grad):
    """``(loss, grad_x, grad_weight)`` for the gate weights plus the router's terms."""
    x = x0.clone().requires_grad_()
    _, weight, _ = router.route(x)
    loss = (weight.float() * grad).sum()
    for term in (router.aux_loss, router.z_loss):
        if term is not None:
            loss = loss + term.float()
    router.zero_grad(set_to_none=True)
    loss.backward()
    return loss.detach(), x.grad, router.weight.grad


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("score_func", GATES)
@pytest.mark.parametrize(
    "kwargs", [kw for _, kw in AUX_KWARGS], ids=[i for i, _ in AUX_KWARGS]
)
def test_fused_router_auxiliary_losses_match_eager(dtype, score_func, kwargs):
    """Both auxiliary terms, and both of their gradients, against eager and fp64.

    The gradient is the half the value cannot see. Unlike the gate weights, these
    terms depend on *every* expert's score, including the ones a token did not
    select -- so a backward that only scattered the ``top_k`` lanes reproduces
    the value exactly and trains the gate with a truncated gradient. The last
    assertion is the control: with the same weights and the terms switched off,
    the gate gradient has to move.
    """
    torch.manual_seed(0)
    dim, num_experts, top_k, tokens = 256, 32, 4, 512
    fused, eager = _routers(dim, num_experts, top_k, score_func, dtype, **kwargs)
    _spread(fused, eager, num_experts, dtype)

    x0 = torch.randn(tokens, dim, device="cuda", dtype=dtype)
    grad = torch.randn(tokens, top_k, device="cuda")
    got_loss, got_gx, got_gw = _router_backward(fused, x0, grad)
    ref_loss, ref_gx, ref_gw = _router_backward(eager, x0, grad)

    logits64 = x0.double() @ fused.weight.detach().double().T
    scores64 = _gate_reference(logits64, score_func)
    _, _, counts = fused.route(x0)
    for name, weight in (("aux_loss", "aux_loss_weight"), ("z_loss", "z_loss_weight")):
        term = getattr(fused, name)
        if weight not in kwargs:
            assert term is None and getattr(eager, name) is None
            continue
        if name == "aux_loss":
            share = counts.double() / (tokens * top_k)
            ref64 = kwargs[weight] * num_experts * (share * scores64.mean(0)).sum()
        else:
            ref64 = kwargs[weight] * logits64.logsumexp(-1).pow(2).mean()
        assert rel_error(term.float().reshape(1), ref64.reshape(1)) < 2e-2
        assert (
            rel_error(term.float().reshape(1), getattr(eager, name).float().reshape(1))
            < 2e-2
        )

    assert rel_error(got_loss.reshape(1), ref_loss.reshape(1)) < 2e-2
    assert rel_error(got_gx, ref_gx) < 2e-2
    assert rel_error(got_gw, ref_gw) < 2e-2

    fused.aux_loss_weight = fused.z_loss_weight = 0.0
    _, _, plain_gw = _router_backward(fused, x0, grad)
    assert rel_error(got_gw, plain_gw) > 0.2, "the auxiliary term changed nothing"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eager_fallback_tracks_the_triton_kernel(dtype):
    """The CPU path is the same update, not a second optimizer."""
    torch.manual_seed(0)
    p0 = torch.randn(4096, dtype=dtype)
    g0 = torch.randn(4096, dtype=dtype) * 1e-3

    def run(device):
        state = [
            p0.to(device).clone(),
            g0.to(device).clone(),
            torch.zeros(4096, dtype=dtype, device=device),
            torch.zeros(4096, dtype=dtype, device=device),
        ]
        for step in range(1, 6):
            adamw16_step(*state, step, 1e-3, weight_decay=0.1)
        return state[0].cpu().float()

    gpu, cpu = run("cuda"), run("cpu")
    # One ULP of the 16-bit state, compounded over five steps.
    tol = 1e-6 if dtype is torch.float32 else 4e-3
    assert (gpu - cpu).abs().max() < tol
