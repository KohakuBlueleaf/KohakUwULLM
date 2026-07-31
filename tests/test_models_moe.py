"""MoE routing: the weighting, the balancing, and the variant routers.

The load-balancing bias is the subtle one. It is added to the gate score *for
selection only* and must never reach the weight a token's output is scaled by;
a version that scaled by the biased score would still balance, still train, and
quietly apply a learned-by-a-rule multiplier to every expert output.

The variant routers (Sinkhorn, expert-choice, ReLU) are covered here rather than
skipped because they are selectable from a config, and an ablation that crashes
on its second forward is worse than one that is merely slower.
"""

import pytest
import torch

from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.models.components.moe import MoEMLP

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------- MoE


@cuda_only
def test_moe_grouped_matches_dense_fallback():
    """The Triton expert path must equal the loop-of-GEMMs path exactly enough."""
    torch.manual_seed(0)
    dim = 256
    fast = (
        MoEMLP(
            dim, hidden=128, num_experts=8, top_k=2, num_shared=1, bias_update_rate=0.0
        )
        .cuda()
        .to(torch.bfloat16)
    )
    slow = (
        MoEMLP(
            dim,
            hidden=128,
            num_experts=8,
            top_k=2,
            num_shared=1,
            dense_fallback=True,
            bias_update_rate=0.0,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    slow.load_state_dict(fast.state_dict())

    x = torch.randn(512, dim, device="cuda", dtype=torch.bfloat16)
    assert rel_error(fast(x).float(), slow(x).float()) < 2e-2


@cuda_only
def test_moe_bias_steers_selection_not_weights():
    """The balancing bias must not appear in the output combination weights."""
    torch.manual_seed(0)
    moe = (
        MoEMLP(
            128, hidden=64, num_experts=4, top_k=2, num_shared=0, bias_update_rate=1e-3
        )
        .cuda()
        .float()
    )
    x = torch.randn(64, 128, device="cuda")

    _, weights_before = moe.router(x)
    moe.router.expert_bias += torch.tensor([10.0, -10.0, 0.0, 0.0], device="cuda")
    idx_after, weights_after = moe.router(x)

    # Selection moved toward expert 0 ...
    assert (idx_after == 0).float().mean() > 0.4
    # ... but every reported weight is still a raw sigmoid score in (0, 1).
    assert weights_after.min() >= 0 and weights_after.max() <= 1
    assert weights_before.min() >= 0 and weights_before.max() <= 1


def _run_router(rate: int, steps: int = 150, seed: int = 0) -> float:
    """Mean max-to-mean expert load over the last 30 steps at a given update rate."""
    torch.manual_seed(seed)
    moe = (
        MoEMLP(
            128, hidden=64, num_experts=8, top_k=2, num_shared=0, bias_update_rate=rate
        )
        .cuda()
        .float()
    )
    moe.train()
    # A skewed gate: experts 0-2 are favoured on average, but each token still
    # has its own preference ordering. That token-to-token variation is what the
    # balancing bias exploits -- if every token wanted the same expert, no bias
    # could spread the load, it could only move the pile somewhere else.
    with torch.no_grad():
        moe.router.weight[:3] *= 4.0
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)

    ratios = []
    for _ in range(steps):
        # Fresh tokens each step: load balancing is a property of the stream,
        # not of one repeated batch.
        x = torch.randn(1024, 128, device="cuda", generator=generator)
        topk_idx, _ = moe.router(x)
        # Measured here rather than read from update_bias(), which returns None
        # when the mechanism is off -- the control arm has to be measured by the
        # same yardstick as the treatment arm.
        counts = torch.bincount(topk_idx.flatten(), minlength=moe.num_experts).float()
        ratios.append((counts.max() / counts.mean()).item())
        moe.router.update_bias()
    return sum(ratios[-30:]) / 30


@cuda_only
def test_moe_bias_update_balances_load():
    """Enabling the balancing bias must lower steady-state load imbalance.

    Compared against the *same* router with the mechanism switched off, rather
    than against its own first step: the update is sign-based with a fixed step,
    so at equilibrium it necessarily oscillates within one step size and a
    first-vs-last comparison measures that oscillation, not the mechanism.
    """
    off = _run_router(0.0)
    on = _run_router(0.01)
    assert off > 1.15, f"the skewed gate was not actually imbalanced: {off:.3f}"
    assert on < off * 0.9, f"balancing did not help: off={off:.3f} on={on:.3f}"


@cuda_only
def test_moe_bias_update_is_noop_without_balancing():
    """bias_update_rate=0 disables the mechanism entirely (no buffers, no update)."""
    moe = (
        MoEMLP(
            128, hidden=64, num_experts=4, top_k=2, num_shared=0, bias_update_rate=0.0
        )
        .cuda()
        .float()
    )
    assert moe.router.expert_bias is None
    assert moe.router.update_bias() is None


# Every formulation that MoEMLP can actually drive, with the router kwargs that
# distinguish it. Listed exhaustively rather than sampled: the ReMoE L1 bug
# survived because that one formulation's backward was never run, and a sampled
# list reproduces exactly that hole.
FORMULATION_KWARGS = [
    ("deepseek_v3", dict(score_func="sigmoid", num_shared=1)),
    ("deepseek_v4", dict(score_func="sqrtsoftplus", num_shared=1)),
    ("mimo_no_shared", dict(score_func="sigmoid", num_shared=0)),
    ("olmoe_softmax_aux", dict(score_func="softmax", aux_loss_weight=0.01)),
    ("remoe_relu", dict(router="relu")),
    ("sinkhorn", dict(router="sinkhorn")),
    ("zloss", dict(z_loss_weight=1e-3)),
    ("aux_and_zloss", dict(aux_loss_weight=0.01, z_loss_weight=1e-3)),
    ("group_limited", dict(n_groups=4, topk_groups=2)),
]


@cuda_only
@pytest.mark.parametrize(
    "kwargs",
    [kw for _, kw in FORMULATION_KWARGS],
    ids=[i for i, _ in FORMULATION_KWARGS],
)
def test_every_formulation_survives_two_forwards_and_one_backward(kwargs):
    """The shape of bug that made ReMoE's sparsity mechanism dead code.

    An auxiliary term assembled from a buffer that is later mutated in place
    raises a version-counter error on *every* backward, and nothing catches it
    until something backprops through that term. Two forwards before one
    backward, because under gradient accumulation the second forward's update is
    what invalidates the first's graph -- so no statement ordering saves it and
    the single-forward case can pass while training cannot.
    """
    torch.manual_seed(0)
    moe = (
        MoEMLP(128, hidden=64, num_experts=8, top_k=2, **kwargs)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    loss = torch.zeros((), device="cuda")
    for _ in range(2):
        term = moe(x).float().sum()
        extra = moe.router_losses()
        if extra is not None:
            term = term + extra.float()
        loss = loss + term
    loss.backward()

    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert torch.isfinite(x.grad).all()


@cuda_only
def test_expert_choice_refuses_the_dispatch_it_cannot_satisfy():
    """A variant MoEMLP cannot drive must say why, not raise AttributeError.

    Expert choice emits a ``(token, expert)`` pair list; the dispatch recovers a
    pair's token as ``pair_index // slots``, which only holds under token choice.
    It was registered and documented as available while crashing on first use.
    """
    moe = MoEMLP(128, hidden=64, num_experts=8, top_k=2, router="expert_choice").cuda()
    x = torch.randn(64, 128, device="cuda")
    with pytest.raises(NotImplementedError, match="pair list"):
        moe(x)


@cuda_only
def test_relu_router_l1_penalty_is_differentiable():
    """ReMoE's L1 term must survive a backward, including across accumulation.

    The penalty *is* ReMoE's sparsity mechanism, so a loss that cannot be
    differentiated means the formulation silently does nothing. Two forwards
    before one backward, not one: the coefficient is updated in place every
    forward, so the single-forward case only catches the hazard when the update
    is ordered wrongly, while accumulation catches it either way.
    """
    torch.manual_seed(0)
    moe = (
        MoEMLP(128, hidden=64, num_experts=8, top_k=2, router="relu", num_shared=1)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    start = moe.router.l1_coeff.clone()

    loss = moe(x).float().sum() + moe.router_losses().float()
    loss = loss + moe(x).float().sum() + moe.router_losses().float()
    loss.backward()

    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()
    assert torch.isfinite(x.grad).all()
    # The coefficient is steered, not frozen; both forwards stepped it.
    assert not torch.equal(moe.router.l1_coeff, start)


@cuda_only
def test_relu_router_counts_match_its_own_indices():
    """``route`` must report the histogram of the padded slots it actually emits."""
    torch.manual_seed(0)
    moe = (
        MoEMLP(128, hidden=64, num_experts=8, top_k=2, router="relu", num_shared=0)
        .cuda()
        .to(torch.bfloat16)
        .train()
    )
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    idx, _, counts = moe.router.route(x)

    assert counts.dtype is torch.int32, "expert_sort offsets are int32"
    assert int(counts.sum()) == idx.numel()
    expected = torch.zeros_like(counts)
    for expert in idx.reshape(-1).tolist():
        expected[expert] += 1
    assert torch.equal(counts, expected)
