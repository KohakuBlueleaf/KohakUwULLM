"""Pins for the closed-form FLOP model every scaling figure divides by.

``bench.flops_analytic`` is the denominator of the headline end-to-end scaling
panels, and a denominator is the one thing a benchmark cannot check for itself: a
census that is 20% low reports a utilisation that is 20% high, and nothing in the
measurement disagrees. So the arithmetic is pinned two ways that do not share a
formula with it.

The load-bearing one is :func:`test_gemm_forward_equals_twice_the_matmul_parameters`.
It builds the real ``LMBackbone`` on ``torch.device("meta")`` and sums the weights of
every ``nn.Linear`` in it, which is an *independent route to the same number* -- one
multiply-accumulate per weight per token, two FLOPs each. Restating the closed form
in the test would pin only that it had been typed twice.

Everything here is meta-device or plain arithmetic. No CUDA, by construction.
"""

import torch
import torch.nn as nn

from kohakuwullm.bench.model.flops_analytic import MAC, budget, causal_pairs_per_token
from kohakuwullm.bench.model.ladder import ladder_census
from kohakuwullm.models import LMBackbone
from kohakuwullm.models.presets import PRESETS, get_preset


def brute_force_pairs(len_min: int, len_max: int, window: int | None) -> float:
    """Attended pairs per token, counted one query position at a time.

    Deliberately the slow way: a closed form here would be the same algebra the
    module uses, and would agree with it while both were wrong.
    """
    pairs = tokens = 0
    for length in range(len_min, len_max + 1):
        for position in range(length):
            attended = position + 1
            pairs += attended if window is None else min(attended, window)
        tokens += length
    return pairs / tokens


def matmul_parameters(name: str) -> int:
    """Weights a token is multiplied against once, for a *dense* preset.

    Every ``nn.Linear`` in the trunk plus the head. The embedding is excluded on
    purpose -- it is a gather, not a matmul, and charging it is the specific error
    the module exists to avoid.
    """
    config = get_preset(name)
    with torch.device("meta"):
        backbone = LMBackbone(config)
    total = sum(
        module.weight.numel()
        for module in backbone.blocks.modules()
        if isinstance(module, nn.Linear)
    )
    return total + config.dim * config.vocab_size


def test_causal_pairs_matches_brute_force():
    """Every regime of the window, against a per-position count.

    The window-longer-than-any-document case is included because it is the one that
    silently reduces to the unwindowed branch: a sign slip in ``length <= window``
    would still produce plausible numbers everywhere else.
    """
    for len_min, len_max, window in [
        (1, 6, None),
        (3, 9, None),
        (3, 9, 4),  # window inside the range: both branches run
        (5, 7, 64),  # window past every document: must equal the unwindowed count
        (100, 160, 128),
        (7, 7, None),  # single length
        (7, 7, 3),
    ]:
        got = causal_pairs_per_token(len_min, len_max, window)
        want = brute_force_pairs(len_min, len_max, window)
        assert abs(got - want) < 1e-9, (len_min, len_max, window, got, want)

    # A window nobody reaches is not a window.
    assert causal_pairs_per_token(5, 7, 64) == causal_pairs_per_token(5, 7, None)
    # ...and one that binds must cost strictly less.
    assert causal_pairs_per_token(100, 160, 128) < causal_pairs_per_token(
        100, 160, None
    )


def test_gemm_forward_equals_twice_the_matmul_parameters():
    """One MAC per weight per token, two FLOPs per MAC -- against the built model.

    Both a preset that spells ``mlp_hidden`` out and one that leaves it to
    ``resolve_hidden``. The second is not redundant: reading the raw config field
    instead of resolving it raises ``TypeError`` on 18 of the 27 presets, and the
    Kohaku ladder is exactly the subset where that mistake is invisible.
    """
    for name in ("Kohaku-200M", "Nano-25M", "Nano-1B"):
        want = MAC * matmul_parameters(name)
        got = budget(name).gemm_fwd
        assert abs(got - want) / want < 1e-12, (name, got, want)


def test_every_preset_has_a_finite_budget():
    """No preset may crash or produce a non-finite term.

    A closed form that only works on the rungs someone happened to plot is a closed
    form nobody can trust on the next one.
    """
    for name in PRESETS:
        row = budget(name)
        for field in ("gemm_fwd", "attn_fwd", "vector_fwd", "pairs_per_token"):
            value = getattr(row, field)
            assert value > 0 and value == value and value != float("inf"), (name, field)


def test_moe_layers_charge_the_active_experts_not_the_bank():
    """A sparse layer costs ``top_k + shared`` experts, never ``num_experts``.

    The whole point of a sparse preset is that its arithmetic tracks *active*
    parameters, so charging the bank would restore precisely the confusion the
    ladder was designed to separate -- and it would do it while still producing a
    number that rises smoothly with size.
    """
    name = "Kohaku-MoE-1B"
    config = get_preset(name)
    active = config.moe_top_k + config.moe_num_shared
    assert active < config.moe_num_experts, "preset is not sparse; test proves nothing"

    row = budget(name)
    # Every layer of this preset is sparse, so scaling the whole MoE term is exact.
    per_expert = MAC * 3 * config.dim * config.moe_hidden * config.depth
    bank = per_expert * config.moe_num_experts
    assert row.gemm_fwd < bank, (row.gemm_fwd, bank)
    # The router matrix is charged too -- it is a real GEMM, just a narrow one.
    assert row.gemm_fwd > per_expert * active


def test_grad_ckpt_charges_the_blocks_twice_and_the_head_once():
    """Checkpointing re-runs blocks, not the head, which sits outside them."""
    plain = budget("Kohaku-200M")
    ckpt = budget("Kohaku-200M", grad_ckpt=True)
    config = get_preset("Kohaku-200M")
    head = MAC * config.dim * config.vocab_size

    assert abs(ckpt.gemm_fwd - (2 * plain.gemm_fwd - head)) / head < 1e-9
    # Backward stays 2x the forward it is the backward of.
    assert abs(ckpt.gemm_bwd - 2 * ckpt.gemm_fwd) / ckpt.gemm_bwd < 1e-12


def test_matrix_and_vector_stay_in_their_own_buckets():
    """Lengthening the documents must move the two buckets by different amounts.

    Both the score/AV matmuls and the softmax scale with attended pairs, so a bug
    that filed one under the other would still track sequence length and still look
    right. What separates them is the *coefficient*: the matmuls carry ``2 * MAC *
    q_dim`` per pair per layer and the softmax carries ``3 * heads``. Pinning the two
    increments independently is what makes the split checkable at all.
    """
    name = "Kohaku-200M"
    config = get_preset(name)
    short = budget(name, len_min=512, len_max=1024)
    long = budget(name, len_min=512, len_max=4096)
    delta_pairs = long.pairs_per_token - short.pairs_per_token
    assert delta_pairs > 0

    q_dim = config.heads * config.head_dim
    assert (
        abs(
            (long.attn_fwd - short.attn_fwd)
            - config.depth * MAC * 2 * delta_pairs * q_dim
        )
        < 1e-6 * long.attn_fwd
    )
    assert (
        abs(
            (long.vector_fwd - short.vector_fwd)
            - config.depth * 3 * delta_pairs * config.heads
        )
        < 1e-6 * long.vector_fwd
    )

    # Weight GEMMs are a property of the model, not of the batch.
    assert short.gemm_only == long.gemm_only
    # The kinds partition the total; nothing is double-counted or dropped.
    assert abs(long.total - (long.matrix + long.vector)) < 1e-9 * long.total


def test_gemm_share_of_6nd_rises_across_the_kohaku_ladder():
    """``gemm / (6 * active)`` climbs 0.75 -> 0.95, and that gradient is the argument.

    It is why the scaling panels use a closed-form FLOP axis rather than a parameter
    count: the 6ND rule charges arithmetic to the untied embedding, which does none,
    and the resulting error is *monotonic in size* -- the shape that bends a fitted
    exponent rather than merely offsetting it. A change that flattened this curve
    would silently make parameter count look like an adequate x-axis again.
    """
    rungs = sorted(ladder_census(), key=lambda c: c.compute_active)
    ratios = [budget(c.name).gemm_only / (6 * c.active) for c in rungs]

    assert 0.70 < ratios[0] < 0.80, ratios[0]
    assert 0.90 < ratios[-1] < 1.00, ratios[-1]
    assert ratios[-1] - ratios[0] > 0.15, ratios
    # Dense and sparse rungs interleave and the sparse ones sit slightly low, so the
    # sequence is not pointwise monotone; the trend within each family is.
    for family in (False, True):
        same = [r for c, r in zip(rungs, ratios) if c.sparse is family]
        assert same == sorted(same), (family, same)
