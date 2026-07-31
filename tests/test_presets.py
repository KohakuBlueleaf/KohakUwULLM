"""The Kohaku preset ladder: does it hold the invariants it was designed under?

Split from ``test_models.py`` because that file reached its 1000-line hard cap,
but the seam is the right one: nothing here builds a forward pass. Every count
comes from ``bench.ladder.census``, which builds the preset on
``torch.device("meta")`` -- so these tests run the rungs at their real depth and
vocabulary, 8B included, on a box with no GPU.

The negative cases are the point. Four of the five things pinned here are exactly
the "obvious fixes" a later reader would apply:

* rounding ``Kohaku-MoE-2B``'s expert to 0.5*dim = 448, which is MXFP8-ineligible;
* returning ``Kohaku-MoE-8B`` to E=64 / top_k=8 like the rest of the ladder;
* letting ``resolve_hidden`` round a feed-forward width up instead of to nearest;
* dropping ``Kohaku-MoE-3B``'s second shared expert.

Each of those trains perfectly well and quietly breaks either hyperparameter
transfer or fp8 eligibility, so each has a test that fails on it.
"""

import pytest

from kohakuwullm import LMBackbone, get_preset
from kohakuwullm.bench.model.ladder import census
from kohakuwullm.models.components.mlp import resolve_hidden
from kohakuwullm.models.presets import KOHAKU_LADDER, PRESETS

# The solved design table the ladder was built to, in millions: (total, active).
# ``active`` is the solver's convention -- body parameters only, no embedding, no
# head, no router -- so it is compared against ``RungCensus.routed_active`` and not
# against ``count_active_parameters``, which is 41-69% higher on a sparse rung.
#
# Restated here rather than imported from ``scripts/bench/e2e/presets.py``: a test's
# expected values are the pin, and moving a preset should have to be justified
# against a literal.
KOHAKU_SOLVED_M: dict[str, tuple[float, float]] = {
    "Kohaku-200M": (204.3, 204.3),
    "Kohaku-MoE-1B": (990.0, 146.2),
    "Kohaku-500M": (546.2, 546.2),
    "Kohaku-MoE-2B": (1951.6, 292.8),
    "Kohaku-1B": (981.5, 981.5),
    "Kohaku-MoE-3B": (2905.2, 480.9),
    "Kohaku-1.5B": (1514.1, 1514.1),
    "Kohaku-MoE-5B": (4931.6, 772.7),
    "Kohaku-MoE-8B": (7712.6, 1163.2),
}
KOHAKU_TARGET_CAPACITY = (200, 350, 500, 700, 1000, 1250, 1500, 1850, 2800)


def test_kohaku_ladder_matches_its_solved_parameter_counts():
    """Measured counts against the closed form that produced the ladder.

    The counts are measured on the meta device, so this pins the *model* against
    the solve rather than the solve against itself -- which is the direction that
    catches an omission, and the solver omitted the router matrix.
    """
    assert set(KOHAKU_LADDER) == set(KOHAKU_SOLVED_M)
    capacities = []
    for name in KOHAKU_LADDER:
        rung = census(name)
        total_m, active_m = KOHAKU_SOLVED_M[name]
        measured_active = rung.total if not rung.sparse else rung.routed_active
        assert rung.total / 1e6 == pytest.approx(total_m, rel=0.01), name
        assert measured_active / 1e6 == pytest.approx(active_m, rel=0.01), name
        capacities.append(rung.capacity / 1e6)

    # The ladder's whole point: one monotone sequence, dense and sparse alternating.
    # A rung that lands out of order is a broken ladder even at the right size.
    assert capacities == sorted(capacities)
    for value, target in zip(capacities, KOHAKU_TARGET_CAPACITY):
        assert value == pytest.approx(target, rel=0.10)


def test_kohaku_ladder_holds_fixed_sparsity_and_expert_width():
    """kappa and expert width are exact, because transfer depends on them.

    Hyperparameter transfer across a ladder is only valid at fixed sparsity, so
    ``top_k / num_experts`` is not approximately 0.125. The two intended
    deviations are named here so that "fixing" either one fails loudly:
    ``Kohaku-MoE-8B`` moves granularity (128/16, not 64/8) and ``Kohaku-MoE-2B``
    takes a 0.571*dim expert because 0.5 * 896 = 448 is not a multiple of 128.
    """
    for name in KOHAKU_LADDER:
        rung = census(name)
        if not rung.sparse:
            continue
        assert rung.kappa == 0.125, name
        assert rung.num_shared >= 1, name
        # Expert width scales with the active expert count; see docs/concepts/presets.md.
        share = 0.25 if rung.top_k == 16 else 0.5
        expected_hidden = 512 if name == "Kohaku-MoE-2B" else int(rung.dim * share)
        assert rung.expert_hidden == expected_hidden, name

    granular = census("Kohaku-MoE-8B")
    assert (granular.num_experts, granular.top_k) == (128, 16)


def test_sparse_rungs_do_not_starve_attention():
    """Active feed-forward per layer over attention per layer, in [5, 7].

    See docs/concepts/presets.md.
    """
    for name in KOHAKU_LADDER:
        rung = census(name)
        if not rung.sparse:
            continue
        attn = 2 * rung.dim * rung.heads * rung.head_dim + 2 * rung.dim * rung.kv_out
        active_ffn = (rung.top_k + rung.num_shared) * 3 * rung.dim * rung.expert_hidden
        ratio = active_ffn / attn
        assert 5.0 <= ratio <= 7.0, f"{name}: active FFN / attention = {ratio:.1f}"


def test_kohaku_ladder_shapes_are_mxfp8_aligned():
    """Every contraction axis a multiple of 128, at every rung.

    ``dim`` and the feed-forward widths are FPROP contraction axes and cannot be
    zero-padded, so this is what makes ``config.mxfp8`` available on the whole
    ladder instead of most of it. ``expert_hidden * num_shared`` is in the list
    because the shared expert is a ``GLUMLP`` at that width, which puts it on
    ``w_out.in_features`` -- the axis that refuses ``MoE-3B-A500M``.
    """
    for name in KOHAKU_LADDER:
        rung = census(name)
        config = get_preset(name)
        widths = {"dim": rung.dim, "kv_out": rung.kv_out, "mlp_hidden": rung.mlp_hidden}
        if rung.sparse:
            widths["expert_hidden"] = rung.expert_hidden
            widths["shared_hidden"] = rung.expert_hidden * rung.num_shared
        for label, width in widths.items():
            assert width % 128 == 0, f"{name}.{label} = {width}"
        assert rung.head_dim == 64, name
        assert 4 <= rung.gqa_ratio <= 8, name
        assert config.vocab_size == 65536 and not config.tie_embeddings, name


def test_kohaku_ladder_is_mxfp8_eligible_at_every_rung():
    """The public path, not the shape rule it is derived from.

    Depth-reduced to two layers (one dense, one sparse) because eligibility is a
    property of the widths; the negative half is the point of the test. Dropping
    ``Kohaku-MoE-2B``'s expert to 0.5*dim = 448 -- the "obvious" fix to its 14%
    granularity deviation -- must refuse, naming the shared expert's projection.
    """
    for name in KOHAKU_LADDER:
        model = LMBackbone(get_preset(name, vocab_size=1024, depth=2, mxfp8=True))
        assert model.mxfp8_projections, name

    with pytest.raises(ValueError) as excinfo:
        LMBackbone(
            get_preset(
                "Kohaku-MoE-2B", vocab_size=1024, depth=2, mxfp8=True, moe_hidden=448
            )
        )
    message = str(excinfo.value)
    assert "shared.w_out" in message
    assert "in_features=448" in message


def test_every_preset_resolves_the_width_its_readers_will_compute():
    """``resolve_hidden`` gives every preset the width its model was actually built with.

    Scope, stated honestly: this pins the *helper against the built module*, so it
    catches the two config fields drifting apart -- the MoE branch resolves with
    ``moe_ratio`` but ``mlp_multiple_of``, an asymmetry only ``backbone._mlp_kwargs``
    knows about, and nothing else would notice it changing. It does **not** protect a
    reader that bypasses the helper: a cost model spelling out
    ``hidden or int(dim * ratio * 2 / 3)`` stays wrong and stays invisible here.

    That is worth having anyway, because the readers are where it went wrong twice.
    ``mlp_hidden`` / ``moe_hidden`` are unset on 8 of the presets, where the width comes
    from the ratio and rounds *up* to ``multiple_of``. `bench/flops_analytic.py` read the
    raw field and raised on 18 of 27; `training/pipeline.py` truncated instead of
    rounding and ran 1.0-4.8% low on 8, moving two stage splits.

    ``PRESETS``, not ``KOHAKU_LADDER``: the ladder spells every width out, so a
    ladder-only check is exactly the one that would have missed both.
    """
    for name in PRESETS:
        config = get_preset(name)
        rung = census(name)
        if config.moe_every > 0:
            resolved = resolve_hidden(
                config.dim,
                config.moe_ratio,
                config.moe_hidden,
                config.mlp_multiple_of,
            )
            built = rung.expert_hidden
        else:
            resolved = resolve_hidden(
                config.dim,
                config.mlp_ratio,
                config.mlp_hidden,
                config.mlp_multiple_of,
            )
            built = rung.mlp_hidden
        assert resolved == built, (
            f"{name}: resolve_hidden gives {resolved}, the built module has {built}; "
            "a cost model reading the raw field would disagree with the model it describes"
        )
        # No 128-alignment assertion here. An explicit `hidden` bypasses rounding by
        # design, and `MoE-1B-A120M` uses 448 on purpose -- the exact width
        # `test_kohaku_ladder_is_mxfp8_eligible_at_every_rung` pins as *ineligible*.
        # Alignment is a property of the Kohaku ladder, not of every preset, and it
        # already has its own test at that scope.


def test_moe_active_parameters_less_than_total():
    """Every sparse preset, including the ladders this one replaces.

    Moved here from ``test_models.py`` with the rest of the preset-level tests: it
    iterates ``PRESETS`` and builds nothing, so it belongs with the ladder.
    """
    # Selected by `moe_every` rather than by a name prefix: a preset rename
    # silently orphaned this test once already, and the `Kohaku-` ladder orphaned
    # it a second time -- "Kohaku-MoE-1B" does not start with "MoE-".
    moe = [name for name in PRESETS if get_preset(name).moe_every > 0]
    assert moe, "no MoE presets registered"
    for name in moe:
        # `census` counts on the meta device, so this runs the preset at its real
        # depth and vocabulary instead of a 4-layer stand-in, and costs nothing.
        # A depth-reduced sparse model also flatters the ratio being asserted:
        # `moe_first_dense` is a larger share of 4 layers than of 40.
        rung = census(name)
        assert rung.active < rung.total / 2, name
