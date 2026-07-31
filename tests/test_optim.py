"""Correctness checks for optimizer construction and the fused AdamW step.

The fused step is pinned against ``torch.optim.AdamW`` -- once in the same
precision, where the two should agree to rounding, and once against an fp64
reference, so that accumulated drift over a run of steps is visible.

The clipping tests matter more than the plain-step ones. Folding the clip
coefficient into the kernel's ``grad_scale`` divisor is an easy place to invert
a reciprocal, clip per parameter group instead of globally, or scale gradients
that were under the threshold to begin with -- all of which still train, just
worse, with nothing in the loss curve to point at.

The Muon tests are negative for the same reason. An embedding in a Muon group, a
3-D expert stack flattened instead of orthogonalized per expert, or the AdamW
weight decay reused against Muon's ~100x larger lr all still train.
"""

import importlib.util
import sys
import types
from copy import deepcopy
from unittest import mock

import pytest
import torch
import torch.nn as nn

from kohakuwullm import LMArchConfig, LMBackbone
from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.registry import OPTIMIZER
from kohakuwullm.training.optim import muon as muon_module
from kohakuwullm.training.optim.build import (
    build_optimizer,
    group_parameters,
    is_hidden_matrix,
)
from kohakuwullm.training.optim.fused_adamw import FusedAdamW
from kohakuwullm.training.optim.muon import (
    NS_CUBIC5,
    NS_PHASES_DIRECT,
    NS_PHASES_GRAM,
    MuonW,
    newton_schulz,
    newton_schulz_cubic,
    orthogonal_update_scale,
)
from kohakuwullm.training.optim.torchao_optim import QUANTIZED_ADAMW

SHAPES = [(64, 32), (128,), (17, 5), (256, 8)]
STEPS = 20


def _params(dtype=torch.float32, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    return [
        nn.Parameter(torch.randn(*s, generator=gen).to(device=device, dtype=dtype))
        for s in SHAPES
    ]


def _grads(step, params, seed=0):
    gen = torch.Generator().manual_seed(seed + 1000 * step)
    for p in params:
        g = torch.randn(*p.shape, generator=gen)
        p.grad = g.to(device=p.device, dtype=p.dtype)


def _run(opt, params, steps=STEPS, seed=0):
    for step in range(steps):
        _grads(step, params, seed=seed)
        opt.step()
    return params


def test_fused_adamw_matches_torch_adamw():
    """Same math in the same precision, and no drift against an fp64 truth."""
    got = _params()
    ref = _params()
    ref64 = _params(dtype=torch.float64)

    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    _run(FusedAdamW(got, **kw), got)
    _run(torch.optim.AdamW(ref, foreach=False, **kw), ref)
    _run(torch.optim.AdamW(ref64, foreach=False, **kw), ref64)

    for g, r, r64 in zip(got, ref, ref64):
        assert rel_error(g, r) < 1e-6
        assert rel_error(g, r64) < 1e-6


def test_fused_adamw_respects_param_groups():
    """Per-group lr and weight decay reach the kernel, not just the defaults."""
    got = _params()
    ref = _params()

    def groups(ps):
        return [
            {"params": ps[:2], "lr": 1e-2, "weight_decay": 0.1},
            {"params": ps[2:], "lr": 1e-4, "weight_decay": 0.0},
        ]

    _run(FusedAdamW(groups(got), betas=(0.9, 0.95), eps=1e-8), got)
    _run(
        torch.optim.AdamW(groups(ref), betas=(0.9, 0.95), eps=1e-8, foreach=False), ref
    )

    for g, r in zip(got, ref):
        assert rel_error(g, r) < 1e-6
    # A group whose lr is 100x smaller must have moved 100x less; equal movement
    # would mean the group's lr was dropped in favour of the default.
    assert (got[0] - _params()[0]).norm() > 50 * (got[2] - _params()[2]).norm()


def test_fused_adamw_clip_matches_clip_then_step():
    """Folded clipping equals ``clip_grad_norm_`` followed by an unclipped step."""
    max_norm = 0.5
    got = _params()
    ref = _params()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_got = FusedAdamW(got, clip_grad_norm=max_norm, **kw)
    opt_ref = torch.optim.AdamW(ref, foreach=False, **kw)

    for step in range(STEPS):
        _grads(step, got)
        _grads(step, ref)
        expected = torch.nn.utils.clip_grad_norm_(ref, max_norm)
        opt_ref.step()
        opt_got.step()
        assert rel_error(opt_got.grad_norm(), expected) < 1e-6
        # The kernel writes the scaled gradient back, so what is left behind
        # must be the clipped gradient, exactly as after ``clip_grad_norm_``.
        for g, r in zip(got, ref):
            assert rel_error(g.grad, r.grad) < 1e-6

    for g, r in zip(got, ref):
        assert rel_error(g, r) < 1e-6


def test_fused_adamw_clip_is_global_not_per_group():
    """The clip coefficient is one number for the whole model."""
    max_norm = 0.5
    one = _params()
    split = _params()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt_one = FusedAdamW(one, clip_grad_norm=max_norm, **kw)
    opt_split = FusedAdamW(
        [{"params": split[:2]}, {"params": split[2:]}],
        clip_grad_norm=max_norm,
        **kw,
    )
    _run(opt_one, one)
    _run(opt_split, split)

    for a, b in zip(one, split):
        assert rel_error(a, b) < 1e-6

    # Adam divides its update by sqrt(v), so a wrong clip coefficient moves the
    # parameters only in the fourth decimal -- the norm itself is the only
    # quantity sharp enough to catch per-group clipping.
    probe = _params()
    opt = FusedAdamW(
        [{"params": probe[:2]}, {"params": probe[2:]}],
        clip_grad_norm=max_norm,
        **kw,
    )
    _grads(0, probe)

    def norm_of(ps):
        return torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in ps]))

    expected = norm_of(probe)
    largest_group = max(norm_of(probe[:2]).item(), norm_of(probe[2:]).item())
    opt.step()
    assert rel_error(opt.grad_norm(), expected) < 1e-6
    assert opt.grad_norm().item() > 1.05 * largest_group


def test_fused_adamw_below_threshold_is_untouched():
    """A norm under the threshold leaves gradients and the update unchanged."""
    got = _params()
    ref = _params()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    # The threshold sits far above any norm these gradients reach, so the
    # coefficient must clamp to exactly 1 rather than to something near it.
    opt_got = FusedAdamW(got, clip_grad_norm=1e6, **kw)
    opt_ref = FusedAdamW(ref, **kw)

    for step in range(STEPS):
        _grads(step, got)
        _grads(step, ref)
        raw = [p.grad.clone() for p in got]
        opt_got.step()
        opt_ref.step()
        for p, g in zip(got, raw):
            assert torch.equal(p.grad, g)

    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_fused_adamw_skips_missing_grads_and_norm_reporting():
    params = _params()
    opt = FusedAdamW(params, lr=1e-2)
    assert opt.grad_norm() is None

    _grads(0, params)
    params[1].grad = None
    before = params[1].detach().clone()
    opt.step()
    assert torch.equal(params[1], before)
    # No gradient means no state: allocating it would cost memory for a
    # parameter that is never updated (a frozen embedding, say).
    assert not opt.state[params[1]]
    assert opt.grad_norm() is None


def test_fused_adamw_rejects_bad_config():
    params = _params()
    with pytest.raises(ValueError, match="betas"):
        FusedAdamW(params, betas=(1.0, 0.95))
    with pytest.raises(ValueError, match="clip_grad_norm"):
        FusedAdamW(params, clip_grad_norm=0.0)
    with pytest.raises(ValueError, match="state_dtype"):
        FusedAdamW(params, state_dtype=torch.float16)
    with pytest.raises(ValueError, match="float32 parameters"):
        opt = FusedAdamW(_params(dtype=torch.float64), state_dtype="bfloat16")
        _grads(0, opt.param_groups[0]["params"])
        opt.step()
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="requires the CUDA kernel"):
            opt = FusedAdamW(_params(), state_dtype="bfloat16")
            _grads(0, opt.param_groups[0]["params"])
            opt.step()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_fused_adamw_refuses_low_precision_parameters(dtype):
    """The kernel accepts bf16 parameters; accepting them here would be the bug.

    ``torch._fused_adamw_`` runs an all-bf16 step happily and rounds to nearest,
    so under ``PRECISION="bf16-true"`` this would train normally until the
    schedule decays lr under half a ULP of the weights and then quietly stop.
    """
    params = _params(dtype=dtype)
    opt = FusedAdamW(params, lr=1e-2)
    _grads(0, params)
    with pytest.raises(ValueError, match="float32 parameters"):
        opt.step()


def test_fused_adamw_state_dict_round_trip():
    params = _params()
    opt = FusedAdamW(params, lr=1e-2, clip_grad_norm=1.0)
    _run(opt, params, steps=5)

    resumed = _params()
    for fresh, trained in zip(resumed, params):
        fresh.data.copy_(trained.data)
    opt_resumed = FusedAdamW(resumed, lr=1e-2, clip_grad_norm=1.0)
    # Deep-copied because `state_dict` hands out the live state tensors, and
    # `load_state_dict` keeps them when dtype and device already match.
    opt_resumed.load_state_dict(deepcopy(opt.state_dict()))

    _run(opt, params, steps=5, seed=99)
    _run(opt_resumed, resumed, steps=5, seed=99)
    for p, q in zip(params, resumed):
        assert rel_error(q, p) < 1e-6


def test_fused_adamw_rehosts_step_on_load():
    """A checkpoint's step count is re-hosted as fp32 on the parameter's device.

    Left alone, ``Optimizer.load_state_dict`` returns ``state["step"]``
    untouched, so a checkpoint written from another device (or, here, another
    dtype) survives the load and the kernel reads it as an fp32 device scalar
    that it is not.
    """
    params = _params()
    opt = FusedAdamW(params, lr=1e-2)
    _run(opt, params, steps=3)

    saved = deepcopy(opt.state_dict())
    for state in saved["state"].values():
        state["step"] = state["step"].double()

    resumed = _params()
    opt_resumed = FusedAdamW(resumed, lr=1e-2)
    opt_resumed.load_state_dict(saved)
    for param in resumed:
        step = opt_resumed.state[param]["step"]
        assert step.dtype is torch.float32
        assert step.device == param.device
        assert step.item() == 3


def test_fused_adamw_is_registered_and_built():
    assert "fused_adamw" in OPTIMIZER.keys()
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    opt = build_optimizer(model, name="fused_adamw", lr=3e-4, weight_decay=0.1)
    assert isinstance(opt, FusedAdamW)
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)
    # Vectors must land in the no-decay group; the fused kernel applies decay
    # per group, so a mis-grouped norm scale would be decayed silently.
    decay, no_decay = group_parameters(model, 0.1)
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim <= 1 for p in no_decay["params"])


def _moe_model() -> LMBackbone:
    return LMBackbone(
        LMArchConfig(
            vocab_size=256,
            dim=64,
            depth=2,
            heads=4,
            kv_heads=2,
            head_dim=16,
            mlp_hidden=128,
            attn="sdpa",
            mlp="swiglu",
            moe_every=1,
            moe_first_dense=1,
            moe_num_experts=4,
            moe_top_k=2,
            moe_hidden=32,
            # The grouped-GEMM path is Triton; these run on CPU.
            moe_router_kwargs={"dense_fallback": True},
            tie_embeddings=False,
        ),
        # Both defaults are Triton kernels; these tests run on CPU.
        head_kwargs={"kernel": "torch"},
    )


def test_muon_split_covers_hidden_matrices_only():
    model = _moe_model()
    named = dict(model.named_parameters())
    chosen = {n for n, p in named.items() if is_hidden_matrix(n, p)}

    assert "blocks.0.attn.q_proj.weight" in chosen
    assert "blocks.0.mlp.w_in.weight" in chosen
    assert "blocks.1.mlp.w_in" in chosen  # the (experts, out, in) stack
    assert "blocks.1.mlp.shared.w_in.weight" in chosen

    assert "embed.weight" not in chosen
    assert "head.weight" not in chosen
    assert "blocks.1.mlp.router.weight" not in chosen
    assert not any(named[n].ndim < 2 for n in chosen)

    groups = group_parameters(model, weight_decay=0.1, lr=3e-4, muon_lr=0.02)
    muon_params = {id(p) for g in groups if g["use_muon"] for p in g["params"]}
    assert muon_params == {id(named[n]) for n in chosen}
    assert all("use_muon" in g for g in groups)


@pytest.mark.parametrize("exponent", [0.0, 0.5])
def test_muon_group_matches_adamw_per_step_decay(exponent):
    """Decoupled decay shrinks by ``lr * wd``; a 100x larger lr needs a smaller wd."""
    model = _moe_model()
    groups = group_parameters(
        model, weight_decay=0.1, lr=3e-4, muon_lr=0.02, muon_mup_exponent=exponent
    )
    adam = next(g for g in groups if not g["use_muon"] and g["weight_decay"] > 0)
    # Every Muon group, not just the first: a per-fan_in lr split has to carry the
    # matching decay or the wider matrices are regularized harder than the narrow.
    for muon in (g for g in groups if g["use_muon"]):
        assert muon["lr"] * muon["weight_decay"] == pytest.approx(
            adam["lr"] * adam["weight_decay"]
        )


def test_muon_grouping_skips_mup_and_stays_off_for_adamw():
    model = _moe_model()
    named = dict(model.named_parameters())
    target = id(named["blocks.0.mlp.w_in.weight"])

    groups = group_parameters(
        model, weight_decay=0.1, lr=3e-4, use_mup=True, base_dim=32, muon_lr=0.02
    )
    # Muon's own sqrt(fan_out/fan_in) already carries the width dependence, so a
    # hidden matrix must appear once, at the plain Muon lr, never also muP-scaled.
    holders = [g for g in groups if any(id(p) == target for p in g["params"])]
    assert len(holders) == 1
    assert holders[0]["use_muon"] and holders[0]["lr"] == 0.02

    plain = group_parameters(model, weight_decay=0.1, lr=3e-4, use_mup=True)
    assert not any("use_muon" in g for g in plain)
    assert any(any(id(p) == target for p in g["params"]) for g in plain)


def test_embedding_is_excluded_from_weight_decay_but_the_head_is_not():
    """The input embedding's gradient is row-sparse; the head's is not.

    A softmax gives every head row a gradient on every step, so decay there is
    the ordinary decoupled kind. An embedding row only gets a gradient on the
    steps where its token appears, so decaying it every step is a pull toward
    zero that rare tokens have no signal to resist. Measured on the real corpus:
    57% of the 65536 rows never appear at all, and decay alone erases 79% of
    such a row's norm over a 100k-step run.
    """
    model = _moe_model()
    named = dict(model.named_parameters())

    for kwargs in ({}, {"use_mup": True, "base_dim": 32}, {"muon_lr": 0.02}):
        groups = group_parameters(model, weight_decay=0.1, lr=3e-4, **kwargs)
        holding = {}
        for group in groups:
            for param in group["params"]:
                holding[id(param)] = group

        embed = holding[id(named["embed.weight"])]
        assert embed["weight_decay"] == 0.0, kwargs

        # Everything that should still decay, still does -- this must not turn
        # into a blanket "no decay on 2-D parameters".
        for name in ("head.weight", "blocks.0.mlp.w_in.weight"):
            assert holding[id(named[name])]["weight_decay"] > 0.0, (name, kwargs)

    # The escape hatch puts it back, and puts it in the *plain* decay group:
    # the embedding's fan-in is the vocabulary, so muP width scaling does not
    # apply to it even when it is being decayed.
    groups = group_parameters(
        model,
        weight_decay=0.1,
        lr=3e-4,
        use_mup=True,
        base_dim=32,
        decay_embeddings=True,
    )
    embed = next(
        g for g in groups if any(p is named["embed.weight"] for p in g["params"])
    )
    assert embed["weight_decay"] == 0.1 and embed["lr"] == 3e-4


def test_changing_the_decay_grouping_refuses_a_stale_optimizer_checkpoint():
    """Moving the embedding between groups must not silently misassign moments.

    ``Optimizer.load_state_dict`` maps saved state onto parameters *by position
    within each group*, so a checkpoint written before this grouping changed
    would attach the embedding's moments to whatever now sits at its old index.
    Torch happens to catch it on the group sizes; pinned because the failure it
    prevents is silent, and a future grouping change that happened to preserve
    the sizes would not be caught at all.
    """
    model = _moe_model()
    old = build_optimizer(
        model, name="adamw", lr=1e-3, weight_decay=0.1, decay_embeddings=True
    )
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    old.step()

    new = build_optimizer(model, name="adamw", lr=1e-3, weight_decay=0.1)
    with pytest.raises(ValueError, match="doesn't match the size"):
        new.load_state_dict(old.state_dict())


def test_stochastic_muon_hands_the_kernel_a_contiguous_update():
    """The SR writeback addresses both operands with one flat offset.

    ``cubic5`` -- the default variant -- iterates on the matrix's short side and
    transposes back, and its last operation is a matmul, whose output is always
    contiguous, so the transpose that follows is not. (The quintic happens to
    come out contiguous because it ends on an add, which propagates its input's
    layout; that is a TensorIterator detail and not something to rely on.) A
    *tall* parameter therefore reaches the writeback as a strided view, which the
    kernel would read wrong -- and ``w_in`` is tall in every preset.

    Spied rather than executed so this runs without CUDA: the bug is in what the
    optimizer hands the kernel, not in the kernel.
    """
    seen = []

    def spy(param, update, seed, *, decay=0.0, alpha=1.0, rng_offset=0):
        seen.append(update.is_contiguous())
        return param

    tall = nn.Parameter(torch.randn(256, 64, dtype=torch.bfloat16))
    wide = nn.Parameter(torch.randn(64, 256, dtype=torch.bfloat16))
    for p in (tall, wide):
        p.grad = torch.randn_like(p)

    opt = MuonW(
        [{"params": [tall, wide], "use_muon": True}],
        lr=0.02,
        rounding="stochastic",
        compile_ns=False,
    )
    with mock.patch.object(muon_module, "stochastic_round_update_", spy):
        opt.step()

    assert len(seen) == 2, "both orientations must reach the writeback"
    assert all(seen), "a strided update reached the flat-offset kernel"


def test_tying_the_embedding_also_removes_decay_from_the_head():
    """One tensor, reported under the embedding's name, so it takes its grouping.

    Pinned rather than left implicit: a tied tensor *does* get a dense gradient
    from the head, so decaying it would be defensible, and someone flipping
    ``tie_embeddings`` should find this stated rather than discover it.
    """
    tied = LMBackbone(
        LMArchConfig(
            vocab_size=256,
            dim=64,
            depth=1,
            heads=4,
            kv_heads=2,
            head_dim=16,
            attn="sdpa",
            tie_embeddings=True,
        ),
        head_kwargs={"kernel": "torch"},
    )
    names = [n for n, _ in tied.named_parameters()]
    assert "head.weight" not in names

    groups = group_parameters(tied, weight_decay=0.1, lr=3e-4)
    shared = dict(tied.named_parameters())["embed.weight"]
    holder = next(g for g in groups if any(p is shared for p in g["params"]))
    assert holder["weight_decay"] == 0.0


@pytest.mark.parametrize("optimizer", ["adamw", "fused_adamw", "muon"])
def test_untouched_embedding_rows_are_left_alone(optimizer):
    """A row whose token never appears must not move at all.

    With no gradient its moments stay zero, so the update term is exactly zero
    and decoupled decay is the only thing that could move it -- which is the
    whole bug. ``lr`` and ``weight_decay`` are far above any real setting so the
    shrink is unmistakable in 20 steps rather than 100k.
    """
    torch.manual_seed(0)
    model = _moe_model()
    opt = build_optimizer(model, name=optimizer, lr=0.1, weight_decay=1.0, muon_lr=0.1)

    seen = torch.arange(8)
    untouched = model.embed.weight.detach()[64:].clone()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        model.embed(seen).pow(2).sum().backward()
        opt.step()

    after = model.embed.weight.detach()[64:]
    assert torch.equal(after, untouched), (
        f"{optimizer}: rows for absent tokens moved by "
        f"{(after - untouched).abs().max():.3e}"
    )
    # The rows that did get a gradient must still be training.
    assert not torch.equal(
        model.embed.weight.detach()[:8], model.embed.weight.detach()[64:72]
    )


def _widths_lr(width: int, **kwargs) -> dict[str, float]:
    """Per-matrix lr at one width, keyed by parameter name."""
    model = LMBackbone(
        LMArchConfig(
            vocab_size=128,
            dim=width,
            depth=2,
            heads=width // 16,
            kv_heads=1,
            head_dim=16,
            mlp_hidden=2 * width,
            attn="sdpa",
            mlp="swiglu",
            tie_embeddings=False,
        ),
        head_kwargs={"kernel": "torch"},
    )
    groups = group_parameters(model, weight_decay=0.1, lr=1.0, base_dim=32, **kwargs)
    by_id = {id(p): g["lr"] for g in groups for p in g["params"]}
    return {n: by_id[id(p)] for n, p in model.named_parameters()}


@pytest.mark.parametrize(
    "name, exponent",
    [
        # AdamW's muP hidden rule is 1/fan_in, so doubling the width halves...
        ("blocks.0.mlp.w_in.weight", 1.0),
        ("blocks.0.attn.q_proj.weight", 1.0),
        # ... every hidden matrix, including the readout, which muP classes as an
        # output weight (fan_in is the width, fan_out the fixed vocabulary).
        ("head.weight", 1.0),
        # The embedding's fan_in is the vocabulary: no width dependence at all.
        ("embed.weight", 0.0),
        ("final_norm.weight", 0.0),
    ],
)
def test_mup_lr_follows_the_predicted_width_exponent(name, exponent):
    narrow, wide = _widths_lr(64, use_mup=True), _widths_lr(128, use_mup=True)
    assert wide[name] / narrow[name] == pytest.approx(2.0**-exponent)


@pytest.mark.parametrize("exponent, predicted", [(0.0, 0.0), (0.5, 0.5)])
def test_muon_mup_exponent_sets_the_width_exponent(exponent, predicted):
    kwargs = dict(use_mup=True, muon_lr=0.02, muon_mup_exponent=exponent)
    narrow, wide = _widths_lr(64, **kwargs), _widths_lr(128, **kwargs)
    name = "blocks.0.mlp.w_in.weight"
    assert wide[name] / narrow[name] == pytest.approx(2.0**-predicted)
    # The default must stay one group at exactly `muon_lr`: a width correction on
    # top of the dualized update scales twice.
    if exponent == 0.0:
        assert wide[name] == 0.02


def test_rms_update_scale_plus_half_exponent_equals_spectral():
    """The two update scales are the same rule, off by the RMS target.

    ``spectral`` puts ``sqrt(fan_out/fan_in)`` in the update and needs no width
    term; ``rms`` puts ``0.2 * sqrt(max(fan_out, fan_in))`` there, which fixes
    the update's RMS and leaves its *spectral* norm growing like ``sqrt(width)``.
    Pinning the identity keeps the two modes from drifting into two rules.
    """
    base_dim, rms_target = 256, 0.2
    for shape in [(1280, 1280), (2560, 1280), (1280, 3456), (64, 1280)]:
        spectral = orthogonal_update_scale(torch.Size(shape), "spectral", rms_target)
        rms = orthogonal_update_scale(torch.Size(shape), "rms", rms_target)
        fan_in = shape[-1]
        corrected = rms * (base_dim / fan_in) ** 0.5
        assert corrected == pytest.approx(
            spectral * rms_target * base_dim**0.5, rel=1e-12
        )


def test_muon_rejects_groups_without_the_flag():
    model = _moe_model()
    with pytest.raises(ValueError, match="use_muon"):
        MuonW([{"params": list(model.parameters())}])
    with pytest.raises(ValueError, match="no singular values"):
        MuonW([{"params": [model.final_norm.weight], "use_muon": True}])


@pytest.mark.parametrize("shape", [(64, 64), (128, 32), (32, 128)])
def test_newton_schulz_keeps_direction_and_flattens_spectrum(shape):
    torch.manual_seed(0)
    grad = torch.randn(*shape)
    out = newton_schulz(grad, steps=5, dtype=torch.float32).double()

    u, _, vh = torch.linalg.svd(grad.double(), full_matrices=False)
    polar = u @ vh
    assert (out * polar).sum() / (out.norm() * polar.norm()) > 0.97

    # The tuned quintic deliberately does not converge to 1; it lands the bulk
    # of the spectrum in a band, and accepting that band is what buys five
    # iterations instead of thirty. Judge it by how much flatter the spectrum
    # got, not by distance to exactly 1.
    got = torch.linalg.svdvals(out)
    ref = torch.linalg.svdvals(grad.double())
    assert got.max() < 1.3 and got.quantile(0.1) > 0.6
    assert got.max() / got.quantile(0.1) < 0.8 * ref.max() / ref.quantile(0.1)


def test_newton_schulz_is_scale_invariant_and_per_expert():
    torch.manual_seed(0)
    stack = torch.randn(3, 24, 16)
    stack[0].zero_()
    out = newton_schulz(stack, steps=5, dtype=torch.float32)

    # A flattened (experts*out, in) or (experts, out*in) view would let the two
    # live experts leak a nonzero update into the dead one.
    assert out[0].abs().max() == 0.0
    for e in (1, 2):
        alone = newton_schulz(stack[e], steps=5, dtype=torch.float32)
        assert rel_error(out[e], alone) < 1e-5

    assert (
        rel_error(newton_schulz(stack * 1e4, steps=5, dtype=torch.float32), out) < 1e-5
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 32), (32, 128), (3, 24, 96)])
def test_cubic_phases_agree_with_the_unfactored_iteration(shape):
    """Grouping steps must not change the operator, only where it is evaluated.

    ``A_{k+1} A_k x`` accumulated in gram space is the same product as two
    successive ``x <- A_k x``; in fp32 the two differ only by rounding. A group
    that dropped a step, applied the product in the wrong order, or reused a
    stale gram would still return a plausible near-orthogonal matrix, so the
    check is against the unfactored path rather than against orthogonality.
    """
    torch.manual_seed(0)
    grad = torch.randn(*shape)
    direct = newton_schulz_cubic(grad, dtype=torch.float32, phases=NS_PHASES_DIRECT)
    for phases in (NS_PHASES_GRAM, (5,), (1, 2, 2), (4, 1)):
        grouped = newton_schulz_cubic(grad, dtype=torch.float32, phases=phases)
        assert rel_error(grouped, direct) < 1e-4, phases

    # The order the group multiplies its own factors in is *not* a correctness
    # question -- every A_k is a polynomial in the initial gram, so they commute,
    # and swapping `step @ product` for `product @ step` changes nothing. The
    # schedule's order does matter, since its coefficients are tuned per step, so
    # that is what this pins.
    reordered = newton_schulz_cubic(
        grad, dtype=torch.float32, schedule=tuple(reversed(NS_CUBIC5))
    )
    assert rel_error(reordered, direct) > 1e-3

    with pytest.raises(ValueError, match="sum to"):
        newton_schulz_cubic(grad, phases=(2, 2))


def test_cubic_keeps_direction_and_flattens_spectrum():
    """The default schedule, which had no test of its own."""
    torch.manual_seed(0)
    grad = torch.randn(96, 256)
    out = newton_schulz_cubic(grad, dtype=torch.float32).double()

    u, _, vh = torch.linalg.svd(grad.double(), full_matrices=False)
    polar = u @ vh
    assert (out * polar).sum() / (out.norm() * polar.norm()) > 0.97

    got = torch.linalg.svdvals(out)
    ref = torch.linalg.svdvals(grad.double())
    # The schedule's polynomials peak at exactly 1.30 and land the bulk of the
    # spectrum in [0.774, 1.30]; it is not a convergent iteration.
    assert got.max() < 1.30 and got.quantile(0.1) > 0.7
    assert got.max() / got.quantile(0.1) < 0.8 * ref.max() / ref.quantile(0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 matmul rounding")
def test_one_gram_group_overshoots_the_band_in_bf16():
    """Why ``NS_PHASES_GRAM`` is 2+3 and not the cheaper single group of 5.

    The accumulated product's gain on the null space is the product of the
    group's ``a`` coefficients -- 119x over all five steps. Rounding a matrix of
    that norm to bf16 leaves an absolute error that the cancelling final product
    cannot recover, and the update's largest singular value leaves the band the
    schedule exists to enforce. Nothing in a loss curve would show that.
    """
    torch.manual_seed(0)
    rows, cols = 128, 640
    spectrum = 1.0 / torch.arange(1, rows + 1, dtype=torch.float64, device="cuda")
    left, _ = torch.linalg.qr(
        torch.randn(rows, rows, dtype=torch.float64, device="cuda")
    )
    right, _ = torch.linalg.qr(
        torch.randn(cols, rows, dtype=torch.float64, device="cuda")
    )
    grad = ((left * spectrum) @ right.mT).float()

    def band_max(phases):
        out = newton_schulz_cubic(grad, dtype=torch.bfloat16, phases=phases)
        return torch.linalg.svdvals(out.float()).max().item()

    assert band_max(NS_PHASES_DIRECT) < 1.31
    assert band_max(NS_PHASES_GRAM) < 1.32
    assert band_max((5,)) > 1.4


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((512, 512), NS_PHASES_DIRECT),
        ((512, 640), NS_PHASES_DIRECT),
        ((512, 2048), NS_PHASES_GRAM),
        ((2048, 512), NS_PHASES_GRAM),
        # An expert stack is judged on its last two dimensions. Folding the
        # expert count into the aspect ratio would send every stack to the
        # grouped path whatever its matrices look like.
        ((8, 512, 512), NS_PHASES_DIRECT),
        ((8, 512, 2048), NS_PHASES_GRAM),
    ],
)
def test_muon_selects_phases_by_matrix_aspect_ratio(shape, expected):
    param = nn.Parameter(torch.zeros(*shape))
    opt = MuonW([{"params": [param], "use_muon": True}], compile_ns=False)
    assert opt._phases[shape] == expected

    off = MuonW(
        [{"params": [param], "use_muon": True}],
        compile_ns=False,
        gram_aspect=float("inf"),
    )
    assert off._phases[shape] == NS_PHASES_DIRECT


def test_muon_step_is_invariant_to_the_batching_budget():
    """Chunking the batch must not change the step, or skip a parameter.

    An off-by-one in the chunk loop leaves the tail of a shape's parameters
    un-updated, which trains -- slightly worse, for no visible reason.
    """
    torch.manual_seed(0)
    start = [torch.randn(24, 96) for _ in range(5)]
    grads = [torch.randn(24, 96) for _ in range(5)]

    results = []
    for budget in (1 << 30, 24 * 96 * 2, 1):
        params = [nn.Parameter(s.clone()) for s in start]
        opt = MuonW(
            [{"params": params, "use_muon": True, "lr": 0.01, "weight_decay": 0.1}],
            ns_dtype=torch.float32,
            compile_ns=False,
            ns_batch_elems=budget,
        )
        for param, grad in zip(params, grads):
            param.grad = grad.clone()
        opt.step()
        results.append([p.detach() - s for p, s in zip(params, start)])

    for other in results[1:]:
        for got, ref in zip(other, results[0]):
            assert rel_error(got, ref) < 1e-5
            assert got.norm() > 0


def test_muon_nesterov_and_heavyball_take_different_steps():
    """The two look-ahead kernels must not be interchangeable.

    Both write a plausible direction into the batch slot, so wiring the
    heavy-ball one into the Nesterov branch silently drops the look-ahead.

    More than one step, and with gradients that differ in direction: on the first
    step the buffer is ``(1 - beta) * grad``, so both look-aheads are positive
    multiples of the same matrix and Newton-Schulz normalizes the difference
    away. A single-step version of this test passes on either wiring.
    """
    torch.manual_seed(0)
    start = torch.randn(24, 96)
    grads = [torch.randn(24, 96) for _ in range(3)]
    steps = []
    for nesterov in (True, False):
        param = nn.Parameter(start.clone())
        opt = MuonW(
            [{"params": [param], "use_muon": True, "lr": 0.01, "weight_decay": 0.0}],
            nesterov=nesterov,
            ns_dtype=torch.float32,
            compile_ns=False,
        )
        for grad in grads:
            param.grad = grad.clone()
            opt.step()
        steps.append(param.detach() - start)
    assert rel_error(steps[0], steps[1]) > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the compiled path")
def test_muon_compiled_step_does_not_recompile_when_lr_changes():
    """A schedule changes lr every step; a float in a compiled graph is a guard.

    ``torch.compile`` specializes on a float's value, so passing the decay and
    step scalars as floats compiles two fresh graphs per shape per distinct lr --
    i.e. on every step of a cosine schedule. They are passed as 0-d tensors for
    exactly this reason.
    """
    counters = importlib.import_module("torch._dynamo.utils").counters
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(64, 256, device="cuda")) for _ in range(2)]
    opt = MuonW(
        [{"params": params, "use_muon": True, "lr": 0.02, "weight_decay": 0.1}],
        compile_ns=True,
    )
    for _ in range(2):
        for param in params:
            param.grad = torch.randn_like(param)
        opt.step()

    compiled = counters["frames"]["ok"]
    for lr in (0.019, 0.018, 0.017):
        opt.param_groups[0]["lr"] = lr
        for param in params:
            param.grad = torch.randn_like(param)
        opt.step()
    assert counters["frames"]["ok"] == compiled


def test_muon_update_scale_conventions():
    # spectral: RMS(dW) = lr / sqrt(fan_in) at every width, so one lr transfers.
    for fan_in in (256, 1024, 4096):
        shape = torch.Size((2 * fan_in, fan_in))
        scale = orthogonal_update_scale(shape, "spectral", 0.2)
        assert scale / max(shape) ** 0.5 == pytest.approx(fan_in**-0.5)
    # rms: RMS(dW) = rms_target whatever the shape, so an AdamW lr carries over.
    for shape in (torch.Size((512, 128)), torch.Size((128, 512))):
        scale = orthogonal_update_scale(shape, "rms", 0.2)
        assert scale / max(shape) ** 0.5 == pytest.approx(0.2)
    with pytest.raises(ValueError, match="unknown update_scale"):
        orthogonal_update_scale(torch.Size((4, 4)), "frobenius", 0.2)


def test_muon_step_ignores_gradient_scale_and_decays_weights():
    torch.manual_seed(0)
    start = torch.randn(32, 16)
    grad = torch.randn(32, 16)

    updates = []
    for factor in (1.0, 1e4):
        param = nn.Parameter(start.clone())
        opt = MuonW(
            [{"params": [param], "use_muon": True, "lr": 0.01, "weight_decay": 0.0}],
            ns_dtype=torch.float32,
        )
        param.grad = grad * factor
        opt.step()
        updates.append(param.detach() - start)
    # Newton-Schulz normalizes, so a loss-scaler or an AMP unscale that missed
    # this group must not change the step it takes.
    assert rel_error(updates[0], updates[1]) < 1e-5

    param = nn.Parameter(start.clone())
    opt = MuonW(
        [{"params": [param], "use_muon": True, "lr": 0.01, "weight_decay": 0.5}],
        ns_dtype=torch.float32,
    )
    param.grad = torch.zeros_like(grad)
    opt.step()
    assert torch.allclose(param.detach(), start * (1 - 0.01 * 0.5))


@pytest.mark.parametrize("nesterov", [True, False], ids=["nesterov", "heavyball"])
@pytest.mark.parametrize(
    "ns_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
)
def test_muon_leaves_its_momentum_buffer_an_ema(nesterov, ns_dtype):
    """Newton-Schulz must not normalize the buffer it was handed.

    Without ``nesterov`` the buffer *is* the tensor passed to the iteration, and
    an in-place normalization there replaces the EMA with a unit-norm matrix --
    which still trains, because the direction is right and the scale is thrown
    away anyway, just with no momentum left.
    """
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(8, 4))
    opt = MuonW(
        [{"params": [param], "use_muon": True, "lr": 0.01, "weight_decay": 0.0}],
        momentum=0.95,
        nesterov=nesterov,
        ns_dtype=ns_dtype,
    )
    grad = torch.randn(8, 4)
    param.grad = grad.clone()
    opt.step()
    assert rel_error(opt.state[param]["momentum_buffer"], 0.05 * grad) < 1e-6


def test_muon_trains_a_model_and_leaves_grads_alone():
    torch.manual_seed(0)
    model = _moe_model()
    opt = build_optimizer(model, name="muon", lr=1e-3, muon_lr=0.02)
    assert isinstance(opt, MuonW)
    assert "muon" in OPTIMIZER.keys()

    tokens = torch.randint(0, 256, (2, 32))
    labels = torch.roll(tokens, -1, dims=1)
    losses = []
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        loss, _ = model.loss(tokens, labels)
        loss.backward()
        before = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        opt.step()
        # The reference implementation writes the Nesterov look-ahead back into
        # .grad; anything reading grads after the step would see garbage.
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.equal(p.grad, before[n]), n
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(p.isfinite().all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA kernel")
def test_fused_adamw_bf16_state_tracks_fp32_state():
    """bf16 moments halve state memory; the parameters must still track fp32."""
    got = _params(device="cuda")
    ref = _params(device="cuda")
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    opt_got = FusedAdamW(got, state_dtype="bfloat16", **kw)
    _run(opt_got, got)
    _run(FusedAdamW(ref, **kw), ref)

    assert opt_got.state[got[0]]["exp_avg"].dtype is torch.bfloat16
    for g, r in zip(got, ref):
        # bf16 has 8 mantissa bits; over 20 steps the moments carry ~1e-2
        # relative error, which is the whole cost of the memory halving.
        assert rel_error(g, r) < 2e-2


def test_quantized_adamw_names_registered_without_torchao():
    """A missing optional dependency must read as an install instruction.

    Registering these names only when torchao imports would turn an uninstalled
    dependency into ``unknown optimizer 'adamw8bit'``, which reads as a typo in
    the config and sends the reader to the wrong place entirely.
    """
    for key in QUANTIZED_ADAMW:
        assert key in OPTIMIZER.keys()
    if importlib.util.find_spec("torchao") is not None:
        pytest.skip("torchao installed; the missing-dependency path is unreachable")
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    with pytest.raises(ImportError, match="torchao"):
        build_optimizer(model, name="adamw8bit", lr=3e-4)


def test_quantized_adamw_receives_this_projects_betas(monkeypatch):
    """torchao's Adam family defaults to beta2=0.999; this project trains at 0.95.

    Pinned against a stand-in module rather than the real torchao, because what
    is under test is ``build_optimizer``'s keyword forwarding -- which matches
    these by registry name, since the classes themselves are behind a deferred
    import -- and because the check must run whether or not torchao is present.
    """
    recorded = {}

    class Probe:
        def __init__(self, params, **kwargs):
            recorded.update(kwargs)
            recorded["groups"] = params

    ao_optim = types.ModuleType("torchao.optim")
    ao_optim.AdamW8bit = Probe
    torchao = types.ModuleType("torchao")
    torchao.optim = ao_optim
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.optim", ao_optim)

    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    build_optimizer(model, name="adamw8bit", lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    assert recorded["betas"] == (0.9, 0.95)
    assert recorded["eps"] == 1e-8
    # `foreach` is a torch.optim keyword; torchao's constructors reject it.
    assert "foreach" not in recorded
    assert all(p.ndim <= 1 for p in recorded["groups"][1]["params"])


def test_build_optimizer_resolves_a_dotted_path_instead_of_calling_it():
    """``OPTIMIZER="pkg.mod.Class"`` must yield the class, not an attempt to build one.

    ``registry.build`` resolves *and calls*, which is what a component the caller wants
    an instance of needs and exactly wrong here: ``build_optimizer`` supplies the
    parameter groups itself, so calling the class at resolution time constructs an
    optimizer with no parameters and raises a bare ``TypeError`` from inside the
    registry. The dotted path is a documented config form, so the assertion is that it
    *works* rather than that it fails politely.

    ``resolve`` covers the registry-name branch too, so this also pins that swapping it
    in did not change what a plain ``"adamw"`` selects.
    """
    model = nn.Linear(8, 8)

    dotted = build_optimizer(model, name="torch.optim.AdamW", lr=1e-3)
    assert isinstance(dotted, torch.optim.AdamW)
    # The explicit betas/eps forward keys off the resolved class, so a dotted path must
    # reach it -- otherwise a dotted AdamW would silently train at torch's 0.999.
    assert dotted.param_groups[0]["betas"] == (0.9, 0.95)

    named = build_optimizer(model, name="adamw", lr=1e-3)
    assert type(named) is type(dotted)
