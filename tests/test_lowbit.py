"""Correctness checks for native low-precision parameters.

The test that matters is :func:`test_nearest_rounding_stalls_bf16_parameters`.
Every other failure mode here is loud; that one is silent -- a bf16 parameter
under round-to-nearest simply stops moving once the update drops below half an
ulp, and the loss curve flattens with nothing to point at. It is written as a
paired test so the stall and its fix are pinned by the same assertion.
"""

import pytest
import torch
import torch.nn as nn

from kohakuwullm import LMArchConfig, LMBackbone
from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.training.optim.lowbit import (
    KEEP_FP32_DEFAULT,
    StochasticAdamW,
    cast_parameters_,
    stochastic_round_,
)

SHAPES = [(64, 32), (128,), (17, 5), (256, 8)]
STEPS = 20


def _params(dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.randn(*s, generator=gen).to(dtype)) for s in SHAPES]


def _grads(step, params, seed=0):
    gen = torch.Generator().manual_seed(seed + 1000 * step)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=gen).to(p.dtype)


def _run(opt, params, steps=STEPS, seed=0):
    for step in range(steps):
        _grads(step, params, seed=seed)
        opt.step()
    return params


def test_stochastic_rounding_is_unbiased_and_exact_on_representable_values():
    """The mean of many draws is the input; a bf16-exact input never moves."""
    torch.manual_seed(0)
    # bf16 has 8 significand bits (7 stored), so its ulp at 1.0 is 2^-7. This
    # sits a quarter of the way to the next value, where an unbiased rule rounds
    # up a quarter of the time and a nearest rule never rounds up at all.
    lo = torch.tensor(1.0)
    step = torch.tensor(2.0**-7)
    src = (lo + 0.25 * step).expand(200_000).contiguous()
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    stochastic_round_(dst, src)

    up = (dst.float() > lo).float().mean().item()
    assert abs(up - 0.25) < 0.01
    assert rel_error(dst.float().mean(), src[0]) < 1e-4
    # Only the two neighbours may ever appear.
    assert torch.isin(dst.float(), torch.stack([lo, lo + step])).all()

    exact = torch.tensor([0.0, -1.0, 2.0, 0.5, -768.0])
    out = torch.empty_like(exact, dtype=torch.bfloat16)
    stochastic_round_(out, exact)
    assert torch.equal(out.float(), exact)


def test_stochastic_rounding_rejects_fp16():
    """fp16 is not a bit-prefix of fp32, so the bit trick must refuse it."""
    with pytest.raises(ValueError, match="bf16-only"):
        stochastic_round_(torch.empty(4, dtype=torch.float16), torch.zeros(4))


def test_nearest_rounding_stalls_bf16_parameters():
    """Sub-ulp updates vanish under nearest rounding and survive under the others.

    A constant gradient makes Adam's update exactly ``lr`` per step, so the
    parameter's true displacement after ``n`` steps is ``n * lr`` and the test
    needs no reference optimizer to know the answer.
    """
    torch.manual_seed(0)
    lr, steps = 1e-5, 1000
    expected = steps * lr
    ulp = 2.0**-7

    def _drive(rounding):
        param = nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
        opt = StochasticAdamW([param], lr=lr, weight_decay=0.0, rounding=rounding)
        for _ in range(steps):
            param.grad = torch.ones_like(param)
            opt.step()
        return (1.0 - param.detach().float()).mean().item()

    # bf16's ulp at 1.0 is 2^-7 = 7.8e-3 and each update is 1e-5, nearly three
    # orders of magnitude below the tie point, so nearest rounding discards
    # every single one and the parameter is bit-identical to its init.
    assert _drive("nearest") == 0.0
    # Stochastic rounding is unbiased across the 4096 elements, so their mean
    # resolves the displacement far below one ulp. Kahan is deterministic and
    # can only land on the grid -- correctness for it means the discarded
    # remainder is still held in the compensation buffer, i.e. within one ulp.
    assert abs(_drive("stochastic") - expected) < 0.2 * ulp
    assert abs(_drive("kahan") - expected) < ulp


def test_fp32_parameters_match_torch_adamw():
    """fp32 takes the same code path and must reproduce the reference exactly."""
    got = _params()
    ref = _params()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    _run(StochasticAdamW(got, **kw), got)
    _run(torch.optim.AdamW(ref, foreach=False, **kw), ref)
    for g, r in zip(got, ref):
        assert rel_error(g, r) < 1e-6


def test_step_does_not_mutate_gradients_or_state_in_place():
    """`.float()` on an fp32 tensor aliases it; the step must not write through."""
    params = _params()
    opt = StochasticAdamW(params, lr=1e-2, clip_grad_norm=1e-3)
    _grads(0, params)
    saved = [p.grad.clone() for p in params]
    opt.step()
    for p, g in zip(params, saved):
        assert torch.equal(p.grad, g)
    # A second moment left divided by its own bias correction would decay the
    # update by a compounding factor that no assertion on one step can see.
    assert opt.state[params[0]]["exp_avg_sq"].max() > 0


def test_clip_is_global_and_below_threshold_is_a_no_op():
    max_norm = 0.5
    clipped = _params()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    opt = StochasticAdamW(clipped, clip_grad_norm=max_norm, **kw)
    _grads(0, clipped)
    expected = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in clipped]))
    opt.step()
    assert rel_error(opt.grad_norm(), expected) < 1e-6

    loose = _params()
    tight = _params()
    _run(StochasticAdamW(loose, clip_grad_norm=1e6, **kw), loose)
    _run(StochasticAdamW(tight, **kw), tight)
    for a, b in zip(loose, tight):
        assert torch.equal(a, b)


def test_rejects_bad_config():
    params = _params()
    with pytest.raises(ValueError, match="betas"):
        StochasticAdamW(params, betas=(1.0, 0.95))
    with pytest.raises(ValueError, match="rounding"):
        StochasticAdamW(params, rounding="nearest-even")
    with pytest.raises(ValueError, match="state_dtype"):
        StochasticAdamW(params, state_dtype=torch.float16)
    with pytest.raises(ValueError, match="clip_grad_norm"):
        StochasticAdamW(params, clip_grad_norm=0.0)


def test_cast_parameters_honours_the_keep_fp32_policy():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    model.register_buffer("counter", torch.zeros(2, dtype=torch.int64))
    model.register_buffer("inv_freq", torch.ones(4))
    summary = cast_parameters_(model, torch.bfloat16, keep_fp32=("1.",))

    assert model[0].weight.dtype is torch.bfloat16
    assert model[1].weight.dtype is torch.float32
    assert summary == {"cast": 72, "kept_fp32": 16}
    # An int buffer cast to bf16 stops counting at 256, and a bf16 `inv_freq`
    # is radians of phase error at long positions. Neither saves any memory
    # worth having, so buffers stay put unless explicitly asked for.
    assert model.counter.dtype is torch.int64
    assert model.inv_freq.dtype is torch.float32
    # The keep-list guards buffers too, so turning casting on does not drag
    # `inv_freq` down with it.
    cast_parameters_(model, torch.bfloat16, cast_buffers=True)
    assert model.inv_freq.dtype is torch.float32
    cast_parameters_(model, torch.bfloat16, keep_fp32=(), cast_buffers=True)
    assert model.inv_freq.dtype is torch.bfloat16
    assert model.counter.dtype is torch.int64


def test_keep_fp32_default_covers_norms_and_router():
    """The default policy must match by the names this repo actually uses."""
    names = [
        "blocks.0.norm1.weight",
        "blocks.0.mlp.router.weight",
        "blocks.0.mlp.router.expert_bias",
        "blocks.0.mlp.router.load_accum",
        "final_norm.weight",
        "pos_enc.inv_freq",
    ]
    for name in names:
        assert any(token in name for token in KEEP_FP32_DEFAULT), name
    assert not any(
        token in "blocks.0.attn.q_proj.weight" for token in KEEP_FP32_DEFAULT
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"attn_sink": True},
        {"attn_bias": True},
        {"qk_norm": False},
        {"post_norm": True},
        {"tie_embeddings": False},
        {"attn_sink": True, "attn_bias": True, "post_norm": True},
        {"moe_every": 1, "moe_num_experts": 8, "moe_hidden": 256},
    ],
)
def test_keep_fp32_default_covers_the_unquantized_tail(overrides):
    """Every tensor a quantized optimizer leaves alone must also stay fp32.

    torchao quantizes a parameter's state only when ``numel() >= 4096`` and
    ``numel()`` divides the block size, and falls back to ``zeros_like(p)``
    otherwise -- which takes the *parameter* dtype. So a tensor that is bf16 here
    and unquantized there gets a bf16 Adam moment, and a bf16 EMA carries ~5e-3
    relative error against ~9e-8 for fp32.

    Swept over config flags rather than over presets: every preset ships
    ``attn_sink`` and ``attn_bias`` off, so a preset-only version of this test
    passes vacuously and keeps passing until someone flips a flag.
    """
    config = LMArchConfig(
        dim=384, depth=2, heads=6, kv_heads=2, head_dim=64, vocab_size=4096, **overrides
    )
    with torch.device("meta"):
        model = LMBackbone(config)

    leaked = [
        name
        for name, param in model.named_parameters()
        if param.numel() < 4096
        and not any(token in name for token in KEEP_FP32_DEFAULT)
    ]
    assert not leaked, leaked
