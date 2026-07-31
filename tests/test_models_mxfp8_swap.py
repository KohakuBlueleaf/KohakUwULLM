"""``swap_mxfp8``: what it converted, and -- the point -- what it did not.

The surgery returns an *accounting*, not a list of successes. Anything it could
not convert is charged to a bucket and blocks the run, because a hard failure is
recoverable and a mislabelled experiment is not: before that, the swap reported
``0 skipped`` on a preset whose routed experts -- 42.4% of per-token matmul --
stayed bf16, one command away from a "MoE fp8 loss" measured on a model that was
two-thirds bf16.

``test_mxfp8_model_trains_on_stale_weights_without_a_refresh`` is a *positive*
test for a failure: the quantized weights are a cache with no invalidation, so a
loop that never refreshes them trains on initialization-time weights for the
whole run with no symptom in the loss. It has to be demonstrated to be believed.

The config-selection path is in ``test_models_mxfp8_config.py``.
"""

import pytest
import torch
from model_fixtures import tiny_config

from kohakuwullm import LMBackbone, SeqInfo
from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.mxfp8_protocol import Matmul
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights, swap_mxfp8

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------- mxfp8


@cuda_only
def test_mxfp8_swap_covers_every_projection_and_reports_skips():
    """Every legal projection swapped, every illegal one *named*, nothing else touched.

    Three ways this surgery fakes an fp8 training result, all pinned here:

    * a projection whose shape ``MXFP8Linear`` rejects is left in bf16 -- so the
      "fp8" arm is partly bf16 and the A/B measures a mixture;
    * the head or the embedding gets swapped -- the one matrix that contracts over
      the vocabulary, where a shared 32-block exponent has no justification;
    * the weight copy is skipped, so the arms no longer start from identical
      weights and step 0 already differs.
    """
    legal = tiny_config(kv_heads=4)  # kv_out = 128
    model = LMBackbone(legal).cuda()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    report = swap_mxfp8(model)
    assert report.skipped == []
    # q/k/v/o + w_in/w_out on every layer, and nothing beyond them.
    assert len(report.swapped) == legal.depth * 6
    assert sum(isinstance(m, MXFP8Linear) for m in model.modules()) == legal.depth * 6
    for name in report.swapped:
        assert name.startswith("blocks.")
        assert ".attn." in name or ".mlp." in name

    assert isinstance(model.embed, torch.nn.Embedding)
    assert not isinstance(model.head, MXFP8Linear)
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} changed across the swap"

    # nn.Linear's autocast contract. Without it the fp32 that a norm hands to
    # q/k/v_proj comes back as fp32, `varlen_attn` refuses the dtype, and
    # attention silently drops to SDPA with a quadratic mask -- a different
    # kernel from the one the bf16 arm runs.
    seen = {}

    def watch(module, args, out):
        seen.setdefault("v", (args[0].dtype, out.dtype))

    handle = model.blocks[0].attn.v_proj.register_forward_hook(watch)
    tokens = torch.randint(0, legal.vocab_size, (96,), device="cuda")
    info = SeqInfo.from_lengths(torch.tensor([32, 64]), "cuda")
    with torch.autocast("cuda", torch.bfloat16):
        hidden = model(tokens, info)
    handle.remove()
    assert seen["v"] == (torch.float32, torch.bfloat16)
    assert torch.isfinite(hidden.float()).all()

    assert refresh_mxfp8_weights(model) == len(report.swapped)

    # Every swapped projection, under both autocast dtypes, on a backbone holding fp32
    # masters. This is the assertion the suite never had: the original pinned the bug
    # (`out_dtype=x.dtype` returning fp32), and when the fix landed it silently migrated
    # to covering the `_Bf16Input` wrapper instead of the module. With the wrapper
    # retired there is nothing left to mask a regression, so the coverage has to be
    # here -- and it has to include **fp16**, because a hardcoded bf16 passes the bf16
    # case and silently truncates the fp16 one, which is the arm fp16 exists for.
    assert (
        model.blocks[0].attn.v_proj.weight.dtype == torch.float32
    ), "the point of this check is fp32 masters; a bf16 backbone cannot see the bug"
    for autocast_dtype in (torch.bfloat16, torch.float16):
        outputs = {}

        # A statement body, not `lambda ...: outputs.setdefault(...)`. A forward hook
        # that returns non-None **replaces the module's output**, and `setdefault`
        # returns a value -- so the lambda form handed every projection's consumer a
        # `torch.dtype` instead of a tensor.
        def recorder(name):
            def hook(_module, _args, out):
                outputs.setdefault(name, out.dtype)

            return hook

        hooks = [
            module.register_forward_hook(recorder(name))
            for name, module in model.named_modules()
            if isinstance(module, MXFP8Linear)
        ]
        with torch.autocast("cuda", autocast_dtype):
            model(tokens, info)
        for handle in hooks:
            handle.remove()
        assert len(outputs) == len(report.swapped)
        wrong = {n: d for n, d in outputs.items() if d is not autocast_dtype}
        assert (
            not wrong
        ), f"under {autocast_dtype} these did not follow autocast: {wrong}"

    # Explicit `linear_cls` even though it is now the default: the assertion below is
    # about `MXFP8Linear` specifically honouring the contract on its own, which is what
    # allowed the `_Bf16Input` wrapper to be retired. Reading the default instead would
    # make this pass for whatever the default happens to become.
    bare = LMBackbone(legal).cuda()
    swap_mxfp8(bare, linear_cls=MXFP8Linear)
    seen.clear()
    handle = bare.blocks[0].attn.v_proj.register_forward_hook(watch)
    with torch.autocast("cuda", torch.bfloat16):
        bare(tokens, info)
    handle.remove()
    assert seen["v"] == (torch.float32, torch.bfloat16)

    # The negative case has to be an unaligned *in_features* now: `out_features` is
    # DGRAD's contraction axis and `MXFP8Linear` zero-pads it exactly, so `kv_out=64`
    # is taken rather than skipped. `dim=192` blocks five of six projections on the
    # axis that cannot be padded -- the same 5-of-6 shape as `Nano-200M-deep`, which
    # is the real preset this protects. It must still be reported, not dropped.
    partial = LMBackbone(tiny_config(dim=192, heads=6)).cuda()
    partial_report = swap_mxfp8(partial)
    skipped = dict(partial_report.skipped)
    assert len(skipped) == partial.config.depth * 5
    assert all("w_out" not in name for name in skipped)
    assert all("in_features=192" in r for r in skipped.values())

    # And the padded axis is *taken*, not merely un-reported. `kv_out=64` is the shape
    # that used to be skipped, and this is the only end-to-end check that the swap and
    # the module agree -- the module test alone would pass with `_reject` unchanged,
    # leaving every affected preset ineligible while looking fixed.
    padded = LMBackbone(tiny_config()).cuda()
    padded_report = swap_mxfp8(padded)
    assert padded_report.skipped == []
    assert len(padded_report.swapped) == padded.config.depth * 6
    kv = padded.blocks[0].attn.k_proj
    assert (kv.out_features, kv.padded_out_features) == (64, 128)

    # An MoE layer's routed experts are a stacked (E, out, in) nn.Parameter with no
    # `nn.Linear` to type-match, so they are reached by *declaration* instead. They
    # stay Parameters -- the fused kernel reads them directly and the optimizer
    # grouping still sees the same names -- so the only evidence the conversion
    # happened is the report, which is why every part of it is asserted.
    sparse = LMBackbone(
        tiny_config(kv_heads=4, moe_every=1, moe_num_experts=4, moe_top_k=2)
    ).cuda()
    sparse_report = swap_mxfp8(sparse)
    assert sparse_report.skipped == []
    assert not sparse_report.blocking
    assert not any("router" in name for name in sparse_report.swapped)
    # Four attention projections plus the shared expert's pair plus the two expert
    # stacks, per layer; the MoE layer's two stacks share one cache, so it is one
    # module and `refresh_mxfp8_weights` counts it once.
    depth = sparse.config.depth
    assert len(sparse_report.swapped) == depth * 8
    assert len(sparse_report.modules) == depth * 7
    assert refresh_mxfp8_weights(sparse) == len(sparse_report.modules)
    assert all(
        isinstance(block.mlp.w_in, torch.nn.Parameter) for block in sparse.blocks
    )
    # The router's matrix is accounted, and accounted as never-fp8. Absent from the
    # census entirely would look identical in `swapped`, so `by_design` is what
    # separates "excluded" from "forgotten".
    assert sum("router.weight" in name for name in sparse_report.by_design) == depth


def _sparse_moe_backbone(**overrides):
    return LMBackbone(
        tiny_config(
            kv_heads=4, moe_every=1, moe_num_experts=4, moe_top_k=2, **overrides
        )
    )


def _matmul_census(model) -> dict[str, int]:
    """Per-token MAC by bucket, computed from shapes rather than from the report.

    An independent second opinion on purpose. The report's denominator is what every
    "x% of matmul is fp8" claim rests on, and a report that both produces the number
    and checks it can be self-consistently wrong -- which is precisely the failure
    being fixed: the old guard's own accounting said 100% while 42% was bf16.
    """
    by_shape = {"linear": 0, "bf16_linear": 0, "routed": 0, "never": 0}
    for module in model.modules():
        if isinstance(module, MXFP8Linear):
            by_shape["linear"] += module.in_features * module.out_features
        elif isinstance(module, torch.nn.Linear):
            by_shape["bf16_linear"] += module.in_features * module.out_features
        elif isinstance(module, MoEMLP):
            by_shape["routed"] += (
                module.w_in[0].numel() + module.w_out[0].numel()
            ) * module.top_k
            by_shape["never"] += module.router.weight.numel()
    by_shape["never"] += model.head.projection.numel()
    return by_shape


def test_mxfp8_declared_matmul_is_never_silently_left_behind():
    """The three ways a MoE model can be *called* fp8 while a third of it is bf16.

    The bug this pins was live: ``swap_mxfp8`` selected by parent module type over
    ``nn.Linear``, an MoE layer's ``w_in``/``w_out`` are bare ``nn.Parameter`` stacks
    fed to a grouped GEMM, and so the surgery reported ``0 skipped`` on a model whose
    routed experts -- 42.4% of ``MoE-2B-A370M``'s per-token matmul -- it never
    touched. Widening the scan to parameter *names* was the tempting fix and is
    strictly worse: ``LMHead.weight`` and a router's ``weight`` are also bare
    parameters called ``weight``, so it would have put the gate in fp8.

    So the fix is a declaration, and each of its three failure modes gets a case:

    * a module declares matmul and nothing converts it -- what the old guard did;
    * a module holds matmul and declares nothing -- the case a declaration
      *cannot* cover, caught by rank instead of by recognition;
    * a declared tensor is refused on shape, which must refuse its siblings too
      rather than half-convert the layer.

    CPU on purpose: none of this touches a kernel, and the guard has to be right
    before a card is spent on the run it authorizes.
    """
    model = _sparse_moe_backbone()
    report = swap_mxfp8(model)
    by_shape = _matmul_census(model)
    assert not report.blocking
    # The report's own numbers against shapes read off the model. Equality, not a
    # tolerance: these are integer MAC counts of the same tensors.
    assert by_shape["bf16_linear"] == 0
    assert report.mac["fp8"] == by_shape["linear"] + by_shape["routed"]
    assert report.mac["never"] == by_shape["never"]
    assert report.mac["skipped"] == 0 and report.mac["unreached"] == 0
    assert report.total_mac == report.mac["fp8"] + by_shape["never"]
    # The routed experts are the bulk of a sparse layer's arithmetic, so a guard that
    # misses them misses most of the model. Asserted as a share so this stays a
    # statement about the stake rather than about one config's numbers.
    assert by_shape["routed"] / report.total_mac > 0.4

    # 1. Declared, unconverted. Deleting the converter reproduces the old guard
    #    exactly -- the declaration is still there, so the arithmetic is still
    #    counted, and that is the whole difference.
    enable = MoEMLP.enable_mxfp8
    del MoEMLP.enable_mxfp8
    try:
        stripped = swap_mxfp8(_sparse_moe_backbone())
    finally:
        MoEMLP.enable_mxfp8 = enable
    assert stripped.blocking
    assert stripped.skipped == []  # the old report's only signal, still empty
    assert len(stripped.unreached) == 2 * tiny_config().depth
    assert stripped.mac["unreached"] == by_shape["routed"]
    assert stripped.mac["fp8"] == by_shape["linear"]
    assert "no enable_mxfp8()" in stripped.unreached[0][1]
    assert stripped.share("unreached") > 0.4

    #    And the same outcome reached deliberately, via the attribution hatch rather
    #    than a missing converter. Both must land in `unreached` and both must refuse --
    #    a caller that opts out of converting the declared tensors is asking for a
    #    mixture, and the report has to keep saying so. The *reasons* differ, and that
    #    is the point: "the module forgot" and "the caller asked" are not the same
    #    diagnosis, and collapsing them would send a reader to the wrong file.
    opted_out = swap_mxfp8(_sparse_moe_backbone(), convert_declared=False)
    assert opted_out.blocking
    assert opted_out.mac["unreached"] == by_shape["routed"]
    assert opted_out.mac["fp8"] == by_shape["linear"]
    assert all("convert_declared=False" in r for _, r in opted_out.unreached)
    assert not any(
        "convert_declared" in r for _, r in stripped.unreached
    ), "a missing converter must not be reported as a caller's choice"

    # 2. Undeclared. A bare matmul-shaped parameter on a module the swap does visit,
    #    under a name it does not know: unreachable by declaration by construction,
    #    so rank is the only handle. The refusal is the point -- a guard that fails
    #    open here is the guard that shipped.
    undeclared = _sparse_moe_backbone()
    undeclared.blocks[0].attn.register_parameter(
        "extra_proj", torch.nn.Parameter(torch.zeros(8, 8))
    )
    stray = swap_mxfp8(undeclared)
    assert stray.blocking
    assert stray.unaccounted == ["blocks.0.attn.extra_proj(8, 8)"]
    # A 1-D parameter is a norm gain or an attention sink, never a matmul, and must
    # not trip the sweep -- if it did, every model with a norm would be refused.
    ok = _sparse_moe_backbone()
    ok.blocks[0].attn.register_parameter("gain", torch.nn.Parameter(torch.zeros(8)))
    assert not swap_mxfp8(ok).blocking

    # 3. Refused on shape. `dense_fallback` is the eager reference path, so an fp8
    #    conversion of it would be a contradiction rather than a slow arm. Both
    #    declared stacks must land in `skipped`, not one: converting the eligible
    #    half of a layer is the partial conversion the whole guard exists to refuse.
    fallback = swap_mxfp8(_sparse_moe_backbone(moe_mlp_kwargs={"dense_fallback": True}))
    assert fallback.blocking
    assert len(fallback.skipped) == 2 * tiny_config().depth
    assert all("dense_fallback" in reason for _, reason in fallback.skipped)
    assert fallback.mac["fp8"] == by_shape["linear"]
    assert fallback.mac["skipped"] == by_shape["routed"]

    # A *mixed* verdict, which no current module can produce -- `dim` and `hidden` are
    # shared by both expert stacks -- but which the protocol has to answer for anyway,
    # since the one-refusal-refuses-the-module rule is what stops a future module from
    # half-converting itself. A stub is the only way to reach it.
    class _MixedVerdict(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.good = torch.nn.Parameter(torch.zeros(4, 4))
            self.bad = torch.nn.Parameter(torch.zeros(4, 4))
            self.converted = False

        def mxfp8_matmul(self):
            return {"good": Matmul(16), "bad": Matmul(16, "unaligned")}

        def enable_mxfp8(self) -> None:
            self.converted = True

    mixed = _MixedVerdict()
    mixed_report = swap_mxfp8(mixed)
    assert not mixed.converted, "one refused tensor must refuse the whole module"
    assert mixed_report.swapped == [] and mixed_report.mac["fp8"] == 0
    assert [name for name, _ in mixed_report.skipped] == ["good", "bad"] or [
        name for name, _ in mixed_report.skipped
    ] == ["bad", "good"]
    assert mixed_report.mac["skipped"] == 32


@cuda_only
def test_mxfp8_model_trains_on_stale_weights_without_a_refresh():
    """A missed ``refresh_mxfp8_weights`` must be detectable at the model level.

    The layer-level version of this is in ``tests/test_kernels.py``. It is worth
    pinning again here because the whole model is where the mistake actually
    happens: an optimizer step moves every weight, the caches all go stale
    together, and the loss curve stays perfectly plausible while the model trains
    against the previous step's weights.

    Run on a sparse model as well as a dense one, and the sparse arm carries a second
    claim: staleness is only observable if the forward really reads the fp8 copies. An
    ``enable_mxfp8`` that declared success without rebinding the routed path would
    leave the eager grouped GEMM reading ``w_in`` directly, so the mutation below would
    show up immediately and the ``torch.equal`` would fail. Nothing in the swap report
    can distinguish those two worlds.
    """
    for overrides in ({}, dict(moe_every=1, moe_num_experts=4, moe_top_k=2)):
        torch.manual_seed(0)
        model = LMBackbone(tiny_config(kv_heads=4, **overrides)).cuda()
        report = swap_mxfp8(model)
        assert not report.blocking
        tokens = torch.randint(0, 512, (64,), device="cuda")
        info = SeqInfo.from_lengths(torch.tensor([64]), "cuda")

        with torch.autocast("cuda", torch.bfloat16):
            before = model(tokens, info).clone()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, MXFP8Linear):
                    module.weight.mul_(1.5)
                elif isinstance(module, MoEMLP):
                    module.w_in.mul_(1.5)
                    module.w_out.mul_(1.5)
        with torch.autocast("cuda", torch.bfloat16):
            stale = model(tokens, info)
        assert torch.equal(stale, before), "caches did not go stale; this test is void"

        assert refresh_mxfp8_weights(model) == len(report.modules)
        with torch.autocast("cuda", torch.bfloat16):
            fresh = model(tokens, info)
        assert not torch.equal(fresh, before)
