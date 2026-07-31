"""``config.mxfp8`` -- the selection path that lets a training config request fp8.

The surgery itself is covered in ``test_models_mxfp8_swap.py``; this pins the *config*
path, whose failure modes are different and quieter:

* a preset the kernel cannot fully take must **raise**, not train as a bf16/fp8 mixture
  that every log and config file calls an fp8 run;
* the swap must land **after** ``initialize_weights``, or an fp8 arm and a bf16 arm start
  from different weights and no A/B between them means anything;
* the flag must default off, since importing Triton is a cost every bf16 run should not
  pay.

Kept separate from the swap tests rather than folded in: together they are ~600 lines,
and they answer different questions. This file asks whether a *config* can request fp8
honestly; that one asks what the surgery reached and what it refused.
"""

import pytest
import torch

from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.moe import MoEMLP

VOCAB = 65536
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def build(preset: str, **overrides) -> LMBackbone:
    return LMBackbone(
        get_preset(preset, vocab_size=VOCAB, **overrides),
        head_kwargs={"kernel": "chunked_ce"},
    )


def fp8_count(model: LMBackbone) -> int:
    return sum(isinstance(m, MXFP8Linear) for m in model.modules())


def test_flag_defaults_off_and_costs_nothing():
    model = build("Nano-200M")
    assert fp8_count(model) == 0
    assert model.mxfp8_projections == ()
    assert model.refresh_mxfp8() == 0


@pytest.mark.parametrize("preset,expected", [("Nano-200M", 96), ("Nano-200M-wide", 60)])
def test_flag_swaps_every_eligible_projection(preset, expected):
    model = build(preset, mxfp8=True)
    assert fp8_count(model) == expected
    assert len(model.mxfp8_projections) == expected
    # The tied head contracts over the vocabulary and the router's logit scale *is* the
    # gate sharpness, so neither belongs in fp8 however eligible its shape looks.
    assert not isinstance(model.head, MXFP8Linear)


def test_an_ineligible_preset_raises_and_says_which_projection_and_why():
    """``Nano-200M-deep`` has dim=704, and dim is FPROP's contraction axis.

    ``out_features`` is zero-padded exactly, so only ``in_features`` can block a
    projection -- padding the contraction axis would mean padding every activation that
    reaches the layer. The message has to carry the projection and the number, because
    "mxfp8 failed" sends the reader to the kernel instead of to the preset.
    """
    with pytest.raises(ValueError) as excinfo:
        build("Nano-200M-deep", mxfp8=True)
    message = str(excinfo.value)
    assert "bf16/fp8 mixture" in message
    assert "in_features=704" in message
    assert "q_proj" in message
    assert "not a multiple of 128" in message


def test_swapped_weights_are_bit_identical_to_the_bf16_arm():
    """The property the whole design rests on, and the reason this is a flag.

    A component spec resolved at build time would have ``MXFP8Linear.__init__`` draw its
    own ``normal_(std=fan_in ** -0.5)``, so the two arms of an A/B would differ before
    the first step. Running the surgery after ``initialize_weights`` and copying what it
    produced is what keeps them equal -- and that is only checkable by comparing them.
    """
    torch.manual_seed(20090220)
    plain = build("Nano-200M")
    torch.manual_seed(20090220)
    swapped = build("Nano-200M", mxfp8=True)

    by_name = dict(swapped.named_modules())
    compared = 0
    for name in swapped.mxfp8_projections:
        reference = dict(plain.named_modules())[name]
        assert isinstance(by_name[name], MXFP8Linear)
        assert isinstance(reference, torch.nn.Linear)
        assert torch.equal(by_name[name].weight, reference.weight), name
        compared += 1
    assert compared == 96


def test_a_sparse_preset_puts_its_routed_experts_in_fp8_not_just_its_shared_one():
    """The MoE half of the flag, and the number that says whether it worked.

    ``MoE-2B-A370M`` used to build without complaint under ``mxfp8=True`` while 42.4%
    of its per-token matmul -- the routed expert bank -- stayed bf16, because the
    stacked ``w_in``/``w_out`` have no ``nn.Linear`` for the swap to type-match. The
    layer declares them now, so the config path either converts them or refuses. What
    makes that checkable is the census: ``mxfp8_projections`` alone looked correct
    before, it was just short.
    """
    model = build("MoE-2B-A370M", depth=4, mxfp8=True)
    linears = fp8_count(model)
    experts = [b.mlp for b in model.blocks if isinstance(b.mlp, MoEMLP)]
    assert len(experts) == 2  # moe_first_dense=2 keeps the first two layers dense
    # Two tensors per MoE layer beyond the linears, but one cache: the refresh count
    # follows modules, and asserting against the tensor list would be off by two.
    assert len(model.mxfp8_projections) == linears + 2 * len(experts)
    assert len(model.mxfp8_modules) == linears + len(experts)
    for name in ("blocks.2.mlp.w_in", "blocks.2.mlp.w_out"):
        assert name in model.mxfp8_projections
    # The router steers selection; quantizing its logits would blunt the gate, and it
    # is a bare `nn.Parameter` named `weight` exactly like the ones that are eligible.
    assert not any("router" in name for name in model.mxfp8_projections)
    assert all(isinstance(mlp.w_in, torch.nn.Parameter) for mlp in experts)


def test_a_sparse_preset_the_kernels_cannot_take_raises_and_names_the_share():
    """``MoE-3B-A500M`` stays ineligible, and for a reason worth writing down.

    ``moe_hidden=448`` is fine for the routed experts -- the grouped kernels read
    ``quantize_mx``'s natural scale layout, so they need ``K % 32``, not the vendor's
    128. It is the *shared* expert that blocks: it is a ``GLUMLP`` at the same width,
    so 448 is its ``w_out``'s ``in_features``, which is FPROP's contraction axis and
    shared with the activation cast. Padding it would mean padding ``h``, the SwiGLU
    output, on every forward -- a contract with the caller, unlike ``out_features``,
    whose zero columns live inside the layer. Making ``moe.py`` produce ``h`` at 512 is
    the fix and it is a parameter-count decision, not this one.

    The share is asserted because it is what tells a reader whether one projection or a
    third of the model is at stake; a bare count does not.
    """
    with pytest.raises(ValueError) as excinfo:
        build("MoE-3B-A500M", depth=4, mxfp8=True)
    message = str(excinfo.value)
    assert "in_features=448" in message
    assert "shared.w_out" in message
    assert "of this model's per-token matmul reached fp8" in message
    assert "bf16/fp8 mixture" in message


@cuda_only
def test_refresh_reports_the_count_a_caller_can_assert_on():
    """A trainer that omits the refresh trains on initialization-time weights forever.

    ``MXFP8Linear.forward`` fills its cache only when empty, so the failure is silent.
    Returning the count is what lets the caller assert the hook fired on every module.

    The sparse arm is here because its count is the one that can be wrong in a way a
    dense model cannot show: an MoE layer holds two fp8 tensors behind one cache, so a
    caller comparing against ``mxfp8_projections`` would demand a number the refresh
    can never return.
    """
    model = build("Nano-200M", mxfp8=True).cuda()
    assert model.refresh_mxfp8() == 96
    assert model.refresh_mxfp8() == 96
    assert build("Nano-200M").cuda().refresh_mxfp8() == 0

    sparse = build("MoE-2B-A370M", depth=4, mxfp8=True).cuda()
    assert sparse.refresh_mxfp8() == len(sparse.mxfp8_modules)
    assert len(sparse.mxfp8_modules) < len(sparse.mxfp8_projections)
    assert build("MoE-2B-A370M", depth=4).cuda().refresh_mxfp8() == 0


def test_quantized_caches_do_not_survive_a_device_or_dtype_move():
    """``_apply`` must drop a quantized cache, never carry it through.

    ``MXFP8Linear._cache`` and ``MoEMLP._packed`` are plain attributes, so ``.to(device)``
    and Lightning's ``.cpu()`` leave them pointing at tensors on the *old* device while
    the parameters move -- a stale cache the next forward reads without complaint. On a
    100k-step run with resume, that surfaces as a wrong answer tens of thousands of steps
    in, from a checkpoint load that looked clean.

    **Registering them as buffers is the obvious fix and is the worse one.** ``_apply``'s
    dtype transforms are gated on ``is_floating_point``, and that is **True for e4m3 and
    e8m0** -- so ``.float()`` would rewrite a registered cache to fp32 and hand a kernel
    that requires fp8 something else. A plain attribute cannot suffer that; registering
    would *introduce* it. ``persistent=False`` avoids the 3x checkpoint but not this.

    Dropping is correct for both, and costs nothing: the cache is a pure function of the
    weights, it is rebuilt after every optimizer step regardless, and both consumers
    (``MXFP8Linear.forward``, ``MoEMLP._routed_mxfp8``) already rebuild lazily when it is
    ``None``.

    CPU, with a planted cache: building a real one needs the Triton quantizer, and what
    is under test is ``_apply``'s treatment of the attribute rather than the quantizer.
    """
    # The premise the whole choice rests on. If this ever became False a registered
    # buffer would be safe, and the reasoning above would need revisiting.
    assert torch.zeros(1, dtype=torch.float8_e4m3fn).is_floating_point()

    fp8 = torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
    linear = MXFP8Linear(128, 64)
    linear._cache = (fp8, fp8, fp8, fp8)
    linear.float()
    assert linear._cache is None, "MXFP8Linear carried its fp8 cache through .float()"

    moe = MoEMLP(dim=128, hidden=64, num_experts=4, top_k=1, num_shared=0)
    moe._packed = object()
    moe.to(torch.float64)
    assert moe._packed is None, "MoEMLP carried its fp8 expert copies through .to()"

    # A move must not disable the rebuild: both consumers key off `is None`, so a cache
    # dropped here has to be reachable again rather than latched off.
    assert linear._cache is None and moe._packed is None
    linear.to(torch.bfloat16)
    assert linear._cache is None
