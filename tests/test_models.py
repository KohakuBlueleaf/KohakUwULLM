"""The backbone itself: forward/backward, packing equivalence, and init scaling.

Split by subject once this file passed the 1000-line cap. Attention lives in
``test_models_attention.py``, MoE in ``test_models_moe.py``, the fp8 surgery in
``test_models_mxfp8_swap.py``, and the collate/split functions -- which are data
code, not model code -- in ``test_packing.py``.

``test_packed_equals_individual_documents`` is the load-bearing one: it is what
says the packed layout is an optimization rather than a different model.
"""

import pytest
import torch
from model_fixtures import tiny_config

from kohakuwullm import LMArchConfig, LMBackbone, SeqInfo
from kohakuwullm.bench.core.timing import rel_error

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------- backbone


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_backbone_forward_backward(dtype):
    model = LMBackbone(tiny_config()).cuda().to(dtype)
    lengths = torch.tensor([30, 50, 20], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, "cuda")
    tokens = torch.randint(0, 512, (100,), device="cuda")
    labels = tokens.roll(-1)

    loss, logs = model.loss(tokens, labels, info)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embed.weight.grad is not None
    assert int(logs["n_tokens"]) == 100


@cuda_only
def test_packed_equals_individual_documents():
    """One packed batch == the same documents forwarded one at a time."""
    torch.manual_seed(0)
    model = LMBackbone(tiny_config()).cuda().to(torch.bfloat16).eval()
    lengths = [17, 33, 9]
    docs = [torch.randint(0, 512, (n,), device="cuda") for n in lengths]

    info = SeqInfo.from_lengths(torch.tensor(lengths, dtype=torch.int32), "cuda")
    packed = model(torch.cat(docs), info)

    offset = 0
    for doc, n in zip(docs, lengths):
        single = model(
            doc, SeqInfo.from_lengths(torch.tensor([n], dtype=torch.int32), "cuda")
        )
        assert rel_error(packed[offset : offset + n].float(), single.float()) < 3e-2
        offset += n


@cuda_only
def test_grad_checkpointing_matches_plain():
    torch.manual_seed(0)
    model = LMBackbone(tiny_config()).cuda().to(torch.bfloat16)
    info = SeqInfo.from_lengths(torch.tensor([64], dtype=torch.int32), "cuda")
    tokens = torch.randint(0, 512, (64,), device="cuda")
    labels = tokens.roll(-1)

    model.grad_ckpt = False
    loss_a, _ = model.loss(tokens, labels, info)
    loss_a.backward()
    grad_a = model.embed.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    model.grad_ckpt = True
    model.train()
    loss_b, _ = model.loss(tokens, labels, info)
    loss_b.backward()
    assert rel_error(model.embed.weight.grad, grad_a) < 5e-2


def test_layer_type_interleave():
    """The last layer is always global -- it is the one that feeds the head."""
    config = LMArchConfig(depth=12, sliding_window=512, global_layer_every=4)
    types = config.layer_types
    assert types[3] == "full" and types[-1] == "full"
    assert types[0] == "sliding"
    assert config.window_for(0) == 512 and config.window_for(3) is None


def test_moe_layer_placement():
    config = LMArchConfig(depth=8, moe_every=2, moe_first_dense=2)
    assert config.moe_layers == [False, False, True, False, True, False, True, False]


# --------------------------------------------------------------------- init


def _init_stds(width: int, **overrides) -> dict[str, float]:
    torch.manual_seed(0)
    model = LMBackbone(
        tiny_config(
            dim=width,
            heads=width // 32,
            mlp_hidden=2 * width,
            moe_every=1,
            moe_first_dense=1,
            moe_num_experts=4,
            moe_top_k=2,
            moe_hidden=width // 2,
            moe_router_kwargs={"dense_fallback": True},
            **overrides,
        )
    )
    named = dict(model.named_parameters())
    return {n: named[n].std().item() for n in named if named[n].ndim >= 2}


@pytest.mark.parametrize(
    "name, exponent",
    [
        ("blocks.0.attn.q_proj.weight", 0.5),
        ("blocks.0.mlp.w_out.weight", 0.5),
        ("blocks.1.mlp.w_in", 0.5),
        ("blocks.1.mlp.w_out", 0.5),
        ("blocks.1.mlp.shared.w_out.weight", 0.5),
        # The embedding's fan_in is the vocabulary; it must not move with width.
        ("embed.weight", 0.0),
    ],
)
def test_mup_init_follows_the_predicted_width_exponent(name, exponent):
    narrow = _init_stds(64, mup_base_dim=32)
    wide = _init_stds(128, mup_base_dim=32)
    assert wide[name] / narrow[name] == pytest.approx(2.0**-exponent, rel=0.05)


def test_init_without_mup_base_dim_is_width_independent():
    """The flag is off by default, and off means the old flat ``init_std``."""
    narrow, wide = _init_stds(64), _init_stds(128)
    flat = "blocks.0.attn.q_proj.weight"
    assert wide[flat] / narrow[flat] == pytest.approx(1.0, rel=0.05)
    # The expert stack keeps MoEMLP's own fan-in init when muP is off, which is a
    # different rule from every dense matrix in the same model.
    assert narrow["blocks.1.mlp.w_in"] / narrow[flat] == pytest.approx(
        64**-0.5 / 0.02, rel=0.05
    )
