"""Trainer-level correctness: FLOP accounting, token counters, previews, resume.

Nothing here needs a GPU -- the point of every test is bookkeeping, and a
bookkeeping bug is invisible in a loss curve.

The negative cases are the ones that earn their keep:

* a FLOP model that charges the embedding *gather* as a GEMM reports an MFU too
  high by the vocabulary's share of the model, and an MoE model that charges all
  ``num_experts`` instead of ``top_k + num_shared`` overstates it further -- both
  in the flattering direction, so nothing in the logs looks wrong;
* a preview that samples from the default RNG stream changes the data order, so
  turning logging on changes the run;
* a resume that restores weights but not the data position trains twice on the
  same tokens and reports a token count that never happened.
"""

import copy
import time
import types
import warnings
from unittest import mock

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from kohakuwullm import LMArchConfig, LMBackbone, SeqInfo, apply_compile
from kohakuwullm.data.loader.microbatch import MicroBatchedStep
from kohakuwullm.data.packing import IGNORE_INDEX, collate_packed
from kohakuwullm.kernels.mxfp8 import quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights
from kohakuwullm.models.flops import FlopCounter, attended_pairs, document_lengths
from kohakuwullm.models.mxfp8_swap import swap_mxfp8
from kohakuwullm.training import LMStage, plan_for
from kohakuwullm.training.loop import trainer as trainer_module
from kohakuwullm.training.loop.callbacks import SampleLogCallback, ThroughputCallback
from kohakuwullm.training.loop.resume import load_rng_state, rng_state
from kohakuwullm.training.loop.tokens import TokenSnapshot
from kohakuwullm.training.loop.trainer import LMTrainer
from kohakuwullm.training.parallel.autotune import LayerCosts, layer_key
from kohakuwullm.training.parallel.uwupipe import (
    PipelineStep,
    batch_signature,
    corpus_steps,
    first_mismatch,
    verify_same_batch,
)
from kohakuwupipe import PipelineGradScaler
from kohakuwupipe.parallel.plan import partition
from kohakuwupipe.parallel.streams import reduce_accumulator
from kohakuwupipe.training.callbacks import Throughput
from kohakuwupipe.training.hooks import Callback as PipeCallback
from kohakuwupipe.training.hooks import CallbackList
from kohakuwupipe.training.loop import StepOutput, build_loss_fn

VOCAB = 64
DIM = 32
DEPTH = 2
CTX = 64


def _cpu_config(**overrides):
    """A tiny model whose every component has a CPU implementation.

    ``sdpa`` rather than the ``varlen`` default, which is CUDA-only; the
    accounting under test is identical either way. ``swiglu`` is now the default
    too, so only the attention backend is overridden here.
    """
    base = dict(
        vocab_size=VOCAB,
        dim=DIM,
        depth=DEPTH,
        heads=4,
        kv_heads=2,
        head_dim=8,
        mlp="swiglu",
        mlp_hidden=64,
        attn="sdpa",
    )
    return LMArchConfig(**{**base, **overrides})


def _moe_config(**overrides):
    sparse = dict(
        moe_every=1,
        moe_first_dense=1,
        moe_num_experts=8,
        moe_top_k=2,
        moe_num_shared=1,
        moe_hidden=16,
    )
    return _cpu_config(**{**sparse, **overrides})


def _counter(config, **kwargs) -> FlopCounter:
    return FlopCounter(LMBackbone(config), **kwargs)


# --------------------------------------------------------------------- flops


def test_flops_do_not_charge_the_embedding_gather():
    """Tying the embedding changes parameters, not arithmetic.

    ``6 * active_parameters`` fails this: untying adds a ``vocab x dim`` matrix
    that no token multiplies by, and the shortcut charges a full GEMM for it.
    """
    tied = _counter(_cpu_config(tie_embeddings=True))
    untied = _counter(_cpu_config(tie_embeddings=False))
    assert tied.per_token(CTX) == pytest.approx(untied.per_token(CTX))
    assert tied.head_per_token == pytest.approx(2.0 * DIM * VOCAB)


def test_moe_charges_routed_experts_not_the_expert_count():
    """Only ``top_k + num_shared`` experts run per token; the rest are memory."""
    config = _moe_config()
    n_moe = sum(config.moe_layers)
    hidden = 16

    base = _counter(_moe_config()).per_token(CTX)
    more_experts = _counter(_moe_config(moe_num_experts=32)).per_token(CTX)
    more_routed = _counter(_moe_config(moe_top_k=4)).per_token(CTX)

    # 4x the experts costs only the router's extra rows, a ``dim x num_experts``
    # GEMM -- not 4x the feed-forward.
    router_delta = 3.0 * 2.0 * (32 - 8) * DIM * n_moe
    assert more_experts - base == pytest.approx(router_delta)
    assert more_experts < 1.2 * base

    # 2 more routed experts per token *is* feed-forward work: w_in is 2h x d and
    # w_out is d x h, so 3hd parameters per expert.
    assert more_routed - base == pytest.approx(3.0 * 2.0 * 3 * DIM * hidden * 2 * n_moe)


def test_attention_flops_follow_the_causal_pair_count():
    counter = _counter(_cpu_config())
    lengths = torch.tensor([4.0, 8.0], dtype=torch.float64)
    pairs = 4 * 5 / 2 + 8 * 9 / 2
    expected = 4.0 * counter.q_dim * DEPTH * pairs
    assert float(counter.attention_flops(lengths)) == pytest.approx(expected)


def test_sliding_window_truncates_the_pair_count():
    """A window charges ``Lw - w(w-1)/2``, not ``Lw``: the ramp is half a window."""
    assert float(attended_pairs(torch.tensor([32.0]), 4)) == pytest.approx(32 * 4 - 6)
    # Shorter than the window: still the full causal triangle.
    assert float(attended_pairs(torch.tensor([3.0]), 8)) == pytest.approx(6.0)

    lengths = torch.tensor([32.0], dtype=torch.float64)
    full = _counter(_cpu_config())
    windowed = _counter(_cpu_config(sliding_window=4))
    assert float(windowed.attention_flops(lengths)) == pytest.approx(
        4.0 * full.q_dim * DEPTH * (32 * 4 - 6)
    )
    assert float(windowed.attention_flops(lengths)) < float(
        full.attention_flops(lengths)
    )

    # Interleaving costs the mix, layer by layer: the last layer feeds the head
    # and is always global, so a 2-layer 1:1 stack is one of each.
    mixed = _counter(_cpu_config(sliding_window=4, global_layer_every=2))
    assert float(mixed.attention_flops(lengths)) == pytest.approx(
        4.0 * full.q_dim * ((32 * 4 - 6) + 32 * 33 / 2)
    )


def test_document_lengths_read_the_layout():
    packed = SeqInfo.from_lengths(torch.tensor([3, 5], dtype=torch.int32))
    assert document_lengths(packed).tolist() == [3.0, 5.0]
    # Padding is charged: a causal SDPA over a padded row computes the whole
    # triangle whether or not the tokens are real.
    assert document_lengths(SeqInfo.padded(2, 16)).tolist() == [16.0, 16.0]


def test_grad_checkpointing_charges_hardware_not_model_flops():
    lengths = torch.tensor([16.0], dtype=torch.float64)
    plain = _counter(_cpu_config()).batch_flops(16, lengths)
    ckpt = _counter(_cpu_config(), grad_ckpt=True).batch_flops(16, lengths)

    assert float(ckpt[0]) == pytest.approx(float(plain[0]))
    assert float(plain[1]) == pytest.approx(float(plain[0]))
    # The extra is one forward through the blocks -- the head is not recomputed.
    counter = _counter(_cpu_config())
    blocks = counter.block_per_token * 16 + float(counter.attention_flops(lengths))
    assert float(ckpt[1] - ckpt[0]) == pytest.approx(blocks)


# ---------------------------------------------------------------- snapshots


def test_snapshot_interval_arithmetic_and_rates():
    early = TokenSnapshot(1000, 600, 1e12, 1.5e12, 1.0)
    late = TokenSnapshot(3000, 1400, 3e12, 4.5e12, 3.0)
    interval = late - early

    assert (interval.seen, interval.trained, interval.elapsed) == (2000, 800, 2.0)
    assert interval.tokens_per_sec == pytest.approx(1000.0)
    assert interval.trained_tokens_per_sec == pytest.approx(400.0)
    assert interval.trained_frac == pytest.approx(0.4)
    assert interval.b_tokens_per_day == pytest.approx(1000 * 86400 / 1e9)
    assert interval.b_trained_tokens_per_day == pytest.approx(400 * 86400 / 1e9)
    assert interval.mfu(2e12) == pytest.approx(0.5)
    assert interval.hfu(2e12) == pytest.approx(0.75)


def test_mfu_above_one_is_reported_not_clamped():
    """270 TFLOP/s is the fp32-accumulate ceiling, not an absolute one.

    fp16 accumulation measures 304.8 TFLOP/s on this card (318.5 at split-K 2),
    so a path that accumulates in fp16 can legitimately exceed 1.0. Clamping
    would have hidden the peak-rate bug that reported MFU above 100% for a plain
    cuBLAS GEMM.
    """
    assert TokenSnapshot(0, 0, 4e12, 4e12, 1.0).mfu(2e12) == pytest.approx(2.0)


def test_token_counters_stay_exact_past_the_float32_range():
    """A float32 counter stops incrementing at 2^24; the odd token is the tell."""
    model = LMTrainer(**_trainer_kwargs())
    model.token_counts += torch.tensor([2**33 + 1, 2**33 - 1], dtype=torch.int64)
    total = model.token_snapshot()
    assert total.seen == 2**33 + 1
    assert total.trained == 2**33 - 1


# ------------------------------------------------------------------ previews


def test_preview_sampling_leaves_the_default_rng_stream_alone():
    """The data order draws from the default stream; a preview must not touch it."""
    model = LMTrainer(**_trainer_kwargs())
    prompt = torch.zeros(1, 3, dtype=torch.long)

    torch.manual_seed(1234)
    reference = torch.randn(4)

    torch.manual_seed(1234)
    model.generate(prompt, max_new_tokens=4, temperature=1.0, top_p=0.95)
    after = torch.randn(4)

    assert torch.equal(reference, after)


def test_preview_sampling_is_reproducible_across_runs():
    prompt = torch.zeros(1, 3, dtype=torch.long)
    kwargs = _trainer_kwargs()

    pl.seed_everything(0)
    first = LMTrainer(**kwargs).generate(prompt, max_new_tokens=6, temperature=1.0)
    pl.seed_everything(0)
    second = LMTrainer(**kwargs).generate(prompt, max_new_tokens=6, temperature=1.0)

    assert torch.equal(first, second)


def test_preview_generation_runs_under_the_precision_plugin_autocast():
    """A preview fires from ``on_train_batch_end``, outside Lightning's autocast.

    Unwrapped, the backbone runs in fp32 and the MXFP8 expert path refuses the
    activation outright -- so the run dies at the *first preview*, however many
    thousand steps in that is, having trained perfectly up to then. The smoke run
    found it at step 20 with `x.dtype=torch.float32 is not one of (bfloat16,
    float16)`.

    Hooking a ``Linear`` rather than the backbone output because a linear is
    autocast-eligible and a norm is not: the backbone returns fp32 either way, so
    asserting on it would pass with the fix reverted.
    """
    prompt = torch.zeros(1, 3, dtype=torch.long)

    def linear_dtypes(model):
        seen = []
        linear = next(
            m for m in model.backbone.modules() if isinstance(m, torch.nn.Linear)
        )
        handle = linear.register_forward_hook(
            lambda _m, _i, out: seen.append(out.dtype)
        )
        try:
            model.generate(prompt, max_new_tokens=2)
        finally:
            handle.remove()
        return seen

    attached = LMTrainer(**_trainer_kwargs())
    attached._trainer = pl.Trainer(
        accelerator="cpu", precision="bf16-mixed", logger=False
    )
    under_autocast = linear_dtypes(attached)

    # The negative twin: unattached, the same hook must report fp32. Without it the
    # assertion above would also pass on a machine where everything is bf16 anyway.
    detached = LMTrainer(**_trainer_kwargs())
    assert detached._trainer is None
    without_autocast = linear_dtypes(detached)

    assert under_autocast and all(d is torch.bfloat16 for d in under_autocast)
    assert without_autocast and all(d is torch.float32 for d in without_autocast)


def test_generate_restores_train_mode_even_when_it_raises():
    model = LMTrainer(**_trainer_kwargs())
    model.backbone.train()

    def boom(_hidden):
        raise RuntimeError("sampling blew up")

    model.backbone.head.logits = boom
    with pytest.raises(RuntimeError):
        model.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=1)
    # Left in eval, gradient checkpointing would silently stop for the whole run.
    assert model.backbone.training


def test_stage_emits_the_declared_boundary_dtype_whatever_its_blocks_return():
    """``PipelineStage`` froze this dtype from ``input_args``; the stage must match it.

    An MoE block's expert combine reduces in fp32, so a sparse stage returns fp32 even
    under bf16 parameters while a dense one returns bf16 -- inferring the boundary from
    the parameter dtype is right for one and backwards for the other, which is why the
    stage is told and not asked. The last stage is exempt: its output feeds the head,
    not a rank boundary, and casting it there would change the loss.
    """
    config = _cpu_config(tie_embeddings=False)
    plans = plan_for(config, 2)
    backbone = LMBackbone(config)
    tokens = torch.zeros(CTX, dtype=torch.long)

    undeclared = LMStage(backbone, plans[0])
    assert undeclared(tokens).dtype is torch.float32, "premise: no cast without a decl"

    declared = LMStage(backbone, plans[0], boundary_dtype=torch.bfloat16)
    assert declared(tokens).dtype is torch.bfloat16

    last = LMStage(backbone, plans[-1], boundary_dtype=torch.bfloat16)
    assert last.boundary_dtype is None
    assert last(torch.zeros(CTX, DIM)).dtype is torch.float32


def _router_loss_stages(num_stages=2, **overrides):
    """A sparse backbone with a router aux loss, split into stages."""
    base = dict(
        depth=4,
        moe_first_dense=0,
        tie_embeddings=False,
        moe_router_kwargs={"aux_loss_weight": 0.5},
        # The grouped-GEMM expert path is Triton-only; this file runs on the CPU.
        moe_mlp_kwargs={"dense_fallback": True},
    )
    config = _moe_config(**{**base, **overrides})
    backbone = LMBackbone(config, head_kwargs={"kernel": "torch"})
    plans = plan_for(config, num_stages)
    return config, backbone, [LMStage(backbone, plan) for plan in plans]


def test_router_loss_rides_a_second_boundary_stream_to_the_head_stage():
    """A stage's MoE auxiliary term must reach the only stage that owns a loss.

    Nothing else can carry it: the term is produced on every stage and applied on
    one. The previous behaviour was to refuse the config outright, and the failure
    mode this replaces is worse than refusing -- a term silently dropped balances
    nothing while the config says it does.

    The last block is the negative case. With no auxiliary weight the boundary has
    to stay a bare tensor, because a tuple would change the shape every
    ``PipelineStage`` on the run has already frozen.
    """
    _config, _backbone, stages = _router_loss_stages()
    tokens = torch.zeros(CTX, dtype=torch.long)
    assert all(stage.router_stream for stage in stages)
    assert all(stage.moe_blocks for stage in stages), "both stages must hold a term"

    hidden, acc = stages[0](tokens)
    assert acc.shape == (1,) and acc.dtype is torch.float32
    first = float(stages[0].router_losses().detach())
    assert float(acc.detach()) == pytest.approx(first, rel=1e-6)

    _out, total = stages[1](hidden, acc)
    second = float(stages[1].router_losses().detach())
    assert float(total.detach()) == pytest.approx(first + second, rel=1e-6)
    assert second > 0.0

    # The accumulator reaches the loss with gradient 1 per element, so the term a
    # stage contributed is worth exactly itself.
    reduce_accumulator(total).backward()
    early = stages[0].blocks[0].mlp.router.weight
    assert early.grad is not None and early.grad.abs().sum() > 0.0

    # Eval has no loss to add it to, and decode declares a bare boundary.
    for stage in stages:
        stage.eval()
    assert isinstance(stages[0](tokens), torch.Tensor)

    _c, _b, plain = _router_loss_stages(moe_router_kwargs={})
    assert not any(stage.router_stream for stage in plain)
    assert isinstance(plain[0](tokens), torch.Tensor)


def test_the_auxiliary_term_does_not_scale_with_the_micro_batch_count():
    """An auxiliary term is a mean per micro-batch, so a step averages them.

    Both loops accumulate over micro-batches, and both had it wrong in opposite
    directions. The pipeline summed 32 means, which trains ``aux_loss_weight=1e-3``
    as 3.2e-2; the Lightning loop folded the term into a sum-reduced CE and then
    divided the total by the step's token count, which trains it as 1.5e-8. Both
    are invisible -- the run trains, at a coefficient nobody chose.
    """
    stub = types.SimpleNamespace(loss=lambda hidden, target: (hidden.sum(), {}))
    hidden, acc = torch.zeros(2, 2), torch.full((1,), 5.0)
    assert float(build_loss_fn(stub, lambda: 1)((hidden, acc), None)) == 5.0
    assert float(
        build_loss_fn(stub, lambda: 1, num_microbatches=8)((hidden, acc), None)
    ) == pytest.approx(5.0 / 8)

    _config, backbone, _stages = _router_loss_stages()
    tokens = torch.zeros(CTX, dtype=torch.long)
    plain, logs = backbone.loss(tokens, tokens, reduction="sum")
    scaled, _ = backbone.loss(tokens, tokens, reduction="sum", router_scale=7.0)
    term = float(logs["router_loss"])
    assert term > 0.0
    assert float(scaled - plain) == pytest.approx(6.0 * term, rel=1e-4)


def test_router_loss_survives_gradient_checkpointing():
    """The term is built inside the checkpointed region and used outside it.

    ``use_reentrant=True`` would run that region under ``no_grad`` and hand back a
    term with no ``grad_fn`` -- a loss that is logged, added, and trains nothing.
    Asserted against the same model without checkpointing, because a zero gradient
    and a merely-different one are not the same failure.
    """
    tokens = torch.zeros(CTX, dtype=torch.long)
    grads = {}
    for ckpt in (False, True):
        torch.manual_seed(0)
        _config, _backbone, stages = _router_loss_stages(grad_ckpt=ckpt)
        for stage in stages:
            stage.train()
        assert stages[0].grad_ckpt is ckpt
        hidden, acc = stages[0](tokens)
        _out, total = stages[1](hidden, acc)
        reduce_accumulator(total).backward()
        grads[ckpt] = stages[0].blocks[0].mlp.router.weight.grad.clone()

    assert grads[True].abs().sum() > 0.0
    assert torch.allclose(grads[True], grads[False], atol=1e-5)

    # The unsplit backbone collects the same terms the same way, so it carries
    # the same hazard and gets the same check.
    torch.manual_seed(0)
    config, backbone, _stages = _router_loss_stages(grad_ckpt=True)
    backbone.train()
    labels = torch.zeros(CTX, dtype=torch.long)
    loss, logs = backbone.loss(tokens, labels)
    assert "router_loss" in logs and float(logs["router_loss"]) > 0.0
    loss.backward()
    assert backbone.blocks[0].mlp.router.weight.grad.abs().sum() > 0.0


def test_pinned_split_beats_the_cost_model_and_rejects_a_bad_one():
    """``LAYERS`` is how a measured split reaches production, so it must validate.

    The cost model puts one layer on the head stage at this rung; the measured
    optimum is three. A pinned split that silently disagreed with the depth
    would drop or duplicate layers, so every mismatch raises -- including the
    string a ``--set LAYERS=[5,4,4,3]`` hands through, which is not a sequence
    of ints however much it looks like one.
    """
    config = _cpu_config(depth=16, tie_embeddings=False)
    auto = [p.num_layers for p in plan_for(config, 4, seq_len=16384)]
    pinned = plan_for(config, 4, seq_len=16384, layers=[5, 4, 4, 3])
    assert [p.num_layers for p in pinned] == [5, 4, 4, 3]
    assert [p.start_layer for p in pinned] == [0, 5, 9, 13]
    assert pinned[0].has_embed and pinned[-1].has_head
    assert sum(auto) == sum(p.num_layers for p in pinned) == 16

    with pytest.raises(TypeError, match="sequence of ints"):
        plan_for(config, 4, seq_len=16384, layers="[5,4,4,3]")
    with pytest.raises(ValueError, match="not depth"):
        plan_for(config, 4, seq_len=16384, layers=[5, 4, 4, 2])
    with pytest.raises(ValueError, match="not 4 stages"):
        plan_for(config, 4, seq_len=16384, layers=[8, 8])


def test_the_split_follows_a_per_layer_cost_vector_not_an_average():
    """Layers stop being interchangeable the moment some of them are sparse.

    ``moe_first_dense`` makes layer 0 cheaper than the rest, and a scalar cost
    cannot express that -- it puts an equal *count* on each stage and calls the
    result balanced. Fed the real per-layer milliseconds this rung measures, the
    partitioner has to reach the split that was verified end to end.
    """
    config = _moe_config(depth=16, moe_first_dense=1, tie_embeddings=False)
    dense, moe, head = 5.16, 6.69, 12.84
    costs = LayerCosts(
        layers=[moe if m else dense for m in config.moe_layers],
        head=head,
        embed=0.4,
        params=[1.0] * config.depth,
        head_params=1.0,
        embed_params=1.0,
    )
    assert [p.num_layers for p in plan_for(config, 4, 8192, costs=costs)] == [
        5,
        4,
        4,
        3,
    ]

    # The same head against cheaper layers moves the answer -- which is the whole
    # point, and why a constant cannot be right at two micro-batch sizes.
    faster = replace_costs(costs, head=25.7, layer=7.87, dense=5.78)
    assert [p.num_layers for p in plan_for(config, 4, 16384, costs=faster)] == [
        5,
        5,
        5,
        1,
    ]

    # A vector and a scalar must agree when every layer really is the same.
    flat = [3.0] * config.depth
    assert partition(16, 4, flat, 3.0) == partition(16, 4, 3.0, 3.0)
    with pytest.raises(ValueError, match="expected 16 per-layer values"):
        partition(16, 4, [1.0, 2.0], 3.0)


def replace_costs(costs, head, layer, dense):
    return LayerCosts(
        layers=[layer if c != costs.layers[0] else dense for c in costs.layers],
        head=head,
        embed=costs.embed,
        params=costs.params,
        head_params=costs.head_params,
        embed_params=costs.embed_params,
    )


def test_the_probe_block_is_the_block_the_model_builds():
    """The autotuner times a block it constructs itself, so it must be the same one.

    Anything re-derived here -- the GLU width correction, the ``multiple_of``
    ceiling, whether this index is sparse -- is a second definition that drifts
    from the model silently, which is how the split ends up disagreeing with the
    thing it is splitting.
    """
    config = _moe_config(depth=4, moe_first_dense=1, tie_embeddings=False)
    model = LMBackbone(config, head_kwargs={"kernel": "torch"})
    for index in range(config.depth):
        probe = LMBackbone.build_block(config, index)
        built = model.blocks[index]
        assert type(probe.mlp) is type(built.mlp)
        assert probe.mlp.hidden == built.mlp.hidden
        assert [p.shape for p in probe.parameters()] == [
            p.shape for p in built.parameters()
        ]
    # Layer 0 is dense and the rest are not, so a single key would be wrong.
    assert len({layer_key(config, i) for i in range(config.depth)}) == 2


def test_corpus_steps_gives_the_loop_the_shape_its_protocol_wants():
    """The loader and the loop disagree on two names, and one of them is a type.

    ``MicroBatchedStep.trained`` is per-microbatch, and the loop divides the loss
    by ``int(batch.trained)`` -- which on a tuple raises, and is the whole reason
    this adapter exists rather than the loader being handed over directly.
    """
    k, micro = 8, 3
    chunks = [
        collate_packed(
            [{"input_ids": [1, 2, 3, 4], "labels": [IGNORE_INDEX, 2, 3, 4]}] * 2,
            pad_to_multiple=k,
        )
        for _ in range(micro)
    ]
    step = MicroBatchedStep.from_chunks(chunks, k)
    assert isinstance(step.trained, tuple)

    stream = corpus_steps([step], torch.device("cpu"))
    out = next(stream)
    assert isinstance(out, PipelineStep)
    assert out.inputs.shape == (k * micro,) and out.target.shape == (k * micro,)
    assert len(out.layout) == micro
    assert isinstance(out.trained, int) and out.trained == sum(step.trained)

    # A one-pass loader must not end the fit: re-iterating reshuffles, and
    # `max_steps` is the only thing that stops a run.
    assert sum(1 for _, _ in zip(range(5), stream)) == 5
    with pytest.raises(RuntimeError, match="no steps on a full pass"):
        next(corpus_steps([], torch.device("cpu")))


def test_a_callback_added_after_the_trainer_still_reaches_the_loop():
    """The loop is what calls callbacks, so the trainer must not hold a copy.

    ``CallbackList`` builds a ``list(callbacks)``, so a trainer that made its own
    would leave ``trainer.callbacks.append(...)`` a silent no-op -- a preview
    configured on and never firing, which is exactly what it did.
    """
    loop = types.SimpleNamespace(callbacks=CallbackList([]))
    trainer = types.SimpleNamespace(loop=loop)
    trainer.callbacks = trainer.loop.callbacks

    marker = PipeCallback()
    trainer.callbacks.append(marker)
    assert marker in loop.callbacks.callbacks


def test_throughput_window_is_trailing_so_one_slow_step_does_not_persist():
    """A checkpoint costs tens of seconds; a running average never forgets it.

    The window is what makes a number after a checkpoint comparable to one
    before it, and the negative case is a cumulative average, which reports the
    stall forever and reads as a regression that never happened.
    """
    rows = []
    # reset@0, report@1 over 10 s, reset, report@2 over 1 s, reset.
    clock = iter([0.0, 10.0, 10.0, 11.0, 11.0])
    callback = Throughput(every_n_steps=1, warmup_steps=0, report=rows.append)
    # The run clock is separate from the window clock, and must not consume it.
    loop = types.SimpleNamespace(
        rank=0, tokens_seen=0, tokens_trained=0, elapsed=lambda: 20.0
    )

    with mock.patch.object(time, "perf_counter", lambda: next(clock)):
        for index in range(3):
            loop.tokens_seen += 1000
            loop.tokens_trained += 800
            callback.on_train_batch_end(loop, _throughput_step(index))

    # 1000 tokens over 10 s, then 1000 over 1 s: the slow window must not bleed.
    assert rows[0]["tokens_per_s"] == pytest.approx(100.0)
    assert rows[1]["tokens_per_s"] == pytest.approx(1000.0)

    # Cumulative counts are totals, not per-window, so they must keep climbing
    # while the trailing rate above swings 10x.
    assert rows[0]["tokens_seen"] == 2000
    assert rows[1]["tokens_seen"] == 3000
    assert rows[1]["tokens_trained"] == 2400
    assert rows[1]["trained_frac"] == pytest.approx(0.8)
    assert rows[1]["tokens_per_s_avg"] == pytest.approx(3000 / 20.0)
    assert rows[1]["b_tokens_per_day"] == pytest.approx(1000.0 * 86400 / 1e9)


def _throughput_step(index: int):
    return StepOutput(index=index, loss=None, seen=1000, trained=1000)


def test_the_rank_data_guard_catches_a_stream_that_drifted():
    """Every rank builds its own loader, and a divergence has no symptom.

    Stage 0's tokens would pair with stage 3's labels from a different batch:
    the loss is wrong and completely plausible, and no assertion downstream
    could tell. What must be caught is a same-shaped, same-token-count batch
    that merely holds different tokens -- so the signature carries content, not
    just geometry.
    """
    base = _signature_step(tokens=[1, 2, 3, 4], labels=[1, 2, 3, 4])
    same = _signature_step(tokens=[1, 2, 3, 4], labels=[1, 2, 3, 4])
    shuffled = _signature_step(tokens=[4, 3, 2, 1], labels=[1, 2, 3, 4])
    other = _signature_step(tokens=[1, 2, 3, 5], labels=[1, 2, 3, 4])

    signatures = [batch_signature(s) for s in (base, same)]
    assert first_mismatch(signatures) is None
    assert first_mismatch([batch_signature(base), batch_signature(other)]) == 1
    # A permutation sums the same, so the guard is a cheap check, not a proof.
    assert first_mismatch([batch_signature(base), batch_signature(shuffled)]) is None
    # No process group: nothing to compare, and it must not raise.
    verify_same_batch(base)


def _signature_step(tokens, labels):
    return PipelineStep(
        inputs=torch.tensor(tokens),
        target=torch.tensor(labels),
        layout=[],
        trained=len(labels),
    )


def test_preview_skips_a_pipeline_split_model():
    """No rank can generate when the model is split; say so once, don't emit garbage."""
    config = _cpu_config(tie_embeddings=False)
    plans = plan_for(config, 2)
    backbone = LMBackbone(config)
    module = types.SimpleNamespace(
        backbone=LMStage(backbone, plans[-1]),
        device=torch.device("cpu"),
        generate=_unexpected_generate,
    )
    trainer = types.SimpleNamespace(global_step=1, is_global_zero=True, logger=None)
    callback = SampleLogCallback(tokenizer=None, every_n_steps=1)

    with pytest.warns(UserWarning, match="pipeline stages"):
        callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback._disabled

    # Warned once, then silent -- and still never generating.
    trainer.global_step = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        callback.on_train_batch_end(trainer, module, None, None, 0)


def test_preview_runs_on_every_rank_when_generate_is_collective():
    """A collective ``generate`` must be entered by every rank, rank 0 or not.

    The rank-0-only guard that a preview normally uses is exactly what deadlocks a
    pipeline: the other ranks never reach the schedule the printing rank is inside.
    """
    config = _cpu_config(tie_embeddings=False)
    plans = plan_for(config, 2)
    backbone = LMBackbone(config)
    calls = []

    def generate(ids, **_kwargs):
        calls.append(ids)
        return torch.zeros(1, ids.shape[1] + 2, dtype=torch.long)

    module = types.SimpleNamespace(
        backbone=LMStage(backbone, plans[-1]),
        device=torch.device("cpu"),
        generate=generate,
        generate_is_collective=True,
    )
    tokenizer = _EchoTokenizer()
    callback = SampleLogCallback(tokenizer=tokenizer, prompts=["a"], every_n_steps=1)

    # A non-zero rank still generates; it just does not decode or print.
    follower = types.SimpleNamespace(global_step=1, is_global_zero=False, logger=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        callback.on_train_batch_end(follower, module, None, None, 0)
    assert not callback._disabled
    assert len(calls) == 1, "a collective generate was skipped on a non-zero rank"

    leader = types.SimpleNamespace(global_step=2, is_global_zero=True, logger=None)
    callback.on_train_batch_end(leader, module, None, None, 0)
    assert len(calls) == 2


class _EchoTokenizer:
    """The two calls SampleLogCallback makes of a tokenizer, and nothing else."""

    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        return "x" * len(ids)


def test_stage_state_dicts_reassemble_the_whole_backbone():
    """Every stage's slice, under global names, must rebuild the model exactly.

    A stage numbers its blocks from zero, so a checkpoint written without the
    ``plan.start_layer`` offset loads the wrong layers into the right slots --
    which raises nothing, and is why this is checked by name.
    """
    config = _cpu_config(tie_embeddings=False)
    backbone = LMBackbone(config)
    plans = plan_for(config, 2)
    stages = [LMStage(backbone, plan) for plan in plans]

    merged = {}
    for stage in stages:
        merged.update(stage.global_state_dict())
    whole = backbone.state_dict()
    assert set(merged) == set(whole)
    for name, tensor in whole.items():
        assert torch.equal(merged[name], tensor), name

    # The later stage must own later blocks, or the offset is not being applied.
    blocks_of = lambda s: {  # noqa: E731
        int(n.split("blocks.")[1].split(".")[0])
        for n in s.global_state_dict()
        if "blocks." in n
    }
    assert max(blocks_of(stages[0])) < min(blocks_of(stages[1]))

    # And the round trip puts them back where they came from.
    fresh = LMBackbone(config)
    reloaded = [LMStage(fresh, plan) for plan in plans]
    for stage in reloaded:
        stage.load_global_state_dict(merged)
    for name, tensor in whole.items():
        assert torch.equal(fresh.state_dict()[name], tensor), name


def test_stage_flops_split_the_whole_model_across_stages():
    """Per-stage FLOP shares must sum to one whole model, head counted once."""
    config = _cpu_config(tie_embeddings=False)
    backbone = LMBackbone(config)
    counter = FlopCounter(backbone)
    plans = plan_for(config, 2)
    lengths = torch.tensor([float(CTX)], dtype=torch.float64)

    whole = counter.batch_flops(CTX, lengths)
    parts = sum(
        counter.stage_flops(CTX, lengths, plan.num_layers / config.depth, plan.has_head)
        for plan in plans
    )
    assert torch.allclose(parts, whole)

    # Exactly one stage carries the head, and it is not free.
    assert sum(plan.has_head for plan in plans) == 1
    headless = counter.stage_flops(CTX, lengths, 1.0, has_head=False)
    assert float(headless[0]) < float(whole[0])


def test_throughput_callback_needs_a_countable_module():
    callback = ThroughputCallback(every_n_steps=1)
    trainer = types.SimpleNamespace(global_step=1, is_global_zero=True)
    module = types.SimpleNamespace()
    with pytest.warns(UserWarning, match="token_snapshot"):
        callback.on_train_batch_end(trainer, module, None, None, 0)
    assert callback._disabled


def _unexpected_generate(*_args, **_kwargs):
    raise AssertionError("generate() ran on a pipeline-split model")


# -------------------------------------------------------------------- resume


def test_rng_state_round_trips():
    before = rng_state()
    drawn = torch.randn(3)
    load_rng_state(before)
    assert torch.equal(torch.randn(3), drawn)


class _CountingLoader:
    """Deterministic packed batches with a resumable cursor.

    Batch content is a pure function of the batch index, so ``served`` -- the
    indices in the order they were handed out -- is a sufficient check for "no
    batch was repeated or skipped". Production order equals consumption order
    because the loader reports a length, and Lightning's fetcher skips
    prefetching when it can compare against one.
    """

    def __init__(
        self,
        num_batches: int = 24,
        docs: int = 2,
        doc_len: int = 8,
        blank: tuple[int, ...] = (),
    ) -> None:
        self.num_batches = num_batches
        self.docs = docs
        self.doc_len = doc_len
        # Batch indices whose every label is masked, i.e. steps that teach nothing.
        self.blank = set(blank)
        self.cursor = 0
        self.served: list[int] = []

    def _batch(self, index: int):
        generator = torch.Generator().manual_seed(1000 + index)
        samples = []
        for _ in range(self.docs):
            ids = torch.randint(1, VOCAB, (self.doc_len,), generator=generator).tolist()
            labels = list(ids)
            # Two masked positions per document, so `seen` and `trained` differ.
            labels[0] = labels[1] = IGNORE_INDEX
            if index in self.blank:
                labels = [IGNORE_INDEX] * len(ids)
            samples.append({"input_ids": ids, "labels": labels})
        return collate_packed(samples)

    def __iter__(self):
        while self.cursor < self.num_batches:
            index = self.cursor
            self.cursor += 1
            self.served.append(index)
            yield self._batch(index)

    def __len__(self) -> int:
        return self.num_batches

    def state_dict(self) -> dict:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = int(state["cursor"])


class _StepRecorder(Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.steps: list[int] = []
        self.metrics: dict[str, float] = {}

    def on_train_batch_end(self, trainer, pl_module, *_args):
        self.losses.append(float(trainer.callback_metrics["train/loss"]))
        self.steps.append(trainer.global_step)
        self.metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}


def _trainer_kwargs(**overrides):
    base = dict(
        preset=None,
        arch_overrides=dict(
            dim=DIM,
            depth=DEPTH,
            heads=4,
            kv_heads=2,
            head_dim=8,
            mlp="swiglu",
            mlp_hidden=64,
            attn="sdpa",
        ),
        vocab_size=VOCAB,
        head_kwargs={"kernel": "torch"},
        learning_rate=1e-2,
        scheduler_config={"lr": {"mode": "cosine", "min_value": 0.1, "end": 16}},
        log_interval=1,
        name="test",
    )
    return {**base, **overrides}


def _fit(
    steps,
    loader,
    ckpt_dir=None,
    ckpt_every=0,
    resume=None,
    extra_callbacks=(),
    **model_kwargs,
):
    pl.seed_everything(0, workers=True)
    model = LMTrainer(**_trainer_kwargs(**model_kwargs))
    recorder = _StepRecorder()
    # The recorder runs last, so it sees whatever the other callbacks logged.
    callbacks: list[Callback] = [*extra_callbacks, recorder]
    if ckpt_every:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                every_n_train_steps=ckpt_every,
                save_top_k=-1,
                filename="step{step}",
            )
        )
    trainer = pl.Trainer(
        max_steps=steps,
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=bool(ckpt_every),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, loader, ckpt_path=resume)
    return model, recorder


def test_resume_continues_where_the_run_stopped(tmp_path):
    """A restart must reproduce the uninterrupted run, not restart it.

    Weights and optimizer state come from Lightning; what this pins is
    everything else -- the data position (no batch repeated, none skipped), the
    token totals, and therefore the loss trajectory step for step.
    """
    total, crash = 8, 4

    ref_inner = _CountingLoader()
    reference, ref_log = _fit(total, ref_inner)

    crash_inner = _CountingLoader()
    _, crash_log = _fit(crash, crash_inner, ckpt_dir=tmp_path, ckpt_every=crash)
    checkpoints = sorted(tmp_path.glob("*.ckpt"))
    assert len(checkpoints) == 1

    resumed_inner = _CountingLoader()
    resumed, resumed_log = _fit(total, resumed_inner, resume=str(checkpoints[0]))

    # The continuation trains the steps the crashed run had not reached.
    assert crash_log.steps == list(range(1, crash + 1))
    assert resumed_log.steps == list(range(crash + 1, total + 1))
    assert resumed_log.losses == pytest.approx(ref_log.losses[crash:], abs=1e-6)
    # ... and is not merely a fresh run that happens to converge the same way.
    assert abs(resumed_log.losses[0] - ref_log.losses[0]) > 1e-4

    # No batch served twice, and none skipped over the seam.
    assert crash_inner.served == list(range(crash))
    assert resumed_inner.served == list(range(crash, total))
    assert ref_inner.served == list(range(total))

    ref_tokens = reference.token_snapshot()
    resumed_tokens = resumed.token_snapshot()
    assert resumed_tokens.seen == ref_tokens.seen
    assert resumed_tokens.trained == ref_tokens.trained
    assert 0 < resumed_tokens.trained < resumed_tokens.seen
    assert resumed_tokens.model_flops == pytest.approx(ref_tokens.model_flops)

    # The counter buffer is a per-rank delta since the last start, which is why
    # the totals travel separately: restoring rank 0's delta onto every rank
    # would all-reduce to world_size copies of rank 0's share.
    resumed_state = resumed.state_dict()
    assert resumed_state.pop("token_counts").tolist() == [
        ref_tokens.seen // 2,
        ref_tokens.trained // 2,
    ]
    for name, expected in reference.state_dict().items():
        if name == "token_counts":
            continue
        assert torch.equal(resumed_state[name], expected), name


def test_resume_restores_the_data_position_with_an_autofilled_schedule(tmp_path):
    """``end: -1`` is the case where Lightning drops the position it saved.

    Resolving it reads ``estimated_stepping_batches``, which builds the combined
    loader from inside ``configure_optimizers`` -- before the loop state is
    restored -- so ``setup_data``'s own restore never runs. The trainer's copy
    has to cover it.
    """
    total, crash = 6, 3
    schedule = {"lr": {"mode": "cosine", "min_value": 0.1, "end": -1}}

    first = _CountingLoader()
    _fit(crash, first, ckpt_dir=tmp_path, ckpt_every=crash, scheduler_config=schedule)
    checkpoint = sorted(tmp_path.glob("*.ckpt"))[0]

    resumed = _CountingLoader()
    _fit(total, resumed, resume=str(checkpoint), scheduler_config=schedule)
    assert resumed.served == list(range(crash, total))


def test_resume_without_state_restoration_still_loads_the_model(tmp_path):
    """``resume_state=False`` keeps the old behaviour: weights and step, no more."""
    total, crash = 6, 3
    inner = _CountingLoader()
    _fit(crash, inner, ckpt_dir=tmp_path, ckpt_every=crash)
    checkpoint = sorted(tmp_path.glob("*.ckpt"))[0]

    fresh = _CountingLoader()
    model, log = _fit(total, fresh, resume=str(checkpoint), resume_state=False)
    assert log.steps == list(range(crash + 1, total + 1))
    # The counters are still continuous -- they are a property of the model's
    # history, not of the opt-in state restoration.
    assert model.token_snapshot().seen > 0


def test_an_all_masked_step_trains_nothing_and_advances_nothing():
    """A step with no trained tokens must not step, log a loss, or count as one.

    Every branch here has to be unanimous across ranks: ``global_step`` counts
    ``opt.step()`` calls under manual optimization and the loss logs are
    ``sync_dist=True`` collectives, so a rank deciding "empty" on its own local
    count would leave the ranks permanently disagreeing about which steps log and
    checkpoint. Hence the reduced token count -- which at world size 1 is just the
    local one, so this test pins the arithmetic, not the reduction.
    """
    loader = _CountingLoader(num_batches=4, blank=(1,))
    model, log = _fit(4, loader)

    # Four batches consumed, three optimizer steps: the blank one did not count.
    assert loader.served == [0, 1, 2, 3]
    assert log.steps == [1, 1, 2, 3]
    # No 0.0 in the loss curve, and the EMA never saw one: the blank batch left
    # the metric at its previous value rather than logging a fresh zero.
    assert log.losses[1] == log.losses[0]
    assert min(log.losses) > 0.0
    assert model.ema_loss > 0.0

    # Its tokens were still computed on, so they are still *seen* -- only the
    # trained count skips them.
    total = model.token_snapshot()
    assert total.seen == 4 * loader.docs * loader.doc_len
    assert total.trained == 3 * loader.docs * (loader.doc_len - 2)


def test_resume_warns_when_the_loader_cannot_restore_its_position():
    class _Blind:
        def __iter__(self):
            return iter(())

    model = LMTrainer(**_trainer_kwargs())
    model._trainer = types.SimpleNamespace(
        train_dataloader=_Blind(), fit_loop=types.SimpleNamespace(_data_source=None)
    )
    assert any("state_dict" in problem for problem in model.resume_warnings())

    model._trainer.train_dataloader = _CountingLoader()
    assert model.resume_warnings() == []


# ------------------------------------------------------- logging in a real fit


class _StubTokenizer:
    """Just enough tokenizer for the preview path: encode, decode, no EOS."""

    eos_token_id = None

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


def test_logging_reports_the_run_without_changing_it():
    """Previews and throughput accounting are observation, not participation.

    The trajectory match is asserted *bitwise*: sampling draws from its own
    generator under ``no_grad``, so a run with previews on has to follow the
    identical path -- otherwise no two runs logged at different cadences are
    comparable.
    """
    plain_model, plain = _fit(4, _CountingLoader())

    previews = SampleLogCallback(
        _StubTokenizer(), prompts=["a", "b"], every_n_steps=1, max_new_tokens=3
    )
    throughput = ThroughputCallback(every_n_steps=1, peak_flops=270e12)
    model, logged = _fit(4, _CountingLoader(), extra_callbacks=[previews, throughput])

    assert logged.losses == pytest.approx(plain.losses, abs=0.0)
    assert model.backbone.training
    assert not previews._disabled

    total = plain_model.token_snapshot()
    assert plain.metrics["train/tokens_seen"] == pytest.approx(float(total.seen))
    assert plain.metrics["train/b_tokens_seen"] == pytest.approx(total.seen / 1e9)
    # Masked prompt positions mean the two counts must not be the same number.
    assert plain.metrics["train/tokens_trained"] < plain.metrics["train/tokens_seen"]
    assert plain.metrics["train/trained_frac"] == pytest.approx(total.trained_frac)
    assert plain.metrics["train/tokens_per_sec"] > 0.0
    assert plain.metrics["train/trained_tokens_per_sec"] > 0.0
    assert plain.metrics["train/b_tokens_per_day"] > 0.0

    # MFU only exists where the peak rate does, and equals HFU without recompute.
    assert "perf/mfu" not in plain.metrics
    assert logged.metrics["perf/mfu"] > 0.0
    assert logged.metrics["perf/hfu"] == pytest.approx(logged.metrics["perf/mfu"])
    assert logged.metrics["perf/tokens_per_sec"] > 0.0
    assert logged.metrics["perf/b_tokens_per_day"] > 0.0


# ------------------------------------------------------------------- mxfp8


def _mxfp8_trainer_kwargs(sparse=False, **overrides):
    """A trainer whose projections ``MXFP8Linear`` will accept.

    Wider than the CPU configs above because ``in_features`` is FPROP's contraction
    axis and must be a multiple of 128 -- the one axis the module cannot pad.
    ``sparse`` makes the one layer an MoE, whose routed experts are reached by
    declaration rather than by the ``nn.Linear`` swap and so hang off a different
    refresh hook.
    """
    arch = dict(
        dim=128,
        depth=1,
        heads=4,
        kv_heads=4,
        head_dim=32,
        mlp="swiglu",
        mlp_hidden=128,
        attn="sdpa",
    )
    if sparse:
        arch |= dict(moe_every=1, moe_num_experts=4, moe_top_k=2, moe_hidden=128)
    return _trainer_kwargs(arch_overrides=arch, **overrides)


class _CacheSnapshot(Callback):
    """Clone the fp8 caches and their masters at ``on_train_end``, still on device.

    The assertion cannot run after ``fit`` returns. Lightning's teardown moves
    parameters back to the CPU, and a quantized cache is dropped on ``_apply``
    precisely so a device-resident cache can never outlive the weights it was derived
    from -- so by then ``_cache`` is ``None`` and the invariant is unobservable.
    Cloning here reads the same values the last optimizer step produced.
    """

    def __init__(self) -> None:
        self.linear: list[tuple] = []
        self.experts: list[tuple] = []

    def on_train_end(self, trainer, pl_module) -> None:
        for module in pl_module.backbone.modules():
            if isinstance(module, MXFP8Linear):
                self.linear.append(
                    (module._cache[0].clone(), module.weight.detach().clone())
                )
        for block in pl_module.backbone.blocks:
            packed = getattr(block.mlp, "_packed", None)
            if packed is None:
                continue
            held = {
                copy: [t.clone() for t in getattr(packed, copy)]
                for copy in ("in_fwd", "out_fwd", "in_dgrad", "out_dgrad")
            }
            self.experts.append(
                (
                    held,
                    block.mlp.w_in.detach().clone(),
                    block.mlp.w_out.detach().clone(),
                )
            )


def _fit_mxfp8(loader, spy_calls, monkeypatch, steps=1, sparse=False):
    """Swap a trainer's projections to MXFP8 and run it on the GPU for ``steps``.

    ``precision="bf16-mixed"``, which every shipped config uses, and not the default
    32-true: the fused MoE expert path **refuses** an fp32 compute dtype rather than
    downcasting it, because ``h``'s storage dtype is what the backward differentiates.
    ``MXFP8Linear`` happens to tolerate fp32, so a dense-only test passes at 32-true and
    the sparse arm is the one that discovers autocast is not optional here.
    """
    pl.seed_everything(0, workers=True)
    model = LMTrainer(**_mxfp8_trainer_kwargs(sparse=sparse))
    report = swap_mxfp8(model.backbone)
    assert report.swapped, "nothing was swapped; this test would prove nothing"
    # `blocking`, not `skipped`: declared-but-unconverted matmul is a mixture with zero
    # skipped projections, which is how a MoE model used to reach a benchmark.
    assert not report.blocking, f"not a pure fp8 arm: {report.summary()}"

    real = trainer_module.refresh_mxfp8_weights

    def spy(module):
        count = real(module)
        spy_calls.append(count)
        return count

    monkeypatch.setattr(trainer_module, "refresh_mxfp8_weights", spy)
    snapshot = _CacheSnapshot()
    trainer = pl.Trainer(
        callbacks=[snapshot],
        max_steps=steps,
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(model, loader)
    return model, report, snapshot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 requires CUDA")
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "moe"])
def test_trainer_requantizes_the_fp8_cache_against_post_step_weights(
    monkeypatch, sparse
):
    """The fp8 copies must reflect the weights ``opt.step()`` produced, not the old ones.

    This is the one MXFP8 mistake with no symptom. ``MXFP8Linear`` builds its cache
    lazily on the **first forward** and never again, so a trainer that omits the
    refresh runs every fp8 GEMM on initialization-time weights for the whole run while
    the fp32 masters move underneath it. The loss is merely bad; nothing points at the
    cause, and ``torch.equal`` on the cache holds exactly from step to step.

    So the assertion is against a **freshly quantized copy of the post-step weight**,
    not against the previous cache. A test that only checked the call happened, or that
    the cache changed, would pass on a refresh placed before the step -- which is
    exactly where a reading of the bench harness puts it.

    The ``moe`` arm exists because the routed experts hang off a *different* hook: they
    are reached by declaration, and ``MoEMLP`` installs ``refresh_quantized_weight`` on
    the instance. Nothing in the dense arm would notice if that hook were missing, and
    the failure is the same silent one -- the whole expert bank frozen at its
    initialization values while the masters move.
    """
    calls: list[int] = []
    model, report, snapshot = _fit_mxfp8(
        _CountingLoader(num_batches=1), calls, monkeypatch, sparse=sparse
    )

    # Modules, not tensors: an MoE layer's two expert stacks share one cache, so on the
    # sparse arm `swapped` is the larger number and comparing against it would demand a
    # count the refresh can never return.
    assert calls == [len(report.modules)], (
        f"refresh fired {calls} for {len(report.modules)} cached modules; it must run "
        "exactly once per optimizer step and cover every module"
    )
    if sparse:
        assert len(report.modules) < len(report.swapped)
        assert snapshot.experts, "no expert caches captured; the arm proves nothing"
        for held_copies, w_in, w_out in snapshot.experts:
            fresh = MXFP8ExpertWeights(w_in, w_out)
            for copy, held_list in held_copies.items():
                for held, want in zip(held_list, getattr(fresh, copy)):
                    assert torch.equal(held, want.to(held.device)), (
                        f"{copy} is stale: the routed experts would run every GEMM on "
                        "pre-step weights while the master moved"
                    )

    # Read from the snapshot taken at `on_train_end`, not from the module: teardown has
    # since moved the parameters and dropped `_cache`, so there is nothing left to read.
    assert snapshot.linear, "no linear caches captured; the arm proves nothing"
    checked = 0
    for cache_q, weight in snapshot.linear:
        fresh_q, _ = quantize_mx_vendor(weight)
        assert torch.equal(cache_q, fresh_q), (
            "fp8 values are stale: the cache holds pre-step weights, so every GEMM "
            "would run on them while the master moved"
        )
        checked += 1
    # Every MXFP8Linear, which on the sparse arm is `swapped` minus the two declared
    # expert stacks per MoE layer -- those are checked above, through their own cache.
    assert checked == len(report.swapped) - 2 * len(model.backbone.moe_blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MXFP8 requires CUDA")
def test_trainer_skips_the_requantize_when_the_step_was_skipped(monkeypatch):
    """An all-masked step does not move the weights, so it must not pay to re-quantize.

    ``opt.step()`` is skipped when nothing was trained, so the masters are unchanged and
    the existing cache is already correct. Re-quantizing anyway costs ~15 ms per step at
    the 8B preset -- and it lands in exactly the regime where no other work is running
    to hide it. The refresh therefore belongs *inside* the ``empty_step`` guard, which a
    placement beside the router-bias update would miss.

    Counted rather than inferred from the cache: on an empty step a stale cache and a
    fresh one are bit-identical, so only the call count can tell them apart.
    """
    calls: list[int] = []
    _fit_mxfp8(_CountingLoader(num_batches=1, blank=(0,)), calls, monkeypatch)
    assert calls == [], f"refresh fired {calls} on a step that trained nothing"


def test_apply_compile_raises_the_recompile_limit_past_the_child_count():
    """Compiling N sibling blocks must not leave Dynamo's limit at its default 8.

    Dynamo caches per *code object*, and N blocks of one class share a single
    ``forward``, so all N contend for one limit. Past it Dynamo **silently falls back
    to eager** for the rest of the run: the symptom is a slow model rather than an
    error, which is exactly why it survived. ``muon.py`` already raised the limit for
    its Newton-Schulz iteration and every other compiled path was left at the default,
    so any preset deeper than eight blocks crossed it during warmup.

    Asserted against the child count rather than a fixed number, so a deeper preset
    keeps the property instead of quietly re-crossing it. No forward is run: the limit
    has to be raised before the first compile, and running one would test Dynamo.
    """
    config = torch._dynamo.config
    before = (config.recompile_limit, config.accumulated_recompile_limit)
    try:
        config.recompile_limit, config.accumulated_recompile_limit = 8, 256
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(12))
        apply_compile(model, {"mode": "module", "targets": ["blocks"], "dynamic": True})
        assert config.recompile_limit >= 12, (
            f"recompile_limit is {config.recompile_limit} for 12 compiled blocks; "
            "Dynamo will silently fall back to eager partway through warmup"
        )
        assert config.accumulated_recompile_limit >= 12
    finally:
        config.recompile_limit, config.accumulated_recompile_limit = before


def test_apply_compile_never_lowers_a_limit_another_path_raised():
    """Two compiled paths share one process-global config, so the raise is a max.

    Muon raises the limit for its Newton-Schulz shapes; the model compile raises it
    for its blocks. Whichever runs second must not undo the first, or the failure
    reappears in whichever path needed the larger number.
    """
    config = torch._dynamo.config
    before = (config.recompile_limit, config.accumulated_recompile_limit)
    try:
        config.recompile_limit, config.accumulated_recompile_limit = 4096, 8192
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(2))
        apply_compile(model, {"mode": "module", "targets": ["blocks"]})
        assert config.recompile_limit == 4096
        assert config.accumulated_recompile_limit == 8192
    finally:
        config.recompile_limit, config.accumulated_recompile_limit = before


def test_composer_warmup_lands_on_the_first_phase_not_the_composer():
    """A composed schedule must still reach its final phase's ``min_value``.

    See docs/guides/training.md.
    """
    from anyschedule.utils import get_scheduler

    from kohakuwullm.utils import autofill_schedule_steps

    total = 100_000
    config = autofill_schedule_steps(
        {
            "mode": "composer",
            "end": -1,
            "schedules": [
                {"mode": "power", "end": 0.9, "s0": 3333, "b": -0.5},
                {"mode": "cosine", "end": 1.0, "min_value": 0.01},
            ],
        },
        total,
        warmup_ratio=0.05,
    )
    assert "warmup" not in config
    assert config["schedules"][0]["warmup"] == 5000

    schedule = get_scheduler(copy.deepcopy(config))
    assert schedule(0) == pytest.approx(0.0)
    assert schedule(5000) == pytest.approx(1.0)
    assert schedule(total - 1) == pytest.approx(0.01, rel=1e-3)

    plain = autofill_schedule_steps(
        {"mode": "cosine", "min_value": 0.05, "end": -1}, total, warmup_ratio=0.05
    )
    assert plain["warmup"] == 5000 and plain["end"] == total


def test_scaler_unscales_then_reports_a_finite_flag():
    """``unscale_`` divides in place and returns 1 only when a gradient is not finite."""
    scaler = PipelineGradScaler(init_scale=1024.0)
    clean = torch.nn.Parameter(torch.zeros(8))
    clean.grad = torch.full((8,), 2048.0)

    assert scaler.unscale_([clean]).item() == 0.0
    assert torch.equal(clean.grad, torch.full((8,), 2.0)), "gradient not unscaled"

    dirty = torch.nn.Parameter(torch.zeros(8))
    dirty.grad = torch.full((8,), float("inf"))
    assert scaler.unscale_([dirty]).item() == 1.0


def test_scaler_backs_off_and_holds_a_floor():
    scaler = PipelineGradScaler(init_scale=8.0, growth_interval=2, min_scale=2.0)
    scaler.update(overflow=True)
    assert scaler.scale_value == 4.0
    scaler.update(overflow=True)
    assert scaler.scale_value == 2.0
    # A run that keeps halving has a real inf; decaying to zero would hide it.
    scaler.update(overflow=True)
    assert scaler.scale_value == 2.0

    scaler.update(overflow=False)
    assert scaler.scale_value == 2.0, "grew before the interval elapsed"
    scaler.update(overflow=False)
    assert scaler.scale_value == 4.0
    assert scaler.overflows == 3


def test_disabled_scaler_is_a_pass_through():
    """A bf16 run must pay nothing and, crucially, must not divide gradients."""
    scaler = PipelineGradScaler(enabled=False, init_scale=1024.0)
    loss = torch.tensor(3.0)
    assert scaler.scale(loss) is loss

    param = torch.nn.Parameter(torch.zeros(8))
    param.grad = torch.full((8,), 2048.0)
    assert scaler.unscale_([param]).item() == 0.0
    assert torch.equal(param.grad, torch.full((8,), 2048.0))


def test_scaler_state_survives_a_round_trip():
    """A resume that lost the scale would re-run the initial backoff."""
    scaler = PipelineGradScaler(init_scale=65536.0)
    scaler.update(overflow=True)
    scaler.update(overflow=False)

    restored = PipelineGradScaler(init_scale=65536.0)
    restored.load_state_dict(scaler.state_dict())
    assert restored.scale_value == scaler.scale_value == 32768.0
    assert restored.clean_steps == 1
    assert restored.overflows == 1


def test_target_reaches_loss_with_its_structure_intact():
    """``loss(hidden, batch)``: the batch is whatever the step carried.

    torch's schedule splits a target with ``tensor_split``, so a tuple or dict
    raises ``AttributeError: 'tuple' object has no attribute 'size'``. The batch
    never crosses a rank -- only the last stage reads it -- so it is split here
    and the schedule is handed an index instead.
    """
    from kohakuwupipe.training.loop import split_target

    batch = {
        "velocity": torch.arange(8.0).view(8, 1),
        "weight": torch.ones(8),
        "scheme": "flow",
    }
    pieces = split_target(batch, 4)
    assert len(pieces) == 4
    assert [tuple(p["velocity"].shape) for p in pieces] == [(2, 1)] * 4
    assert torch.equal(pieces[2]["velocity"].flatten(), torch.tensor([4.0, 5.0]))
    # A non-tensor leaf is config, not data: every microbatch gets it whole.
    assert all(p["scheme"] == "flow" for p in pieces)

    nested = (torch.arange(4.0), [torch.arange(4.0), None])
    parts = split_target(nested, 2)
    assert isinstance(parts[0], tuple) and isinstance(parts[0][1], list)
    assert torch.equal(parts[1][0], torch.tensor([2.0, 3.0]))
    assert parts[0][1][1] is None

    # The control: this is what the schedule would have done, and why it can't.
    from torch.distributed.pipelining.microbatch import TensorChunkSpec, _split_tensor

    with pytest.raises(AttributeError):
        _split_tensor(batch, TensorChunkSpec(0), 4)
