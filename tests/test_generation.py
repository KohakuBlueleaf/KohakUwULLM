"""Cached generation must be indistinguishable from re-running the prefix.

A KV cache produces fluent, plausible, wrong output when its position offset,
its GQA head count or its window is off by one, so the load-bearing test here is
token-for-token equality against the cache-free path rather than any property of
the cache itself.
"""

import pytest
import torch
from model_fixtures import tiny_config

from kohakuwullm import LMBackbone, SeqInfo
from kohakuwullm.bench.core.timing import rel_error
from kohakuwullm.generation import engine
from kohakuwullm.models.cache import KVCache
from kohakuwullm.training import LMStage, plan_for
from kohakuwullm.training.loop.sampling import PreviewSampler

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Every attention feature that changes what a cached step is allowed to see.
MOE = dict(moe_every=1, moe_num_experts=4, moe_top_k=2, moe_hidden=128)
ARCHITECTURES = {
    "dense": dict(),
    # The grouped-GEMM expert path is CUDA-only; routing is what a cache can break.
    "moe": dict(MOE, moe_mlp_kwargs=dict(dense_fallback=True)),
    "sliding_window": dict(sliding_window=8),
    "sink": dict(attn_sink=True),
    "mqa": dict(kv_heads=1),
    "no_qk_norm": dict(qk_norm=False),
    "sdpa_backend": dict(attn="sdpa"),
    "partial_rope": dict(rope_partial=0.5),
}
CUDA_ARCHITECTURES = {**ARCHITECTURES, "moe_fused": MOE}


# At the default init_std these weights decode one token and then repeat it
# forever, which no cache bug can disturb. See docs/concepts/architecture.md.
CHAOTIC_INIT = 0.3


def _backbone(seed: int = 0, **overrides) -> LMBackbone:
    torch.manual_seed(seed)
    return LMBackbone(tiny_config(init_std=CHAOTIC_INIT, **overrides)).eval()


# ------------------------------------------------------------------ equivalence


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_cached_generation_matches_the_cache_free_path(arch):
    """Greedy decoding with a cache must emit the identical token sequence."""
    backbone = _backbone(**ARCHITECTURES[arch])
    sampler = PreviewSampler()
    prompt = torch.randint(0, 512, (2, 5))

    cached = sampler.generate(
        backbone, prompt, max_new_tokens=32, temperature=0.0, use_cache=True
    )
    plain = sampler.generate(
        backbone, prompt, max_new_tokens=32, temperature=0.0, use_cache=False
    )
    assert torch.equal(cached, plain)
    assert cached.shape == (2, 37)
    _assert_the_comparison_can_fail(cached[:, prompt.shape[1] :])


def _assert_the_comparison_can_fail(generated: torch.Tensor) -> None:
    """Require the decoded stream to depend on its context.

    A model that emits one token forever satisfies token-for-token equality
    against any cache, correct or not.
    """
    distinct = [len(set(row.tolist())) for row in generated]
    assert min(distinct) > 4, f"degenerate decode, {distinct} distinct tokens"


def test_cached_decode_hidden_states_match_a_full_forward():
    """Decoding token by token must reproduce the full forward's hidden states."""
    backbone = _backbone(sliding_window=6)
    tokens = torch.randint(0, 512, (2, 12))
    reference = backbone(tokens)

    cache = KVCache.from_config(backbone.config, batch_size=2, max_length=12)
    stepwise = [backbone(tokens[:, i : i + 1], cache=cache) for i in range(12)]
    assert rel_error(torch.cat(stepwise, dim=1), reference) < 1e-5


@pytest.mark.parametrize("window", [None, 6])
def test_prefill_then_decode_matches_a_full_forward(window):
    """Prefill, a multi-token continuation and a single step share one position axis.

    The multi-token continuation is the only caller of the masked cached path;
    a one-token step takes the mask-free decode branch.
    """
    backbone = _backbone(sliding_window=window)
    tokens = torch.randint(0, 512, (1, 12))
    reference = backbone(tokens)

    cache = KVCache.from_config(backbone.config, batch_size=1, max_length=12)
    for start, end in ((0, 8), (8, 11), (11, 12)):
        chunk = backbone(tokens[:, start:end], cache=cache)
        assert rel_error(chunk, reference[:, start:end]) < 1e-5
    assert cache.length == 12


def test_cache_respects_the_sliding_window():
    """A cached step must not see past the window the training path enforces.

    Perturbing a token that has fallen out of the window may not change the
    output; the token just inside it must.
    """
    window = 4
    backbone = _backbone(sliding_window=window, depth=1)
    tokens = torch.randint(0, 512, (1, 10))

    def decode_last(prefix):
        cache = KVCache.from_config(backbone.config, batch_size=1, max_length=10)
        backbone(prefix[:, :-1], cache=cache)
        return backbone(prefix[:, -1:], cache=cache)

    base = decode_last(tokens)
    outside = tokens.clone()
    outside[:, 0] = (outside[:, 0] + 1) % 512
    inside = tokens.clone()
    inside[:, -window] = (inside[:, -window] + 1) % 512

    assert torch.equal(decode_last(outside), base)
    assert not torch.equal(decode_last(inside), base)


# ------------------------------------------------------------------ the buffers


def test_cache_holds_kv_heads_not_query_heads():
    backbone = _backbone(heads=4, kv_heads=2, head_dim=32)
    cache = KVCache.from_config(backbone.config, batch_size=3, max_length=7)
    backbone(torch.randint(0, 512, (3, 7)), cache=cache)

    assert cache.length == 7
    assert cache.keys[0].shape == (3, 7, 2, 32)
    assert cache.values[0].shape == (3, 7, 2, 32)
    assert len(cache.keys) == backbone.config.depth


def test_cache_refuses_to_run_past_its_preallocation():
    """Overflow raises; wrapping or truncating would corrupt the prefix silently."""
    backbone = _backbone()
    cache = KVCache.from_config(backbone.config, batch_size=1, max_length=4)
    backbone(torch.randint(0, 512, (1, 4)), cache=cache)

    with pytest.raises(ValueError, match="overflow"):
        backbone(torch.randint(0, 512, (1, 1)), cache=cache)


def test_cache_rejects_the_packed_layout():
    backbone = _backbone()
    cache = KVCache.from_config(backbone.config, batch_size=1, max_length=8)
    with pytest.raises(ValueError, match="padded"):
        backbone(torch.randint(0, 512, (8,)), cache=cache)


def test_cache_positions_continue_the_prefix():
    cache = KVCache(layers=2, batch_size=2, max_length=16, kv_heads=2, head_dim=32)
    cache.advance(5)
    info = cache.seq_info(torch.zeros(2, 3, dtype=torch.long))

    assert not info.packed
    assert torch.equal(info.position_ids[0], torch.tensor([5, 6, 7]))


def test_cache_reset_keeps_the_buffers():
    backbone = _backbone()
    cache = KVCache.from_config(backbone.config, batch_size=1, max_length=8)
    backbone(torch.randint(0, 512, (1, 8)), cache=cache)
    buffers = [k.data_ptr() for k in cache.keys]

    cache.reset()
    backbone(torch.randint(0, 512, (1, 8)), cache=cache)
    assert cache.length == 8
    assert [k.data_ptr() for k in cache.keys] == buffers


# ------------------------------------------------------------ the training path


def test_a_cacheless_forward_is_untouched_by_the_cache_plumbing():
    """The packed training path must reach the same numbers with the cache in tree."""
    backbone = _backbone()
    lengths = torch.tensor([5, 7], dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths)
    tokens = torch.randint(0, 512, (12,))

    first = backbone(tokens, info)
    second = backbone(tokens, info, cache=None)
    assert torch.equal(first, second)


@cuda_only
def test_cached_decode_stays_close_in_bf16():
    """Token equality is an fp32 statement; bf16 pins the error stays numerical.

    A structurally wrong cache -- a lost position, a dropped key -- puts the
    hidden states O(1) apart, which no dtype hides.
    """
    backbone = _backbone().cuda().to(torch.bfloat16)
    tokens = torch.randint(0, 512, (2, 12), device="cuda")
    reference = backbone(tokens)

    cache = KVCache.from_config(backbone.config, batch_size=2, max_length=12)
    stepwise = [backbone(tokens[:, i : i + 1], cache=cache) for i in range(12)]
    assert rel_error(torch.cat(stepwise, dim=1).float(), reference.float()) < 5e-2


@cuda_only
@pytest.mark.parametrize("arch", sorted(CUDA_ARCHITECTURES))
def test_cached_generation_matches_the_cache_free_path_on_cuda(arch):
    backbone = _backbone(**CUDA_ARCHITECTURES[arch]).cuda()
    sampler = PreviewSampler()
    prompt = torch.randint(0, 512, (2, 5), device="cuda")

    cached = sampler.generate(
        backbone, prompt, max_new_tokens=32, temperature=0.0, use_cache=True
    )
    plain = sampler.generate(
        backbone, prompt, max_new_tokens=32, temperature=0.0, use_cache=False
    )
    assert torch.equal(cached, plain)
    _assert_the_comparison_can_fail(cached[:, prompt.shape[1] :])


class _FirstStepRunsTwice:
    """A schedule that forwards once for metadata before its first real step."""

    def __init__(self, module) -> None:
        self.module = module
        self.pending = True

    def step(self, tokens=None):
        if self.pending:
            self.pending = False
            self.module(torch.zeros_like(tokens))
        return self.module(tokens)


def test_decode_absorbs_the_schedules_first_extra_forward(monkeypatch):
    """The extra forward a schedule makes on its first step must miss the cache.

    A ``PipelineStage`` declared with ``input_args`` alone infers its output
    metadata by running the stage once on an uninitialized receive buffer.
    Reaching the generation cache, that forward stores one garbage key/value and
    shifts every position after it; rebuilding the schedule per call repeats it.
    """
    torch.manual_seed(0)
    config = tiny_config(init_std=CHAOTIC_INIT, attn="sdpa")
    stage = LMStage(LMBackbone(config), plan_for(config, 1)[0]).eval()

    built = []

    def build(module, chunks, loss_fn=None, kind="gpipe"):
        built.append(chunks)
        return _FirstStepRunsTwice(module)

    monkeypatch.setattr(engine, "build_schedule", build)

    attached = []
    set_cache = stage.set_cache

    def record(cache):
        if cache is not None:
            attached.append(cache)
        set_cache(cache)

    monkeypatch.setattr(stage, "set_cache", record)

    # A cache dtype an uncast CPU attention can take.
    stage.boundary_dtype = torch.float32
    generator = engine.PipelineGenerator(stage, stage, rank=0, world=1, microbatches=1)
    prompt = torch.zeros(2, 3, dtype=torch.long)
    for _ in range(2):
        generator.generate(prompt, max_new_tokens=4, temperature=0.0)

    steps = prompt.shape[1] + 4 - 1
    assert attached[-1][0].length == steps
    assert built == [1], "the schedule was rebuilt, so the extra forward ran again"
