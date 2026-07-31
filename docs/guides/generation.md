# Generation

Two engines, one API. `LocalGenerator` runs a whole model on one rank;
`PipelineGenerator` runs a model split across ranks. Both take padded `(B, S)`
prompts and return `(B, S + n)`.

```python
from kohakuwullm.generation import build_generator
from kohakuwullm.models import LMBackbone, get_preset

model = LMBackbone(get_preset("Kohaku-500M")).cuda().to(torch.bfloat16).eval()
gen = build_generator(backbone=model)

prompt = tokenizer(["a cat sitting on a windowsill"], return_tensors="pt").input_ids
out = gen.generate(prompt.cuda(), max_new_tokens=128, temperature=0.8, top_p=0.95)
```

`build_generator` picks the engine once, from whether you hand it a `stage`:

```python
build_generator(backbone=model)                       # LocalGenerator
build_generator(stage=stage, head_module=inner,       # PipelineGenerator
                rank=rank, world=world)
```

## Sampling

`sample()` applies three filters in a fixed order: **top-k** bounds the candidate
count, **top-p** bounds their mass, **min-p** bounds their ratio to the peak.
`temperature <= 0` is greedy and bypasses all three.

```python
from kohakuwullm.generation import sample

sample(logits, temperature=0.0)                       # greedy
sample(logits, temperature=0.8, top_k=50)             # 50 candidates
sample(logits, temperature=0.8, top_p=0.95)           # smallest 95% of mass
sample(logits, temperature=1.0, min_p=0.05)           # >= 5% of the peak
sample(logits, temperature=0.8, top_k=50, min_p=0.02) # composed
```

`sample` returns `(B, 1)`, ready to concatenate onto the running sequence. Each
filter is exported separately so it can be tested alone: `filter_top_k` takes
logits, `filter_top_p` and `filter_min_p` take probabilities.

The generator owns a **private RNG stream**, seeded from `SAMPLE_SEED`, not the
default one. A preview that drew from the default stream would change the data
order, so turning logging on would change the run.

## The KV cache

```python
from kohakuwullm.models.cache import KVCache

cache = KVCache.from_config(config, batch_size=4, max_length=1024,
                            device="cuda", dtype=torch.bfloat16)
hidden = model(prompt, cache=cache)     # prefill; advances the cache by S
hidden = model(next_token, cache=cache) # decode; one token, positions continue
```

The cache is **padded-layout only**. Packed varlen is a training concern, and a
cache that tried to serve both would need per-document offsets it has no way to
carry. It raises rather than guessing.

Cached and cache-free generation produce **identical tokens** under greedy
decoding; `tests/test_generation.py` pins that, along with the traps that make a
cache produce fluent-but-wrong output:

- **positions continue the prefix** — the new token's position is the cache
  length, not zero
- **the cache holds `kv_heads`, not query heads** — GQA
- **the sliding window is respected**, so decode attends over the prefix training
  would have
- **a cacheless forward is untouched** by the plumbing

## Pipelined generation

**Cached decode is what makes pipelined generation possible at all.**
`PipelineStage` freezes its boundary shape at construction; a cache-free
generator grows its input every token, so no frozen boundary can ever fit it.
With a cache, the per-step input is exactly one token and the boundary is
constant.

```python
from kohakuwullm.generation import PipelineGenerator
from kohakuwullm.training.parallel.pipeline_lightning import build_stage

# `decode=True` declares the padded single-token boundary; the third-to-last
# argument is rows *per microbatch*, not the whole batch.
module, stage, plan, _ = build_stage(
    config, rank, world, device, 1, seq_len=1024, decode=True
)
inner = getattr(module, "module", module)

gen = PipelineGenerator(stage, inner, rank, world)
out = gen.generate(prompt, max_new_tokens=64, temperature=0.0)
```

Every rank must call `generate` the same number of times with the same shapes.
Only stage 0 needs the prompt and only the last stage produces logits; the
sampled token is broadcast back so stage 0 can feed the next step.

**The microbatch split is frozen at stage construction.** `build_stage` declares
the boundary as `(rows_per_microbatch, 1)`, so how the batch divides is a
build-time choice, not a runtime one.

**Use one microbatch.** Splitting the batch is a *training* optimization — it fills
the pipeline bubble — and it does not carry over to decode. Measured at
Kohaku-500M, pp=2, 8 rows x 32 new tokens:

| microbatches | ms/token | tok/s | % of roofline |
|---|---|---|---|
| **1** | **24.77** | **323** | **2.1%** |
| 2 | 47.86 | 167 | 1.1% |
| 4 | 95.40 | 84 | 0.5% |
| 8 | 176.29 | 45 | 0.3% |

Almost exactly linear in the wrong direction. Decode is latency-bound, not
throughput-bound: each microbatch is its own schedule step with its own P2P round
trip, so splitting 8 rows into 8 microbatches multiplies the sequential round
trips by 8 while shrinking every GEMM to a single row. There is no compute left to
overlap. `PipelinedLMTrainer.generate` passes `microbatches=1` for this reason.

Run `scripts/dist/pp_generate_smoke.py` under `torchrun --nproc_per_node=2` to
check a split model generates, and `pp_generate_bench.py` to reproduce the table.

## Speed

Decode is **memory-bound**: each step reads the active parameters once, so the
ceiling is `active_bytes / 1791 GB/s`, amortized across the batch.
`scripts/bench/gen/decode.py` reports achieved bandwidth against that ceiling.

The single largest lever measured so far is graph capture:

| | ms/token | tok/s | |
|---|---|---|---|
| eager | 26.10 | 38 | 0.8% of roofline |
| `torch.compile(mode="reduce-overhead")` | **0.97** | **1035** | **27x** |
| raw CUDA graph replay | 1.34 | 749 | 19.6x |

Eager decode issues **1646 device ops per token** and spends 275 ms of host time
against 37 ms of device time — it is launch-bound, not bandwidth-bound. Compile
beats a raw graph because Inductor also fuses the pointwise chains. MoE routing
captures without trouble.

Swapping the RoPE implementation is worth 1.07x by itself, which is the ceiling
its 11% share of host time allows. The gap was never one kernel; it was 1646 of
them. See [../performance/benchmarking.md](../performance/benchmarking.md) for
how these are measured.

**Pipelining decode is slower than not pipelining it, and that is the expected
result.** At Kohaku-500M the pipelined path reaches 2.1% of roofline against a
single card's ~23% compiled. Splitting a model across stages adds serialization
without adding parallelism: one token must still traverse every stage in order, so
the stages take turns rather than working at once. Pipelined generation exists so
that a model too large for one card can be previewed **at all** — not to make
generation faster. If the model fits on one card, use `LocalGenerator`.

## The schedule's first step runs an extra forward

`PipelineStage` supports two metadata modes. Given both `input_args` and
`output_args` it is *static* and knows every boundary shape up front. Given
`input_args` alone — which is what `build_stage` and `decode_stage` pass — it is
*dynamic*: it infers each stage's **output** metadata by running that stage once,
on the first `step()` of a freshly constructed schedule. On any stage past the
first, the input to that inference forward is the receive buffer *before anything
has been received into it*, i.e. uninitialized `torch.empty` memory.

That forward is real. It runs the model, and during generation it therefore runs
against whatever KV cache is attached: it stores one garbage key/value at
position 0 and calls `advance`, so every real decode step afterwards attends over
a poisoned prefix and sits one position too late. Roughly 1 in 32 random 16-bit
words is a NaN pattern, so the garbage is usually merely wrong and occasionally
NaN — and a NaN written into a cache is permanent, because the poisoned prefix is
re-attended every step. The visible failure is `multinomial` asserting
`probability tensor contains either inf, nan or element < 0` many tokens later,
with the stage forward that surfaces the sticky CUDA error blamed for it.

Two properties keep it out of the way:

- `PipelineGenerator` builds its schedule **once** and holds it. A schedule
  constructed per call resets `_stage_forward_initialized`, so the inference
  forward re-runs for every prompt.
- `PipelineGenerator.prepare` takes that first step against throwaway caches
  before `generate` attaches the ones it decodes from.

`tests/test_generation.py::test_decode_absorbs_the_schedules_first_extra_forward`
pins both, by asserting the decode cache advanced exactly once per decode step.

The same inference forward happens once per run on the training path, where
there is no cache to poison. It does add one microbatch of garbage routing counts
to `load_accum` before the first optimizer step, which the first `update_bias`
then clears.
