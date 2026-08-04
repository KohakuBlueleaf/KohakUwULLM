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

**`sample` casts logits to fp32 first.** Top-p cumulates over the whole
vocabulary, and summing 65536 fp16 terms loses percent-level accuracy — the same
rule as every other reduction in this repo. The finite check before
`torch.multinomial` is there because `multinomial` only asserts device-side, many
launches later, with an unrelated frame blamed for it.

`PreviewSampler` (`training/loop/sampling.py`) is a `LocalGenerator` that takes
its backbone per call instead of holding one. It exists so the training loop does
not have to construct a generator per preview; it adds no decode logic of its
own, and defaults to `static=False`.

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

`static=True` — what `LocalGenerator` passes by default — writes with
`index_copy_` at a device-side position and returns the whole buffer instead of a
growing prefix slice, so every decode step has one shape. That is what makes the
step compilable; see [The static KV cache](#the-static-kv-cache-is-what-made-that-possible)
below. `static=False` keeps the growing-slice layout and is what
`PipelineGenerator` builds.

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

# `decode=True` makes `boundary_example` declare a padded `(rows, 1)` boundary,
# so the `microbatch_tokens` positional is read as rows *per microbatch* here,
# not as a token count and not as the whole batch.
module, stage, plan, _ = build_stage(
    config, rank, world, device, 1, seq_len=1024, decode=True
)
inner = getattr(module, "module", module)

# `microbatches=1` is not optional; see below.
gen = PipelineGenerator(stage, inner, rank, world, microbatches=1)
out = gen.generate(prompt, max_new_tokens=64, temperature=0.0)
```

Every rank must call `generate` the same number of times with the same shapes.
Only stage 0 needs the prompt and only the last stage produces logits; the
sampled token is broadcast back so stage 0 can feed the next step.

**`microbatches` defaults to `None`, which means one row per microbatch — and
that combination is currently broken.** `generate` computes
`chunks = self.microbatches or rows`, then allocates `chunks` caches of
`rows // chunks` rows each. Under the default decode path (below) there is
exactly *one* forward per step, not `chunks` of them, so a batch of 8 rows meets
a cache built for 1 and `KVCache.append` raises
`cache holds batch 1, got 8` on the first step. Pass `microbatches=1` explicitly.
Both trainers already do; `scripts/dist/pp_generate_smoke.py` does not, so it is
expected to fail at its default 2 rows.

### The default decode path does not use the schedule

`PipelineGenerator(forward_only_decode=True)` — the default — drives each decode
step with a plain `dist.recv` / stage forward / `dist.send` chain
(`PipelineGenerator.forward_only`). It builds no schedule and carries no autograd
state, so a decode step costs one hop per stage instead of a training step's
bookkeeping.

Passing `forward_only_decode=False` restores the old path, which runs the token
through a `ScheduleGPipe` built once in `prepare`. Only that path is subject to
the extra-forward hazard at the end of this page.

**The per-step token slice is `clone`d, not `contiguous()`d.** A one-element
slice `tokens[:, pos:pos+1]` reports itself contiguous while keeping the source
row stride, so `contiguous()` is a no-op on it and the send would carry the wrong
stride to the next stage.

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

Run `scripts/dist/pp_generate_smoke.py`, which spawns its own two ranks, to
check a split model generates, and `pp_generate_bench.py` to reproduce the table.

### Four ranks, from `configs/lm/smoke/pipegen_bench.py`

`configs/lm/smoke/pipegen_bench.py` runs Kohaku-MoE-1B over four ranks with the
pipelined preview selected — `SAMPLE_LOCAL = False`, `SAMPLE_FORWARD_ONLY =
True` — so one short run reports both the decode path and what the training loop
around it does.

Preview decode through `PipelineGenerator.forward_only`, fp16, 4 rows x 64 new
tokens:

| | wall | ms/token | tok/s |
|---|---|---|---|
| steady state | 1.31 s | 20.5 | 195 |
| first preview, cold | 2.96 s | — | — |

The cold preview costs 2.3x the steady one, so a preview interval short enough
that the first one dominates will misreport the path.

Training in the same run, **uncompiled**, `NUM_MICROBATCHES = 8` and
`MICRO_TOKENS = 16384` (131k tokens per step), synthetic data:

| ms/step | tok/s | trained_frac | trained tokens/day |
|---|---|---|---|
| 502.3 | 261k | 0.9969 | 22.5B |

**Neither figure has a JSON artifact.** Both were read from the run's console
output; there is no file under `out/bench*/` to cite for either, and no plot
script regenerates them. Re-run the config to reproduce them.

## Speed

Decode *should* be **memory-bound**: each step reads the active parameters once,
so the ceiling is `active_bytes / 1791 GB/s`, amortized across the batch.
`scripts/bench/gen/decode.py` reports achieved bandwidth against that ceiling.

It is not. Kohaku-MoE-1B, bf16, 197M active
(`out/bench_old/gen/decode/Kohaku-MoE-1B.json`):

| batch | tok/s | GB/s | % of roofline |
|---|---|---|---|
| 1 | 35.1 | 13.9 | 0.78% |
| 4 | 141.3 | 14.0 | 0.78% |
| 16 | 567.7 | 14.4 | 0.80% |

Read the last column, not the second. Achieved bandwidth is **flat** across a 16x
batch, and tok/s scales exactly with the batch, so a decode step costs the same
wall time whether it carries 1 row or 16. A bandwidth-bound step would not do
that. The per-step cost is dispatch, not DRAM: decode here is **launch-bound**,
and the batch is free until something else binds.

### `torch.compile` is worth 4.1x, once the decode step has one shape

`scripts/tools/sample.py` is the only harness in the tree that compiles the decode
step: `--set COMPILE=<mode>` calls `model.compile(mode=...)` and `--set
BENCH_STEPS=N` times N greedy steps, best of 3 after a warmup call, with the EOS
stop disabled so every arm does identical work.

Kohaku-MoE-1B at step 64000, fp16 parameters and fp16 autocast, MXFP8 **off**,
8 rows x 64 steps (`out/bench/gen/decode/compile_modes.json`):

| arm | ms/step | tok/s | vs eager |
|---|---|---|---|
| eager | 35.80 | 223 | 1.00x |
| `mode="reduce-overhead"` | 9.88 | 810 | 3.6x |
| `mode="default"` | **8.66** | **923** | **4.1x** |

```bash
kogine run scripts/tools/sample.py --config configs/lm/tipo/tipo_moe_1b_uwupipe.py \
    --set CKPT=out/ckpt/tipo-moe-1b-uwupipe/step-64000.ckpt \
    --set SAMPLE_COUNT=8 --set BENCH_STEPS=64 --set MXFP8=False \
    --set COMPILE=default
```

**The win is not CUDA graphs.** `reduce-overhead` is the mode that captures them,
and it is the *slower* of the two compiled arms. What compile buys here is one
traced kernel set that every step reuses, which collapses the per-step dispatch
the roofline table above identified as the binding cost.

`model.compile(...)` is `nn.Module.compile` — compilation applied in place —
not `torch.compile(model)`, which returns a wrapper and prefixes every checkpoint
key with `_orig_mod.`.

### The static KV cache is what made that possible

A compiled decode step is only reusable if its shapes never move, and the cache
was what moved them. `KVCache.append` returned `keys[:, :end]`, one token longer
every step, so Inductor re-traced per token and paid more in compilation than it
saved.

`KVCache(static=True)` — the default for `LocalGenerator` — removes the moving
dimension:

- the write is `index_copy_` at `self.pos`, a **device-side** scalar, so no Python
  int enters the graph;
- the read hands back the **whole** `(B, max_length, kv_heads, head_dim)` buffer
  rather than a prefix slice;
- `key_mask(steps)` is what makes that correct: `(1, 1, steps, max_length)`,
  causal within the step and bounded by the committed prefix, so a query can never
  read a slot that has not been written. `window_mask` does the same for the
  sliding-window case;
- buffers are allocated with `torch.zeros` rather than `torch.empty`, since the
  unwritten tail is now inside every attention's key axis;
- `seq_info` reads positions from `pos` and declares `max_seqlen = max_length`.

`PipelineGenerator` does **not** use it: `_caches` builds its `KVCache` without
`static=`, so pipelined decode keeps the growing-slice layout. Only the local path
is compiled today.

> **Superseded logs.** `out/sample_arms.log`, `sample_arms2.log`, `gen_speed.log`
> and `arm_compiled.log` hold an eager spread of 28.43-35.14 ms/step and a single
> surviving `reduce-overhead` run at 31.40 ms/step, alongside three
> `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
> by a subsequent run` aborts raised from `models/block.py`. **Those predate both
> the static cache and whole-model compilation** — the aborts came from compiling
> per block — and must not be pooled with the table above.

> **Unreproduced.** An earlier revision of this page reported eager 26.10
> ms/token against 0.97 ms/token compiled (27x), a raw CUDA-graph replay at 1.34
> ms/token, "1646 device ops per token", "275 ms of host time against 37 ms of
> device time", and a 1.07x RoPE-swap win. **No harness in the tree produces any
> of those numbers**, and the measured compile win above is 4.1x, not 27x.
> `scripts/bench/gen/decode.py` and `scripts/bench/model/generate.py` never
> compile; nothing counts device ops; there is no raw-graph decode arm and no RoPE
> decode A/B. Treat every one of those figures as unsourced.

**Pipelining decode is slower than not pipelining it, and that is the expected
result.** At Kohaku-500M, pp=2, the pipelined path reaches 2.1% of its own
roofline. There is no single-card Kohaku-500M decode figure in the tree to put
beside it — `decode.py` has only been run at Kohaku-MoE-1B — so treat the ratio
between the two paths as unmeasured, and the argument below as the reason to
expect it rather than as a result. Splitting a model across
stages adds serialization
without adding parallelism: one token must still traverse every stage in order, so
the stages take turns rather than working at once. Pipelined generation exists so
that a model too large for one card can be previewed **at all** — not to make
generation faster. If the model fits on one card, use `LocalGenerator`.

Both training loops now take that further and **do not pipeline previews at all
by default**. `SamplePreview(local=True)` — the default, and what
`scripts/train/lm_pipe.py` ships as `SAMPLE_LOCAL = True` — `all_gather_object`s
every stage's slice onto rank 0, loads it into a whole `LMBackbone`, and decodes
there with `LocalGenerator`. The gather is collective and every rank must reach
it; only rank 0 holds the merged model and decodes. `SAMPLE_LOCAL = False` falls
back to pipelined decode, with `SAMPLE_FORWARD_ONLY` choosing between the
forward-only path and the schedule.

## The schedule's first step runs an extra forward

This applies only to `forward_only_decode=False`, which is no longer the default
decode path. It is kept because the hazard is invisible when you hit it and the
schedule path is still reachable.

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
