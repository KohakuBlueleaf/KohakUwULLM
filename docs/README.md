# KohakUwULLM

An extensible **decoder-only LLM** training framework. The philosophy: *the framework
provides slots; you select or drop in what you need.* Reproducing a Llama-shaped dense
model, a Gemma-shaped sliding-window model or a DeepSeekMoE sparse model is a **config**,
not a code change. Adding a norm / MLP / attention / position encoding / router / data
source is a registry entry or a dotted import path — **no core edits**.

Target hardware is 4x RTX 5090 (32 GB, sm_120). Target sizes are dense up to ~1.5B and
MoE up to ~8B total. The corpus is the KohakuVault caption/tag databases rendered into
TIPO-style prompt-generation examples.

## Documentation map

Four sections. Each has its own index; start with the one matching what you are
about to do.

### [concepts/](concepts/README.md) — what the framework is

| Doc | What it covers |
|---|---|
| [architecture.md](concepts/architecture.md) | The backbone, `SeqInfo` and packed varlen, blocks, every swappable component, how presets compose them, the KV cache |
| [presets.md](concepts/presets.md) | The Kohaku ladder: each rung, the constraints it was solved under, measured parameter counts |

### [guides/](guides/README.md) — how to do a thing

| Doc | What it covers |
|---|---|
| [training.md](guides/training.md) | The Lightning trainer, schedules, optimizer defaults, token accounting, callbacks, previews, resume |
| [writing-configs.md](guides/writing-configs.md) | KohakuEngine config files, the knob catalogue, overrides and sweeps |
| [writing-scripts.md](guides/writing-scripts.md) | Writing a new training, bench or inference script |
| [generation.md](guides/generation.md) | Sampling, the KV cache, pipelined decode, and where decode's speed is |
| [extending.md](guides/extending.md) | Adding a component, router, MoE formulation, data source, renderer or optimizer |

### [internals/](internals/README.md) — how it is built

| Doc | What it covers |
|---|---|
| [data.md](internals/data.md) | KohakuVault sources, TIPO rendering, tokenization, loss masking, packing, the loader |
| [optimizers.md](internals/optimizers.md) | Muon, parameter grouping and weight decay, muP, low-bit state, stochastic rounding |
| [kernels.md](internals/kernels.md) | The Triton kernels, their numerics, and the trap in each |
| [mxfp8.md](internals/mxfp8.md) | fp8 training: what is converted, what is verified, what is still open |
| [pipeline.md](internals/pipeline.md) | Pipeline parallelism: stage splitting, boundary dtype, DDP vs pipeline |
| [moe-router-loss.md](internals/moe-router-loss.md) | The router's auxiliary losses in the fused kernel, and the stream that carries them across stages |

### [kohakuwupipe/](kohakuwupipe/README.md) — the pipeline trainer, on its own

`src/kohakuwupipe/` imports nothing from `kohakuwullm` and is meant to be lifted
into its own repository. Its docs live beside it, and every `See docs/…` in that
package resolves into this folder.

| Doc | What it covers |
|---|---|
| [module.md](kohakuwupipe/module.md) | `PipelineModule`: the contract, and what survives being split across ranks |
| [loop.md](kohakuwupipe/loop.md) | The step, loss normalization, callbacks |
| [streams.md](kohakuwupipe/streams.md) | Multi-stream boundaries, and the dense-gradient trap |
| [plan.md](kohakuwupipe/plan.md) | Splitting a layer stack, and why the cost model is not enough |
| [distributed.md](kohakuwupipe/distributed.md) | Process groups, and why `device_id` is omitted |
| [checkpoint.md](kohakuwupipe/checkpoint.md) | Whole-model checkpoints from per-rank stages |
| [logging.md](kohakuwupipe/logging.md) | The rank-aware structured logger |
| [bench.md](kohakuwupipe/bench.md) | Measuring pipeline latency without lying to yourself |

### [performance/](performance/README.md) — what it costs, and how that was measured

| Doc | What it covers |
|---|---|
| [performance.md](performance/performance.md) | Measured throughput across the ladder, and where it goes |
| [benchmarking.md](performance/benchmarking.md) | How this repo measures, so a number means something |
| [ab-testing.md](performance/ab-testing.md) | Running a trustworthy A/B: noise floors, block bootstrap, admissibility |
| [upstream-cutlass-findings.md](performance/upstream-cutlass-findings.md) | Why CUTLASS grouped block-scaled GEMM is unusable on sm_120 |

Coding conventions live in [../CLAUDE.md](../CLAUDE.md). Benchmark *results* live in
[../out/bench_old/README.md](../out/bench_old/README.md); the docs here explain the
methods, that index holds the numbers. `out/bench/` holds the current re-runs and
carries no index of its own yet.

## The test suite is absent from the working tree

Docs across this tree cite `tests/test_*.py` by name and by test function. **None
of those files are present**: `tests/` does not exist in the working tree and is
not tracked by git. The suite is pending re-implementation from scratch, so the
citations name what a test *must* pin rather than a file you can run today. Read
every `tests/...` reference in the docs as a specification, not as a path.

Affected: `test_generation.py`, `test_models.py`, `test_models_posenc.py`,
`test_kernels.py`, `test_kernels_cpu_fallback.py`, `test_presets.py`,
`test_training.py`, `test_iterative_loader.py`, `test_lowbit.py`,
`test_stochastic_round.py`.

## Install and smoke-test

```bash
uv pip install -e ".[dev,bench]"

# absent from the working tree; see the note above
.venv/bin/python -m pytest tests/test_models.py -q

# 40 steps of the full shipping path on one card: MXFP8, Muon, real corpus
kogine run scripts/train/lm.py --config configs/lm/smoke_mxfp8.py
```

## Train something

```bash
# dense 500M, 4 cards
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py

# sparse 1B, 4 cards
kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py

# ad-hoc override, no file edit
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py --set LR=2e-4
```

Configs are plain Python. A config names a **preset**, overrides architecture fields,
lists data **sources**, and sets the training knobs:

```python
PRESET = "Kohaku-500M"
ARCH_OVERRIDES = {"max_position": 4096, "qk_norm": True, "mxfp8": True}
SOURCES = [{"name": "danbooru", "repeat": 3}]
OPTIMIZER = "muon"
MAX_STEPS = 100_000
```

See [writing-configs.md](guides/writing-configs.md) for the full catalogue.

## The one mental model

1. A **backbone** is `model(tokens, seq_info) -> hidden`. It knows nothing about the
   loss. Norm / MLP / attention / position encoding are swappable components; Llama,
   Gemma and DeepSeekMoE are **presets** over one backbone, not separate classes.
2. A **`SeqInfo`** says how the batch is laid out — packed (varlen, the training path)
   or padded (eval). Only attention reads it; everything else is a last-dim op that
   works on both.
3. The objective lives in **`LMHead`** and the trainer, so changing it never touches
   the trunk.
4. A **trainer** (Lightning, manual optimization) wires data → backbone → head, with
   schedules, compile and gradient accumulation resolved once and called directly.
5. Everything swappable lives in a **registry**; `build(spec, REGISTRY)` resolves a
   name / dotted path / dict / class / instance into a concrete object at build time.

The two rules that follow from this:

**Select, don't dispatch.** Configuration resolves a concrete class or callable *once*,
at build time. There is no per-step `if mode == ...` in any training loop. A runtime
branch on a config value belongs in `__init__`.

**The backbone is a pure function.** It maps tokens to hidden states and nothing else.

## Where the corpus stands

One **unweighted** pass over the caption corpus is 13.478B raw tokens across 76.3M
records, which is **51,413 optimizer steps** at a 262144-token global batch. That
count is `repeat: 1` on every source and includes `imagenet`, which no config trains
on. Every production config weights danbooru x3 and danbooru_tagger x2, so one pass of
the mixture actually trained is **20.890B raw, ~18.9B trained, about 79,700 steps**.
`scripts/data/token_census.py` regenerates the unweighted count; the breakdown is in
[data.md](internals/data.md).
