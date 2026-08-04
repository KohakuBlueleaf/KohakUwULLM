<p align="center">
  <strong>KohakUwULLM</strong>
</p>
<p align="center">
  <strong>The machine for training decoder-only LLMs, so a new model shape costs a config instead of a fork.</strong>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/torch-2.13%2B-red" alt="torch 2.13+">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License">
</p>

---

## See it run (60 seconds)

```bash
uv pip install -e ".[dev,bench]"

# tokenizer: DeepSeek-V4 pruned to 64k + 1536 special slots = 65536
.venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer

# smoke run: 25M model, one GPU, 50 steps
kogine run scripts/train/lm.py --config configs/lm/smoke/debug.py
```

That is a real training loop with packed varlen batches, token-exact gradient
accumulation and in-training sample previews. Swapping to a 1B sparse model is a
different config file and nothing else:

```bash
kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m.py       # dense 546M
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo/tipo_moe_1b_uwupipe.py  # MoE 991M
```

The second one splits itself across every GPU it finds and measures the split
rather than guessing it.

## Is this for you?

**You probably want KohakUwULLM if** you are training decoder-only models from
scratch and keep changing the architecture, you want low-precision training you
can verify instead of trust, you have a handful of consumer GPUs rather than a
cluster, or you want the kernels and the benchmarks in the same repo as the
trainer.

**You probably do not if** you are fine-tuning existing checkpoints (use PEFT or
Axolotl), you need multi-node training at cluster scale (use Megatron or
TorchTitan), or you want a stable API. This repo changes its internals whenever a
measurement says it should.

## What KohakUwULLM is

KohakUwULLM is a framework for building LLM training runs, not a trainer script.

Most training repos hard-code one architecture and hand you flags. Changing the
norm, the router, the attention kernel or the position encoding means editing the
core. KohakUwULLM puts every one of those behind a registry, so reproducing a
Llama-shaped dense model, a Gemma-shaped sliding-window model or a DeepSeekMoE
sparse model is a config entry rather than a patch.

Two rules hold the design together.

**Select, do not dispatch.** Configuration resolves a concrete class once, at
build time, through `build(spec, REGISTRY)`. The result is a plain attribute and
gets called directly. There is no per-step branch on a config value anywhere in a
training loop. If a runtime branch on config appears, it belongs in `__init__`.

**The backbone is a pure function.** `LMBackbone` maps `(tokens, seq_info)` to
hidden states and knows nothing about the loss. Objectives live in `LMHead` and
the trainer, so changing the objective never touches the trunk.

Everything else follows from those two. `transformers` is used for the tokenizer
and nothing else.

## Where it fits

|  | Full stack | Framework | Utility |
|--|-----------|-----------|---------|
| **Fine-tune** | Axolotl, LLaMA-Factory | PEFT, TRL | Unsloth |
| **Pretrain** | Megatron-LM, TorchTitan | ***KohakUwULLM***, litgpt | nanoGPT |
| **Kernels** | | ***KohakUwULLM***, Liger | FlashAttention |

Megatron and TorchTitan target clusters and assume a fixed model family.
nanoGPT is a teaching artifact. KohakUwULLM sits where a small lab actually
works: a few consumer cards, an architecture that keeps moving, and a need to
know whether a kernel is both fast and correct.

## Key features

- **Own architecture implementation.** GQA, QK-norm, RoPE with linear, NTK and
  YaRN scaling, sliding windows with global-layer interleave and a per-layer
  choice of attention kernel, attention sinks, DeepSeekMoE with aux-loss-free
  load balancing, and z-loss. Sinkhorn, expert-choice and ReLU routers are
  registered beside the default. Embeddings are untied by default.
- **Packed variable-length training.** No padding is ever computed on and
  attention cannot cross a document boundary. Worth 2.8x to 5.8x forward on this
  corpus, rising with the padding fraction it removes.
- **Native MXFP8 training, verified rather than assumed.** e4m3 values with
  UE8M0 per-32 block scales on the attention projections, the feed-forward pair
  and the routed experts. Measured 1.08x to 1.22x end to end on dense and 1.53x
  to 2.31x on MoE, at a loss gap inside run-to-run noise.
- **Triton kernels where they win**, each benchmarked and precision-tested
  against an fp64 reference in both fp16 and bf16. Fused RMSNorm, fused SwiGLU,
  a grouped GEMM for MoE experts, fully fused MXFP8 expert kernels, a
  block-scaled attention kernel, and a chunked z-loss that never materializes
  logits.
- **Two trainers.** A Lightning path with token-exact gradient accumulation and
  DDP, and `kohakuwupipe`, an extractable pipeline-parallel engine that stays
  architecture-free and measures its own stage split at startup.
- **Portable checkpoints.** Weights and optimizer state are both stored under
  whole-model names, so a file written by one pipeline split loads into any other
  split, into DDP, or onto a single GPU.
- **Export to the ecosystem.** `to_gguf.py` produces a llama.cpp model,
  `to_hf.py` produces a transformers repository with its own architecture and
  `trust_remote_code`.
- **Benchmarks as a deliverable.** `scripts/bench/` produces figures that show
  accuracy next to speed, because a kernel that is fast and wrong is not a
  result.

## Quick start

> **Recommended Python**: 3.13. `requires-python = ">=3.10"`, and torch 2.13 or
> newer is mandatory for `F.linear_cross_entropy` and
> `torch.nn.attention.varlen.varlen_attn`.

### 1. Install

```bash
git clone https://github.com/KohakuBlueleaf/KohakUwULLM.git
cd KohakUwULLM
uv pip install -e ".[dev,bench]"
```

### 2. Build the tokenizer

```bash
.venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer
```

### 3. Train something

```bash
kogine run scripts/train/lm.py --config configs/lm/smoke/debug.py        # 25M, one GPU
kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m.py    # dense 546M
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo/tipo_moe_1b_uwupipe.py
```

Every UPPER_CASE global in a train script is a knob a config may override, and
`--set KEY=VALUE` overrides one from the command line.

### 4. Sample, export, quantize

```bash
kogine run scripts/tools/sample.py --config configs/lm/tipo/tipo_moe_1b_uwupipe.py \
    --set CKPT=out/ckpt/.../step-50000.ckpt --set TEMPERATURE=1.0 --set MIN_P=0.1

kogine run scripts/tools/to_gguf.py --config <same> --set CKPT=... --set OUT=model.gguf
kogine run scripts/tools/to_hf.py   --config <same> --set CKPT=... --set OUT=out/hf
```

## The preset ladder

Nine rungs on a fixed 65536 vocabulary and untied embeddings, with sparsity
pinned at `top_k/experts = 0.125` so hyper-parameters transfer between rungs.
Counts are measured by building each rung on the meta device rather than solved,
because a closed form omits the norms, the router matrix and the second embedding
matrix.

| dense | total | | sparse | total / active |
|---|---|---|---|---|
| Kohaku-200M | 204M | | Kohaku-MoE-1B | 991M / 248M |
| Kohaku-500M | 546M | | Kohaku-MoE-2B | 1953M / 411M |
| Kohaku-1B | 982M | | Kohaku-MoE-3B | 2907M / 617M |
| Kohaku-1.5B | 1514M | | Kohaku-MoE-5B | 4934M / 943M |
| | | | Kohaku-MoE-8B | 7713M / 1371M |

The rungs interleave. Their effective capacities `sqrt(active x total)` form one
smooth sequence: 204, 381, 546, 756, 982, 1182, 1514, 1953 and 2902 M. The
`active` in that expression excludes the embedding, the head and the router,
which is the quantity the ladder's targets were solved in. The `active` in the
table is the repo-wide one, which includes all three and runs 20 to 30 percent
higher on every sparse rung. `bench/ladder.py` reports both under different names
for exactly that reason.

## Core mental model

### One backbone, swappable parts

```text
     tokens + SeqInfo
            |
            v
    +----------------+
    |   embedding    |
    +----------------+
            |
            v
    +----------------+      norm    <- rmsnorm / layernorm / gemma / dyt
    |     block      |      attn    <- varlen / triton / mxfp8 / sdpa / flex
    |   (x depth)    |      mlp     <- swiglu / geglu / gelu / moe
    +----------------+      posenc  <- rope / nope
            |
            v
    +----------------+
    |  final norm    |
    +----------------+
            |
            v
        hidden          -> LMHead owns the loss, the trunk does not
```

A `SeqInfo` says how a batch is laid out, packed or padded. Only attention reads
it. Everything else is a last-dim operation that works on both.

### Packed varlen is the training layout

Every sequence is concatenated into one flat token axis and `cu_seqlens` carries
the document boundaries. Nothing is padded. For 50 to 600 token samples against a
2048 context, a padded batch would be roughly 80 percent padding.

The invariant that makes it correct is that **attention must never cross a
document boundary**. `varlen` gets this from the kernel. The SDPA and Flex
fallbacks build an explicit block-diagonal mask. A test pins it, because a
backend that forgets the boundary still trains, just worse, for reasons nothing
in the loss curve explains.

### Two trainers, one model

```text
        LMBackbone + LMHead
           /            \
          v              v
  Lightning LMTrainer   kohakuwupipe
  manual optimization   pipeline-parallel engine
  DDP, grad accum       1F1B, measured stage split,
                        architecture-free and extractable
```

`kohakuwupipe` knows nothing about tokens or tokenizers. Anything that does
lives in `kohakuwullm`, so the engine can be lifted out and used elsewhere.

## Choose your path

### I want to train something now

- [Quick start](#quick-start)
- [docs/guides/writing-configs.md](docs/guides/writing-configs.md): the full knob catalogue
- [docs/concepts/presets.md](docs/concepts/presets.md): the ladder and how it was solved

### I want to change the architecture

- [docs/guides/extending.md](docs/guides/extending.md): adding a component, router, renderer or kernel
- [docs/concepts/architecture.md](docs/concepts/architecture.md): backbone, components, MoE, packing

### I want to understand the performance work

- [docs/performance/performance.md](docs/performance/performance.md): measured results on RTX 5090
- [docs/performance/gemm.md](docs/performance/gemm.md): beating cuBLAS on one shape, and the cost model
- [docs/internals/kernel-convention.md](docs/internals/kernel-convention.md): how a kernel gets written here
- [docs/internals/mxfp8.md](docs/internals/mxfp8.md): what is swapped, what is verified, what is not

### I want to scale past one card

- [docs/internals/pipeline.md](docs/internals/pipeline.md): pipeline parallelism and the measured split
- [docs/kohakuwupipe/](docs/kohakuwupipe/): the extractable engine's own docs

### I want to ship a trained model

- [docs/guides/generation.md](docs/guides/generation.md): sampling, KV cache, decode speed
- `scripts/tools/to_gguf.py` and `scripts/tools/to_hf.py`

### I want to work on the framework

- [CLAUDE.md](CLAUDE.md): coding conventions, and the two architectural rules
- [docs/internals/kernel-dev.md](docs/internals/kernel-dev.md): the five budgets
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Selected measurements (4x RTX 5090, sm_120)

Every figure below was measured on this box, with both arms of every comparison
on the same card, and the losing configuration reported as readily as the winning
one.

- **MXFP8 is worth more the larger the model.** At micro 8192 under 4-card
  pipelining: dense 1.081x, 1.139x and 1.221x at 200M, 500M and 1B, and sparse
  1.533x, 1.661x, 1.895x and **2.308x** at MoE-1B, 2B, 5B and 8B. The loss gap
  over 400 steps is 1.2x the bf16-against-bf16 noise floor, which puts it inside
  run-to-run scatter.
- **Where that came from, separated rather than claimed.** Measured as the ladder
  actually built: eager experts to grouped bf16 is **5.7x**, grouped bf16 to
  grouped fp8 is **1.49x**, grouped fp8 to fully fused is **1.65x**. The shipped
  bf16 baseline is already grouped, so the end-to-end ratios measure only the last
  two.
- **On MoE, fp8 also saves 40 to 44 percent of peak memory**, because the fused
  epilogues never materialize the `(tokens*top_k, hidden)` intermediates. On dense
  it costs about 2 bytes per swapped parameter instead. The two families are
  opposite and neither number transfers to the other.
- **FlashAttention-4 does not run on consumer Blackwell.** It is built on TMEM,
  an sm_100 feature. FA2-class is the ceiling, and PyTorch 2.13's `varlen_attn`
  is exactly that, natively, with a trainable backward.
- **fp16 accumulation is 1.4x to 1.55x faster** than fp32 accumulation on GeForce
  tensor cores, 325 against 232 TFLOP/s. Error grows with reduction depth and
  split-K buys it back. Used only in the MoE grouped GEMM, never in a backward.
- **The LM head is a memory problem.** At vocab 65536 and 16k tokens the naive
  path takes 12.61 GiB against the chunked path's 0.93 GiB, and the chunked path
  is the *most* accurate of the three rather than the least.
  `F.linear_cross_entropy(options=None)` silently means the materializing path.
- **Packing is worth 2.8x to 5.8x** forward at the length spread this corpus has,
  rising with the padding fraction it removes.
- **Pipelining beats DDP on this box**, and the reason is the fabric. There is no
  NVLink, so every pair goes over the PCIe host bridge, and a ring all-reduce of
  the whole gradient is the access pattern that handles worst. Pipeline boundary
  sends are 12 to 29 MB point-to-point and overlap with compute. DDP rows show 4
  to 16 percent step spread where pipeline rows sit at 0.1 to 0.6 percent.
- **Decode is launch-bound, and a static KV shape fixes it.** Returning a growing
  cache slice gives every step a new shape, so no compiled kernel set is ever
  reused. Scattering at a device-side position and masking instead takes decode
  from 35.80 to 8.66 ms per step at 8 rows, a 4.1x speedup with identical tokens.

Full details and the figures live in
[docs/performance/performance.md](docs/performance/performance.md).

## FAQ

**Does this need a cluster?**
No. It targets a single box with a few consumer cards. The reference machine is
4x RTX 5090, and the pipeline trainer exists because 32 GB cards and no NVLink is
the constraint that shaped this repo.

**Why write your own kernels instead of using Liger or FlashAttention?**
Where an existing kernel wins, it is used. `varlen_attn` from PyTorch is the
training default. The Triton kernels exist where measurement said they were
worth it, and each one carries a precision test against an fp64 reference in both
fp16 and bf16.

**Is MXFP8 actually safe to train in?**
On this hardware and these shapes, measured rather than assumed. The loss gap
against bf16 over 400 steps sits inside the bf16-against-bf16 noise floor. The
attention kernel is a separate question and is documented separately in
[docs/internals/mxfp8-attention.md](docs/internals/mxfp8-attention.md).

**Can I run a trained model outside this repo?**
Yes, two ways. `to_gguf.py` writes a llama.cpp model, quantizable to Q8_0 at
0.0041 mean KL divergence and 98.4 percent top-token agreement against f16.
`to_hf.py` writes a transformers repository with its own architecture, loaded
with `trust_remote_code=True`.

**Why is the tokenizer 65536 tokens?**
DeepSeek-V4's tokenizer pruned to 64k merges plus 1536 special slots. The power
of two matters for the chunked cross-entropy head and for MXFP8 alignment.

**What is `kohakuwupipe`?**
The pipeline-parallel engine, kept architecture-free so it can be lifted out of
this repo and used on any model. It owns stages, schedules, checkpoints and
collectives. It knows nothing about tokens.

## Requirements

Python 3.10 or newer with 3.13 recommended, torch 2.13 or newer, triton 3.7 or
newer, and an NVIDIA GPU. The kernels target sm_120 and are measured there.

## Related projects

Sibling repositories sharing the same conventions and stack:
[KohakuLatentMaid](https://github.com/KohakuBlueleaf/KohakuLatentMaid) for
diffusion and
[KohakuTerrarium](https://github.com/KohakuBlueleaf/KohakuTerrarium) for agents.

## License

Apache-2.0. See [LICENSE.md](LICENSE.md).
