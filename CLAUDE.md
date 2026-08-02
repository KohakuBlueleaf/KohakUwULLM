# KohakUwULLM

An extensible decoder-only LLM training framework. Sibling to
[KohakuLatentMaid](https://github.com/KohakuBlueleaf/KohakuLatentMaid) (diffusion)
and [KohakuTerrarium](https://github.com/KohakuBlueleaf/KohakuTerrarium) (agents);
same house conventions, same tech stack.

## Project Overview

The framework provides slots; you select or drop in what you need. Reproducing a
Llama-shaped dense model, a Gemma-shaped sliding-window model, or a DeepSeekMoE
sparse model is a **config**, not a code change. Adding a new norm / MLP /
attention / position encoding / router is a registry entry or a dotted import
path -- no core edits.

Targets: dense models up to ~1B, MoE up to ~3B, trained on 4x RTX 5090 (32 GB,
sm_120). Primary corpus is the KohakuVault caption/tag databases, rendered into
TIPO-style prompt-generation examples.

## Code Conventions

### File Organization
- Source: `src/kohakuwullm/`
- Training / bench / tokenizer scripts: `scripts/`
- KohakuEngine config files: `configs/`
- Docs: `docs/`
- Max lines per file: 600 (hard max 1000). Split modules before they grow.
- Highly modularized -- one responsibility per module.

### Coding style rules

**Code files carry only *what*. Every *why* and *how* belongs in `docs/`.**
Source is not a place to write documentation.

1. **Docstring — what a module / class / function is for.** A straightforward
   description, and the caller-facing contract (arguments, units, dtype/layout
   requirements, what is returned). Typically one to three lines.
2. **Comment — what a piece of code is doing.** Name the logical step, which
   matters most in a kernel, or say what the code cannot say for itself. One
   line; two at the outside.
3. **NO rationale, derivation, measurement, history, alternative-considered or
   memo/noting comments anywhere in source.** If it explains *why this way* or
   *how it works*, it goes to `docs/` and the code gets at most
   `See docs/<file>.md`.
4. NO editorial comments, and none addressed to a future reader as a note.
5. AVOID long multi-line inline comments. A comment longer than the code it sits
   above is wrong.

### Import Rules
1. No imports inside functions (except optional deps and lazy imports that avoid
   long init time -- e.g. Lightning in `training/__init__.py`).
2. Grouping order: built-in, third-party, `kohakuwullm.*`.
3. Within a group: `import` before `from`, shorter dotted paths first, then
   alphabetical.

### Python Style
- Target Python 3.10+. Modern type hints: `list`, `dict`, `tuple`, `X | None`.
  Never `List`, `Dict`, `Optional`, `Union` from `typing`.
- Prefer `match-case` over deeply nested `if-elif-else`.
- `black src/ scripts/` and `ruff check src/ scripts/` before commit.

### The two architectural rules

**Select, don't dispatch.** Configuration resolves a concrete class or callable
**once**, at build time, via `build(spec, REGISTRY)`. The result is held as a
plain attribute and called directly. There is no per-step `if mode == ...`
anywhere in a training loop. If you find yourself adding a runtime branch on a
config value, the branch belongs in `__init__`.

**The backbone is a pure function.** `LMBackbone` maps `(tokens, seq_info) ->
hidden`. It knows nothing about the loss. Objectives live in `LMHead` and the
trainer, so changing the objective never touches the trunk.

### Numerics rules (this repo trains in low precision)

- **Never trust a low-precision scalar reduction.** Summing 16k bf16 terms loses
  several percent. Reduce in fp32 -- `LMHead.token_loss` asks the fused CE for
  `reduction="none"` and reduces itself, precisely for this reason.
- **Every Triton kernel needs a precision check** against an fp64 reference, in
  *both* fp16 and bf16, forward and backward. See `docs/internals/kernel-dev.md`.
- **Judge error in ULP, not absolutely.** An absolute tolerance that passes in
  fp32 is meaningless in bf16. Use `bench.timing.ulp_error`, and pick its mode:
  `"elementwise"` for elementwise kernels, `"rms"` for GEMMs and reductions
  (where a near-zero output is cancellation, not a small true value).
- **Token counts are int64.** A run passes 2^31 tokens in under an hour and a
  float32 counter stops incrementing at 2^24.

### Post-impl tasks
1. Verify the rules above, especially in-function imports.
2. `black` and `ruff`.
3. Logically separated commits.

### Testing

**`tests/` is removed for now and will be added back deliberately.** Do not
recreate the directory, do not add test files, and do not treat a missing test
as a blocker. Docs that cite `tests/test_*.py` name a specification, not a path
(see `docs/README.md`).

Verify with a throwaway script under `/tmp` instead, and hold it to the same bar:
it must **fail on the unfixed code**. A check that passes before and after
proves nothing.

### Audit loop (multi-step work: REQUIRED)

Do not stop at "it runs". Loop until it converges:

1. **Implement** the slice.
2. **Construct the negative case** -- the bug you'd accidentally introduce --
   and confirm the current code is actually wrong on it.
3. **Run lint.**
4. **Audit the diff** for clear bugs (typos, off-by-ones), integrity bugs
   (invariants broken, state drifted), and behavior bugs (does what's typed but
   the wrong thing for the spec).
5. **If a bug slipped past a check, fix the check first**, confirm it fails on
   the unfixed code, then fix the bug.
6. **Loop.**

Also audit your *benchmarks* this way. Three real bugs in this repo's history
were benchmark bugs that made the code look wrong: a non-leaf input tensor that
made every timed backward fail, a per-element ULP metric that reported 24000 ULP
for a numerically perfect GEMM, and an fp64 reference that OOMed at 131k tokens.
A measurement you have not audited is not evidence.

## Architecture

### The one mental model

1. A **backbone** is `model(tokens, seq_info) -> hidden`. Norm / MLP / attention /
   position-encoding are swappable components; Llama, Gemma and DeepSeekMoE are
   **presets** over one backbone, not separate classes.
2. A **`SeqInfo`** says how the batch is laid out -- packed (varlen, the training
   path) or padded (eval). Only attention reads it; everything else is a last-dim
   op that works on both.
3. A **trainer** (Lightning, manual optimization) wires data -> backbone -> head,
   with schedules, compile and grad accumulation selected once and called directly.
4. Everything swappable lives in a **registry**; `build(spec, REGISTRY)` resolves
   a name / dotted path / dict / class / instance at build time.

### Packed varlen is the training layout

Every sequence is concatenated into one flat token axis; `cu_seqlens` carries the
document boundaries. Nothing is padded. For TIPO-shaped data (50-600 token
samples against a 2048 context) a padded batch would be ~80% padding, so this is
close to a 4x throughput multiplier before any kernel work -- measured at
**2.8x to 5.8x forward**, rising with padding fraction, in
`scripts/bench/kernel/attention.py`. The fwd+bwd column is not quotable until
that bench is re-run: it was measured without a per-iteration grad reset.

The invariant that makes it correct: **attention must never cross a document
boundary.** `varlen` gets this from the kernel; the SDPA and Flex fallbacks build
an explicit block-diagonal mask. Nothing currently pins it, and a backend that
forgets the boundary still trains -- just worse, for reasons nothing in the loss
curve explains. Re-check it by hand after touching any attention backend.

### Hardware notes (RTX 5090, sm_120)

- **FlashAttention-4 does not run here.** FA4 is built on TMEM, an sm_100
  (B200) feature. Consumer Blackwell is sm_120 and keeps the sm_80-era `mma.sync`
  model. FA2-class is the ceiling; `torch.nn.attention.varlen.varlen_attn` is that,
  natively, with a trainable backward.
- **fp16 accumulation is ~1.5x faster than fp32 accumulation** on GeForce tensor
  cores. Measured: 325 vs 210 TFLOP/s. Error grows with reduction depth K;
  split-K buys it back. Only worth it where we own the kernel (the MoE grouped
  GEMM), and never in a backward.
- **The LM head is memory-bound, not compute-bound.** At vocab 65536 and 16k
  tokens: naive 12.61 GiB, chunked 0.93 GiB, and the chunked path is the *most*
  accurate of the three, not the least. `options=None` on
  `F.linear_cross_entropy` silently means *reference* (materializing) -- the
  chunked path needs an explicit `LinearCrossEntropyOptions()`.

## Project Structure

```
src/kohakuwullm/
  registry.py          build(spec, REGISTRY) -- the one dispatch point
  utils.py             import_class, parameter counting, schedule autofill
  compile_utils.py     per-module / whole-model torch.compile
  models/
    presets.py         LMArchConfig + named presets (dense + MoE)
    backbone.py        embeddings -> blocks -> final norm; owns the block loop
    block.py           pre-norm attention + feed-forward, optional post-norms
    head.py            fused linear+cross-entropy, z-loss, soft-cap
    components/
      seqinfo.py       packed vs padded layout descriptor
      attention.py     varlen / sdpa / flex, GQA, QK-norm, sinks, windows
      mlp.py           SwiGLU / GeGLU / GELU
      moe.py           DeepSeekMoE: shared + routed experts, aux-loss-free balance
      norm.py          RMSNorm / LayerNorm / Gemma / DyT
      posenc.py        RoPE (+ linear / NTK / YaRN scaling), NoPE
  kernels/             Triton: rmsnorm, swiglu, grouped_gemm, zloss
  data/
    vault.py           KohakuVault record sources (danbooru + path-keyed)
    tipo.py            record -> (user_text, output_text)
    packing.py         tokenize, loss-mask, pack, split
  training/
    trainer.py         Lightning LMTrainer (manual optimization)
    optim.py           parameter grouping, muP, optimizer registry
    pipeline.py        cost-balanced pipeline stage split
    callbacks.py       sample previews, throughput / MFU
  bench/               shared timing, precision metrics, plot styling
  tokenizer/prune.py   BPE vocabulary pruning
```

## Benchmarks

`scripts/bench/` is part of the deliverable, not a scratch area. Every figure
shows **throughput and accuracy together** -- a kernel that is fast and wrong is
not a result, and a chart reporting only the fast half invites exactly that
mistake. Run them after any kernel change:

```bash
.venv/bin/python scripts/bench/kernel/hgemm_acc.py    # fp16 vs fp32 accumulation
.venv/bin/python scripts/bench/kernel/attention.py    # backends, packing, GQA, windows
.venv/bin/python scripts/bench/kernel/kernels.py      # norm, swiglu, MoE GEMM, head
.venv/bin/python scripts/bench/data/data.py         # loader throughput
```

## Development Setup

```bash
uv pip install -e ".[dev,bench]"
kogine run scripts/train/lm.py --config configs/lm/debug.py
```

Never use `sys.path` hacks in scripts; always import from the installed package.
