# Architecture

This document explains how a model is put together here, in the order you would
need to know it: the shape of the whole thing first, then each swappable slot,
then how a preset composes them. Read it end to end once and you should be able
to add a component without reading the backbone.

The companion documents are [extending.md](../guides/extending.md) for the mechanical
contracts a new component must satisfy, [presets.md](presets.md) for the model
ladder, [writing-configs.md](../guides/writing-configs.md) for the knobs, and
[performance.md](../performance/performance.md) for where the measured numbers below came from.

---

## 1. The one mental model

1. A **backbone** is `model(tokens, seq_info) -> hidden`. Norm, feed-forward,
   attention and position encoding are swappable components; a Llama-shaped
   dense model, a Gemma-shaped sliding-window model and a DeepSeekMoE sparse
   model are **presets** over one backbone, not separate classes.
2. A **`SeqInfo`** says how the batch is laid out — packed (variable-length
   documents concatenated, the training path) or padded (`(B, S)`, the eval
   path). Only attention reads it.
3. A **head** owns the objective. `LMHead` fuses the vocabulary projection into
   cross-entropy so the logits never exist.
4. Everything swappable lives in a **registry**, and `build(spec, REGISTRY)`
   resolves a name, dotted path, dict, class or instance **once**, at build time.

Two rules follow from that, and they are the ones worth internalising.

**Select, don't dispatch.** Configuration resolves a concrete class or callable
once and the result is held as a plain attribute and called directly. There is no
per-step `if mode == ...` in a training loop anywhere. If you find yourself adding
a runtime branch on a config value, the branch belongs in `__init__`. You will see
this pattern repeatedly below: `TopKRouter.use_fused`, `MoEMLP._routed`,
`RoPE.impl`, `LMHead.kernel` are all decisions made at construction and never
revisited.

**The backbone is a pure function.** `LMBackbone` knows nothing about the loss, owns
no cache and holds no sampling loop — a `KVCache` is passed in by the caller and
written through, never kept (§15). That is what lets the objective change
— plain cross-entropy, distillation, an auxiliary head — without touching the
trunk.

---

## 2. The backbone

```
tokens ──► embed ──► block 0 ──► ... ──► block L-1 ──► final_norm ──► hidden
                        ▲                                                │
                    posenc (built once per forward)                      ▼
                                                                      LMHead
```

`LMBackbone.forward` owns the block loop explicitly rather than delegating to an
`nn.Sequential`, because the loop is where gradient checkpointing, per-layer
window selection, hidden-state taps and pipeline stage boundaries all attach.
Keeping it explicit means none of those need the blocks to cooperate.

The position encoding carrier is built **once per forward** and handed to every
layer, so the trigonometry is computed once regardless of depth.

Nothing in `models/` imports `transformers`. That is deliberate: the tokenizer is
worth borrowing, the modeling code is not — it hard-codes an attention interface,
a cache layout and a config schema that this framework replaces.

### The block

```
x = x + attn(norm(x))
x = x + ffn(norm(x))
```

Pre-norm on both branches, with optional **post-norms** on each branch output
(Gemma 2/3, OLMo 2). A post-norm costs one more reduction per branch and buys a
lot of stability at depth, because it bounds what a branch can add to the residual
stream regardless of how the branch's own scale drifts.

`residual_scale` multiplies each branch before it joins the stream; `depth ** -0.5`
is the usual choice for very deep stacks. It is a *persistent* forward multiplier,
which is what distinguishes it from the initialization scaling below.

`DecoderBlock` does not know whether its feed-forward is dense or sparse — `mlp` is
built from a spec, so an MoE layer is `{"name": "moe", ...}` and nothing in the
block changes.

### Initialization

`LMBackbone.initialize_weights` is normal init with two corrections.

**Depth scaling.** Every residual branch adds variance to the stream, so with `L`
layers the stream grows like `sqrt(L)` unless the branch outputs shrink to match.
The two *output* projections — attention `o_proj` and feed-forward `w_out` — are
scaled by `1/sqrt(2L)`, which holds residual variance roughly constant with depth.
This is the GPT-2 rule and still what everyone uses. It is an *initial condition*,
not a persistent multiplier: training is free to grow the branch back, which is
why depth-wise hyperparameter transfer needs `residual_scale` as well and not just
this.

**Width scaling (muP).** With `mup_base_dim` set, every hidden matrix is
initialized at `init_std * sqrt(base / fan_in)` instead of a flat `init_std`.
Without it the residual-to-branch ratio at step 0 drifts with width, so a learning
rate tuned at one width starts the next one from a different regime. The embedding
is excluded, because its fan-in is the vocabulary and does not scale with the
model. The init exponent is a property of the *architecture*, not the optimizer:
AdamW, Muon and Shampoo share it (arXiv 2602.20937 Table 2).

The MoE expert stacks need a special case. They are `nn.Parameter`s of rank 3, not
`nn.Linear` children, so the generic `apply(_basic)` pass never sees them. The
backbone finds them **structurally** — "does `mlp` have a 3-D parameter called
`w_out`?" — rather than by `isinstance(mlp, MoEMLP)`, because a formulation that
*wraps* an inner MoE (`latent_moe`) exposes the same stacked weights without
inheriting, and a type check would send it down the dense branch to dereference
`.weight` on a `Parameter`.

---

## 3. `SeqInfo`: two layouts, one model

| | packed (training) | padded (eval / generation) |
|---|---|---|
| hidden shape | `(T, D)` | `(B, S, D)` |
| `cu_seqlens` | `(N+1,)` document offsets | `None` |
| `position_ids` | `(T,)`, restarts per document | `(B, S)` |
| padding computed on | none | up to ~80% |

Only attention reads `SeqInfo`. Norms, feed-forwards and the head are all last-dim
ops, so they serve both layouts unchanged — which is why supporting two layouts
costs one dataclass instead of two model classes. `position_ids` restarting at 0
per document is what makes RoPE correct across a packed batch.

Packed varlen is the training layout because rendered TIPO samples run 50–600
tokens against a 2048 context: a padded batch would be ~80% padding, so packing is
close to a 4x throughput multiplier before any kernel work. Measured at **6.3x**
at 82% padding in `scripts/bench/kernel/attention.py`
(`out/bench/kernel/attention/attention.json`, `sweep="ragged"`,
`pad_frac=0.8204`): 5.4-5.8x forward, 6.0-6.3x forward+backward, consistent
across all four backends that run packed.

### The invariant

**Attention must never cross a document boundary.**

`varlen` and `triton` get this from the kernel, which takes `cu_seqlens` directly.
The SDPA and Flex fallbacks build an explicit block-diagonal mask from `_doc_ids`.
Forgetting it produces a model that still trains, just worse, for reasons the loss
curve will not explain — and only on the slow fallback path, which is the worst
place for a correctness difference to hide.

`tests/test_models.py::test_packed_attention_does_not_cross_documents` pins it by
perturbing document 0 and asserting document 1's output is bit-identical.

---

## 4. Selecting components

Everything below is chosen by a *spec*: a registry name (`"swiglu"`), a dotted
import path (`"my_pkg.MyMLP"`), a dict (`{"name": "moe", "num_experts": 64}`), or a
class. `build(spec, REGISTRY, **kwargs)` resolves it once, at construction.

| slot | registry | built-ins |
|---|---|---|
| norm | `NORM` | `rmsnorm`, `rmsnorm_triton`, `layernorm`, `gemma_rmsnorm`, `dyt` |
| feed-forward | `MLP` | `swiglu`, `swiglu_triton`, `geglu`, `gelu`, `moe`, `latent_moe` |
| attention | `ATTENTION` | `varlen`, `triton`, `sdpa`, `flex` |
| position | `POSENC` | `rope`, `ndrope`, `ggrope`, `none` |
| MoE router | `ROUTER` | `topk`, `sinkhorn`, `expert_choice`, `relu` |

Importing `kohakuwullm.models.components` is what registers the built-ins; the
package `__init__` imports every module for that side effect alone.

---

## 5. Normalization

All norms operate on channel-last `(..., D)` tensors and are shape-agnostic in
every other dimension, so one module serves both layouts without a reshape.

- **`rmsnorm`** — the default; every modern decoder LM uses it.
- **`rmsnorm_triton`** — the fused Triton implementation. Which of the two is
  faster depends on shape and on whether the block is compiled, so both stay
  registered and `scripts/bench/kernel/kernels.py` measures them.
- **`layernorm`** — the classic, for reproducing older recipes.
- **`gemma_rmsnorm`** — zero-centered weight, `(1 + w)`. The zero-init weight means
  the layer starts as pure normalization and the scale is learned as a deviation,
  which interacts better with weight decay than a one-init weight does.
- **`dyt`** — Dynamic Tanh, `w * tanh(a * x) + b`. Normalization-free, so it fuses
  into neighbouring elementwise work under `torch.compile`. Registered for
  ablations.

`RMSNorm(affine=False)` is also what QK-norm uses inside attention.

---

## 6. Position encodings

The contract is two methods:

- `prepare(position_ids, device, dtype)` returns a *carrier*, or `None`;
- the carrier exposes `apply(q, k)`, rotating `q`/`k` in place of any additive
  term. `q` and `k` arrive as `(..., H, D)` with the position axis immediately left
  of the head axis, which covers packed `(T, H, D)` and padded `(B, S, H, D)`
  alike.

### RoPE

`rope` is the default. Scaling for context extension (`linear`, `ntk`, `yarn`) is
applied to the inverse frequencies **at build time**, never per step:

- **linear** divides every inverse frequency by `factor` — uniform interpolation.
- **ntk** raises the base instead, by `factor ** (d / (d - 2))`, so the highest
  frequency is untouched and the lowest is interpolated by `factor`.
- **yarn** interpolates per frequency band, ramping between the two, and also
  rescales attention logits by `0.1 * log(factor) + 1` to hold softmax entropy
  constant as the context grows.

`partial_rotary_factor < 1` rotates only a prefix of each head and passes the rest
through unchanged (GPT-NeoX / Phi style).

The rotation itself has **three peer implementations** — `triton`, `compiled` and
`eager` — behind one signature. `RoPE` takes the name as `impl`, resolves it to a
callable once in `__init__`, and hands it to each `RotaryCache` it builds:

```python
ARCH_OVERRIDES = {"posenc": {"name": "rope", "impl": "triton"}}
```

`triton` is the default. At 131072 tokens it and `compiled` are at parity — within
the repeat spread on both forward and forward+backward — and identical in accuracy
at 0.5 ULP, so the default is chosen on the things that are not throughput: no
compile step, no shape guards, no recompile budget. Pick `compiled` when the
rotation sits inside an already-compiled region, where Inductor can fuse it into
its neighbours and a Triton call cannot. The measured table and the reason
`torch.compile`d eager counts as a kernel here are in
[../internals/kernels.md](../internals/kernels.md).

### `none` (NoPE)

A causal decoder can infer position from the mask alone, and dropping RoPE on a
subset of layers is a real length-generalization technique (SmolLM3 interleaves
NoPE layers). Registered so a per-layer pattern can select it.

### N-D RoPE and ggRoPE

Standard 1-D RoPE rotates channel pair `i` by `w_i * t`. The N-D generalization
replaces the scalar position with a projection: pair `i` rotates by
`w_i * <u_i, t>` for a unit direction `u_i` and a position vector `t` in R^N.
Axial RoPE is the special case where every `u_i` is a standard basis vector.

**Why axial is not good enough.** With axial 2-D RoPE the first half of a query is
rotated only by `x` and the second half only by `y`, so the first half's
contribution to an attention score is *identical regardless of the key's y*. The
query therefore cannot ask for "the token two to my left and one up" as a single
relative offset — it can only ask about each axis independently. Arbitrary
directions fix this, because each pair measures position along its own direction
and the set of directions spans the plane.

**ggRoPE** (Golden Gate RoPE) picks those directions deterministically: direction
`i` sits at angle `i * pi / phi` with `phi` the golden ratio. The golden angle is
the standard low-discrepancy choice — it never repeats and fills the circle as
evenly as possible for *any* prefix length, so a truncated set of directions is
still well spread. That gives mixed-RoPE's expressivity without mixed-RoPE's
sensitivity to frequency initialization, since nothing is learned.

Above two dimensions the directions come from the generalized golden ratio (the
positive root of `x^(N+1) = x + 1`) used as a Kronecker sequence, mapped through
the inverse Gaussian CDF and normalized. The inverse CDF matters: normalizing
uniform lattice points directly would clump the directions toward the cube corners.

Frequency magnitudes are log-spaced over `[omega_min, omega_max]`. With
coordinates normalized to `[-1, 1]` the reference range is
`omega_min in [0.2, 1.0]` and `omega_max in [20, 100]`. A fraction of frequencies
can be zeroed (`zero_freq_ratio`); those pairs carry no position at all and act as
the head's NoPE channels, which measurably improved length generalization in the
reference experiments. The *lowest* frequencies are the ones zeroed, since they
already vary least across a sequence and so carry the least positional information
to lose.

For text these are drop-in alternatives to `rope`: a 1-D position is just `N = 1`,
where ggRoPE degenerates to standard RoPE with a fixed frequency ladder. They earn
their keep when positions are genuinely multi-dimensional — image patch grids, or a
`(document, offset)` pair.

---

## 7. Attention

One `BaseAttention` owns the projections, the GQA head layout, QK-norm and sinks.
The four registered subclasses differ **only** in the inner kernel call.

### Shared machinery

**GQA** (`kv_heads < heads`) is the default. Measured: 20x smaller KV cache and
*faster*, because the KV projections shrink. Note that GQA saves memory traffic,
not arithmetic — every query head still scores against its group's key, which is
why the FLOP model below does not divide by the group size.

**QK-norm** — RMSNorm over `q` and `k` before the kernel — is on by default. It is
what Gemma 3, Qwen 3 and OLMo 2 all adopted for training stability, and it
*replaced* logit soft-capping rather than supplementing it.

QK-norm and RoPE both run in fp32 (autocast promotes normalization, and the
rotation is worth full precision regardless), so `q` and `k` come back wider than
`v`. They are cast back to `v`'s dtype at the end, because the flash kernels reject
a mixed-precision triple and accept only fp16/bf16 at all.

**Attention sinks** — a learnable per-head logit that sits in the softmax
denominator without contributing a value — are implemented through the log-sum-exp
the kernel already returns:

```
out_with_sink = out * sigmoid(lse - sink)
```

The kernel returned `out = sum_j p_j v_j` with `p` normalized by `exp(lse)`. Adding
a sink column of logit `s` that carries no value rescales every probability by
`exp(lse) / (exp(lse) + exp(s))`, i.e. `sigmoid(lse - s)`. This is algebraically
exact, costs one extra elementwise op, and — crucially — works *with* the flash
kernel instead of forcing a materialized score matrix.

**Sliding windows** are per layer. `sliding_window` + `global_layer_every` gives the
Gemma-3 (5:1) / OLMo-3 (3:1) interleave; the last layer is always global, because it
is the one that feeds the head. `window_pattern` is the general form — an explicit
list cycled over the depth, so widths can vary rather than only local-or-global.
`[512, 1024, 2048, None]` gives an increasing-receptive-field stack with every
fourth layer global.

### The four backends

| backend | kernel | use |
|---|---|---|
| `varlen` | PyTorch 2.13 `varlen_attn` (FA2) | training default, full-causal layers |
| `triton` | ours, `kernels/attention/flash_attn.py` | sliding-window layers |
| `sdpa` | `F.scaled_dot_product_attention` | reference and eval |
| `flex` | compiled FlexAttention | masks the flash kernels have no flag for |

Measured on an RTX 5090 (sm_120) in `scripts/bench/kernel/attention.py`:

- **full-causal layers → `varlen`.** Its fused backward reads q/k/v once; ours takes
  two passes and loses ~19% on forward+backward at 8k.
- **sliding-window layers → `triton`.** It skips out-of-window key blocks more
  aggressively and runs up to 20% faster on the forward at windows 1k–4k.

`LMArchConfig.attn_sliding` exists precisely so a mixed local/global stack can take
both wins: it overrides `attn` on sliding layers only.

Both flash backends fall back to SDPA when handed a padded batch or a dtype the
kernel does not take (fp32 debugging runs), with the document mask built explicitly
so the fallback matches numerically.

All four also share one path that is none of these: when a `KVCache` is passed,
every backend attends over the cached prefix through `BaseAttention.attend_cached`.
See §15.

`flex` must be compiled. Uncompiled, `flex_attention` falls back to an eager path
that materializes the full score matrix — at 16k packed tokens by 20 heads that is
a 20 GiB allocation and an instant OOM. The fused kernel only exists after
`torch.compile`, so compilation is mandatory, not an optimization; it is compiled
once per process and shared across layers.

`sdpa`'s packed path allocates a `(T, T)` bool mask, which is exactly the quadratic
allocation the varlen kernel exists to avoid. It is the reference and the eval
path, not a training path.

**On FlashAttention-4.** FA4 is built on TMEM, an sm_100 (B200) feature. Consumer
Blackwell is sm_120 and keeps the sm_80-era `mma.sync` model, so FA4 has no sm_120
path upstream (Dao-AILab issue #2307, unanswered). Community PRs #2329/#2330/#2333
add one, and a patched build measures ~25% over FA2 on a 5060 Ti — an out-of-tree
option worth revisiting if attention ever becomes the bottleneck. FA2-class is the
ceiling here, and `varlen_attn` is that, natively, with a trainable backward.

---

## 8. Dense feed-forward

All feed-forwards are last-dim ops, so they serve both layouts unchanged.

`swiglu` is the default for every modern decoder LM. The gate and value share one
fused `w_in` projection — one GEMM instead of two, which matters more than it looks
at small hidden sizes where launch overhead dominates. `fused_gate=False` splits
them into separate `gate_proj` / `up_proj` parameters, which is what most Hugging
Face checkpoints store.

`resolve_hidden` applies the conventional `2/3` correction for GLU variants: a GLU
spends its parameter budget over three matrices instead of two, so the correction
keeps the parameter count equal to a plain `ratio * dim` MLP. An explicit `hidden`
overrides it.

**`swiglu`, not `swiglu_triton`, is the default.** Compiled, the eager version is
31% faster on forward+backward, and the difference is entirely the backward: the
kernel writes two gradients that the chunk backward concatenates, where Inductor
fuses it. `swiglu_triton` stays registered and wins on an uncompiled path, where it
saves the `silu(gate)` intermediate — the single largest activation in an
uncompiled feed-forward.

`geglu` (tanh-approximate GELU gate) and `gelu` (plain two-layer, no gating) are
registered for comparability with older recipes.

---

## 9. Sparse feed-forward (MoE)

`MoEMLP` slots into the same `MLP` registry as a dense feed-forward, so making a
layer sparse is `{"name": "moe", ...}` and nothing in the block or backbone moves.

The formulation is `shared(x) + sum_k w_k * expert_k(x)`, following DeepSeek-V3.

### The three DeepSeekMoE pieces

**Fine-grained routed experts.** Many narrow experts instead of a few wide ones,
`top_k` of them per token. Narrow experts give the router more combinations to
express at the same active-parameter cost.

**Shared experts.** A small always-on expert every token passes through, absorbing
the common-to-everything computation so routed experts do not each have to relearn
it. Their widths add to the routed one.

**Aux-loss-free load balancing.** A per-expert bias is added to the gate score *for
selection only* — never to the weight a token's output is scaled by — and is nudged
toward whichever experts are underloaded once per optimizer step. This balances the
load without an auxiliary gradient competing with the language objective, which is
what the older aux-loss approach does. The classic loss is still available
(`aux_loss_weight`) but defaults off; it costs nothing extra to select, and it
reaches the loss under pipeline parallelism through a second boundary stream —
see [moe-router-loss.md](../internals/moe-router-loss.md).

Three details of the bias update matter:

- It is a **buffer, not a parameter.** It is updated by a rule, not a gradient, so
  it must not receive weight decay or optimizer state.
- The load counter is **all-reduced** before the update. It fills per rank, but the
  quantity being balanced is the load over the whole step; without the reduce the
  bias is driven by `1/world` of each batch and the other ranks' imbalance never
  reaches it — while DDP's buffer broadcast then overwrites every rank with rank
  0's anyway.
- The step is `sign(mean - load)`, not the raw error, so the step size stays bounded
  regardless of how skewed the load got. That is what keeps it stable.

The update runs once per **optimizer** step, not per micro-batch, so the load
estimate covers the whole step rather than chasing accumulation noise.

### Routing

`TopKRouter` supports three gate score functions:

| `score_func` | used by | property |
|---|---|---|
| `sigmoid` | DeepSeek-V3, GLM-5.2, MiMo-V2.5, MiniMax-M3 | decouples experts; scores do not compete |
| `softmax` | DeepSeek-V2, Qwen3, OLMoE | experts compete for a fixed budget |
| `sqrtsoftplus` | DeepSeek-V4 | unbounded above; preference is not clipped at 1 |

`sqrtsoftplus` is `sqrt(softplus(x))`. The square root keeps growth sub-linear in
the logit, which is what stops one expert's score running away the way a bare
softplus would; being unbounded above is what lets a strongly-preferred expert
express that preference instead of saturating at 1 the way sigmoid does.

`group_limited` routing (DeepSeek-V3 node-limited) restricts a token to
`topk_groups` of `n_groups` expert groups before the final top-k. A group's strength
is its top-2 sum rather than its max alone, which is less jumpy.

**The fused router.** `TopKRouter.use_fused` is resolved in `__init__` and covers
the common case in a single Triton launch. It requires `sigmoid` or `sqrtsoftplus`,
no groups, and at most 128 experts. Both fused gates are elementwise, which is what
lets the backward work from the `top_k` selected scores alone; a softmax gate
couples all `E` in the *forward*, and group-limited routing needs the full row to
pick its groups, so both stay eager.

The two auxiliary losses (`aux_loss_weight`, `z_loss_weight`) need every expert's
score in *both* directions — the forward reduces the full score row, the backward
assembles a dense `(T, E)` gradient tile — and both are fused.
[moe-router-loss.md](../internals/moe-router-loss.md) has the derivation, the cost,
and how the term crosses a pipeline stage boundary.

Device-eligibility is checked separately, per call, because a module is built on
CPU and moved afterwards: a CPU tensor is a real input property rather than a
config value, Triton has no CPU backend, and the tests build MoE models on CPU on
purpose.

The fused path's gate multiply follows the activation dtype rather than upcasting
to fp32 the way the eager path does. That upcast bought nothing — a bf16 input is
represented exactly by the tensor cores either dtype lowers to, so the only
difference is summation order — and fp32 stays fp32, which Triton lowers to TF32,
the same thing eager gets under the repo-wide `set_float32_matmul_precision("high")`.

**Counting rows with `scatter_add`, never `bincount`.** `bincount` sizes its output
from `input.max()`, which is a device-to-host copy. One host stall per MoE layer per
micro-batch drains the CPU's run-ahead for the whole step, and makes the layer
impossible to CUDA-graph capture. Every router here uses `scatter_add_` into a
pre-sized tensor. `route()` hands those counts to `MoEMLP`, which needs exactly that
tensor to build the grouped-GEMM offsets — so passing it over is what lets the
dispatch drop its own count pass.

### Dispatch and the expert GEMMs

Tokens are sorted by expert so each expert owns a contiguous row range, then all
experts run in **one grouped GEMM launch** (`kernels/moe/grouped_gemm.py`). Expert
weights live as one stacked `(E, out, in)` tensor per matrix so the kernel can index
them by expert id without a gather.

The alternative — a Python loop of `E` small GEMMs — is **14x slower at 64 experts**
and is launch-bound long before it is compute-bound. `dense_fallback=True` selects
that loop with identical semantics; it is what the tests compare against.

With `mxfp8` enabled the routed experts go further, to a fully fused fp8 path
(GEMM1+SwiGLU, then GEMM2+gate+combine) that never materializes the
`(tokens*top_k, hidden)` intermediate: the gather becomes GEMM1's A-index and the
gate scale and scatter become GEMM2's epilogue. Worth 1.53–2.31x end to end on the
sparse rungs — see [mxfp8.md](../internals/mxfp8.md).

The bf16 path deliberately uses eager `silu(gate) * value` rather than the fused
`swiglu_mul`, even though the fused kernel would save the intermediate. This is the
**bf16 control arm** of an fp8 A/B, and the two arms should differ in dtype, not in
which elementwise kernel each happens to reach.

`active_parameters()` counts what one token actually touches: `top_k` experts plus
the shared ones plus the router matrix. `param_summary()` reports it separately from
the total, because active is what governs FLOPs and total is what governs memory,
and for a sparse model those differ by 4x.

### Router variants

None of these is the house default. They exist so a routing ablation is a config
change, because the 2026 open-weight MoE models do not agree on the important
choices.

**Shared experts: contested.** DeepSeek-V4 (384 routed + 1 shared) and Kimi K2.6
keep them; Qwen3-30B-A3B deliberately dropped them — 128 experts, 8 active, all
treated equally — as did Qwen3 generally. The argument for is that a shared expert
absorbs what every token needs so routed experts stop duplicating it; the argument
against is that it spends always-on FLOPs a well-balanced router would allocate
better. `num_shared=0` gets the Qwen shape.

**Granularity: diverging.** DeepSeek-V4 runs 384 routed experts, GLM-5.2 8-of-256,
Qwen3 8-of-128. Finer granularity gives the router more combinations at equal active
cost, and is the direction the frontier moved.

**`sinkhorn`** — balanced assignment by Sinkhorn normalization instead of a bias
nudge. Alternating row and column normalization drives the score matrix toward
doubly-stochastic, i.e. balanced *by construction* rather than by a correction that
lags a step behind. The costs are a few extra normalization passes per forward, and
that the balancing is *within the batch*: a batch that genuinely wants one expert
gets spread anyway. Assignment comes from the balanced matrix while the output
weight comes from the raw scores, so the balancing never distorts the output
magnitude.

**`expert_choice`** — experts pick tokens instead of tokens picking experts. Every
expert takes its top-`capacity` tokens, so load is *exactly* uniform with no
auxiliary loss and no bias: the balancing problem is defined away. Two consequences:
a token may be chosen by zero experts (it then passes through only the shared expert
and the residual), and the routing is not causal across the batch, which makes it
unsuitable for autoregressive inference without a per-position variant.

It is **not wired to `MoEMLP`**. It emits a flat `(token, expert)` pair list, while
the dispatch expects a `(T, slots)` index matrix and recovers a pair's token as
`pair_index // slots` — an identity that only holds when every token has the same
number of slots, which is exactly what expert choice gives up. Wiring it needs
`expert_sort` to take an explicit per-pair token index. `route()` raises with that
explanation rather than an `AttributeError` from the middle of a forward.

### ReMoE: ReLU routing

The discrete top-k is what makes a standard router non-differentiable — the gate
learns only through the `k` scores that happened to be selected, and the boundary
itself carries no gradient. ReLU routing removes the selection step entirely: an
expert is active iff its score is positive, so the gate is differentiable
everywhere and the number of active experts per token is free to vary.

Sparsity then becomes a *constraint* rather than a structure, enforced by an L1
penalty whose multiplier is steered toward the target activation rate:
`lambda *= alpha ** sign(target - observed)`. Load balancing folds into the same
penalty, weighted per expert by its activation share — an over-used expert pays
proportionally more for staying positive — so there is no separate balance loss and
no bias buffer.

It is autoregressive-safe, unlike expert choice: a token's routing depends only on
its own hidden state, and the batch coupling lives entirely in the L1 term, which is
a training-time loss rather than a forward-path dependency.

Two implementation details are load-bearing:

- The L1 coefficient is a buffer that is both **read into the loss and updated
  in-place**. The loss must use a `clone()`: autograd saves the multiplier to
  differentiate the penalty, so feeding the live buffer in makes every backward
  fail on a version-counter mismatch — and under gradient accumulation it fails from
  the *next* micro-batch's update, not this one's, so no ordering of the two
  statements fixes it.
- The coefficient update stays device-side. `alpha ** sign(...).item()` would stall
  the host once per MoE layer per micro-batch, which is the same cost the fused
  router exists to remove.

**The sentinel bucket.** The router emits a fixed `max_slots` width so the dispatch
keeps a static shape, and inactive slots go to a sentinel bucket with no weight
matrix — *not* to expert 0. Expert 0 would be **computed**: the dispatch sizes its
GEMM from the bucket histogram, so a padded slot carrying weight 0 still costs a
full expert's FLOPs, and at `max_slots = 4 * top_k` that is 4x the arithmetic the
formulation actually asks for. `MoEMLP` truncates the offsets to the first
`num_experts` buckets, so the sentinel's rows are never computed rather than
computed and discarded, and the bound stays host-known so the layer is still
graph-capturable.

**Measured cost, and its ceiling.** At `E=96, k=8, dim=1536, T=8192` (forward, device
time), against top-k routing's 2379 us:

| state | time | ratio |
|---|---|---|
| untrained, every padded slot active | 8494 us | 3.57x |
| L1 penalty at its sparsity target | 3748 us | 1.58x |

The untrained figure is not an artifact: a random ReLU gate has ~E/2 positive scores
per token and `max_slots` sits below E/2, so nothing is inactive to skip. Before the
sentinel existed, inactive slots went to expert 0 and were computed, so the 3.57x was
paid forever, trained or not.

The residual 1.58x is the row buffer, still sized for `max_slots`: the gather writes
all `T * max_slots` rows, `silu(gate) * value` runs over the whole buffer including
the uncomputed region, and the combine launches a program per row before skipping
most. Compacting it to the active count would recover ~840 us and land at ~135 us per
M active parameter against top-k's 99 — still losing by ~1.4x, and costing either a
host sync or a compaction pass. **It cannot reach parity, so it is deliberately not
built.** ReMoE is here to be measurable, and what it measures is a formulation that is
fine and a dispatch that is not.

### Latent MoE

Kimi-K3 style: the routed experts live in a compressed latent space. Standard MoE
makes every routed expert a `dim -> hidden -> dim` map, so the bank costs
`E * 3 * dim * hidden`. Projecting once into a latent of width `dim / alpha` and
running the experts there costs `E * 3 * latent * hidden` plus two shared
projections, so the bank shrinks by `alpha` for two extra GEMMs per layer.

Three things it deliberately does *not* change, all of which distinguish it from
routing in a compressed space:

- **The router still reads full width.** Routing quality is what the whole mechanism
  depends on, and it is cheap — a `dim x E` matmul against the expert bank's
  `E * 3 * latent * hidden`. This is why `MoEMLP` takes a separate `router_dim`: the
  routing width is separable from the expert width, and the gate should decide from
  the full residual, where the information actually is.
- **Shared experts stay at full width.** They see every token, so their capacity is
  never the thing being economised. The inner `MoEMLP`'s own shared expert is
  disabled and the outer class provides them.
- **Expert hidden width is untouched.** Only the *input* dimension moves.

The published motivation is cross-node all-to-all volume, which does not apply on a
single node with replicated or pipeline-split experts. The local reason to want it is
different: dispatch is a gather/scatter of `T * top_k` rows whose cost is linear in
the width of what is moved, and on a profile where dispatch is 485–915 us against
678–1819 us of expert GEMM, halving that width targets the larger of the two
overheads.

Untested below `dim=2048` in the literature. A rank-320 bottleneck on a 640-wide
residual is not the same object as 3584-on-7168, so treat small `dim` as unvalidated
rather than merely aggressive.

### Per-layer embeddings

`PerLayerEmbedding` implements Gemma-3n-style PLE: each block gets its own small
token-indexed vector, projected up and mixed into the residual stream. The point is
that an embedding table lives in ordinary memory and is *looked up*, not multiplied,
so it buys representational capacity without growing the attention/FFN stack or the
high-speed working set. Gemma 3n E2B/E4B are 5B/8B raw parameters running in a
~2B/4B accelerator footprint on exactly this trick.

That trade is aimed at on-device inference, where the table can sit in slower memory.
For training on a 5090 the table is on the same HBM as everything else, so it is
capacity per *parameter*, not per byte of fast memory. It is here to be measurable,
not because it is obviously right at this scale. The up-projections are zero-initialized
so the module starts as a no-op on the residual stream.

---

## 10. The head

At vocab 65536 and 16k packed tokens a materialized logit tensor is
`16384 x 65536 x 4B = 4.0 GiB` in fp32 — on a 32 GB card that single tensor decides
the batch size. `LMHead` has two loss kernels and neither materializes it.

**`kernel="chunked_ce"` (default)** — ours, `kernels/loss/chunked_ce.py`.
Vocabulary-major, so the fp32 `dW` accumulator spans one vocabulary block rather
than all of `(V, D)`. It takes `compute_dtype` from the caller rather than reading
an ambient autocast state, because a kernel that inspects global precision state
cannot be benchmarked honestly.

**`kernel="torch"`** — ATen's `F.linear_cross_entropy`, which is what the old code
reached liger / cut-cross-entropy for before it landed in 2.13.

Three measured facts govern the settings:

1. **`options=None` is the *reference* path and still materializes.** The chunked
   path only engages when an explicit `LinearCrossEntropyOptions()` is passed.
   At `N=16384, D=1280, V=65536`: naive 12.34 GiB, reference 6.23 GiB, chunked
   **0.69 GiB**.
2. **The returned scalar carries the *input* dtype.** Reducing 16k bf16 terms inside
   the op loses ~6% of the loss value — tolerable for a gradient, useless for logging
   or for token-weighted normalization. So the head *always* asks for
   `reduction="none"` and does its own reduction in fp32.
3. **The ATen chunked path is not autocast-registered**, so it runs the `dim x vocab`
   GEMM in fp32 on the vector cores rather than bf16 on the tensor cores. At
   `T=8192, D=1792, V=65536` with an fp32 parameter under autocast, forward+backward:
   chunked 169.7 ms / 1.33 GiB against unchunked 32.2 ms / 6.05 GiB. Hence
   `chunked=False` by default on that path; `chunked_ce` has no such penalty.

`retain` is the fraction of forward logit tiles cached for the backward to reuse
instead of recomputing, and defaults to 1. At `T=8192, V=65536` in bf16 that is the
fast end of the time/memory frontier — 26.93 ms at 1.328 GiB, against the
materializing path's 27.58 ms at 3.000 GiB and `retain=0`'s 35.48 ms at 0.391 GiB.
Lower it only where the scratch is genuinely unaffordable. Any value above 0 makes a
second backward on the same graph raise, since the epilogue consumes the cached tiles
in place.

**z-loss** (`logsumexp(logits)^2`) keeps the softmax normalizer near 1 so logits
cannot drift upward until they lose precision. `kernels/loss/zloss.py` walks the
batch in chunks under `no_grad` and recomputes each chunk's softmax in the backward,
so it never materializes the vocabulary either. `logs` carries the cross-entropy
separately from the z-loss, so reported perplexity is not polluted by the regularizer.

**It is off in every shipped config, and should stay off.** It is a second full
pass over `dim × vocab`, not part of the fused head: at MoE-1B and 8192 tokens the
head goes from 12.83 ms to 54.82 ms, and since under pipeline parallelism the head
is the critical-path stage, that is 182.0k → 114.5k tok/s end to end on 4 cards
(1.59x). No frontier recipe requires
it — DeepSeek-V3 trained 14.8T tokens with no auxiliary loss at all. See
[moe-router-loss.md](../internals/moe-router-loss.md#the-head-z-loss-is-a-different-thing-and-it-is-off).

**`soft_cap`** (`cap * tanh(logits / cap)`, Gemma 2) is not expressible in either
fused op, so setting it forces the materializing path. Prefer QK-norm, which is what
the models that dropped soft-capping moved to.

`loss(..., reduction="sum")` is what a token-weighted training loop wants: divide by
the *global* trained-token count yourself, so gradient accumulation and DDP averaging
stay exact regardless of how tokens are split across micro-batches and ranks.

### Tying

**`tie_embeddings` defaults to `False`.** Tying only holds up on a corpus and batch
large enough that every vocabulary row is seen often; below that the one matrix
serves two objectives and collapses, because the head's dense softmax gradient and
the lookup's sparse gradient fight over the same rows. Untying costs a second
`vocab x dim` matrix — 58.7M at dim 896, 84M at 65536 x 1280 — so it is a real
parameter cost, and every Kohaku rung's target count absorbs it.

`logits(hidden)` materializes and exists for generation and analysis only. Call it on
a short sequence or a single position, never in training — which, with a KV cache,
is one row per step (§15).

---

## 11. FLOP accounting

`FlopCounter` is the numerator behind MFU, and an MFU number is only as trustworthy
as its FLOP model. The two ways the usual `6 * parameters` shortcut goes wrong both
*flatter* the run.

**The embedding is a gather, not a GEMM.** With tied embeddings the shortcut happens
to be right — the one `vocab x dim` weight really is used once, by the head. Untied,
it invents a second `vocab x dim` GEMM per token that nobody runs, and reports an MFU
too high by the fraction of the model that is vocabulary (17% of Nano-500M). So the
count is `active - embedding + head`, which is identical for a tied and an untied
model of the same shape — as it must be, since they do the same arithmetic.

**Attention is quadratic and gets dropped.** At 2048-token documents the score and AV
matmuls are ~8% of Nano-500M's forward, and the share grows with context; a model
that ignores them under-reports MFU exactly where a context change is what you are
trying to measure. They are charged from the batch's *own* document lengths rather
than a nominal context, because a packed batch of 50–600 token TIPO samples attends
nothing like a 2048-token one.

Attention costs `4 * q_dim` per attended pair: two FLOPs per multiply-accumulate,
once for `q · k` and once for `p · v`, each over the full query width.

A window of `w` truncates every row past the `w`-th to `w` keys, giving
`Lw - w(w-1)/2` once `L >= w` — not `Lw`, which would over-charge the ramp at the
start of every document by half a window. On this corpus most documents are shorter
than the window, so the ramp *is* the common case.

Sparsity is the one thing the parameter count already gets right:
`MoEMLP.active_parameters()` counts `top_k + num_shared` experts rather than all of
them. Counting every expert would *overstate* FLOPs and therefore overstate MFU — a
sparse model would look like it was using the card better than it is.

The counter reports **two** totals. *Model FLOPs* is what the architecture owes:
forward plus backward, with the backward charged 2x (every forward GEMM becomes a
grad-input and a grad-weight GEMM of the same shape). *Hardware FLOPs* is what the
GPU is actually asked to run, which under gradient checkpointing includes a second
forward through the blocks. Report only the first and recompute looks free; only the
second and it looks like progress.

Document lengths are carried in **fp64**, because the pair count is quadratic: a
2048-token document is 2.1M pairs, and a batch of them overflows fp32's
exact-integer range (2^24) after eight documents. The padded layout reports
`num_seqs` rows of `max_seqlen` with padding included on purpose, since a causal SDPA
over a padded row computes the whole triangle whether or not the tokens are real.

---

## 12. MXFP8 surgery

`config.mxfp8 = True` replaces the eligible projections with MXFP8 linears **after
construction**, not through a component spec. [mxfp8.md](../internals/mxfp8.md) covers what is
converted and what has been verified; this section covers why the swap is shaped the
way it is, because that shape is a piece of architecture.

**Why surgery, not a spec.** The swap has to happen after `initialize_weights`,
because the depth- and width-scaled init is what makes two arms of an A/B start from
bit-identical weights. A spec resolved at build time would instead have
`MXFP8Linear.__init__` draw its own `normal_(std=fan_in ** -0.5)`, and the two models
would differ before the first step. The swap copies what the scaled init produced.

**Children are selected by parent module type, not by attribute name.** `LMHead` and
every router hold their matrix as a bare `nn.Parameter` named `weight`, so a name scan
would reach them; typing the parent makes their exclusion an invariant instead of an
accident. Neither belongs in fp8 — the head contracts over the vocabulary, and a
router's logit scale *is* the gate sharpness.

**Matmul held as a bare parameter is declared, not discovered.** That is the other
half of the same trade. A module implements `mxfp8_matmul()` (the `DECLARE`
attribute) returning `{name: Matmul(mac_per_token, refusal, never)}`, stating what it
owns and what it is worth per token, and `enable_mxfp8()` (the `CONVERT` attribute) to
perform the conversion. `MoEMLP` declares its two expert stacks *and* `router.weight`,
which keeps the never-in-fp8 rule for routers in the one module that owns one and
covers routers written later.

Before that protocol existed, `swap_mxfp8` reported `0 skipped` on `MoE-2B-A370M`
while the routed experts — 42.4% of the model's per-token matmul — stayed bf16. That
is one command away from a "MoE fp8 loss" measured on a model that was two-thirds
bf16.

**Every non-fp8 outcome blocks the run rather than shrinking it.** A hard failure is
recoverable; a mislabelled experiment is not. Three outcomes refuse, not one: a shape
refusal, a *declared* matmul nothing converted, and a matmul-shaped parameter nobody
claimed. The last is found by a closing sweep that fails *closed* — it cannot
*recognise* an unclaimed GEMM (that is the name scan this design rejects), but it can
find one by not recognising it, since a 1-D parameter is a norm gain or an attention
sink and never a matmul. The refusal message quotes the fraction of arithmetic at
stake, because a bare count does not say whether a third of the model or one
projection is involved.

One refused tensor refuses its whole module. Converting the eligible half would leave
a layer that is half fp8, and for an MoE layer the two halves share `dim`, so a split
verdict there would always mean the report and the arithmetic disagree.

**Which shapes are eligible.** Only `in_features` can block an `nn.Linear`:
`out_features` is DGRAD's contraction axis and is zero-padded, which is exact — a zero
never raises an MX block's `amax`, so the real columns keep the scale they would have
had. So `kv_heads * head_dim` no longer rejects anything, and `Nano-200M-wide` and
`MoE-2B-A370M` (both `kv_out=192`) are fully swappable. `Nano-200M-deep` is still
blocked on five of its six projections by a `dim` of 704, and that one cannot be padded
away: it is FPROP's contraction axis, so padding it would mean padding every activation
that reaches the layer.

The MoE experts are stricter still in one respect and looser in another. Their
constraint is 32, not the vendor's 128, because the grouped kernels read
`quantize_mx`'s natural scale layout rather than cuBLAS's `SWIZZLE_32_4_4` and so do
not inherit that alignment. But it applies to *both* contraction axes — `dim` for
GEMM1 and `hidden` for GEMM2 — and neither can be zero-padded, since padding `hidden`
would mean padding the SwiGLU output on every forward, which is a contract with the
caller rather than zero columns living inside the layer.

**The refresh.** Quantized copies are derived from the masters and must be rebuilt
after **every optimizer step** — `refresh_mxfp8_weights(model)`, or
`LMBackbone.refresh_mxfp8()`. It is a per-step rule, not a per-micro-batch one: under
gradient accumulation the weights are unchanged across micro-batches, so a per-call
dirty check tests state the caller already knows the answer to, and calling it per
micro-batch multiplies the cost by the accumulation factor. That cost is fully exposed
below ~16k tokens per step — measured at ~15 ms, 15% of an 8192-token step, against
1.4% at 16384.

Omitting it entirely is the dangerous case: the layer then trains every fp8 GEMM
against initialization-time weights while the masters move underneath, with nothing
visible in the loss. The refresh returns a count so a caller can assert it fired —
against `len(report.modules)`, which counts *modules*, since an MoE layer's two expert
stacks share one cache.

The quantized caches are dropped on any `_apply` (device or dtype transform) and
rebuilt lazily. They cannot be registered buffers, because `is_floating_point` is True
for e4m3 and `.float()` would rewrite them rather than leave them alone.

Two escape hatches on `swap_mxfp8`, both deliberately *reported*: `scope` restricts
conversion to qualified-name substrings (everything outside is charged to
`out_of_scope`, a stated decision rather than a failure), and `convert_declared=False`
leaves declared bare-parameter matmul in bf16 and charges it to `unreached` so the
report still refuses. `scope` exists because the dense projections go through vendor
`_scaled_mm` and are verified, while the routed-expert path is a custom Triton kernel,
so the two need to be selectable independently. `convert_declared` is an
*attribution* hatch — splitting a loss gap between the two needs an arm with only the
first, and reconstructing that arm by monkeypatching the converter puts the decision
somewhere no reader of the module would find it.

---

## 13. How presets compose it

`LMArchConfig` is everything the backbone needs to build itself. Component fields
(`norm` / `mlp` / `attn` / `posenc` / `moe_router`) are specs. Two derived quantities
are computed from the shape fields and then just read:

- **`layer_types`** — per-layer `"full"` or `"sliding"`, from `sliding_window` and
  `global_layer_every` (or `window_pattern`). Interleaving a cheap local layer with an
  occasional global one is how Gemma 3 (5:1) and OLMo 3 (3:1) keep the KV cache bounded
  without losing long-range reach.
- **`moe_layers`** — which layers are sparse, from `moe_every` and `moe_first_dense`.
  Keeping the first layers dense is standard (DeepSeek-V3 keeps 3): early layers learn
  generic features every expert would otherwise have to duplicate.

`attn_for(layer)` returns `attn_sliding` on sliding layers when it is set, and `attn`
otherwise — the per-layer backend choice described in §7.

`moe_mlp` selects the sparse *formulation* (`"moe"` is DeepSeek-shaped, `"latent_moe"`
runs the experts in a compressed space) while the gate function and shared-expert count
stay separate fields, so a formulation is a config change rather than a code path.

The named presets and the constraints they were solved under are in
[presets.md](presets.md).

---

## 14. Adding a component

The mechanics — registry names, dotted paths, constructor signatures — are in
[extending.md](../guides/extending.md). What that document does not say is which invariants the
rest of this file has just established, so here they are as a checklist:

1. **Take `**kwargs`.** The backbone passes every slot the full set; a component that
   wants two of them must tolerate the rest.
2. **Be a last-dim op**, unless you are attention. That is what lets one module serve
   packed `(T, D)` and padded `(B, S, D)` without a reshape.
3. **If you are attention, respect document boundaries** when `seq_info.packed`. Take
   `cu_seqlens`, or build a block-diagonal mask with `_doc_ids` + `_causal_mask`.
4. **Resolve choices in `__init__`.** If your forward branches on a config value, the
   branch is in the wrong place. Device-dependent branches are the exception: a module
   is built on CPU and moved afterwards.
5. **Balancing state is a buffer, not a parameter.** It is updated by a rule, so it must
   not receive weight decay or optimizer state.
6. **Never `bincount` on the device path.** It syncs. `scatter_add_` into a pre-sized
   tensor.
7. **If you hold a matmul as a bare `nn.Parameter`, declare it** — implement
   `mxfp8_matmul()`, and `enable_mxfp8()` if it can be converted. Otherwise the closing
   sweep will refuse every fp8 run that reaches your module, correctly.
8. **Give it a test.** New behaviour gets a test that pins it, and for a kernel, a
   precision test against an fp64 reference in both fp16 and bf16.
9. **If you are attention, take a `cache`.** The block passes one positionally.
   `BaseAttention.attend_cached` already implements it; a subclass that only
   overrides the packed kernel call inherits a correct cached path (§15).

---

## 15. Generation and the KV cache

Training never needs a cache: every position is known, so one packed forward
computes all of them. Generation does, because without one, step `t` recomputes
the keys and values of every earlier token — the same arithmetic `t` times over,
so the *arithmetic* of a sample grows with the square of its length. Whether the
clock notices is a separate question, answered under Cost below, and the answer on
this hardware at preview scale is mostly no.

### What it holds

`KVCache` (`models/cache.py`) is a plain Python object, not a module and not a
buffer: it is per-generation state with a lifetime of one `generate()` call, and a
buffer would follow the model into checkpoints and `.to()` calls that have nothing
to do with it.

Per layer it holds one key and one value tensor of
`(batch, max_length, kv_heads, head_dim)`, allocated once at full length and
filled in place. Three details in that shape are load-bearing:

- **`kv_heads`, not `heads`.** Under GQA the cache is the whole reason `kv_heads`
  exists — the 20x memory saving in §7 is this tensor. Storing query heads would
  work and quietly cost 8x.
- **Allocated to `max_length` up front**, so decoding never reallocates or
  concatenates. Running past it raises; it does not grow, wrap or truncate. A cache
  that silently dropped its oldest entry would keep generating fluent text with a
  corrupted prefix.
- **dtype and device are taken from the first appended tensor**, not from the
  constructor. Under autocast the projections emit bf16 while the module's
  parameters are fp32, and a cache sized from the parameter dtype would either
  upcast every write or reject it.

`length` is one scalar for the whole cache, not one per layer: layers advance in
lockstep, and per-layer counters would let a bug in one layer's bookkeeping stay
invisible. `LMBackbone.forward` calls `advance()` once, after the block loop —
appends inside the loop all write at the same offset, which is what makes them
order-independent.

### Padded only

The cache serves the padded `(B, S)` layout and refuses `(T,)`. Packing is a
*training* layout: it exists to keep 80% padding out of a batch of 50–600 token
documents (§3). Generation runs a handful of equal-length rows, so packing buys
nothing, while a packed cache would need per-document offsets in the append path
and a block-diagonal mask over a growing prefix — a second correctness surface for
no throughput.

For the same reason all four attention backends share one cached path,
`BaseAttention.attend_cached`, which is SDPA. `varlen_attn` consumes `cu_seqlens`
and has no cached-prefix form; Flex would need a new block mask per step.

That path has two branches, and which one runs is decided by the query length, not
by config:

- **One query row (`_decode_attend`) needs no mask at all.** The single new token
  is at the end of the prefix, so every cached key is already visible to it, and a
  window becomes `k[:, -window:]` — a slice, not a comparison matrix. This is worth
  its own branch rather than a general mask: building the mask cost more than the
  attention it described, and made the cache a *net loss* at 128 tokens (0.74x
  before the branch, 1.05x after).
- **A multi-token step (prefill, or a chunked continuation) builds one mask**, with
  `q_offset` placing the queries. At `(S, prefix)` this is a strip, not the `(T, T)`
  matrix §7 warns about.

### The position contract

**The new token's position is the cache length, not zero.** This is the mistake the
whole test file exists to catch, because a model whose RoPE angles restart at 0
every step still emits fluent, plausible text.

Two places consume the offset, and both must agree:

1. **RoPE.** `cache.seq_info(tokens)` builds the `SeqInfo` whose `position_ids` run
   `length … length + S`. The backbone asks the cache for it rather than
   constructing one itself.
2. **The mask**, on multi-token steps. `_causal_mask` takes `q_offset`: query row
   `i` is at absolute position `q_offset + i`, key column `j` at `j`. Without it a
   continuation would mask away the prefix it is supposed to attend over. The
   one-row decode branch never builds a mask, so it cannot get this wrong — which
   is also why the suite needs a chunked continuation to exercise the argument at
   all.

`is_causal=True` is used only when the query and key lengths agree, which for a
cached step means the prefill. PyTorch's `is_causal` alignment for unequal lengths
is ambiguous, and the explicit mask is the same computation with the alignment
written down.

**Sliding windows carry over**, as a mask on a continuation and as a slice on a
decode step, so a cached step sees exactly the span the training path would have
given it. A cache that ignored the window would attend over a prefix the model was
never trained on — slightly better-informed, entirely off-distribution.

**QK-norm and RoPE are applied before the write.** The cache holds normalized,
rotated keys, because both are functions of the token and its absolute position,
neither of which changes later. Appending raw `k` and normalizing the whole prefix
each step would be correct and would also make the cache pointless.

### The equivalence guarantee

Greedy generation with a cache emits the **identical token sequence** as the
cache-free path. Not close — identical, `torch.equal` on the token ids, over 32 new
tokens, for every architecture in `tests/test_generation.py::ARCHITECTURES` (dense,
MoE, sliding window, attention sink, MQA, no QK-norm, the `sdpa` backend, partial
RoPE) in fp32, plus the fused MoE expert path on CUDA.

That test is worth more than everything else in the file, and it has one failure
mode worth knowing about: **at the default `init_std` it proves nothing.** A
randomly initialized model at `init_std=0.02` decodes one token and then repeats it
forever — a fixed point of its own output map. Every cache bug tried against that
fixture (zeroed RoPE offset, off-by-one append, dropped mask offset, ignored
window, query-head-shaped buffers) still produced a byte-identical token stream.
The fixture is built at `init_std=0.3`, where the decode visits ~26 distinct tokens
in 32 steps, and `_assert_the_comparison_can_fail` pins that property so a future
fixture change cannot quietly re-hollow the test.

Two weaker tests sit underneath it and localize a failure the token test only
reports: a token-by-token decode must reproduce a full forward's hidden states to
`rel_error < 1e-5`, and a prefill-then-decode split must match one forward over the
same tokens. The second is parametrized over a window, and it is the *only* caller
of the masked cached path — a one-token step takes the mask-free decode branch
below, so without a multi-token continuation in the suite the `q_offset` argument
is dead code that nothing would catch being wrong.

**Token identity is an fp32 claim, and it does not survive bf16.** Measured on a
154M dense model: fp32 gives a bit-identical 128 tokens with a per-step logit
difference of 5e-7; bf16 gives 13 identical tokens and then parts, at a per-step
logit difference of 1.2e-2. Nothing is wrong in the bf16 case — an untrained
model's top-1/top-2 logit gap is 3.1e-2, the same order as the difference between
two legitimate orderings of the same sum, so the argmax is deciding a coin flip.
A *trained* model has a decisive argmax and does not have this problem, but the
suite must not depend on that, which is why the equivalence tests are fp32.

**MoE has no bit-exact story at all, in any dtype.** Two identical cache-free
forwards of the same MoE model, no cache involved, differ by 1.8e-2 — the routed
combine is an atomic scatter, so its summation order changes between runs. The
dense model is bitwise identical across reruns. Any cached-vs-uncached comparison
on a sparse model therefore has to be read against that floor, which is what
`scripts/bench/model/generate.py` measures alongside it: cached-vs-plain must sit
inside plain-vs-plain, not at zero.

### Cost

`scripts/bench/model/generate.py`, one RTX 5090, bf16, 32-token prompt unless
stated:

| model | batch | new | cache-free | cached | speedup |
|---|---|---|---|---|---|
| Kohaku-200M | 1 | 128 | 1887 ms | 1786 ms | **1.06x** |
| Kohaku-200M | 1 | 512 | 7569 ms | 7148 ms | 1.06x |
| Kohaku-200M | 4 | 128 | 1870 ms | 1949 ms | 0.96x |
| Kohaku-200M | 4 | 512 | 7434 ms | 7982 ms | 0.93x |
| Kohaku-200M, window 128 | 1 | 512 | 11110 ms | 7202 ms | **1.54x** |
| Kohaku-MoE-1B | 1 | 128 | 3297 ms | 3153 ms | 1.05x |
| Kohaku-200M, 1024-token prompt | 1 | 128 | 1903 ms | 1817 ms | 1.05x |

**This is not the 5x the quadratic argument predicts, and the reason is in the
`host%` column: every row is 100% host-bound.** A batch-1 decode step is ~14 ms of
Python and kernel dispatch for a 154M model — roughly 500 launches whose *contents*
are one token wide. The arithmetic the cache removes was never what the clock was
measuring. Quadrupling the batch or quadrupling the sample length changes the
per-token cost by nothing (71 tok/s at 128 new tokens and at 512), which is the
signature of a dispatch-bound loop.

Two consequences worth stating plainly:

- **At preview scale the cache buys ~5%, and at batch 4 it loses 4-7%.** Its extra
  per-step host work — one buffer write and one prefix slice per layer — is real,
  and it is competing against attention work that is already hidden under
  dispatch. The reason to have it is that generation now scales, not that a
  128-token preview got faster.
- **The windowed row is the one unambiguous win (1.54x)**, and not because of the
  KV saving: the cache-free path has to build an explicit `(S, S)` mask every step
  because a window has no `is_causal` fast path, while a cached step slices the
  last `window` entries of the prefix and needs no mask at all.

Where the cache would show its asymptotics — a long context, a large batch, a
model big enough to be compute-bound — is also where a decode step stops being
free, so the honest summary is that this is a correctness and scaling change first
and a throughput change a distant second. CUDA graphs or a compiled decode step,
not a better cache, is what the 14 ms is waiting on.
