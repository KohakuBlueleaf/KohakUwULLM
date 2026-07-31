# Optimizers

This document covers everything between the backward pass and the next forward:
how parameters are split into groups, which optimizer runs on which group, and
what changes when the parameters themselves are held in 16 bits.

Three decisions are worth understanding before any of the details:

1. **Weight decay is a property of a parameter's *shape and gradient pattern*,
   not of its name.** Getting the grouping wrong costs a measured 79% of a rare
   token's embedding norm over a production run, and nothing in the loss curve
   says so.
2. **Muon is the production optimizer** for the hidden matrices, with an
   internal AdamW for everything a spectral update does not suit. It is worth
   1.51x on the dense step and 1.22x on MoE.
3. **16-bit parameters are an optimizer problem, not a model problem.** Casting
   the model is one line; keeping it training is a rounding rule.

| registry name | class | notes |
|---|---|---|
| `adamw` / `adam` / `sgd` | `torch.optim.*` | `foreach=True` on CUDA |
| `fused_adamw` | `FusedAdamW` | fused ATen kernel, clipping folded in |
| `muon` | `MuonW` | Muon + internal AdamW, selected per group |
| `adamw8bit` / `adamw4bit` / `adamwfp8` | torchao | quantized moments, optional dep |
| (dotted path) | anything | e.g. `...optim.lowbit.StochasticAdamW` |

Configuration is `OPTIMIZER` plus `OPTIMIZER_KWARGS`; see
[writing-configs.md](../guides/writing-configs.md) for the knob table. The shipping
setup is three learning rates in one optimizer:

```python
OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}
LR = 5e-4                       # everything else, on the internal AdamW
WEIGHT_DECAY = 0.1
```

Muon at 2e-3 is in spectral-norm units and is not comparable to the AdamW rate
beside it; the embedding gets 2e-3 on AdamW because a row-sparse gradient wants a
larger step than a dense one. Directly:

```python
from kohakuwullm.training.optim.build import build_optimizer, group_parameters

opt = build_optimizer(model, "muon", lr=5e-4, weight_decay=0.1,
                      muon_lr=2e-3, embed_lr=2e-3)

# What that produced, which the startup log prints and you should read once:
for g in opt.param_groups:
    n = sum(p.numel() for p in g["params"])
    print(f"{n/1e6:8.1f}M  lr={g['lr']:.2e}  wd={g['weight_decay']:.4f}  "
          f"muon={g.get('use_muon')}")
```

At `Kohaku-MoE-1B` that is four groups:

```
    51.1M  lr=5.00e-04  wd=0.1000  muon=False     the LM head
     0.0M  lr=5.00e-04  wd=0.0000  muon=False     norm scales, router bias
    50.3M  lr=2.00e-03  wd=0.0000  muon=False     the input embedding
   889.4M  lr=2.00e-03  wd=0.0250  muon=True      every hidden matrix
```

Two things to read off it. The embedding carries `wd=0` — the row-sparse argument
below. And Muon's decay is **0.025, not 0.1**: it is rescaled by `lr / muon_lr` so
the per-step shrink matches what the AdamW groups get, which is what makes one
`WEIGHT_DECAY` knob mean the same thing across a 4x learning-rate difference.

---

## Parameter grouping

`group_parameters` in `training/optim/build.py` is the single place that decides
what decays, what gets a muP-scaled learning rate, and what Muon is allowed to
touch. Every optimizer in the table above receives its parameters through it.

### Weight decay applies to matrices, not to vectors

Norm scales, biases and the MoE router's balancing bias are 1-D. Decoupled decay
on a 1-D parameter pulls a learned scale toward zero and buys nothing back:
there is no high-dimensional direction for the decay to shrink, so the entire
effect is a bias toward smaller activations that the model then has to re-learn.
These land in a `weight_decay=0.0` group.

### The input embedding is excluded, and that one is not taste

The embedding table has the same `vocab x dim` shape as any other matrix, but
its **gradient is row-sparse**: a row is updated only on the steps where its
token appears. Decoupled weight decay is not sparse — it shrinks every row on
every step, whether or not that row saw a gradient.

Measured on this corpus with the production tokenizer:

| quantity | value |
|---|---|
| rows never appearing in a 36M-token sample | 37,336 of 65,536 (**57%**) |
| steps the median *appearing* row is present on | 1.3% |
| norm a never-touched row loses to decay alone | **79%** |
| norm a row touched on 0.1% of steps loses | 72% |

The simulation behind the last two rows runs the real cosine schedule at
`lr=3e-4`, `wd=0.1`, 100k steps. The rare-token embeddings — the ones with the
least signal to resist it — are exactly the ones the effect hits hardest.

Decay is essentially the *whole* of that effect. The other thing a dense
optimizer step does to an untouched row is apply an update carried over from
momentum since its last appearance, and that term decays as
`(beta1 / sqrt(beta2))**k`: over 98k steps it moves an untouched row by 0.3%.
This is why the fix is a grouping rule and not a lazy or sparse optimizer.
Skipping untouched rows entirely would buy the remaining 0.3% and cost a per-row
step counter to do it.

`decay_embeddings=True` puts the embedding back in the decay group if you want
to measure the alternative.

### The LM head is *not* excluded

Same shape, opposite gradient pattern: a softmax gives every row of the head a
gradient on every step, so ordinary decoupled decay is correct there and the
argument above simply does not apply.

**One consequence worth remembering: `tie_embeddings=True` also drops weight
decay from the head.** Tying makes them one tensor, reported under the
embedding's name, so the embedding exclusion catches it. If you want a tied model
whose shared matrix decays, pass `decay_embeddings=True`.

The two rules are matched on path *components* (`"embed" in name.split(".")`), so
a wrapper module that renames the root cannot silently re-enable decay on the
embedding, and a substring like `embedding_head` cannot accidentally match.

### muP

With `use_mup=True`, hidden matrices are bucketed by fan-in and each bucket gets
`lr * (base_dim / fan_in)`: a wider layer receives proportionally smaller
updates, which is what keeps the update-to-activation ratio constant across
widths, so a hyperparameter found at `base_dim` transfers to a wider model
without a re-sweep. That matters here specifically because the plan is a family
of sizes (25M to 1B dense, plus MoE) trained from one recipe — see the preset
ladder in [architecture.md](../concepts/architecture.md).

muP scales weight decay by the same factor (`weight_decay / scale`) so that the
decay-to-update ratio is width-invariant too; without that the wider model ends
up under-regularized. Embeddings and 1-D parameters keep the base rate — the
embedding because its fan-in is the vocabulary rather than a feature axis, which
is the same predicate as the decay exclusion rather than a second spelling of it.

### Checkpoint compatibility

**Existing optimizer checkpoints will not resume across the parameter-regrouping
change.** The group layout changed shape, and `torch.optim.Optimizer.load_state_dict`
matches state to parameters positionally within a group. Rather than silently
assigning a matrix's Adam moments to a different matrix, the load raises on the
group sizes. Model weights still load; the optimizer restarts cold. For a run
mid-flight, the practical options are to finish it on the old code or accept a
few hundred steps of momentum re-warmup.

---

## Muon

Muon replaces a matrix parameter's momentum update with its nearest
semi-orthogonal matrix. Every singular value of the resulting step is 1, so the
step length is set by the spectral norm alone instead of by whichever few
directions the gradient happened to be largest in. In practice that means the
update is a *direction* with the magnitude discarded, applied uniformly across
the matrix's spectrum.

```python
from kohakuwullm.training.optim.muon import (
    MuonW, newton_schulz, newton_schulz_cubic, NS_PHASES_GRAM,
)

opt = MuonW(groups, lr=5e-4, betas=(0.9, 0.95))   # groups carry `use_muon`

# The primitive underneath, batched over leading dimensions: 8 same-shape matrices,
# each orthogonalized on its own, which is what the MoE expert stack needs.
g = torch.randn(8, 1536, 1536, device="cuda")
u = newton_schulz_cubic(g, phases=NS_PHASES_GRAM)   # the default, cubic5
u = newton_schulz(g)                                # the Jordan quintic
```

`MuonW` holds both algorithms — Muon and a decoupled-decay AdamW — in one
optimizer object, selected per parameter group by a `use_muon` flag. That is a
deliberate structural choice: the trainer keeps a single optimizer and a single
LR schedule, instead of two objects whose schedules can drift apart. A group
arriving without the flag **raises**, because the default would quietly run
AdamW on precisely the matrices Muon exists for.

### Which parameters qualify

Orthogonalizing is only meaningful for a parameter that *is* a linear map
between two feature spaces. `is_hidden_matrix` says yes to attention q/k/v/o,
the SwiGLU pair, the shared expert, and the stacked routed-expert weights, and
no to:

- **`embed` and `head`** — one axis is indexed by token id, not by features. A
  step touches only the rows of tokens present in the batch, and orthogonalizing
  smears that step across the whole vocabulary. This is the one case where
  equalizing singular values destroys information instead of balancing it.
- **`router`** — its logit scale *is* the gate sharpness, and the aux-loss-free
  bias rule in `models/components/moe.py` assumes that scale drifts slowly,
  while an orthogonalized step moves it at a rate fixed by lr alone. No published
  run settles this either way; pass a different `muon_filter` to try the other
  choice.
- **every 1-D parameter** — no singular values to equalize.

Those all fall through to the internal AdamW.

**MoE expert weights are a batched case, not a reshape case.** They are stored
as one `(experts, out, in)` stack, and `newton_schulz` batches over leading
dimensions, so each expert's matrix is orthogonalized on its own — correct,
because each expert *is* a separate linear map. Reshaping to `(experts * out, in)`
would fuse the experts into one operator, and `(experts, out * in)` — what a
conv-style flatten does, and what the published reference implementations do to
any tensor above 2-D — turns each expert into a single row and destroys the
update entirely.

### Newton-Schulz: the iteration

The polar factor `U V^T` is computed by a fixed-length Newton-Schulz iteration
rather than an SVD. Two details are load-bearing:

**Normalization.** The iteration converges only for spectral norm ≤ 1, which the
Frobenius norm bounds, so the input is divided by its Frobenius norm first. That
norm is reduced in **fp32** — a bf16 sum over a million terms loses several
percent, and this scalar divides the whole matrix. The safety margin is
*relative* (1.01x) rather than additive: an additive margin would make the update
depend on the gradient's absolute scale, which is the very thing orthogonalizing
exists to remove.

**The copy.** `grad.to(dtype, copy=True)` is copied unconditionally. Without
Nesterov, `grad` *is* the momentum buffer, and `.to(dtype)` on a tensor already
in `dtype` returns it unchanged — so the in-place divide would replace the EMA
with a unit-norm matrix.

The iteration also runs on the **short** side: iterating there costs
`(4*aspect + 2) n^3` instead of the transpose of that, and the polar factor of
the transpose is the transpose of the polar factor, so this is free.

### The quintic schedule

`NS_COEFFS = (3.4445, -4.7750, 2.0315)` is the Jordan quintic, tuned to maximize
the slope at zero. It does *not* converge to exactly 1 everywhere — its fixed
point spreads singular values over roughly `[0.7, 1.3]` — which costs nothing
measurable in training and is exactly what lets five iterations replace the ~30 a
convergent iteration would need.

### The cubic schedule (`cubic5`, the default)

`NS_CUBIC5` is a per-iteration cubic schedule from arXiv 2606.00371. A cubic step
drops one of the quintic's three matmuls — specifically the **cheapest** one,
since `gram @ gram` is `2m^3` against `2m^2n` — so the saving is
`1 / (2*aspect + 1)`, not a flat third:

| shape | saving |
|---|---|
| square | 33% |
| our skinniest expert stack | 15% |

The schedule lands singular values in `[0.774, 1.30]` — the band the Jordan
quintic reaches by accident — for 2/3 of the FLOPs.

**Its length is not a knob.** There is no `steps` argument, because the
coefficient schedule *is* the iteration count: truncating it leaves singular
values at 0.153, which is under-orthogonalized rather than merely less accurate.

### Phase grouping

Each cubic step is `x <- A_k x` with `A_k = a_k I + b_k x x^T`. A group of `L`
consecutive steps is therefore one product `A_{k+L-1} ... A_k` that can be
accumulated in `m x m` gram space and applied to the `(m, n)` matrix once. Over
five steps, `P` groups cost:

```
4 P m^2 n  +  6 (5 - P) m^3
```

`P = 5` is the unfactored iteration; fewer groups trade the `m^2 n` term for the
`m^3` one. The two cost the same at `n / m = 1.5`, which is why `gram_aspect`
defaults to 1.5 — the feed-forward matrices are above it and the attention
projections are not. This is a **per-shape** choice resolved once in
`MuonW.add_param_group`, never a global mode, and never a runtime branch.

Grouping also removes the iteration's traffic over the big matrix. The
per-step `a x + b (gram @ x)` is an elementwise pass over `m x n`, while the
grouped form's elementwise work is all on `m x m`. At `P = 5` the arithmetic is
identical to the unfactored loop and only that traffic is saved, which is why
there is no separate `P = 5` fallback expression.

**Why 2+3 and not one group of 5.** The accumulated product's gain on the null
space is the product of the group's `a` coefficients — **119x** over all five
steps — and bf16 rounding at that norm leaves an error the cancelling final
product cannot recover. Measured at `m=128` on a `1/i` spectrum:

| grouping | null-space gain | top singular value of the update |
|---|---|---|
| one group of 5 | 119x | 1.98 (against the schedule's cap of 1.30) |
| 3 + 2 | 22x | — |
| **2 + 3** | **13.8x** | within 1% of the unfactored iteration |

So `NS_PHASES_GRAM = (2, 3)`, and `NS_PHASES_DIRECT = (1, 1, 1, 1, 1)` is the
unfactored iteration expressed in the same form.

The quintic has no grouped form: its `A_k = a I + b G + c G^2` needs `G^2` as
well, which costs the group an extra `m^3` per step. Since `cubic5` is the
default, that path was not worth building.

### Update scaling

The orthogonalized update is a pure direction, so something has to set its
length. `orthogonal_update_scale` offers two rules, and the choice interacts with
muP:

**`"spectral"` (default).** Multiply by `sqrt(fan_out / fan_in)`. This is the
gradient dualized under the RMS→RMS operator norm, which makes one learning rate
correct at every width (at fixed aspect ratio) **with no muP scaling on top**.
The `max(1.0, ...)` clamp is what every reference implementation does: a
down-projection then takes a step `sqrt(fan_in/fan_out)` larger than the dualized
one, but that factor is fixed by the architecture, so the global lr absorbs it
and width transfer survives.

**`"rms"` (Moonlight's alternative).** Multiply by
`rms_target * sqrt(max(fan_out, fan_in))`, forcing the update's RMS to
`rms_target` — AdamW's empirical update RMS, 0.2 — so an AdamW-tuned lr carries
over unchanged. This is *not* width-invariant and needs muP's `1/fan_in`.

That difference is why `group_parameters` takes a `muon_mup_exponent` rather than
a mode string. The Muon lr is scaled by `(base_dim / fan_in) ** e`:

| `update_scale` | `muon_mup_exponent` | reason |
|---|---|---|
| `"spectral"` | `0.0` | the dualized update already has spectral norm `sqrt(fan_out/fan_in)` |
| `"rms"` | `0.5` | fixing the *RMS* leaves the spectral norm growing like `sqrt(width)`; half a power of `fan_in` removes exactly that |

At `e = 0.5` the two modes are algebraically identical up to the 0.2. They are
the same rule, so this is one exponent and not two code paths.

Note also that `muon_lr` pulls the hidden matrices into their own group *before*
the muP split, for the same reason: applying `base_dim / fan_in` on top of a
spectral scale would scale twice, and that is one of the two heuristics
arXiv 2512.05620 §G measures as failing to transfer.

At `e = 0` the Muon groups are not keyed by fan-in at all, since every bucket
would carry the same lr and one group per fan-in is one more optimizer group for
the schedule and the startup log to carry for nothing.

**Decay is rescaled with the lr.** `muon_lr` is in spectral-norm units, roughly
100x the AdamW rate. Decoupled decay shrinks by `lr * weight_decay` per step, so
reusing the AdamW `weight_decay` against a 100x larger lr would decay those
matrices 100x harder. The Muon group's decay is set to
`weight_decay * lr / muon_lr` to match the intended shrink.

### Batching, and what the step costs

`newton_schulz` normalizes with `dim=(-2, -1)`, so stacking same-shape matrices
on a new leading axis leaves each matrix's normalization and its short-side
transpose untouched. A 500M-class model has ~220 eligible tensors over ~8
distinct shapes, and batching them is **27x fewer launches for identical
arithmetic**.

The momentum look-ahead is written *straight into* that batch. The obvious
alternative — `torch.stack` over per-parameter look-aheads — reads and writes the
whole batch a second time for nothing, and at 344M eligible parameters that pass
costs more than the launches the batching saves.

The batch is capped by **elements**, not by count (`ns_batch_elems`, default 64M):
an expert stack is already ~84M elements on its own, so batching all of them
would allocate a temporary larger than the model.

**The cost profile.** The iteration's arithmetic does not scale with the batch —
it is set by the parameter shapes alone — so Muon is a large *fixed* cost per
step:

| model | Muon | step | AdamW, same model |
|---|---|---|---|
| Nano-500M | 61.6 ms | 353 ms | 19.8 ms |
| MoE-1B | 156.6 ms | 343 ms | 19.8 ms |

**None of it is launch-bound.** A CUDA-graph replay of the step matches the eager
one to within 1% at every configuration measured, so the ~1000 launches cost
nothing that pipelining does not already hide. The two things that *do* cost are
matmul FLOPs (69% of the step) and elementwise passes over fp32 matrices at the
DRAM roofline. `phases` addresses the first and compilation the second; there is
nothing here for a CUDA graph to fix.

End to end, the shipping configuration measures **1.51x on the dense step and
1.22x on MoE** ([performance.md](../performance/performance.md)).

### Compilation

`compile_ns` compiles the Newton-Schulz iteration and the two elementwise passes
around it. The iteration is a fixed sequence of matmuls on a shape that repeats
every step, which is what makes compilation pay here rather than in the model:
measured **1.16x on the whole optimizer step** at Nano-500M, essentially all of
it from fusing the elementwise terms — the launch count only falls 1367 → 1267,
and a CUDA-graph replay of the step is the same time as the eager one either way.

The elementwise helpers matter more than the iteration does. Unfused, the
momentum EMA, the look-ahead and the gather into the batch are three passes over
an fp32 matrix where one suffices.

Two compilation details that are easy to lose:

- **`compile_ns=None` resolves from the parameters, not from `cuda.is_available()`.**
  Compiling emits Triton, which has no CPU backend, so a CPU model on a CUDA host
  would still fail at the first step.
- **The recompile limit is raised to `2 * shapes + 8`.** Two graphs per shape — a
  full batch and the remainder chunk — since the iteration is called once per
  (batch, matrix shape, phases) combination. Six of those on a dense 500M and more
  than eight on any MoE preset, so the default budget is spent partway through the
  first step.

### Two writeback details

**The step's scalars are 0-d tensors, not floats.** `torch.compile` guards on a
float's *value*, so an lr schedule would compile a fresh graph on every step —
measured, 2 graphs per parameter shape per distinct lr. A tensor is never
value-specialized. Those scalars are refilled in place rather than reallocated,
which is safe because the fill is queued on the same stream as the kernels that
read it, so the previous chunk's kernels have already consumed the previous value.

**The writeback is `addcmul_`, not `param * keep + update * alpha`.** A 0-d tensor
loses type promotion against a dimensioned one, so multiplying the bf16 update by
an fp32 0-d alpha would compute *in bf16* and round the step. `addcmul_` promotes
to the output dtype and is bit-identical to the `add_(update, alpha=float)` form
it replaced. Both promote the bf16 update inside the kernel; an explicit `.to()`
would materialize a second fp32 copy of the largest tensor.

**The momentum look-ahead is written out-of-place.** The reference implementation
writes it back into `.grad`, which corrupts anything reading gradients after the
step — the throughput callback logs the pre-clip grad norm.

---

## FusedAdamW

The Adam step is pure memory traffic. At fp32 the floor is **28 bytes per
parameter** — read `p, g, m, v`, write `p, m, v` — and `torch._fused_adamw_`
already hits it in one kernel, so there is nothing here for a hand-written Triton
kernel to win. What is *not* at the floor is everything around the step:

| path | bytes per parameter | why |
|---|---|---|
| `torch._fused_adamw_` | 28 | the floor |
| `foreach=True` (torch default here) | ~80 | every `_foreach_*` op is its own pass |
| gradient clipping, separately | +12 | one pass to reduce the norm, one to apply it |

`foreach=True` also materializes `sqrt(v)` into a fresh tensor list the size of
the parameters — 4 more bytes per parameter of *peak* memory, on a card where
Nano-1B under DDP is already at the edge of 32 GB.

**The clip pass is the one that disappears.** `torch._fused_adamw_` divides each
gradient by a device-resident `grad_scale` before the update — that is what makes
it AMP-aware — and a clip coefficient is exactly such a divisor. So clipping
becomes a scalar the kernel was already going to read, the read-modify-write pass
over every gradient is gone, and no value crosses to the host.

```python
from kohakuwullm.training.optim.fused_adamw import FusedAdamW

opt = FusedAdamW(groups, lr=3e-4, clip_grad_norm=1.0, state_dtype="bfloat16")
opt.step()
norm = opt.grad_norm()      # pre-clip, a device tensor; reading it is your sync, not the step's
```

When `clip_grad_norm` is set the optimizer *owns* clipping and **the trainer must
not clip again**: the gradients it leaves behind are already clipped, so a second
clip is a no-op that costs a full pass. `StochasticAdamW` takes the same argument
for the same reason, so the two are interchangeable from a config — though unlike
`FusedAdamW` it leaves the *unclipped* gradient behind, because it applies the
coefficient to a working copy rather than folding it into a kernel argument.

Three implementation notes:

- **Why a separate class rather than `torch.optim.AdamW(fused=True)`.** That class
  sets `_step_supports_amp_scaling`, and Lightning refuses to clip gradients for
  any optimizer carrying that flag — including under `bf16-mixed`, where there is
  no scaler and nothing to unscale. This is also why `build_optimizer` does *not*
  default `fused=True` for the plain `adamw` registry entry.
- **`fused=True` sits in the defaults dict** for `Optimizer.load_state_dict`,
  which otherwise leaves `state["step"]` on whatever device the checkpoint was
  written from — a CPU scalar the CUDA kernel would then read as a device pointer.
  The flag is what tells it to host the step count on the parameter's device, as
  fp32.
- **Tensors are bucketed by hand**, not through
  `_group_tensors_by_device_and_dtype`, because that helper keys on one dtype for
  all five lists — wrong as soon as the state is bf16 and the parameters are fp32.

**`state_dtype="bfloat16"`** keeps `exp_avg` / `exp_avg_sq` in bf16 against fp32
parameters, halving optimizer state from 8 to 4 bytes per parameter. The CUDA
kernel dispatches to a mixed-precision path for this; the CPU kernel has no such
path and will reject it, so the class raises with a message that says so rather
than letting the kernel report "expected scalar type Float but found BFloat16".

Both `FusedAdamW` and `StochasticAdamW` reduce the clip norm in fp32. Per-tensor
norms come back in the *gradient* dtype, and stacking several hundred bf16 norms
and reducing them there loses percent-level accuracy in the one number that gates
the whole step. The reduction is returned as a device tensor, so the step costs no
host synchronisation and the caller decides when to pay for the logged value.

---

## Quantized optimizer state (torchao)

`adamw8bit` / `adamw4bit` / `adamwfp8` quantize the Adam moments. That is only
half the memory problem, and it is the half that does **not** include the
parameters:

| state | bytes per parameter | end-to-end under `bf16-mixed` |
|---|---|---|
| fp32 moments | 8 | 16 |
| 8-bit (uint8 code + fp32 scale per 256-element block) | 2.03 | 10.0 |
| 4-bit | 1.06 | 9.1 |

The fp32 parameters and gradients — the other 8 of the 16 bytes — do not move.
Halving those is the job of the 16-bit path below, and the two compose.

torchao rather than bitsandbytes because these are a few hundred lines of plain
PyTorch behind `torch.compile` over a tensor-subclass state, so there is no
compiled CUDA extension that has to have been built for sm_120.

The quantizers are the published ones: Dettmers' block-wise *dynamic* map for the
signed first moment (arXiv 2110.02861), and at 4-bit a zero-excluding *linear*
map for the strictly positive second moment (arXiv 2309.01507). The two moments
do not share a map because a second moment that quantizes to zero sends
`1/sqrt(v)` to infinity.

A parameter's state is quantized only when `numel() >= 4096` and `numel()` divides
by `block_size`; the rest keeps fp32 state. Measured across this repo's presets
that leaves under 0.11M parameters — the norm scales — in fp32, so the byte counts
above are effectively the whole model.

The names are registered **unconditionally**, behind a deferred factory that
imports torchao at build time. Naming one of these in a config is what should fail
when torchao is absent, not importing the module — which turns the failure into an
install instruction instead of `unknown optimizer`.

One trap `build_optimizer` handles explicitly: torchao's Adam family defaults to
`beta2=0.999`. The quantized names are matched by *name* (they are factories, not
classes) so that changing `OPTIMIZER` cannot silently also change beta2 away from
this project's 0.95.

---

## 16-bit parameters

Under `bf16-mixed` the parameters are fp32 and `torch.autocast` casts every weight
on the way into every matmul. Lightning constructs that autocast with
`cache_enabled=False` (`lightning/pytorch/plugins/precision/amp.py`), so the cast
is **not amortised across micro-batches**: each weight is read at 4 bytes and
written at 2 on every forward, and the cast's backward moves 6 more. Holding the
parameter in bf16 deletes that traffic and halves the parameter and gradient
footprint.

### Why this is not `model.to(bfloat16)`

bf16 carries 8 significand bits, so a relative update below `2^-8` — about 0.4% —
rounds to nothing. With a typical `|update| / |weight|` around `1e-4` at
mid-training, round-to-nearest discards essentially every update and the run
**silently stalls**: the loss flattens, and nothing else says why.

`torch._fused_adamw_` rounds to nearest, which is why `FusedAdamW` *refuses*
low-precision parameters outright rather than quietly stalling on them.

### Two fixes, both implemented

Which one wins is a measurement, not a derivation, so `StochasticAdamW` carries
both.

**Stochastic rounding.** Round up with probability equal to the distance to the
upper neighbour. The update is then unbiased — small updates land with low
probability instead of never — at the cost of one 16-bit random draw per parameter
and no extra state.

The implementation is a bit trick, and it works because **bf16 is the top 16 bits
of fp32**: truncation *is* rounding toward zero, and the discarded low 16 bits
*are* the fractional position between the two bf16 neighbours. Adding a uniform
16-bit integer before truncating therefore carries into the bf16 mantissa with
exactly that probability. The sign bit needs no special case — IEEE-754 orders
magnitudes monotonically within a sign, so a carry always moves away from zero,
which is the correct neighbour on both sides.

It is restricted to bf16 deliberately. fp16 is *not* a bit-prefix of fp32 (the
exponent fields are 5 and 8 bits wide), so the same trick would silently produce
garbage near the fp16 range limits rather than merely being slow.

**Kahan summation.** Keep a compensation buffer holding what the last writeback
discarded and fold it into the next update. Deterministic, but costs 2 bytes per
parameter. The compensation is computed against the *decayed* working value rather
than the pre-update parameter, so the weight-decay rounding error is compensated
too.

Both are from [Zamirai et al., "Revisiting BFloat16
Training"](https://arxiv.org/abs/2010.06192); the stochastic variant is carried to
6.7B by [Ozkara et al.](https://arxiv.org/abs/2502.20566).

### What `StochasticAdamW` keeps in fp32

The moments stay fp32 by default and the **whole update is computed in fp32**;
only the writeback into the parameter is low precision, and only that writeback is
rounded by the selected rule. That split is the point — Adam's second moment is a
sum of squares over a whole run, and bf16 there is a separate and much less
forgiving decision from bf16 parameters.

The bias corrections are why the update is assembled in fp32 *even when the state
is bf16*: `1 - beta2**step` is within one bf16 ULP of 1.0 after ~2000 steps, and a
bf16 divisor there would quantise the second moment's scale for the rest of the
run.

fp32 parameters take the exact same path with a no-op writeback rule, so a model
that keeps some tensors in fp32 needs no second optimizer.

### `KEEP_FP32_DEFAULT`: what never gets cast

`cast_parameters_` casts a module's floating-point parameters in place, skipping
any name containing an entry of `KEEP_FP32_DEFAULT`. Together these are well under
0.1% of the parameters.

One constraint binds the list before any per-tensor judgement: **it must be a
superset of the tensors a quantized optimizer declines to quantize.** torchao skips
`numel() < 4096` and falls back to a plain `zeros_like(p)`, which inherits the
*parameter* dtype — so a tensor that is bf16 here and unquantized there gets a bf16
Adam moment, and a moment is a running EMA.
`tests/test_lowbit.py::test_keep_fp32_default_covers_the_unquantized_tail` pins the
containment, because the two sets are defined by unrelated rules (a name here, a
numel there) and agree only by coincidence. That argument alone covers `sink` and
`bias`. Past that floor:

| name | why it stays fp32 |
|---|---|
| `norm` | multiplies the entire residual stream, so its quantisation error lands on every activation instead of being averaged away over a contraction the way a projection's is |
| `router` | decides a discrete top-k, where a bf16 tie between two experts flips the assignment. The same prefix covers `expert_bias` (stepped by ±1e-3 per optimizer step, with no optimizer to round it stochastically) and `load_accum` (a token counter summed over a whole step) |
| `inv_freq` / `freq_dirs` | multiplied by position indices up to the context length, so 8 significand bits is radians of phase error at the long end of the sequence |
| `sink` | a per-head logit *inside* the softmax denominator, so its error is averaged over nothing. The attention biases do not share this reason: only q/k reach the scores, v is added after the softmax has weighted the values, and o lands after attention has closed |
| `bias` | matched on the bare word rather than on `attn.`, so a future biased projection anywhere is covered by construction. Over-matching only keeps a matrix in fp32, which is the harmless direction to be wrong in |

`cast_buffers` defaults **off**, unlike Lightning's `bf16-true`, which casts
everything. No float buffer in this repo is large enough for its dtype to show up
in a memory total, and all of them are either constants consumed by fp32 math or
counters, so the default trades nothing for one fewer way to corrupt a run.

The function returns how many parameters moved and how many stayed, for the
startup log: a keep-list that silently matches nothing looks exactly like a
keep-list that works.

### Selecting it

Nothing in the 16-bit path changes a default. From a config:

```python
PRECISION = "bf16-true"
OPTIMIZER = "kohakuwullm.training.optim.lowbit.StochasticAdamW"
OPTIMIZER_KWARGS = {"rounding": "stochastic"}
```

`rounding="nearest"` exists so a run can isolate the cost of bf16 parameters from
the cost of the rounding rule. It is not a sane production setting for bf16
parameters.

### Muon with stochastic rounding

`MuonW` takes `rounding="stochastic"` too, and it needs it *more* than AdamW does,
not less. Muon's update is a polar factor — direction with the magnitude discarded
— so every entry of a well-conditioned matrix takes a step of similar size. At
bf16's seven mantissa bits a uniformly small step is uniformly lost, rather than
lost only where the gradient was small.

The Muon path uses a fused Triton kernel
(`kernels/optim/stochastic_round.stochastic_round_update_`) that folds decay,
scale, round and store into one pass, replacing the compiled writeback. Two
consequences:

- **It takes plain floats, not the 0-d tensors the compiled writeback takes.**
  Those exist to stop `torch.compile` specializing on an lr schedule's value, and
  this path is not compiled. Reading a 0-d tensor back with `float()` would sync
  the device once per parameter per step — hundreds of syncs to save nothing.
- **The update is made contiguous at runtime, not by orientation.** `cubic5` ends
  on a matmul so its `.mT` is strided; the quintic ends on an add so it is not.

`nearest` stays the default because SR is only meaningful for low-precision
parameters and it is not free: a run that keeps fp32 masters would pay a rounding
rule it cannot benefit from.

The RNG is one monotonic counter, not a per-step seed. The stream is
`randint(seed, offset + i)` and the offset advances by each parameter's `numel`,
so every (step, parameter, element) triple is already distinct — and a counter
cannot accidentally repeat the way a step-derived seed does on resume. Both the
seed and the offset are host-side ints, so walking them costs no synchronisation.
