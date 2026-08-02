# MoE router losses: the fused kernel, and getting them across a pipeline

Two auxiliary terms come out of the router, and both used to force it onto the
eager path:

| term | formula | what it does |
|---|---|---|
| load balance (`aux_loss_weight`) | `w · E · Σ_e f_e · P_e` | equalizes expert load through a gradient |
| router z-loss (`z_loss_weight`) | `w · mean_t (logsumexp_e z_te)²` | pins the gate's softmax normalizer near 1 |

with `f_e = c_e / (T·k)` the fraction of routed slots expert `e` won and
`P_e = (1/T) Σ_t s_te` the mean score it received.

Neither is on by default. The shipping recipe is DeepSeek-V3's **aux-loss-free**
balancing (`bias_update_rate`), which steers selection with a per-expert bias
updated by a rule rather than by a gradient — see
[architecture.md](../concepts/architecture.md). The terms exist so an ablation is
a config change, and this document is about what it costs to make that config
change cheap and correct.

---

## Why the fused router used to refuse them

`fused_router` computes the gate GEMM, the activation, the balancing bias, the
top-k selection and the load histogram in one launch, and its backward saves only
`(T, top_k)` — the selected scores. That is enough for the gate weights, because
both fused activations are *elementwise*: a selected score's gradient depends on
that score alone.

It is not enough for either auxiliary term. `P_e` is a mean over **every** token,
including the ones that did not select `e`; `logsumexp_e z_te` couples the whole
row. So `TopKRouter.use_fused` used to require `aux_loss_weight == 0.0 and
z_loss_weight == 0.0`, and a config that set either dropped silently to the eager
path — an fp32 `(T, E)` GEMM, a `topk`, a `gather` and a `scatter_add`, roughly
six launches where the fused router is one.

That mattered most in exactly the setup where the terms are interesting. At
MoE-1B (depth 16, `moe_first_dense=1`, so 15 sparse layers) with 32 micro-batches,
the eager router is ~2400 extra launches per optimizer step.

## What it costs now

Kohaku-MoE-1B, pp4 on 4x RTX 5090, `micro_tokens=8192`, 32 micro-batches, bf16,
Muon, 24 steps, throughput read after the 8-step warmup. One knob apart, produced
by `scripts/train/lm_pipe.py` on the **kohakuwupipe** loop with `DATA_KIND=synthetic`:

| arm | tok/s | ms/step | peak GiB |
|---|---|---|---|
| no auxiliary term | 182.0k | 1441 | 18.75 |
| `AUX_LOSS_WEIGHT=1e-3` | 181.6k | 1444 | 18.78 |
| `ARCH_OVERRIDES={"z_loss_weight": 1e-4}` | 114.5k | 2289 | 18.75 |

The router term costs **0.2%** and 30 MiB. The head's z-loss — a different thing
with the same name, below — costs **1.59x**.

The 182.0k baseline is **not** comparable to
[performance.md](../performance/performance.md)'s 159,907 for the same rung and
nominal shape: that table is the Lightning loop, this one is kohakuwupipe. Only
the deltas within this table mean anything.

```bash
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py \
    --set MAX_STEPS=24 --set AUX_LOSS_WEIGHT=1e-3
```

`AUX_LOSS_WEIGHT` and `ROUTER_Z_LOSS_WEIGHT` are the router's; the head's own
z-loss is an arch field, reached through `ARCH_OVERRIDES={"z_loss_weight": ...}`.

## What the forward adds

Two reductions, alongside the load histogram that was already there:

```python
if NEED_AUX:
    column = tl.sum(tl.where(mask_t[:, None], scores, 0.0), axis=0)
    tl.atomic_add(stats_ptr + offs_e, column, mask=mask_e)          # score_sum, (E,)
if NEED_Z:
    lse, _prob = _logsumexp(acc, mask_e)
    tl.atomic_add(
        stats_ptr + E + tl.arange(0, 1),
        tl.sum(tl.where(mask_t[:, None], lse * lse, 0.0), axis=0),  # Σ lse², scalar
    )
```

Both land in one `(E + 1,)` buffer, so enabling the terms costs one extra memset
launch and no extra pass over the data. `NEED_AUX` and `NEED_Z` are `constexpr`,
so a run with the terms off compiles a kernel that does not contain them.

The two loss values are then three small ops on the host side:

```python
aux = torch.dot(counts.float(), stats[:e]) * (aux_weight * e / (t * top_k * t))
z = stats[e] * (z_weight / t)
```

`counts` is an int32 histogram accumulated by an int32 atomic, so it is exact
regardless of order; the fp32 atomics are `stats` and `load_accum`.
`score_sum` accumulates in fp32 across `cdiv(T, 64)` programs and is therefore
order-dependent at the 1e-7 level — which **only affects the reported value**.
Neither gradient reads it (see below), so nothing about training depends on the
atomic order.

### Why the value and the gradient can disagree about determinism

```
∂aux/∂z_te = w · E · c_e / (T² · k) · gate'(s_te)
∂z  /∂z_te = w · (2 · lse_t / T) · softmax(z)_te
```

The aux gradient reads `c_e` (an integer count) and the per-element score; the
z-loss gradient reads a per-token `lse` and a per-row softmax. Neither reads the
accumulator its *value* came from. This is worth knowing when a loss curve is
compared across runs: the logged `router_loss` is reproducible to ~1e-7, the
weights are reproducible exactly.

## What the backward adds

`_router_bwd_logits` already wrote a dense `(T, E)` gradient tile — but sparsely,
scattering the `top_k` lanes into a zeroed tensor and touching nothing else. Both
auxiliary terms put gradient on *every* lane, so the kernel gains a `DENSE`
specialization that assembles the whole row in registers instead:

```python
dense = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
for k in range(TOP_K):                            # the selected lanes, as before
    slot = mask & (offs_k[None, :] == k)
    value = tl.sum(tl.where(slot, grad, 0.0), axis=1)[:, None]
    lane = tl.sum(tl.where(slot, idx, 0), axis=1)[:, None]
    dense += tl.where(offs_e[None, :] == lane, value, 0.0)

logits = tl.load(logits_ptr + ..., mask=row_mask, other=0.0)
if NEED_AUX:
    counts = tl.load(counts_ptr + offs_e, mask=mask_e, other=0).to(tl.float32)
    dense += (aux_scale * tl.load(gaux_ptr)) * counts[None, :] * _gate_grad(
        _gate(logits, SCORE_FUNC), SCORE_FUNC
    )
if NEED_Z:
    lse, prob = _logsumexp(logits, mask_e)
    dense += (z_scale * tl.load(gz_ptr)) * lse * prob
```

The scatter is a broadcast compare per slot rather than a `tl.store` at a
computed address, which is what lets the selected and the dense contributions be
summed before a single store. A token's `top_k` indices are distinct, so the
lanes never collide.

`gaux_ptr` and `gz_ptr` are the **incoming gradients of the two loss outputs**,
read from device memory. They are almost always exactly 1.0, but reading them
with `.item()` would put a host sync inside every MoE layer's backward.

Three consequences of this design worth stating outright:

- The forward saves `logits` as an extra `(T, E)` fp32 tensor when either term is
  on. At `T=8192, E=64` that is 2 MiB per layer per microbatch — against a
  `(8192, 2048)` bf16 activation at 32 MiB, it is noise.
- The sparse path is untouched. `DENSE` is `False` when no term is on, and the
  kernel compiles to what it compiled to before.
- The backward's `torch.zeros` becomes `torch.empty` on the dense path, because
  every valid `(t, e)` is written.

## The eligibility rule now

```python
self.use_fused = (
    fused
    and score_func in ("sigmoid", "sqrtsoftplus")
    and n_groups == 1
    and num_experts <= 128
)
```

A softmax gate and group-limited routing still force the eager path, for the
reason they always did: the *forward* needs the full row, not just the backward.

## Getting the term to the loss under pipeline parallelism

A pipeline stage that is not the last one has no loss to add anything to. Every
stage produces router terms; exactly one stage owns the head. Before this, an
`LMStage` raised at construction if it held a layer with a non-zero
`aux_loss_weight`, which was honest but not a solution.

The mechanism that carries it is the multi-stream boundary from
[kohakuwupipe](../../src/kohakuwupipe/parallel/streams.py): a stage's boundary is
a tuple, and one of its streams is a `(1,)` fp32 **accumulator** every stage adds
to.

```python
def forward(self, x, aux=None):
    ...
    if not (self.router_stream and self.training):
        return x
    carried = accumulator(x.device) if aux is None else aux
    return x, accumulate(carried, (self.router_losses(),))
```

and on the last stage the loss function unpacks it:

```python
def loss_fn(output, target):
    streams = split_streams(output)
    loss, _ = stage_module.loss(streams[0], target)
    loss = loss / max(denom(), 1)
    for stream in streams[1:]:
        if stream is not None and stream.shape[-1] == 1:
            loss = loss + reduce_accumulator(stream)
    return loss
```

`d(total)/d(acc) == 1` at every hop, so a term a stage contributed reaches the
optimizer worth exactly itself — no `1/num_stages`, no double counting.

## The coefficient you set has to be the coefficient you get

Both loops accumulate over micro-batches, and both got this wrong, in opposite
directions. The term is a **mean over one micro-batch's tokens**, so a step must
*average* the micro-batches' terms, not sum them, and must not divide them by a
token count.

| loop | what it did | `aux_loss_weight=1e-3` trained as |
|---|---|---|
| pipeline (`build_loss_fn`) | summed 32 micro-batch means | 3.2e-2 |
| Lightning (`LMTrainer`) | folded into a sum-reduced CE, then `/ step_tokens` | 1.5e-8 |
| both, now | mean over micro-batches | 1e-3 |

Neither has a symptom. The run trains; the coefficient is simply not the one in
the config. The pipeline arm is the visible one only because 32x is large enough
to dominate the loss — MoE-1B opens at 24.8 rather than the `ln(65536) = 11.09`
a fresh model should.

The fix is one line on each side. In the pipeline, the accumulator is scaled
before it joins the loss:

```python
aux_scale = 1.0 / max(num_microbatches, 1)
...
loss = loss + aux_scale * reduce_accumulator(stream)
```

In the Lightning loop, `LMBackbone.loss` gained a `router_scale` the caller sets
to undo the normalization it is about to apply to the CE:

```python
router_scale = denom / len(micro)          # denom is the step's trained tokens
loss, logs = self.backbone.loss(..., reduction="sum", router_scale=router_scale)
self.manual_backward(loss / denom)         # -> mean_CE + router / len(micro)
```

`router_scale` defaults to 1.0, which is already correct for a
`reduction="mean"` caller — every benchmark script is one.

`tests/test_training.py::test_the_auxiliary_term_does_not_scale_with_the_micro_batch_count`
pins both halves.

### The four things that make this correct rather than merely working

**`router_stream` is a whole-model property, not a per-stage one.** It is
answered from `backbone.blocks`, not `self.blocks`:

```python
self.router_stream = plan.num_stages > 1 and router_stream_enabled(backbone.blocks)
```

A stage that happens to hold only dense layers still opens the stream and passes
the accumulator through untouched. If it decided for itself, a dense-prefix model
would have rank 0 sending one tensor while rank 1 expected two, and
`PipelineStage` freezes those shapes at construction.

**The accumulator requires grad from the start.** `accumulator()` returns a leaf
with `requires_grad=True`. A boundary tensor with no backward edge gets no
gradient send at all, so a stage with no terms of its own would break the chain
for every stage behind it.

**`reduce_accumulator`, not `.sum()`.** `stream.sum()` builds its gradient by
expanding a scalar to stride 0, and NCCL rejects a non-dense tensor for P2P:

```
Tensors for P2P must be non-overlapping and dense
```

`(stream * torch.ones_like(stream)).sum()` is the same value with a materialized
gradient. Widening the stream and calling `.contiguous()` both fail — the
non-dense tensor is the *gradient*, not the stream. [kohakuwupipe/streams.md](../kohakuwupipe/streams.md) has the bisection that found it.

**The stream is training-only.** Generation builds its own `PipelineStage` with a
decode-shaped boundary and runs under `eval()`, where there is no loss to add a
term to; `LMStage.forward` returns a bare tensor there. Running a training-shaped
stage in eval mode would change the boundary arity, which is why the sampling
path toggles `train()`/`eval()` around the whole model rather than around a
submodule.

### What a stage emits, declared before it runs

`PipelineStage` needs the boundary shape before any forward has happened, so
whether a router emits a term has to be answerable from the config alone:

```python
self.emits_loss = aux_loss_weight > 0.0 or z_loss_weight > 0.0   # TopKRouter
self.emits_loss = True                                            # ReLURouter (ReMoE)
```

`ReLURouter` is `True` unconditionally: its L1 penalty *is* its sparsity
mechanism, not an option. `SinkhornRouter` and `ExpertChoiceRouter` are `False`.

## The head z-loss is a different thing, and it is off

`LMHead(z_loss_weight=...)` regularizes the *vocabulary* logits and is unrelated
to the router term of the same name. It is not on the head's optimized path:
`kernels/loss/zloss.py` is a second full pass over `dim × vocab`. Measured at
MoE-1B, `micro_tokens=8192`, `vocab=65536`:

| head | forward+backward |
|---|---|
| `chunked_ce`, no z-loss | 12.83 ms |
| `chunked_ce`, `z_loss_weight=1e-4` | 54.82 ms |

That 4.27x lands entirely on the stage that owns the head, which is the pipeline's
critical path, and shows up end to end as the 182.0k → 114.5k tok/s in the table
above. No config in `configs/lm/` sets it any more, and nothing in the literature
asks for it: DeepSeek-V3 trained 14.8T tokens with **no** auxiliary loss of any
kind, z-loss included, and reported the run as "remarkably stable".

## Running the tests

```bash
.venv/bin/python -m pytest tests/test_kernels.py -k auxiliary -q     # kernel, fp16+bf16
.venv/bin/python -m pytest tests/test_training.py -k router_loss -q  # the stream
.venv/bin/python scripts/kohakuwupipe/streams_demo.py --case aux
```

The kernel test runs the terms at `aux_loss_weight=50.0`, far above any training
value. At a shipping coefficient the term is a ~1e-3 correction to the gate
gradient, and no comparison against eager could separate it from GEMM noise — so
the test would pass with the dense contribution deleted.

---

## Scheduling the balancing bias

`bias_update_rate` is fixed when the router is built, and for most of a run that
is what you want. DeepSeek-V3 did not leave it fixed to the end: it ran γ=0.001
for the first 14.3T tokens of 14.8T and then set it to **0.0** for the last 500B,
so the router specializes freely once the experts are trained.

`RouterBiasSchedule` does that, as a curve rather than a switch. It multiplies
the built rate by an AnySchedule factor every step:

```python
BIAS_SCHEDULE = {
    "mode": "composer",
    "end": -1,
    "schedules": [
        {"mode": "constant", "end": 0.9},
        {"mode": "cosine", "end": 1.0, "min_value": 0.0},
    ],
}
```

That holds the rate flat for 90% of the run and anneals it to zero over the last
10%. A plain `{"mode": "cosine", "end": -1, "min_value": 0.0}` decays from the
first step instead, which balances least where balance matters most, so prefer
the composer. `end: -1` is filled from `MAX_STEPS` by `autofill_schedule_steps`,
the same way the learning rate is. The config is read through
`anyschedule.utils.get_scheduler`, which takes a config and returns a callable
without an optimizer to wrap.

Two properties worth knowing:

- **A zero rate freezes the bias but not the reporting.** `update_bias` still
  runs, so `load_accum` still fills and `moe/load_imbalance_max` still reports.
  Freezing by skipping the call would have gone blind exactly when the routing
  stops being steered.
- **The factor multiplies the built rate, not the current one.** The callback
  reads the base once at train start. Re-reading it each step would compound the
  factor into a geometric decay that no config describes, and a resume would land
  somewhere off the curve. `tests/test_training.py::test_router_bias_schedule_scales_the_built_rate_without_compounding`
  pins both halves.

### What to watch while it anneals

`moe/load_imbalance_max` alone cannot tell you whether annealing was safe,
because the same value means two different things. A router that genuinely wants
near-uniform load and one that is being forced there by a large bias both report
about 1.1.

The distinguishing quantity is the **magnitude of `expert_bias`**, which is not
logged today. If the spread is small compared with the gap between adjacent
expert ranks, roughly `1/E` for sigmoid scores, the router is doing the balancing
and annealing is safe. If the spread is large and still growing, the balancer is
carrying the load and removing it will move the routing.
