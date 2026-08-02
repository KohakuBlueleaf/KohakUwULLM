# Pipeline parallelism

## Why, on this hardware

DDP replicates the whole model on every card, so the model *plus its optimizer
state* must fit on one. At bf16 weights and fp32 AdamW moments that is ~16 bytes
per parameter:

| model | parameters | DDP state per card | fits in 32 GB? |
|---|---|---|---|
| Kohaku-500M | 546M | ~8.7 GB | yes, comfortably |
| Kohaku-1B | 982M | ~15.7 GB | yes |
| Kohaku-1.5B | 1514M | ~24 GB | yes, tightly |
| Kohaku-MoE-3B | 2907M | ~47 GB | **no** |
| Kohaku-MoE-8B | 7713M | ~123 GB | **no**, by 4x |

Note that DDP state is charged on **total** parameters, not active ones: every
expert is replicated on every card whether a token routes to it or not. That is
why the sparse rungs fall off this table so much earlier than their active count
suggests -- `Kohaku-MoE-3B` is 617M active and still does not fit.

Pipelining splits the *parameters* across cards -- each holds `1/N` -- and ships
only the boundary activations between neighbours.

It suits MoE particularly well. A sparse model reaches a given capacity with a
narrower `dim` than a dense one, and the tensor crossing the pipeline seam is
`(tokens, dim)`. So the dimension that grew (expert count) never crosses the
wire, and the dimension that does cross is smaller than the dense equivalent.

## Pipelining also wins on speed here, and the reason is the interconnect

`nvidia-smi topo -m` reports `NODE` for every GPU pair on this box -- no NVLink,
every transfer across the PCIe host bridge. A ring all-reduce of the whole
gradient is the access pattern that fabric handles worst, and it is a barrier:
the slowest leg sets the step. Pipeline boundary sends are 12-29 MB
point-to-point between adjacent stages and overlap with compute under 1F1B.

It shows up as variance as much as as mean. DDP rows measure 4-16% step spread
where pipeline rows sit at 0.1-0.6%, and MXFP8 makes DDP *worse* precisely
because it works: faster compute shrinks the step, so the collective becomes a
larger share of it.

The caveat on that sweep is in [performance.md](../performance/performance.md): its DDP arm runs
with gradient checkpointing and the dense pipeline arm does not, so the two are
not a clean throughput comparison -- only the mxfp8-vs-bf16 ratios *within* each
strategy are.

## The split is cost-balanced, not layer-balanced

`plan_for(config, num_stages)` charges each piece its real cost:

- a **block** costs its attention projections + the quadratic attention term +
  its feed-forward (with MoE experts counted at `top_k + num_shared`, since that
  is what a token actually touches);
- the **embedding** costs ~0 compute -- it is a gather;
- the **head** costs a full `dim x vocab` GEMM per token, which at vocab 65536 is
  more than several transformer blocks.

Splitting evenly by layer count would give the last stage the head *on top of* a
full share of blocks, and every other stage would idle waiting for it. The
planner gives the last stage fewer blocks instead.

```python
from kohakuwupipe.parallel.plan import describe
from kohakuwullm.models import get_preset
from kohakuwullm.training import plan_for

config = get_preset("Kohaku-MoE-3B", vocab_size=65536)
plans = plan_for(config, num_stages=4)
print(describe(plans))
```

`plan_for` is the LM-aware entry point: it turns a `config` into the per-layer and
head costs and hands them to the generic `plan_stages(depth, num_stages,
layer_cost, head_cost, ...)` in `kohakuwupipe.parallel.plan`. Call `plan_for`;
`plan_stages` does not take a config. `describe` takes only the plans.

```
pipeline split: 4 stages over 27 layers
  stage      layers    n         ends  cost share
      0        0..7    8        first       25.3%
      1       8..15    8            -       25.3%
      2      16..23    8            -       25.3%
      3      24..26    3         last       24.2%
```

The cost-share column above has **not been re-run** against the current cost
model; the columns are `describe`'s current ones and the split is the shape the
prose below depends on, but treat the percentages as illustrative.

The last stage gets 3 blocks where the others get 8. That gap *is* the head:
at vocab 65536 a `dim x vocab` GEMM per token costs more than five of this
model's blocks.

### The feed-forward width comes from the model, not from the config field

`_ffn_hidden` calls `resolve_hidden`, the same function the blocks use, rather
than reading `mlp_hidden` directly. The raw field is `None` on 8 of the 27
presets, and the fallback this replaced omitted the `multiple_of` ceiling --
predicting a width **1.0-4.8% under** the one actually built, so the stage split
disagreed with the model it was splitting. The MoE branch takes
`mlp_multiple_of`, matching what `_mlp_kwargs` hands the block.

This is the third place an FFN width had been defined independently. There is now
one.

### The tie-break is memory, not cost

`partition` (in `kohakuwupipe.parallel.plan`) is a dynamic program over
contiguous splits: `f[s][i]` is the best
achievable bottleneck for placing blocks `i..depth` over `s` remaining stages, so
the head's fixed cost is carried by the last stage *during* the recursion rather
than corrected afterwards.

Minimizing the bottleneck alone is under-determined once one stage dominates --
any arrangement of the rest ties. The tie is broken on **parameter count**
(minimizing the sum of squared per-stage parameter counts), not on cost, because
that is what decides whether a card runs out of memory: compute counts only the
experts that route, while memory holds all of them. `_block_params` (in
`training/parallel/pipeline.py`, feeding `plan_stages(layer_params=...)`)
therefore counts every expert, not the `top_k` that run.

## Equal FLOPs are not equal milliseconds

`plan_for` charges cost in FLOPs, but blocks and the head reach very different
fractions of peak. The head is one huge `dim x vocab` GEMM; a block is a stack of
smaller ones, and the two respond differently to dtype, autocast and vocabulary
size. `head_scale` is the correction:

| source | value |
|---|---|
| `HEAD_INEFFICIENCY`, the default | **1.55** |
| measured head cost, Nano-1B at vocab 65536 | 2.9 block-equivalents |
| FLOP-model prediction for the same | 1.88 |
| the same measurement against the old chunked head | 15.2 |

The default is calibrated from `scripts/bench/e2e/stage_balance.py`. **Recalibrate
when the head implementation changes** -- the 15.2 row is what the chunked path
gave, because it never reached the tensor cores at all.

`measure_head_scale` does that calibration at startup: it times one block and the
head and returns the ratio of measured to predicted head cost, so `plan_for`
plans against milliseconds instead of FLOPs. Two things keep it cheap enough to
run every time:

- **A one-layer copy of the config**, not the whole model, so this stays
  affordable even at 8B.
- **`probe_tokens` caps the measurement size** at 2048. Both the block and the
  head are linear in tokens, so the *ratio* barely moves with it, while the peak
  the probe itself allocates does: at 8192 tokens the unchunked head alone wants
  ~6 GiB. Measuring at the full microbatch size leaves rank 0 -- and only rank 0
  -- with a fragmented allocator immediately before its stage is built, which
  slows the stage every other rank waits on.

**The measurement is broadcast from rank 0, never taken locally.** Identical
hardware still yields slightly different timings, and ranks that disagreed about
the split would build mismatched stages. If calibration raises, every rank falls
back to `HEAD_INEFFICIENCY` with a warning.

Calibration is on by default because the FLOP model's default correction is
fitted on one dense preset and does not transfer to MoE, whose blocks are far
more expensive and so make the head a smaller share of a stage.

## Sequence metadata does not cross the wire

`torch.distributed.pipelining` chunks pipelined tensors along dim 0 to form
microbatches. For a packed batch that would slice through the middle of a
document and give the remainder wrong position ids.

So only the hidden states are pipelined. Each `LMStage` derives its own
`SeqInfo` from per-microbatch lengths registered before the step:

```python
stage.set_seq_info([info_for_microbatch_0, info_for_microbatch_1, ...])
```

Every stage computes the same list from the same lengths, so nothing has to be
sent. Lengths are a tiny int tensor; broadcast them once at step start.

`LMStage.forward` is `forward(x, aux=None)`. It ships one tensor in the ordinary
case and a tuple when a side channel is open:

| stage | signature |
|---|---|
| first | `tokens (T,) -> hidden (T, D)` |
| middle | `hidden -> hidden` |
| last | `hidden -> hidden` |

With `router_stream` on **and the module training**, every row above gains a
trailing `(1,)` accumulator: `-> (hidden, aux)`. See the arity note below and
[moe-router-loss.md](moe-router-loss.md).

The loss is applied outside the last stage's forward, so the schedule's `loss_fn`
stays the only place that sees the target.

## `input_args` freezes the boundary, and everything follows from that

`PipelineStage` must be constructed with explicit `input_args`. Without them it
runs a meta-device shape-inference pass, which **deadlocks** against a stage that
carries per-microbatch state.

The consequence is that the boundary's shape *and dtype* are frozen at
construction time, which constrains the order of three otherwise-unrelated
operations:

- **`microbatch_tokens` must be identical on every rank**, since it is what fixes
  the boundary shape.
- **Parameters must be cast before the stage is built.** A cast applied afterwards
  leaves the declaration describing a tensor nothing sends, and the runtime raises
  `PipeliningMetadataError` on the first step of every low-precision pipeline run.
- **The stage must *produce* the declared dtype**, rather than whatever its last
  block happened to return. `LMStage.boundary_dtype` casts on the way out.

That last one cannot be inferred from the parameter dtype, which gets dense right
and sparse backwards: a dense block ends in a residual add that stays bf16 under
bf16 parameters, while an MoE block's expert combine reduces in fp32 either way.
So the boundary carries `param_dtype` explicitly, declared and produced from the
same value.

The *arity* is frozen the same way. A boundary is a tuple when the model needs a
side channel alongside the activation — today that means a router auxiliary loss,
which is produced on every stage and can only be applied on the one that owns the
head. `LMStage.router_stream` decides it, once, from the **whole** backbone rather
than from this rank's slice, so a model with a dense prefix cannot end up with
rank 0 sending one tensor while rank 1 expects two. See
[moe-router-loss.md](moe-router-loss.md#getting-the-term-to-the-loss-under-pipeline-parallelism).

## Autocast belongs to the stage

A pipeline schedule calls the stage directly, so there is no trainer step and no
Lightning precision plugin in the path. Two things go wrong if the stage does not
supply its own autocast: the matmuls run in fp32, and attention **silently** drops
to the masked-SDPA path, since `varlen_attn` accepts only fp16/bf16.

`AutocastStage` wraps the stage's `forward` and `loss` in
`torch.autocast("cuda", dtype=...)`, and the Trainer runs at `precision="32-true"`
so nothing else tries to manage precision as well.

Lightning has no first-class pipeline strategy, so `PipelineOnlyStrategy`
subclasses `DDPStrategy` for **launch only** -- process launch, rank/device
assignment and NCCL process-group init -- and overrides `configure_ddp` to a no-op,
which is what would otherwise wrap the module in DDP. The loop stays in
`PipelinedLMTrainer`, an ordinary `LightningModule` under an ordinary
`pl.Trainer`:

```python
trainer = pl.Trainer(
    accelerator="cuda",
    devices=4,
    strategy=PipelineOnlyStrategy(process_group_backend="nccl"),
    precision="32-true",          # AutocastStage owns precision
    accumulate_grad_batches=1,    # microbatches ARE the accumulation
)
trainer.fit(PipelinedLMTrainer(preset="Kohaku-MoE-3B", ...), loader)
```

`accumulate_grad_batches` must stay 1: `num_microbatches` is the gradient
accumulation, and Lightning stacking a second factor on top would multiply them.

## A tied head cannot span two ranks

If the split puts the embedding on stage 0 and the head on the last stage,
`tie_embeddings=True` is impossible: keeping the weight tied would require
all-reducing a `vocab x dim` matrix every step, which exceeds the boundary
activation the pipeline exists to minimise.

`LMStage` unties it into a stage-local copy and warns. Set `tie_embeddings=False`
in the arch config to make that cost explicit rather than discovering it in a
warning. `plan_for` already accounts for it: `head_params` is zero when the
config says tied, so the memory tie-break does not double-charge a matrix that
only exists once.

## Running one

`scripts/train/lm_pipe.py` is the production pipeline trainer, on the
`kohakuwupipe` loop. It is a KohakuEngine script that spawns its own ranks, so
one config file drives it and every knob is reachable:

```bash
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py

# ten steps, no network, synthetic data, one checkpoint
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py \
    --set MAX_STEPS=10 --set DATA_KIND=synthetic --set WANDB_PROJECT=
```

`GPUS` sets the rank count (0, the default, uses every GPU); a rank group the
caller already started — `RANK` in the environment — is used as-is.

`DATA_KIND="corpus"` reads KohakuVault through the `pipeline` loader;
`"synthetic"` is the benchmark stream and reuses one packed batch, which is what
makes it a measurement of the step rather than of the loader.

`--set` coerces from the script's declared default, and it does **not** parse a
list literal — pin `LAYERS` in a config file, not on the command line.

### The measured configuration, and why it is not the default one

On 4x RTX 5090 at Kohaku-MoE-1B, one flag at a time, 32-step runs:

| change | tok/s |
|---|---|
| baseline: analytic split, 8192x32, bf16 | 182.0k |
| `MXFP8=True` | 247.4k |
| split `5/4/4/3` at 8192x32 | 263.3k |
| `MICRO_TOKENS=16384, NUM_MICROBATCHES=16` | 277.2k |
| `AUTOTUNE=True` derives `5/5/5/1` at that shape | **313.9k** |
| all of the above, on the real corpus | 307k |

Two things here are worth more than the numbers.

**The optimal split moves with the micro-batch size.** `5/4/4/3` is right at
8192 tokens and `5/5/5/1` is right at 16384 — measured 277.7k vs 313.9k for the
two at 16384, a 13% swing in the *opposite* direction to the one at 8192. The
MoE layer is overhead-bound at 8192 (6.69 ms) and gets relatively cheaper at
16384 (7.87 ms for twice the work) while the head scales linearly (12.84 ->
25.69 ms), so the balance point moves. No constant can be right at both.

**1F1B pays for the spread, not just the maximum.** In isolation `5/4/4/3` moved
the slowest stage only 31.55 -> 30.67 ms (2.8%) at 8192, but end to end it was
worth 13%. Rank a candidate with the measured model; do not quote its predicted
speedup.

## Autotuning the split

`AUTOTUNE = True` (the default) measures instead of predicting. At startup it
times one block of each **distinct layer type**, the head and the embedding, at
the real micro-batch shape, dtype and fp8 setting, then partitions the measured
milliseconds:

```
measured stage costs: head 25.67 ms, embed 0.47 ms, 1x layer 5.82 ms, 15x layer 7.89 ms
```

A layer type is `(is_moe, attention backend, window)` — at most a handful even
at depth 33, so the whole measurement is seconds. `LMBackbone.build_block` is
the single place a block is constructed, so the probe is the block the model
builds and cannot drift from it. Costs are broadcast from rank 0: identical
cards still time differently, and ranks that disagreed would build mismatched
stages.

The analytic model remains as the fallback and is wrong three ways, all of which
measurement sidesteps: `HEAD_INEFFICIENCY = 1.55` overcharges the head; callers
pass `micro_tokens` as `seq_len`, which in `_block_cost` is the *document*
length, so a 50-600 token corpus overcharges attention roughly 25x; and every
layer is priced as MoE even when `moe_first_dense` makes the first one dense.

`LAYERS = [5, 4, 4, 3]` still pins a split explicitly and skips both.

## Wiring it up

`LMStage` wraps a contiguous slice of an existing `LMBackbone`, so the stages
share the backbone's parameter objects rather than copying them:

```python
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training import LMStage, plan_for

config = get_preset("Kohaku-MoE-3B", vocab_size=65536)
backbone = LMBackbone(config)
plans = plan_for(config, num_stages=dist.get_world_size())
stage_module = LMStage(backbone, plans[dist.get_rank()]).cuda()

stage = PipelineStage(
    stage_module, dist.get_rank(), len(plans), torch.device("cuda"),
)
schedule = Schedule1F1B(stage, n_microbatches=8, loss_fn=loss_fn)
```

`build_stage` in `training/parallel/pipeline_lightning.py` does all of the above
in the right order, including calibration, the cast and the `input_args`
declaration. Prefer it to assembling the pieces by hand.

Schedule choice, cheapest first in bubble terms:

| schedule | when |
|---|---|
| `ScheduleGPipe` | simplest; largest bubble. Fine for a smoke test. |
| `Schedule1F1B` | the default. Same bubble as GPipe but far less activation memory. |
| `ScheduleInterleaved1F1B` | smaller bubble, needs >1 stage chunk per rank. |
| `ScheduleZBVZeroBubble` | near-zero bubble; splits the backward into dW/dX. |

Only the first two are selectable. Both trainers build **one** stage per rank,
and the bottom two schedules are multi-stage: they take a *list* of stage chunks
and fail with `'PipelineStage' object is not subscriptable` when handed one.
Offering them as a `SCHEDULE` value was advertising a knob that could only
crash, so `SCHEDULES` now holds `1f1b` and `gpipe` and an unknown name raises
with the pair it does have. Reinstating interleaving means assigning each rank
several non-contiguous chunks, which `plan_for` does not model.

Each stage owns a disjoint parameter set, so its optimizer is genuinely local:
there is no cross-rank reduction to arrange.

**`build_schedule(scale_grads=False)`, unlike torch's default.** The schedule
would otherwise divide gradients by the microbatch count, which double-counts when
`loss_fn` already normalizes by the step's total trained tokens -- yielding
gradients exactly `1/n_microbatches` of the correct value. Leaving it False and
letting the loss own normalization is also what keeps variable-length microbatches
weighted by tokens rather than by count.

Every rank must call `run_step` exactly once per step and in the same order; a
rank that skips or returns early hangs the others.

## Combining with DDP

Pipeline and data parallelism compose: build a 2-D device mesh
(`pp` x `dp`), pipeline within a `pp` group and all-reduce across `dp`. On four
5090s the useful configurations are `pp=4, dp=1` (biggest model) or `pp=2, dp=2`
(better throughput if the model fits in half the cluster).

Note `find_unused_parameters=True` is often cited as required for MoE under DDP:
routing sends different tokens to different experts each step, so an expert can
receive no tokens in a given backward.

**`scripts/train/lm.py` does not actually turn it on for any shipping MoE config.**
The guard is `bool(ARCH_OVERRIDES.get("moe_every")) or (preset or "").startswith("MoE")`;
no `Kohaku-MoE-*` name starts with `MoE`, and no shipping MoE config sets
`moe_every` in `ARCH_OVERRIDES`, so it evaluates False for all of them. It fires
only for the retired `MoE-*-A*` names. Whether the flag is needed here is
separately doubtful — the experts are stacked `(E, ...)` parameters that take a
grouped-GEMM gradient every step — but the guard as written does not do what its
name suggests.

**The loss divisor is the third place a `world_size` factor hides**, after
`build_schedule(scale_grads=False)` and the DDP gradient average. The rule, in
one line: divide by the trained-token count summed over the **data-parallel group
only**, then by that group's size.

* Pure pipeline (`dp=1`): the DP group is a single rank, every stage sees the same
  step, and stages own disjoint parameters -- there is no cross-rank averaging, so
  a step-local count is already correct.
* DDP (`pp=1`): what `LMTrainer.training_step` does. DDP *averages* gradients, so a
  rank dividing by its own count yields the mean of per-rank means; dividing by
  `global / world_size` recovers the true per-token mean.
* Hybrid (`pp>1, dp>1`): reduce over the `dp` process group and **never** across
  the `pp` dimension. Every stage of a pipeline processes the same tokens, so
  summing across stages counts each token once per stage and shrinks the gradient
  by the stage count -- silently, and absorbed by whatever LR was tuned around it.

One consequence for the token-budget loaders: the per-step token reduction is a
collective every rank must reach the same number of times, so a rank that runs out
of batches first now hangs there rather than in the gradient reduction. Same
failure one step earlier, and the same fix -- pin `batches_per_epoch`, which
`build_ddp_loader` already warns about above one rank.

## Checkpoints must hold the whole model, not this rank's stage

Each rank's `LightningModule` *is* its stage, so the default
`strategy.lightning_module_state_dict()` returns one slice and Lightning writes it
from rank 0 alone. That produces a file which **loads without error and is wrong**:
at `pp=4` it carries a quarter of the layers, renumbered from zero, so they land in
the wrong positions of any full backbone that reads it.

Measured on a 2-stage 8-layer smoke run before the fix: 7.08M of 14.16M parameters,
block indices `0,1,2,3` only.

`PipelineOnlyStrategy` therefore redirects both directions:

| hook | what it does |
|---|---|
| `lightning_module_state_dict` | `all_gather_object` over every stage's slice, keyed by whole-backbone names |
| `load_model_state_dict` | hands each rank only the tensors its own stage owns |
| `optimizer_state` | `gather_object` onto rank 0, kept as a **per-rank list** |
| `load_optimizer_state_dict` | gives each rank the entry at its own index |

Weights are merged into one model; optimizer moments are not. Stages own disjoint
parameters, so there is nothing to merge, and a rank loading another stage's
moments fails loudly -- `loaded state dict contains a parameter group that doesn't
match the size of optimizer's group` is what a resume did before this landed.

**The `kohakuwupipe` loop has moved past that, and the Lightning path has not.**
`kohakuwupipe/io/checkpoint.py` now stores optimizer state under whole-model
*parameter names* (`named_optimizer_state`), gathered and merged the same way the
weights are, so a file written at `pp=4` resumes into `pp=2`, into DDP, or on one
GPU. The per-rank list above is still read when it is the only thing in the file,
and is never written by that loop -- but reading it now **raises** unless this
run has the world size that wrote it, instead of handing rank *r* some other
split's stage *r* and letting torch report a param-group size mismatch.

Note what that means for a run in flight: the layout a process writes is the one
it imported at startup, so a job launched before the change keeps producing
`pipeline_stage_optimizer_states` files until it is restarted, and those files
resume only at their own rank count. Check a file before planning a re-split.

`PipelineOnlyStrategy` -- the table above -- still writes
`pipeline_stage_optimizer_states`, so a Lightning-pipeline checkpoint remains
split-locked. See
[../kohakuwupipe/checkpoint.md](../kohakuwupipe/checkpoint.md#optimizer-state-is-keyed-by-whole-model-parameter-name).

The renumbering is the part worth stating explicitly. A stage numbers its blocks
from zero; the backbone numbers them from `plan.start_layer`. `LMStage.global_names`
is the one place that mapping exists, and both directions go through it:

```python
stage.global_state_dict()                 # blocks.0 -> blocks.{start_layer}
stage.load_global_state_dict(checkpoint)  # and back
```

The artifact is a plain `LMBackbone` state dict, so a pipeline-trained checkpoint
loads on one card for inference with no pipeline involved:

```python
backbone = LMBackbone(config)
backbone.load_state_dict(torch.load("last.ckpt")["state_dict"], strict=False)
```

`scripts/dist/pp_checkpoint_smoke.py` pins both directions, and pins them **by
tensor name**. A count-based check passes by coincidence at `pp=2`: the module
registers its stage twice (as `backbone` and inside `stage_module`), so a
half-saved model has exactly the whole model's parameter count. The resume leg
sets `max_steps` to what the checkpoint already reached, so `fit` restores and
stops without training and the weights must come back bit-identical:

```
[resume] rank0 recovered 41 tensors bit-exactly, blocks.0.attn.k_norm.weight .. embed.weight
[resume] rank1 recovered 42 tensors bit-exactly, blocks.4.attn.k_norm.weight .. head.weight
```

Note which ranks report which names -- that is the assertion. A resume that gave
every rank stage 0's slice would print the same range twice.

## Sample previews are collective, whichever path they take

Whatever else changes, the invariant does not: **every rank must enter a preview
the same number of times.** The rank-0-only guard a preview callback would
normally use is what hangs the run.

There are now three paths, and the default is no longer the schedule.

| path | selected by | what a decode step costs |
|---|---|---|
| **gather + local decode** (default) | `SamplePreview(local=True)`, `SAMPLE_LOCAL = True` | one `all_gather_object` per preview, then no pipeline at all |
| forward-only pipelined | `local=False`, `SAMPLE_FORWARD_ONLY = True` | one `dist.send`/`recv` hop per stage per token |
| schedule pipelined | `local=False`, `SAMPLE_FORWARD_ONLY = False` | a full `ScheduleGPipe` step per token |

The default gathers every stage's slice onto rank 0, loads it into a whole
`LMBackbone`, and decodes there with `LocalGenerator`. Only the gather is
collective; the decode is not. Previewing by pipelining every token cost one
pipeline traversal per token, which is what this replaced.

`PipelinedLMTrainer.generate` (the Lightning path) does not gather. It builds a
`PipelineGenerator` with `microbatches=1` and takes that generator's default
`forward_only_decode=True`, so it too no longer runs the schedule.

`SampleLogCallback` reads `generate_is_collective` off the module: when it is set,
every rank runs the generation loop and only rank 0 decodes and prints. Without it,
the callback keeps the old behaviour and disables itself with a warning.
`PipelinedLMTrainer` sets it; the `kohakuwupipe` loop uses `SamplePreview`
instead, which is collective by construction.

Two details that are easy to get wrong on either pipelined path:

- **The decode stage is not the training stage.** `PipelineStage` freezes its
  boundary at construction, and training declares a packed `(micro_tokens,)` one
  while cached decode needs a padded `(rows, 1)`. `decode_stage` builds the second
  over the *same module*, so it carries the trained weights and allocates no
  second copy.
- **The head runs outside the stage forward.** `AutocastStage` wraps `forward` and
  `loss`, but the generator calls `head.logits` itself, between decode steps. It
  therefore takes the stage's autocast dtype and re-enters it; without that, an
  fp32 hidden meets a bf16 projection and the last rank raises while every other
  rank has already returned.

See [../guides/generation.md](../guides/generation.md).
