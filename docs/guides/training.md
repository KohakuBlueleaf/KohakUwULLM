# The training loop

`LMTrainer` (`training/loop/trainer.py`) is a Lightning `LightningModule` that
trains a causal LM on packed variable-length batches. This document explains the
four decisions that shape it, the accounting it keeps, what a checkpoint has to
carry to be resumable, and the callbacks that make a long run legible while it is
still going.

For what to *set*, see [writing-configs.md](writing-configs.md). For the
optimizers it builds, see [optimizers.md](../internals/optimizers.md).

**There are two training loops.** This one, on Lightning, is the DDP path and
drives `scripts/train/lm.py`. A model split across cards runs on
`scripts/train/lm_pipe.py` instead, over the `kohakuwupipe` loop — different
entry point, different config globals, same corpus and same checkpoint format.
See [pipeline.md](../internals/pipeline.md#running-one).

## Running one

```bash
kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py

# resume from a crash: weights, optimizer, schedule, RNG, data position.
# CHECKPOINT_PATH alone loads weights only; TRAINER_RESUME is what continues the run.
kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py \
    --set CHECKPOINT_PATH=out/ckpt/TIPO-MoE-1B/last.ckpt --set TRAINER_RESUME=True

# one knob, no file edit
kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py --set LR=3e-4
```

The schedule half of a config, which is the part with the least obvious defaults:

```python
MAX_STEPS = 100_000
LR = 5e-4                        # AdamW groups
OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}
WEIGHT_DECAY = 0.1

SCHEDULER_CONFIG = {
    "lr": {
        "mode": "composer",
        "end": -1,               # autofilled from MAX_STEPS
        "schedules": [
            {"mode": "power", "end": 0.9, "s0": 2500, "b": -0.5},
            {"mode": "cosine", "end": 1.0, "min_value": 0.01},
        ],
    }
}
SCHED_WARMUP_RATIO = 0.02        # 2000 steps here -- a fixed count, not a fraction
```

**Warmup is a fixed 2000 steps at every run length**, expressed as a ratio only
because that is the knob's unit. The mechanism it addresses — preconditioned
sharpness early in training — does not scale with the total step count, so a run
4x longer does not want 4x the warmup.

`s0` sets the power schedule's halving distance: the multiplier is
`((step + s0) / s0) ** -0.5`, so the rate halves at `3 * s0` and again at `15 * s0`.
Use **2500** for 100–250k steps, **3333** for 250–500k, **5000** beyond that. The
last 10% is a cosine to `0.01x` the peak.

---

## Four decisions

**Manual optimization.** `automatic_optimization = False`. Gradient accumulation
is done by hand so that every backward can be scaled by its micro-batch's share
of the step's *trained tokens*, not by `1/K`. That is the whole reason for the
manual loop; the next section says why `1/K` is wrong here.

**Select, don't dispatch.** Architecture, optimizer, scheduler and compile policy
resolve once in `__init__` and bind as attributes. The training step calls them.
There is no per-step branch on a config value anywhere in the loop.

**Token accounting is first-class.** For a language model the only meaningful
x-axis is tokens — two runs at different batch sizes are not comparable on steps.
Seen, trained, the trained fraction and both rates are tracked and globally
reduced.

**A checkpoint has to be enough to continue.** Lightning restores weights,
optimizer, schedule position and loop counters. The token totals *as global
numbers*, the RNG stream, and a dataloader position that lives on its dataset are
picked up here.

---

## Token-exact gradient accumulation

A dataloader batch is one optimizer step. `GRAD_ACC` splits that batch into
micro-batches; it does not multiply it.

The split respects the layout, read from the batch rather than configured, so one
dataloader choice flows through without a second switch to keep in sync:

- **packed** batches split on *document* boundaries, never at token offsets —
  cutting a document in half would misplace the positions of the remainder;
- **padded** batches split along the batch axis.

Because packing gives the micro-batches unequal token counts, scaling each
backward by `1/K` would reweight tokens by how they happened to be packed. Each
backward is instead divided by the step's total trained token count, which makes
the step gradient the true per-token mean.

**The divisor is global, not rank-local.** DDP *averages* gradients, so a rank
dividing by its own count yields the mean of per-rank means — equal to the true
per-token mean only if every rank packed the same number of trained tokens, which
packed varlen guarantees they did not. The step therefore all-reduces the trained
token count and divides by `global_tokens / world_size`.

That `world_size` factor is one of three places a world-size term hides; the
other two are in the pipeline path and are enumerated in
[pipeline.md](../internals/pipeline.md).

## Nothing in the step returns early

The token reduction, the backward and the logs are all **collectives**. A rank
that bailed out would hang the ranks that did join. So the empty-batch case is
reduced to a flag and handled, never short-circuited:

- **An all-masked step still runs its backward.** Every label being `IGNORE`
  makes the gradient exactly zero, so it is collective bookkeeping with no effect
  on weights — and skipping it hangs the ranks that had tokens.
- **`opt.step()` *is* skipped on an all-masked step.** With a zero gradient the
  only thing a step applies is decoupled weight decay, and decaying on a batch
  that taught the model nothing is not the intent. This is safe only because
  `empty_step` is the *reduced* flag: `global_step` counts `opt.step()` calls, so
  a rank skipping alone would fall permanently behind and the ranks would stop
  agreeing about which steps checkpoint.
- **Logging is gated on the same reduced flag.** `sync_dist=True` makes each
  `self.log` a collective, so every rank has to log the same set of keys. A step
  with no trained tokens also has no loss to report, and logging its `0.0` would
  spike the curve and drag the EMA.

The same rule governs `on_save_checkpoint`: nothing in it may move under an
`is_global_zero` guard, because the snapshot all-reduces and the loader's
`state_dict` all-gathers every rank's position. It is safe as written because
`ModelCheckpoint` calls `trainer.save_checkpoint` on every rank — only the file
write is rank-zero.

## Order of operations in a step

1. Split the batch into micro-batches by layout.
2. All-reduce the trained token count; compute the divisor.
3. For each micro-batch: forward, `loss / denom`, `manual_backward`.
4. **Unscale** gradients if the precision plugin has a scaler. Under fp16 the
   gradients are still scaled at this point, so measuring before unscaling would
   log a norm that tracks the scaler rather than the model. The scaler records the
   unscale per optimizer, so the subsequent clip does not repeat it.
5. Record the pre-clip global gradient norm — a run's first visible symptom of
   instability is a norm spike, not a loss spike.
6. Clip, then `opt.step()` unless the step was empty.
7. **`refresh_mxfp8_weights(self.backbone)`.** Omitting this is the one MXFP8
   mistake with no symptom: the quantized cache is built lazily on first forward
   and never invalidated, so every fp8 GEMM would run on initialization-time
   weights while the masters moved underneath. It sits here rather than in the
   micro-batch loop, where accumulation 4 would pay its ~13 ms four times for one
   result. See [mxfp8.md](../internals/mxfp8.md).
8. `sched.step()`, then `update_router_bias()` — a per-optimizer-step rule, so it
   updates here rather than in the forward; the load counter has to cover the
   whole step or the bias chases accumulation noise.
9. Update counters and log.

---

## Token accounting

Two token counts matter, not one. **Seen** is what the GPU computed on; **trained**
is what carried a gradient. They differ by the prompt half of every sample, which
is masked to `-100`, so a run whose renderer starts folding prompts into the
target moves one and not the other. A single "tokens" number hides that.

`TokenSnapshot` (`training/loop/tokens.py`) is a frozen dataclass holding
cumulative, globally-reduced progress at one instant: `seen`, `trained`,
`model_flops`, `hardware_flops`, `elapsed`. Subtracting two snapshots gives an
interval with the same rate properties, so `(now - last).tokens_per_sec` and
`now.tokens_per_sec` are the instantaneous and cumulative versions of one metric
rather than two separately-maintained ones.

Three rules this file exists to keep:

**Counts are int64 end to end**, including through the DDP all-reduce. A run
passes 2^31 tokens in under an hour; float32 stops incrementing at 2^24 and int32
wraps into negatives. The accumulators upstream are Python ints, which do not
overflow at all — int64 is only the wire format for the reduction. `all_reduce_`
reduces in the tensor's own dtype for the same reason: an int64 count summed as
float32 would round every total past 16.7M tokens.

**FLOPs are float64.** 270 TFLOP/s at 40% MFU overflows int64 in a day, and a FLOP
total is never compared for equality, so an exponent beats exact digits. The FLOP
accumulator is deliberately *not* a registered buffer: `bf16-true` converts a
module's float buffers, and fp64 → bf16 would leave the counter with 8 bits of
mantissa.

**A snapshot is a sync point.** It all-reduces and reads back to the host, so the
meter does not maintain a per-consumer interval. Consumers take a snapshot on
their own cadence and subtract their own previous one. Two consumers with
different intervals then cost two reductions instead of two counter sets that can
drift apart.

The rank-local counters are a registered int64 buffer; the run's *global* totals
restored from a checkpoint are kept apart from them as offsets. That separation is
load-bearing — a rank-local count restored onto every rank would all-reduce to
`world_size` times rank 0's share.

Both raw counts and billions are logged (`train/tokens_seen` and
`train/b_tokens_seen`). `self.log` stores float32, which stops resolving single
tokens past 2^24: harmless on an axis read in billions, misleading on one read as
a count. Nothing in `_log_progress` uses `sync_dist=True`, because the values are
already global and averaging the ranks' identical copies would divide the run's
progress by the world size.

`_elapsed()` measures training wall-clock only, excluding the gap between a crash
and a restart.

## MFU and HFU

FLOPs are charged from the **unsplit** batch. The micro-batch split keeps documents
whole, so the attention cost is identical either way, and one call keeps the device
syncs at zero instead of one per micro-batch.

Two utilization numbers are logged, not one:

- **`perf/mfu`** — model FLOPs utilization: the arithmetic the *architecture* owes.
- **`perf/hfu`** — hardware FLOPs utilization: adds the second forward that gradient
  checkpointing runs through the blocks.

Without gradient checkpointing they are equal; the gap between them is exactly what
recompute costs.

Neither is clamped to 1.0. `ThroughputCallback.PEAK_TFLOPS` is the **fp32-accumulate**
ceiling (270 TFLOP/s on a 5090), and fp16 accumulation genuinely reaches 325, so a
kernel accumulating in fp16 can legitimately report above 100%. That peak figure is
measured, not the whitepaper's 209.5 — cuBLAS bf16 reaches 227.4 here, so the spec
figure reports MFU above 100% for trivial code. A clamp would have hidden that bug
rather than surfacing it. The FLOP model itself lives in `models/flops.py` and is
documented in [performance.md](../performance/performance.md).

`ThroughputCallback` owns the *policy* (how often, against what peak) and none of the
accounting. Its first call only establishes a baseline: an interval measured against
the start of the run would report the average, not the rate. `peak_flops` defaults to
the detected device's entry in `PEAK_TFLOPS`; pass it explicitly for anything not in
that table.

---

## Resume

`trainer.fit(ckpt_path=...)` restores weights, optimizer, schedule position and the
loop counters. Since Lightning 2.x it also calls `load_state_dict` on a train
dataloader that has one (see `_FitLoop.setup_data`, which restores the loader state
and only then builds the iterator). Three things are still missing for a run that
has to survive a crash at hour 40, and `training/loop/resume.py` supplies them.

Everything the trainer adds sits under one checkpoint key (`"kohakuwullm"`), so a
checkpoint that simply predates the resume support is distinguishable from one whose
state failed to save.

### RNG

Nothing in a Lightning checkpoint records it, so a resumed run would draw a different
stream than the one it is continuing. `rng_state()` snapshots Python's `random`,
numpy's MT19937, torch CPU and — only if CUDA is already initialized — torch CUDA.
Asking for the CUDA state unconditionally initializes CUDA as a side effect, which
would create a context on a CPU-only run for nothing.

Everything written has to survive `torch.load(weights_only=True)`, which is Lightning's
default: tensors, and Python primitives inside lists / tuples / dicts. That is why
numpy's generator key is stored as a tensor rather than as the ndarray
`np.random.get_state` hands out, and why the Python state's middle element is
re-tupled on load — a tuple survives pickling, but a checkpoint that has been through
a json or yaml round trip comes back as lists, which `setstate` rejects.

**Rank 0's RNG is restored onto every rank.** The run starts with
`pl.seed_everything(SEED)`, which seeds every rank identically; ranks diverge by
*which shard they read*, not by which numbers they draw. Restoring one saved stream
everywhere therefore reproduces the run's own starting condition, whereas re-seeding
the other ranks from something else would put the resumed run in a state the original
never had.

### The dataloader position

Saved even when Lightning would save it too. Lightning's copy is applied inside
`_FitLoop.setup_data`, and `setup_data` returns early once the combined loader
exists — which it does whenever anything asked for `trainer.estimated_stepping_batches`
before the loop state was restored. One redundant restore of the same position is
idempotent; a silently skipped one costs a re-run of every batch since the last
checkpoint.

**Only the loader object is asked, never its dataset.** A dataset can report the last
batch it *produced*, and Lightning prefetches, so that position runs ahead of what the
trainer consumed — by one batch with a sized loader, by up to
`prefetch_factor * num_workers` without. On the production loader config
(`prefetch_factor=2`, `num_workers=16`, `k=262144`) that upper bound is 32 batches,
i.e. ~8M tokens skipped per resume, and nothing in a loss curve would show it.
Recording *consumption* needs something between the dataset and the trainer, which is
what `data.resume.ResumableLoader` is; see [data.md](../internals/data.md).

A loader without the state-dict protocol is not an error, but it is announced: the
startup log prints a `[resume]` warning saying the run will restart its data stream
from the beginning, and a checkpoint carrying a position that has nowhere to go warns
that the run will repeat data it has already trained on.

### Hook ordering

Three hooks each do one part, and the ordering is not incidental:

| hook | what it does | why there |
|---|---|---|
| `on_load_checkpoint` | stash the totals; restore RNG and loader position | the fit loop builds its iterator inside `setup_data`, which precedes every train hook — a position restored later would apply to an iterator already prefetching |
| `on_train_start` | move the stashed totals into the offsets | first hook that runs *after* the checkpoint is restored under every strategy: FSDP restores after `on_fit_start`, single-device and DDP before it |
| `on_save_checkpoint` | snapshot totals, RNG and loader position | collective; see the rule above |

The token totals wait for `on_train_start` rather than being applied in
`on_load_checkpoint`, because the state-dict load that happens between them would
overwrite the buffer. When they are applied, the local counters are **zeroed** and the
run's totals live entirely in the offsets — the restored buffer holds rank 0's *local*
count, and the local counters must stay a per-rank delta.

**No trainer attached means `load_from_checkpoint`**: weights lifted into a *new* run.
The old RNG stream and data position are not restored, but the token totals are, since
they describe how much this model has been trained.

### The `estimated_stepping_batches` trap

Reading `trainer.estimated_stepping_batches` builds the train dataloader from inside
`configure_optimizers`, which runs *before* Lightning restores the loop state.
`setup_data` then returns early for the rest of the run, and the dataloader position
Lightning saved is never applied.

`LMTrainer` therefore asks for it **only if a schedule needs it** — that is, only when
a scheduler entry leaves `end` at `-1` or `None`. Configs that autofill `end` ahead of
time (what `autofill_schedule_steps` is for) never reach that code; those that do at
least pay the cost knowingly.

---

## Callbacks

### `StepProgressBar`

Progress against the run's total step target, not against an epoch.

The default Lightning bar measures an epoch, which needs the epoch to have a length.
Ours does not: the loader packs a data-dependent number of batches, and under the
iterative variant it is an `IterableDataset` with no `__len__` at all — so the bar
either shows an unbounded counter or a total invented from a pinned
`batches_per_epoch`. Neither answers the only question being asked, which is how far
through the run we are.

The bar therefore spans the whole run and is driven by `trainer.global_step`. That is
also what makes it correct across a resume, where an epoch-relative bar restarts at
zero while the run is half finished, and correct under gradient accumulation, where
batches and steps differ by the accumulation factor. `on_train_epoch_start`
deliberately does not call `super()`, which would reset `n` to zero and retotal to the
epoch's batch count.

The step counter is *assigned* from `global_step`, never incremented: an optimizer step
does not land on every batch under accumulation, and a skipped step (non-finite
gradients) advances neither.

### `SampleLogCallback`

Generates a few completions every `SAMPLE_INTERVAL` steps and prints them, mirroring to
a wandb table when one is present.

For a prompt-generation model this is the cheapest real signal there is. The loss curve
cannot tell you that the model has stopped emitting `<|task|>` markers or has started
looping a tag; a glance at four samples can.

Three things keep it from perturbing the run it is observing:

- `LMTrainer.generate` samples from **its own generator**, so the default RNG stream —
  which decides data order — is untouched no matter how often previews are logged.
- Only rank 0 generates, and it calls the *unwrapped* backbone, so nothing here enters
  a collective the other ranks would have to match.
- Under pipeline parallelism no single rank owns the whole model (the last stage has
  the head but not the embedding), so generation is **skipped with one warning** rather
  than left to emit whatever a headless stage returns. The check is duck-typed for a
  `StagePlan` through up to four wrapper layers, because the module may be behind
  `AutocastStage`, DDP or `torch.compile`.

**Previews must run under the precision plugin's forward context.** A preview is
triggered from `on_train_batch_end`, which is *outside* the autocast Lightning wraps
`training_step` in. Without re-entering it the backbone sees fp32 activations — which
the MXFP8 expert path refuses outright, killing the run at the first preview rather
than at step 1. `LMTrainer.generate` takes the plugin's own context rather than an
autocast dtype of its own, so there stays one definition of the run's precision.

### `ThroughputCallback`

Covered under [MFU and HFU](#mfu-and-hfu) above.

---

## Preview sampling

`PreviewSampler` (`training/loop/sampling.py`) decodes through a `KVCache` by
default, and `use_cache=False` gives back the cache-free path that recomputes the
whole prefix every step. The two are not alternatives to choose between: the
cache-free path is the *reference* the cached one is tested against, token for
token, and keeping it callable is what makes that test possible.

This was deliberately cache-free for a long time, on the grounds that a cache is a
correctness surface a logging path has no business exercising. What changed is not
the risk assessment but the evidence: the equivalence tests in
`tests/test_generation.py` are checked against five deliberately broken caches, and
the cache buys the project a generation path it did not have. See
[architecture.md](../concepts/architecture.md) §15 — including the measurement that a preview
at this scale is dispatch-bound, so the cache is worth about 5% here rather than
the factor the recompute argument suggests.

The sampler owns a per-device RNG stream seeded from a **fixed** constant
(`SAMPLE_SEED = 20090220`), not from the run seed, so the same step in two runs draws
the same sampling noise — which is what makes two previews comparable at all.

`top_p_sample` keeps the first token that crosses the threshold, so the candidate set
is never empty. The backbone's training flag is restored in a `finally`: left in eval,
gradient checkpointing (which keys off `training`) would be off for the rest of the
run, and nothing in the loss would say so.

---

## Startup log

`on_train_start` prints, on rank zero only. For `Kohaku-500M` at vocab 65536:

```
[lm] 22L d1280 heads 20/4
  parameters   total 546.30M | active 546.30M | embedding 167.77M
  layer types  22 full / 0 sliding, 0 MoE / 22 dense
  model FLOPs  3.12B/token fwd+bwd at ctx 2048
```

The FLOPs line is reported at a fixed reference context (2048). Only the attention
term depends on context length, so this is a reference point rather than a
configuration knob — the *charged* cost always uses the batch's own document lengths.

The `FlopCounter` is built from the **uncompiled** module: it reads parameter counts
and shape constants once, and a compile wrapper only obscures them.

A resumed run adds a line with the tokens seen and trained so far, followed by any
`[resume]` warnings about state that will not survive a restart.
