# The loop, the trainer, and the callbacks

`PipelineLoop.step` is five lines of schedule plus bookkeeping, so a throughput
regression is a regression in that file.

```python
self.optimizer.zero_grad(set_to_none=True)
self._set_layout(batch.layout)
self.schedule.step(batch.inputs)                       # first stage
self.schedule.step(target=batch.target, losses=losses) # last stage
self.schedule.step()                                   # middle stages
self.optimizer.step()
```

The first and last stage are **not** exclusive: at `world == 1` one rank is
both, and it has to pass the inputs *and* the target. Written as
`if is_first: ... elif is_last:` the single-rank case reaches the schedule with
no target at all, and torch raises inside `_compute_loss` rather than anywhere
near the branch.

Two device reads survive per step, and both are deliberate: the loss scaler's
overflow flag has to branch on the host, and `build_loss_fn` does `int(target)`
per micro-batch to pick that micro-batch's held-back target. The second is
`num_microbatches` syncs, not one — see *The target never leaves the rank*
below.

`inputs` and `target` are opaque — the loop hands them to the stage and to
`loss` without looking inside. See [module.md](module.md).

`PipelineTrainer` owns construction order — cast before the stage, stage before
the schedule — which is what a caller most often gets wrong. See
[module.md](module.md) for what you supply.

## The loss is normalized once, by the caller

`build_loss_fn(stage_module, denom, num_microbatches)` divides the head loss by
the step's trained-token count. Two consequences:

**Micro-batch losses sum, they do not average.** Each one was already divided by
the *step's* total, so summing them reconstructs the per-token mean. This is
what keeps variable-length micro-batches weighted by tokens rather than by
count, and it is why `scale_grads=False` is passed to the schedule — torch's
default would divide gradients by the micro-batch count a second time, yielding
gradients exactly `1/n_microbatches` of the correct value.

**Auxiliary streams join after the division**, averaged over micro-batches. See
[streams.md](streams.md#normalizing-what-an-accumulator-carries).

`loss_fn` takes `stage_module`, not the module it wraps: the schedule calls it
*outside* the stage forward, so only the wrapper carries any autocast. Passing
the unwrapped module runs the head in fp32 and can fail outright on a
low-precision expert path.

## The target never leaves the rank

Only the last stage owns the target, and nothing about it is shipped. What the
schedule receives is `arange(num_microbatches)` — a handle it splits the same
way it splits everything else — and `loss_fn` uses `int(target)` to index the
list of pieces the loop filled before entering the schedule. The batch itself is
never sent between ranks, and `target` can be any structure at all.

The cost is one host read per micro-batch, because that handle is a device
tensor. Moving it to the CPU would remove them, but the target's first element
is also what torch's schedule passes to `_prepare_backward_infra`, so that is
not a free change; measure before making it.

## Counters

`tokens_seen` and `tokens_trained` are Python ints, and must stay that way. A
run passes 2^31 in under an hour; a float32 accumulator stops incrementing at
2^24, and a device tensor would put a sync in the step. They ride in the
checkpoint under `progress`, with the wall clock, so a resumed run continues its
totals instead of restarting them.

## Reading the loss

Only the last stage computes one. `broadcast_loss` sends it to every rank as a
device tensor, so logging, checkpointing and early-stop decisions can be
unanimous without any rank reading a value it does not have.

## Callbacks

`Callback` carries 23 hook names matching `LightningModule`'s where the concept
survives being split across ranks. `CallbackList.call` **raises** on an unknown
hook: a typo would otherwise be a callback that never fires, which is
indistinguishable from one that had nothing to say.

Every built-in callback is interval-gated, because reading a device tensor costs
a synchronization and the loop deliberately leaves them unread.

| callback | note |
|---|---|
| `Throughput` | tokens/s and ms/step over the window **since the last report** |
| `LossLog` | reads the step loss on an interval, with an EMA |
| `ProgressBar` | rank-0 tqdm; `postfix` names the `extra` keys it shows |
| `Checkpoint` | whole-model write; collective, so every rank must reach it |

None of them derives a model-specific metric. `LossLog` reports `loss` and
`loss_ema` and stops there: perplexity is `exp(loss)` for a language model and
nonsense for a diffusion one, so it belongs to whatever reporter the training
script installs. `ProgressBar.postfix` defaults to `("scale",)` for the same
reason — a model's own keys are passed in by the caller.

`Throughput`'s window is trailing, not cumulative. A checkpoint or a preview
costs tens of seconds, and a running average carries that into every later
number — a stall reported forever, reading as a regression that never happened.
Measured on a 4-card MoE-1B run: 267.0k tok/s, then 32.3k for the window holding
a preview and a checkpoint, then 271.4k. A cumulative average never recovers.

**Adding a callback after the trainer exists.** `PipelineTrainer.callbacks`
*is* the loop's list, deliberately — `CallbackList` builds a `list(callbacks)`,
so a trainer holding its own copy would make `trainer.callbacks.append(...)` a
silent no-op. Prefer passing callbacks to the constructor anyway.

## Fitting

```python
trainer.fit(steps, max_steps=100_000)
```

`steps` is any iterator of objects with `inputs`, `target`, `layout` and
`trained`. Every rank must consume the **same** stream: the first stage reads
the inputs and the last reads the target, so a divergence pairs one batch's
inputs with another's targets and produces a loss that is wrong and entirely
plausible. If each rank builds its own loader, verify that once at startup.

`on_exception` fires on every rank before the error propagates, so a callback
can tear its own state down rather than leaking a process group.
