# The loop, the trainer, and the callbacks

`PipelineLoop.step` is five lines of schedule plus bookkeeping. Nothing in it
reads a device tensor on the host, so a throughput regression is a regression in
that file.

```python
self.optimizer.zero_grad(set_to_none=True)
self._set_layout(batch.layout)
self.schedule.step(batch.inputs)                       # first stage
self.schedule.step(target=batch.target, losses=losses) # last stage
self.schedule.step()                                   # middle stages
self.optimizer.step()
```

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
| `Checkpoint` | whole-model write; collective, so every rank must reach it |

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
