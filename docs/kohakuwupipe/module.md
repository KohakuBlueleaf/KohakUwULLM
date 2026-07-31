# `PipelineModule`

What a `LightningModule` is, for a model that does not fit on one card.

A subclass owns its stage, its optimizer and its objective, so the loop stays a
driver. Hook names match `LightningModule` wherever the concept survives being
split across ranks.

```python
class MyModule(PipelineModule):
    def configure_model(self, plan, rank, world, device):
        """This rank's slice. Called once, before anything else."""

    def configure_optimizers(self):
        """`optimizer`, or `(optimizer, scheduler)`, over this stage only."""

    def boundary_example(self, plan, device):
        """The zero tensor `PipelineStage` freezes as the boundary shape."""

    def loss(self, hidden, target):
        """`(loss, logs)`. Last stage only. A sum, not a mean."""
```

**`target` is not "labels".** This package never looks inside it — it is
whatever the step carried, handed straight through. Naming it `labels` in a
framework that also has to run a DiT would be baking one objective into the
contract; the LM adapter is free to call it `labels` in its own implementation,
because there that is what it is.

**One constraint, and it comes from torch, not from here.** `target` must be a
**tensor or `None`**. `Schedule._step` guards `None` explicitly, and anything
else goes to `_split_tensor(target, TensorChunkSpec(0), n_microbatches)` —
verified: a tuple or a dict raises `AttributeError: 'tuple' object has no
attribute 'size'`. A denoising objective carrying noise *and* a timestep has to
pack them into one tensor, or carry the second through a
[boundary stream](streams.md) instead.

`inputs` has no such limit — it is the schedule's `*args`, split by
`split_args_kwargs_into_chunks`, so a tuple is fine there.

## What is different from Lightning, and why it has to be

**`loss` is not `training_step`.** The schedule owns forward and backward across
ranks, calling `loss` once per micro-batch from inside itself. A stage supplies
the objective; it does not drive the pass. `training_step` still exists, but only
for bookkeeping that needs the raw batch — it returns nothing and does not
backward.

**`self.stage_module` is what the schedule calls and what the checkpoint reads.**
An autocast wrapper belongs *there*, not around `forward`, or the loss runs
outside it.

**Optimizers are genuinely local.** Each stage owns a disjoint parameter set, so
there is no cross-rank reduction to arrange. That also means a scheduler is
per-stage; every rank must step it the same number of times.

**`log()` records without reading.** It stores whatever you hand it —
device tensors included — and `pop_metrics()` takes them. Nothing syncs until a
callback decides to.

## Construction order

`setup(plan, rank, world, device)` calls `configure_model` and records the rank
wiring. `PipelineTrainer` calls it before `boundary_example`, so the example may
depend on what the model turned out to be — a boundary that is a tuple only when
the model needs a side channel, for instance. See [streams.md](streams.md).

The order that matters, and the failure if you get it wrong:

1. **Build the stage.**
2. **Cast the parameters.** After `PipelineStage` exists, a cast leaves the
   declared boundary describing a tensor nothing sends, and the runtime raises
   `PipeliningMetadataError` on the first step.
3. **Wrap in autocast.**
4. **Declare `input_args`** — and the stage must *produce* that dtype, not
   whatever its last layer happened to return.

## Checkpoint hooks

`on_save_checkpoint` / `on_load_checkpoint` are for state the stage owns that is
not a parameter — a data-loader position, a step counter of your own. Both are
**collective** if what they touch is: a loader that all-gathers its position
across ranks must be called on every rank or the run hangs.

See [checkpoint.md](checkpoint.md) for how the per-rank slices become one file.
