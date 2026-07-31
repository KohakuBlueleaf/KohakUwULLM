# Checkpoints

A checkpoint must hold the **whole model**, not this rank's stage. A file that
holds a slice is a file only a run with the identical split can read — and the
split is exactly the thing you change when you move to different hardware.

```python
trainer.save_checkpoint("out/last.ckpt")     # collective
trainer.load_checkpoint("out/last.ckpt")     # collective
```

## Renumbering is the whole problem

A stage numbers its blocks from zero. The model numbers them from
`plan.start_layer`. Write the stage's `state_dict` as-is and every rank produces
keys `blocks.0..N`, which collide; load it back and rank 1 gets rank 0's layers.

Neither raises. The keys match, the shapes match, and the model trains — worse,
for reasons nothing in the loss curve explains.

`global_names()` maps local keys to whole-model ones, and it is applied on
**both** sides. A save that renumbers and a load that does not is the same bug
wearing a different hat, and it only shows up on a stage whose `start_layer` is
non-zero — so rank 0 passes and the others do not.

## The block prefix moves when you wrap

An autocast wrapper makes the keys `module.blocks.N`, not `blocks.N`, so the
renumbering finds nothing and silently does nothing. The prefix is therefore
passed explicitly (`block_attr`) rather than assumed, and the module that built
the wrapper is what reports it.

## Optimizer state is per rank

Stages share no parameters, so there is no meaningful merge: each rank's
optimizer state is stored under its own index and read back by index. A resume
into a different world size is refused rather than reinterpreted.

## What else belongs in the file

Whatever a resume needs that is not a parameter — most importantly the **data
position**. A checkpoint that restores the weights and the step count but not
the loader trains a second time on tokens it has already seen, and reports a
token count that never happened. `PipelineModule.on_save_checkpoint` is where
that goes; see [module.md](module.md#checkpoint-hooks).

Callbacks get `on_save_checkpoint` / `on_load_checkpoint` too, and their state
rides in the same payload.

## Verifying a resume

Bit-exactness is the only check worth running. Load into a freshly built model
and compare every tensor against the pre-save values — and assert that
*something* changed, or a load that quietly did nothing passes:

```
resumed [step=32, tensors=71, changed=67, layers=0..4]
```

`changed=0` would mean the checkpoint never landed. `changed=71` on a stage
whose `start_layer` is 0 proves less than it looks, because the identity mapping
hides a renumbering bug — check a later stage.
