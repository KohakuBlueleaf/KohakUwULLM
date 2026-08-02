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

## Optimizer state is keyed by whole-model parameter name

Weights are not the only thing a split renumbers. AdamW's moments are keyed by
the optimizer's *positional index*, and a stage numbers its parameters from zero
exactly as it numbers its blocks -- so a per-rank state list is readable only by
the split that wrote it.

`gather_optimizer_state` therefore does to the moments what `gather_state_dict`
does to the weights: it maps each entry through `parameter_names()` to its
whole-model name, `all_gather_object`s across the ranks and merges. Stages own
disjoint parameters, so the per-name entries never collide. The result rides in
the file under `named_optimizer_state`.

**Param groups are deliberately not stored.** Only the per-parameter `state` is
portable. A rank's group *count* is not a property of the model but of that
rank's slice: Muon separates matrix from non-matrix parameters, so a stage
holding only blocks builds a different number of groups than one that also holds
the embedding or the head. Storing them would make the file readable only by the
split that wrote it -- the exact failure the name keying exists to remove. Each
rank rebuilds its own groups, with their own hyperparameters, from its slice.

`load_optimizer_state` inverts the gather: build this rank's `index ->
whole-model name` map, take the named entries it finds, and hand
`load_state_dict` a positionally-keyed dict whose `param_groups` are this rank's
own, untouched. A checkpoint written at `pp=4` therefore resumes into `pp=2`,
into DDP, or onto a single GPU. Verified on CPU by a discriminating test: a
4-way save loaded 2-way, with every parameter's `exp_avg` uniquely identifiable,
restored each one to the right parameter.

**The legacy layout is still readable and is never written.**
`pipeline_stage_optimizer_states` -- a list indexed by rank -- is what older
files carry, and `load_optimizer_state` falls back to it when
`named_optimizer_state` is absent. That path loads **only** under the world size
that wrote it, and it now says so: a 4-entry list read by a 2-rank run raises
`ValueError` naming the file, the count and this run's world size.

Before that guard the fallback silently took `parts[rank]`, so rank 1 of a
2-way split resumed stage 1 of a 4-way split, and torch reported `loaded state
dict contains a parameter group that doesn't match the size of optimizer's
group` -- an error that names neither the file nor the topology and reads like a
bug in the optimizer.

**A run already in flight keeps writing whatever layout its process imported at
startup.** A long run started before this change produces legacy files for its
whole life, however new the source is, and those files resume only at their own
world size. What a file actually holds:

```python
zipfile.ZipFile(path).read("archive/data.pkl").count(b"named_optimizer_state")
```

The Lightning pipeline path has *not* been converted:
`PipelineOnlyStrategy.optimizer_state` in
`kohakuwullm/training/parallel/strategy.py` still gathers a per-rank list under
`pipeline_stage_optimizer_states`. Only the `kohakuwupipe` loop writes
name-keyed state. See
[../internals/pipeline.md](../internals/pipeline.md#checkpoints-must-hold-the-whole-model-not-this-ranks-stage).

## What else belongs in the file

Whatever a resume needs that is not a parameter — most importantly the **data
position**. A checkpoint that restores the weights and the step count but not
the loader trains a second time on tokens it has already seen, and reports a
token count that never happened. `PipelineModule.on_save_checkpoint` is where
that goes; see [module.md](module.md#checkpoint-hooks).

Callbacks get `on_save_checkpoint` / `on_load_checkpoint` too, and their state
rides in the same payload.

`PipelineTrainer.save_checkpoint` is what assembles all of it:

| key | what a resume loses without it |
|---|---|
| `state_dict`, `optimizer_states` | the model |
| `global_step` | where the run is |
| `progress` | the token and wall-clock totals |
| `lr_schedulers` | the LR schedule — a resume re-runs warmup |
| `grad_scaler` | the fp16 scale — a resume re-finds it, overflowing on the way |
| `callbacks`, plus whatever the module added | loader position, callback state |

**The periodic `Checkpoint` callback routes through the trainer** for exactly
this reason. Calling `checkpoint.save` itself, as it used to, wrote weights,
optimizer and `progress` and nothing else — so every file a long run produced
between its start and its end silently lacked the data position, while the one
written by `trainer.save_checkpoint` at the end lacked `progress` instead. Two
save paths, each missing what the other had. Which keys a file actually holds
reads out of its pickle without materializing a tensor, the same way the
optimizer layout does above.

`PipelineTrainer.load_checkpoint` warns when the file carries no LR schedule,
and `LMPipelineModule.on_load_checkpoint` warns when it carries no loader
position, because both failures are otherwise invisible in the loss curve.

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
