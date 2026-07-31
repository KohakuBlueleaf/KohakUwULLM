# Splitting a layer stack

`plan_stages` cuts `depth` layers into `num_stages` contiguous ranges,
minimizing the slowest stage. Ties break on parameter count, so two splits with
the same predicted runtime pick the one that balances memory.

It is model-agnostic on purpose: costs arrive as plain numbers, and nothing here
knows what a layer is.

```python
from kohakuwupipe import describe, plan_from_layers, plan_stages

plans = plan_stages(
    depth=16, num_stages=4,
    layer_cost=21.9e6,        # per-layer forward cost, any consistent unit
    head_cost=78.0e6,         # what the last stage carries beyond its layers
    layer_params=..., head_params=..., embed_params=...,   # the tie-break
)
print(describe(plans))
```

`allow_empty_last` lets the head-carrying stage hold no layers at all, which a
large vocabulary can want.

## Pinning a split you measured

The cost model is a model. When you have measured the real thing, say so:

```python
plans = plan_from_layers([5, 4, 4, 3], layer_cost=..., head_cost=...)
```

The costs are then carried for reporting only. This is worth reaching for more
often than it sounds, because **the cost model optimizes the wrong quantity**.

`partition` minimizes the maximum stage. A 1F1B schedule pays for the *spread*:
a stage that finishes early idles on every slot, not just once. Measured on
Kohaku-MoE-1B over four stages, `5/4/4/3` against the cost model's `5/5/5/1`
moved the slowest stage only 31.55 → 30.67 ms (2.8%) but was worth **13%** end
to end, because `5/5/5/1` left the head stage at 18.37 ms against 31.55.

The second failure mode is the head cost itself. Whatever constant you use is a
guess until it is measured, and a head charged at 3.56 layers when it is really
1.4 will strip layers off the stage that owns it. Time one layer and the head,
once, and feed the ratio in.

## Reading the table

```
pipeline split: 4 stages over 16 layers
  stage      layers    n        extra  cost share
      0        0..4    5        embed       27.4%
      1        5..8    4            -       21.9%
      2       9..12    4            -       21.9%
      3      13..15    3         head       28.8%
```

The cost share is what the model *predicts*, not what it measured. Divergence
between this table and per-stage timings is the signal to recalibrate — the
share column looking balanced is not evidence that the run is.
