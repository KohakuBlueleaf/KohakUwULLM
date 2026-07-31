# Loss scaling across a pipeline

fp16 has ~5 decimal digits of dynamic range below 1. A gradient of `1e-4` is
perfectly representable; its *square* is not, and `1e-8` is below fp16's
smallest subnormal (`5.96e-8`). Anything that squares or accumulates gradients —
an Adam second moment, a long reduction — loses them to zero. Scaling the loss
by a large constant moves the whole gradient distribution up into the range fp16
represents, and the optimizer divides it back out before stepping.

bf16 needs none of this: it keeps fp32's exponent range and pays for it in
mantissa. That is why `GRAD_SCALER = "auto"` turns the scaler on only when
`fp16` appears in the parameter or autocast dtype.

## Why not `torch.amp.GradScaler`

The stock scaler is per-process and keyed on the optimizer. It records that
`scale(loss)` happened, and `unscale_(optimizer)` raises if it did not.

In a pipeline **only the last stage computes a loss.** Ranks 0..n-2 never call
`scale()`, so they can never call `unscale_()` — but they hold gradients that
are scaled, because the scale rode backward to them through the boundary
activations. The stock scaler has no way to express "this rank's gradients are
scaled by a factor it never saw."

`PipelineGradScaler` keeps the scale as a plain Python float that every rank
holds identically, and never asks who produced the loss.

## The overflow vote

Each rank divides its own gradients and returns a device flag. The flag is
all-reduced with `MAX` before anyone reads it:

```python
found    = scaler.unscale_(parameters)   # device tensor, no host read
overflow = bool(scaler.agree(found).item())
```

`agree` is what makes the step atomic. If rank 2 saw an inf and stepped anyway
while rank 3 skipped, the stages would hold updates from different batches — the
model is silently no longer a single model, and nothing downstream reports it.
A rank whose gradients are all finite must still skip when a peer's are not.

This `.item()` is the **only** host synchronization in a scaled step, and it is
the reason the scaler is not free. Without one, the loop reads no device tensor
at all.

## Backoff, growth, and the floor

On an overflow the scale halves and the step is discarded. After
`growth_interval` clean steps it doubles. The default `init_scale` of 65536 is
deliberately too high: the first few steps overflow, cost nothing, and the run
settles at the largest power of two that does not — on MoE-1B that is 32768,
reached by step 3.

`min_scale` refuses to back off below 1.0. A run that keeps halving is not
suffering a transient; it has a real inf in the graph, and letting the scale
decay to `1e-30` converts a loud failure into a silent one where every step is
skipped and the loss never moves.

## What the log shows

`scale` and `overflow` are reported alongside the loss. `broadcast_loss` divides
by the scale that was in effect for that step — captured *before* `update()`, or
an overflow step would unscale by the already-halved value and report a loss 2x
too high.

A healthy fp16 run shows `scale` flat and `overflow=0`. Occasional overflows are
normal. `scale` ratcheting steadily downward is the signal to look at the model,
not the scaler.

## Checkpoints

`state_dict` carries the scale, the clean-step counter and a cumulative overflow
count. A resume that restored the weights but not the scale would re-run the
initial backoff, discarding real steps at a point in training where they are no
longer cheap.
