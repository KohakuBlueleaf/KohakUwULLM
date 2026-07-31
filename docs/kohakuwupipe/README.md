# kohakuwupipe

A pipeline-parallel training loop that knows nothing about transformers.

`torch.distributed.pipelining` gives you a schedule; it does not give you a
trainer. This package is the rest: a module contract, a step loop, callbacks,
checkpointing, and the boundary plumbing — with **zero imports from
`kohakuwullm`**, so it can be lifted into its own repository unchanged.

These docs travel with the package. Every `See docs/<name>.md` in
`src/kohakuwupipe/**` resolves here.

| Doc | Covers |
|---|---|
| [module.md](module.md) | `PipelineModule` — the `LightningModule` analogue, and what survives being split across ranks |
| [loop.md](loop.md) | `PipelineLoop` and `PipelineTrainer`: the step, loss normalization, callbacks |
| [streams.md](streams.md) | Multi-stream boundaries: accumulators, constants, skips — and the dense-gradient trap |
| [plan.md](plan.md) | Cost-balanced contiguous splitting of a layer stack |
| [distributed.md](distributed.md) | Process-group setup, and why `device_id` is deliberately omitted |
| [checkpoint.md](checkpoint.md) | Whole-model checkpoints assembled from per-rank stages |
| [logging.md](logging.md) | The rank-aware structured logger |
| [bench.md](bench.md) | Measuring pipeline latency and the overlap it buys |

## The shape of a run

```python
from kohakuwupipe import PipelineTrainer, init_pipeline, plan_stages, shutdown

ranks = init_pipeline()
plans = plan_stages(depth=16, num_stages=ranks.world, layer_cost=1.0, head_cost=1.4)
trainer = PipelineTrainer(
    MyModule(), ranks, plans, micro_tokens=16384, num_microbatches=16
)
trainer.fit(steps, max_steps=100_000)
shutdown()
```

`MyModule` is a [`PipelineModule`](module.md): it builds one rank's slice, owns
its optimizer, and supplies the objective. The loop drives; it never decides.

## The rules this package follows

- **Nothing in the step reads a device tensor.** A `.item()` in the loop is a
  synchronization on every step; metrics stay device-resident and callbacks pay
  for them on their own cadence.
- **Collectives are unconditional.** Every rank calls every collective the same
  number of times, in the same order, or the run hangs — not fails.
- **Unknown hook names raise.** A typo in a callback is a callback that never
  fires, which looks exactly like a callback that had nothing to say.
