# Measuring a pipeline

`kohakuwupipe.bench.latency` answers one question: does pipelining actually
overlap anything, or is it just splitting a model across cards?

```python
from kohakuwupipe.bench.latency import SyntheticStage, hop_cost, overlap, roofline
```

The comparison that matters is **one micro-batch at a time** against **many
micro-batches in flight**. A stage split with a single micro-batch has no
overlap by construction: each rank waits for the one before it, and the total is
the sum of the stages. The ratio between that and the pipelined version is the
only number that shows whether the schedule is doing its job.

## The roofline has two terms, not one

With infinite micro-batches each card reaches its own compute or bandwidth
ceiling — but the boundary sends do not disappear. A roofline for tokens/s has
to charge:

- **compute or memory bandwidth per stage**, whichever binds first, and
- **the hop**: `micro_tokens * dim * dtype_bytes` per boundary, both directions,
  over the interconnect that actually carries it.

A roofline built from FLOPs alone will sit far above anything achievable and
make a healthy run look broken.

## Things that make a pipeline measurement lie

**Warmup must cover every shape.** A varlen stream compiles new kernels for a
while; timing that window charges compilation as throughput. `Throughput`
excludes `warmup_steps` for this reason.

**Use a trailing window, not a running average.** A checkpoint or a preview
costs tens of seconds and a cumulative mean never forgets it. See
[loop.md](loop.md#callbacks).

**A per-stage timing has no seam in it.** Timing each stage alone, on its own
card, tells you the imbalance the schedule then waits on — but it deliberately
excludes the send/recv. Do not compare it to an end-to-end number and call the
difference "communication"; the bubble is in there too.

**Isolated per-iteration timing serializes host and device.** Synchronizing
around each iteration stops the host running ahead into the next one, which a
real schedule does. A device-busy percentage measured that way can look far
worse than the run it is meant to describe — check any launch-bound conclusion
against a real run before acting on it.

**One job at a time.** Two runs sharing the cards invalidates every timing on
both, and the symptom is a plausible-looking number rather than an error.
