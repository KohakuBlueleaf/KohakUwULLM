# Process groups

```python
from kohakuwupipe import init_pipeline, shutdown

ranks = init_pipeline()      # .rank, .world, .device
...
shutdown()
```

`init_pipeline` reads the launcher's environment (`RANK`, `WORLD_SIZE`,
`LOCAL_RANK`), initializes NCCL, sets the CUDA device, and configures the logger
with this rank's id.

## `device_id` is omitted deliberately

`init_process_group(device_id=...)` puts NCCL in **eager initialization**: the
group builds one communicator up front, and every point-to-point send and
receive shares it. For a pipeline that is exactly wrong — the boundary sends are
the hot path, and sharing one communicator serializes them.

Omitting it selects lazy initialization, where each send/recv pair gets its own
2-rank communicator on first use. The cost is a warning per pair, once:

```
An unbatched P2P op (send/recv) was called on this ProcessGroup with size 4.
In lazy initialization mode, this will result in a new 2-rank NCCL communicator
to be created.
```

That warning is the *desired* state. `eager_init_in_use()` reports which mode is
active and `warn_on_eager_init()` returns a message when the slow one is, so a
run that silently regressed into shared communicators says so.

The trade is a barrier warning at startup instead:

```
barrier(): using the device under current context. You can specify `device_id`
in `init_process_group` to mute this warning.
```

Muting that one costs the P2P behaviour above. It stays.

## What every rank must do identically

A collective that one rank skips does not fail — it hangs, usually far from the
cause. The invariants:

- **Every rank calls every collective the same number of times, in the same
  order.** No rank may return early from a step, including on an empty batch.
- **Creating a process group is itself collective.** `dist.new_group` must be
  reached by every rank, which is why a lazily-created group is cached rather
  than built on demand.
- **Shapes and dtypes at a boundary are frozen at construction**, so every rank
  must agree on them before the first step, not discover them during it.

When ranks build their own data independently, that agreement is an assumption,
not a guarantee. Check it once: hash the first step's content on each rank and
compare. The failure it catches — one stage's inputs paired with another's
targets — produces a loss curve that looks entirely normal.
