# Logging

A rank-aware structured logger. One line per event, fields as `key=value`, and
the rank in every record so a four-process log is readable without splitting it.

```python
from kohakuwupipe import get_logger

log = get_logger(__name__)
log.info("stage built", layers="0..4", params=292_167_808)
```

```
[21:23:44] [rank0] [training.module] [INFO] stage built [layers=0..4, params=292167808]
```

`configure(rank=...)` sets the rank prefix; `init_pipeline` calls it, so a
standalone user of the package gets it for free.

## Field names that would shadow a `LogRecord` raise

`logging` reserves a set of attribute names on every record — `args`, `module`,
`name`, `msg`, `exc_info`, `levelname`, and notably **`extra`**. Passing one as a
field does not warn: the record is constructed with a conflicting attribute and
the whole line is dropped or mangled.

That is a genuinely nasty failure, because the code that produced it looks
correct and the operator sees fewer lines than they expected rather than an
error. `Logger._log` therefore raises on a reserved name:

```python
log.info("stage cost", extra=1.2)   # ValueError, not a silently vanished line
```

Rename the field. `extra_ms`, `stage_extra`, anything unreserved.

## What to log from a training loop

Not much, and never on the hot path. Reading a device tensor to log it costs a
synchronization on every step; the loop keeps metrics device-resident and
callbacks pay for them on their own interval. See [loop.md](loop.md).

Rank 0 only, for anything an operator reads. The default reporter in
`kohakuwupipe.training.callbacks` already gates on rank, so a custom `report`
callable should too — four copies of every metric line is how a log becomes
unreadable.
