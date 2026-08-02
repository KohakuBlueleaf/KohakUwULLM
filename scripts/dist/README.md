# scripts/dist

Multi-process correctness harnesses. These are **not** benchmarks and **not** pytest
tests: each spawns its own real ranks and checks that a distributed path agrees
with a reference, so they cannot run inside a test process and they measure nothing.

| script | checks |
|---|---|
| `parallel_equivalence.py` | gradients under DDP/pipelining match a single-GPU reference |
| `pp_torch_smoke.py` | a pipeline stage builds, steps and produces finite loss |

They lived in `tests/` and pytest never collected them, so a harness that silently
stopped working would have looked exactly like one that was passing. Benchmarks that
happen to spawn ranks belong in `scripts/bench/e2e/` instead.

Run one directly; its `GPUS` global sets how many ranks it spawns, and an existing
rank group (`RANK` in the environment) is used as-is.

    .venv/bin/python scripts/dist/parallel_equivalence.py
