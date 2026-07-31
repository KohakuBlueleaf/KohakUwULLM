# scripts/dist

Multi-process correctness harnesses. These are **not** benchmarks and **not** pytest
tests: each spawns real ranks with `torchrun` and checks that a distributed path agrees
with a reference, so they cannot run inside a test process and they measure nothing.

| script | checks |
|---|---|
| `parallel_equivalence.py` | gradients under DDP/pipelining match a single-GPU reference |
| `pp_torch_smoke.py` | a pipeline stage builds, steps and produces finite loss |

They lived in `tests/` and pytest never collected them, so a harness that silently
stopped working would have looked exactly like one that was passing. Benchmarks that
happen to use `torchrun` belong in `scripts/bench/e2e/` instead.

    torchrun --standalone --nproc_per_node=4 scripts/dist/parallel_equivalence.py
