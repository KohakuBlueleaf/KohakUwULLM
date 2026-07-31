"""What a pipeline seam costs, and when pipelining is worth paying it.

Two measurements, both model-free:

* :func:`hop_cost` -- per-hop latency and bandwidth around the stage chain, so
  the seam is a number rather than an assumption;
* :func:`overlap` -- a synthetic stage of known compute against the same chain,
  which is the only way to see whether microbatches actually overlap.

Sweeping the compute-per-stage against a fixed seam is what tells you where
pipelining starts winning. See docs/kohakuwupipe/bench.md.
"""

import time

import torch
import torch.distributed as dist

SIZES = (4 << 10, 24 << 10, 96 << 10, 1 << 20, 16 << 20)


def hop_cost(
    rank: int, world: int, device, sizes=SIZES, iters: int = 100
) -> list[dict]:
    """Per-hop seconds and GB/s, from a full circulation of the chain.

    Collective: every rank must call with the same ``sizes``.
    """
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    rows = []
    for size in sizes:
        buf = torch.empty(size, dtype=torch.uint8, device=device)
        for count in (max(iters // 10, 2), iters):
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(count):
                if rank == 0:
                    dist.send(buf, nxt)
                    dist.recv(buf, prv)
                else:
                    dist.recv(buf, prv)
                    dist.send(buf, nxt)
            torch.cuda.synchronize()
            hop = (time.perf_counter() - start) / count / world
        rows.append({"bytes": size, "hop_s": hop, "gbps": size / hop / 1e9})
        del buf
        torch.cuda.empty_cache()
    return rows


class SyntheticStage(torch.nn.Module):
    """A stage whose cost is a knob: ``depth`` square matmuls of width ``dim``.

    Stands in for a real stage so the seam can be measured against a compute
    budget that is known exactly rather than estimated.
    """

    def __init__(self, dim: int, depth: int, dtype=torch.bfloat16) -> None:
        super().__init__()
        self.weights = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(dim, dim, dtype=dtype) * dim**-0.5)
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for weight in self.weights:
            x = torch.nn.functional.gelu(x @ weight)
        return x

    def flops(self, rows: int) -> float:
        """Forward FLOPs for ``rows`` inputs, one multiply-accumulate as 2."""
        dim = self.weights[0].shape[0]
        return 2.0 * rows * dim * dim * len(self.weights)


def overlap(schedule_fn, microbatch_counts, rank: int, iters: int = 5) -> list[dict]:
    """Wall time for the same total work split into N microbatches.

    ``schedule_fn(n)`` returns a zero-argument callable running one step of
    ``n`` microbatches. A ratio near 1 between N=1 and N>1 means the stages are
    taking turns rather than overlapping.
    """
    rows = []
    for count in microbatch_counts:
        run = schedule_fn(count)
        run()
        samples = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        samples.sort()
        rows.append(
            {"microbatches": count, "seconds": samples[len(samples) // 2], "rank": rank}
        )
    base = rows[0]["seconds"] if rows else 1.0
    for row in rows:
        row["speedup_vs_one"] = base / row["seconds"]
    return rows


def roofline(stage_seconds: float, hop_seconds: float, stages: int) -> dict:
    """Where a pipeline's ceiling comes from: the slowest stage, or the seam.

    ``saturated`` is steady-state steps/s with many microbatches; ``latency`` is
    one microbatch end to end.
    """
    bound = "seam" if hop_seconds > stage_seconds else "stage"
    return {
        "stage_seconds": stage_seconds,
        "hop_seconds": hop_seconds,
        "bound": bound,
        "saturated_steps_per_s": 1.0 / max(stage_seconds, hop_seconds),
        "latency_seconds": stage_seconds * stages + hop_seconds * (stages - 1),
        "ideal_speedup": stage_seconds * stages / max(stage_seconds, hop_seconds),
    }
