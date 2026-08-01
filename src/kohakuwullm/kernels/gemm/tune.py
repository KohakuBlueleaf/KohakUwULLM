"""Time the planner's shortlist once per shape and keep the winner.

The model ranks every legal tile; only its top few are ever run. The result is an
offline table keyed by shape, not a re-benchmark whenever a dimension moves.

See docs/performance/gemm.md section 5e.
"""

import json

import torch

from kohakuwullm.kernels.gemm.device import Device
from kohakuwullm.kernels.gemm.plan import Plan, plan_topk
from kohakuwullm.kernels.gemm.streamk import StreamKGemm

_CACHE: dict[tuple, Plan] = {}


def _time(fn, warmup: int = 8, iters: int = 20) -> float:
    """Best-of-``iters`` device time in ms, after ``warmup`` untimed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        beg[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return min(s.elapsed_time(e) for s, e in zip(beg, end))


def key(m: int, n: int, k: int, elem_bytes: int) -> tuple:
    return (m, n, k, elem_bytes)


def tuned_plan(
    m: int,
    n: int,
    k: int,
    dev: Device,
    elem_bytes: int = 2,
    shortlist: int = 4,
    dtype=torch.bfloat16,
) -> Plan:
    """Best of the model's top ``shortlist`` candidates, timed once and cached."""
    ck = key(m, n, k, elem_bytes)
    if ck in _CACHE:
        return _CACHE[ck]

    a = torch.zeros((m, k), device="cuda", dtype=dtype)
    b = torch.zeros((k, n), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)
    best, best_ms = None, float("inf")
    for cand in plan_topk(m, n, k, dev, elem_bytes, shortlist):
        try:
            g = StreamKGemm(m, n, k, dev, elem_bytes, p=cand)
            g(a, b, c)
            if g.handle.n_spills > cand.bm * cand.bn // (32 * cand.warps):
                continue
            ms = _time(lambda: g(a, b, c))
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = cand, ms
    del a, b, c
    torch.cuda.empty_cache()
    if best is None:
        best = plan_topk(m, n, k, dev, elem_bytes, 1)[0]
    _CACHE[ck] = best
    return best


def save(path: str) -> None:
    """Write the tuned table so a later run does not repeat the timing."""
    rows = [{"key": list(k), "plan": v.__dict__} for k, v in _CACHE.items()]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)


def load(path: str) -> None:
    """Load a tuned table written by `save`."""
    with open(path) as fh:
        for row in json.load(fh):
            _CACHE[tuple(row["key"])] = Plan(**row["plan"])


def clear() -> None:
    _CACHE.clear()


class TunedGemm:
    """`StreamKGemm` whose plan came from timing the model's shortlist."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dev: Device,
        elem_bytes: int = 2,
        shortlist: int = 4,
        dtype=torch.bfloat16,
    ):
        p = tuned_plan(m, n, k, dev, elem_bytes, shortlist, dtype)
        self.inner = StreamKGemm(m, n, k, dev, elem_bytes, p=p)
        self.plan = p

    def __call__(self, a, b, c=None):
        return self.inner(a, b, c)
