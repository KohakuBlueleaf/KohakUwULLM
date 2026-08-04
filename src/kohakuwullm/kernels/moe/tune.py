"""Time the planner's shortlist once per routing shape and keep the winner.

The model ranks every legal tile; only its top few are ever run. The result is an
offline table keyed by shape, not a re-benchmark whenever a dimension moves.

See docs/internals/fused-moe-16bit.md.
"""

import dataclasses
import json

import torch
import triton

from kohakuwullm.kernels.gemm.device import RTX_5090, Device
from kohakuwullm.kernels.moe import fused_experts as K
from kohakuwullm.kernels.moe.plan import GroupedShape, MoePlan, plan_topk

# Rows are bucketed so a routing count that jitters by a few tokens reuses a plan
# instead of retuning; the tile choice does not move inside a bucket.
ROW_BUCKET = 4096
SHORTLIST = 6

_CACHE: dict[tuple, "FusedPlans"] = {}
_DEVICE: Device | None = None


@dataclasses.dataclass(frozen=True)
class FusedPlans:
    """One tile per kernel in the fused expert path."""

    gemm1: MoePlan
    gemm2: MoePlan
    gemm2_dgrad: MoePlan
    gemm1_dgrad: MoePlan

    @property
    def min_block_m(self) -> int:
        return min(
            p.bm for p in (self.gemm1, self.gemm2, self.gemm2_dgrad, self.gemm1_dgrad)
        )


def set_device(dev: Device) -> None:
    """Override the card description the planner scores against."""
    global _DEVICE
    _DEVICE = dev
    _CACHE.clear()


def device() -> Device:
    if _DEVICE is not None:
        return _DEVICE
    name = torch.cuda.get_device_name()
    if "5090" in name:
        return RTX_5090
    raise ValueError(
        f"no measured Device for {name}; measure mma_peak, dram_bw and l2_bw on "
        "this card and pass it to moe.tune.set_device(); see "
        "docs/internals/kernel-dev.md section 1"
    )


def shapes(rows: int, dim: int, hidden: int, experts: int, elem: int) -> dict:
    """The four grouped GEMMs the fused path launches, as planner inputs."""
    return {
        # GEMM1 stores `pre` and `h`; GEMM2 and GEMM1's DGRAD only scatter-add into
        # a token-sized buffer, which never reaches DRAM.
        "gemm1": GroupedShape(
            rows, hidden, dim, experts, accs=2, epilogue_bytes=3 * elem
        ),
        "gemm2": GroupedShape(rows, dim, hidden, experts),
        # The swiglu backward holds two extra accumulator-sized tiles live.
        "gemm2_dgrad": GroupedShape(
            rows, hidden, dim, experts, epilogue_bytes=4 * elem, live_tiles=2
        ),
        "gemm1_dgrad": GroupedShape(rows, dim, 2 * hidden, experts),
    }


def _scratch(rows, dim, hidden, experts, dtype, device_str="cuda"):
    """Operands with balanced routing, enough to launch every kernel once."""
    per = rows // experts
    offsets = torch.arange(experts + 1, device=device_str, dtype=torch.int32) * per
    offsets[-1] = rows
    token_of = torch.arange(rows, device=device_str, dtype=torch.int32) % max(
        rows // 8, 1
    )
    order = torch.arange(rows, device=device_str, dtype=torch.int32)
    tokens = int(token_of.max().item()) + 1
    return dict(
        x=torch.zeros(tokens, dim, device=device_str, dtype=dtype),
        w_in=torch.zeros(experts, 2 * hidden, dim, device=device_str, dtype=dtype),
        w_out=torch.zeros(experts, dim, hidden, device=device_str, dtype=dtype),
        h=torch.zeros(rows, hidden, device=device_str, dtype=dtype),
        pre=torch.zeros(rows, 2 * hidden, device=device_str, dtype=dtype),
        dpre=torch.zeros(rows, 2 * hidden, device=device_str, dtype=dtype),
        out=torch.zeros(tokens, dim, device=device_str, dtype=dtype),
        gate=torch.zeros(rows, device=device_str, dtype=dtype),
        dgate=torch.zeros(rows, device=device_str, dtype=torch.float32),
        offsets=offsets,
        token_of=token_of,
        order=order,
        experts=experts,
        block_e=max(16, triton.next_power_of_2(experts)),
    )


def _runners(s):
    """A no-argument launch of each kernel, given a plan."""
    return {
        "gemm1": lambda p: K.launch_gemm1(
            s["x"],
            s["token_of"],
            s["w_in"],
            s["h"],
            s["pre"],
            s["offsets"],
            s["experts"],
            s["block_e"],
            p,
        ),
        "gemm2": lambda p: K.launch_gemm2(
            s["h"],
            s["w_out"],
            s["out"],
            s["token_of"],
            s["order"],
            s["gate"],
            s["offsets"],
            s["experts"],
            s["block_e"],
            p,
        ),
        "gemm2_dgrad": lambda p: K.launch_gemm2_dgrad(
            s["out"],
            s["token_of"],
            s["order"],
            s["gate"],
            s["w_out"],
            s["pre"],
            s["dpre"],
            s["dgate"],
            s["offsets"],
            s["experts"],
            s["block_e"],
            p,
        ),
        "gemm1_dgrad": lambda p: K.launch_gemm1_dgrad(
            s["dpre"],
            s["w_in"],
            s["x"],
            s["token_of"],
            s["offsets"],
            s["experts"],
            s["block_e"],
            p,
        ),
    }


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


def _fastest(cands, run) -> MoePlan:
    """Time each candidate; fall back to the model's first choice if all fail."""
    best, best_ms = None, float("inf")
    for cand in cands:
        try:
            run(cand)
            ms = _time(lambda c=cand: run(c))
        except Exception:
            continue
        if ms < best_ms:
            best, best_ms = cand, ms
    return best or cands[0]


def key(rows: int, dim: int, hidden: int, experts: int, elem: int) -> tuple:
    bucket = -(-rows // ROW_BUCKET) * ROW_BUCKET
    return (bucket, dim, hidden, experts, elem)


def tuned_plans(
    rows: int,
    dim: int,
    hidden: int,
    experts: int,
    dtype: torch.dtype,
    shortlist: int = SHORTLIST,
) -> FusedPlans:
    """Best of the model's top ``shortlist`` candidates per kernel, timed once."""
    elem = torch.finfo(dtype).bits // 8
    ck = key(rows, dim, hidden, experts, elem)
    if ck in _CACHE:
        return _CACHE[ck]

    dev = device()
    bucket = ck[0]
    ranked = {
        name: plan_topk(shape, dev, elem, shortlist)
        for name, shape in shapes(bucket, dim, hidden, experts, elem).items()
    }
    scratch = _scratch(bucket, dim, hidden, experts, dtype)
    runners = _runners(scratch)
    picked = {n: _fastest(ranked[n], runners[n]) for n in ranked}
    del scratch, runners
    torch.cuda.empty_cache()

    plans = FusedPlans(**picked)
    _CACHE[ck] = plans
    return plans


def save(path: str) -> None:
    """Write the tuned table so a later run does not repeat the timing."""
    rows = [
        {"key": list(k), "plans": {n: getattr(v, n).__dict__ for n in _FIELDS}}
        for k, v in _CACHE.items()
    ]
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)


def load(path: str) -> None:
    """Load a tuned table written by `save`."""
    with open(path) as fh:
        for row in json.load(fh):
            _CACHE[tuple(row["key"])] = FusedPlans(
                **{n: MoePlan(**row["plans"][n]) for n in _FIELDS}
            )


def clear() -> None:
    _CACHE.clear()


_FIELDS = tuple(f.name for f in dataclasses.fields(FusedPlans))
