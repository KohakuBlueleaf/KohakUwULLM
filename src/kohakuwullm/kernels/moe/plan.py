"""Choose tiles for the grouped MoE kernels from the routing shape, analytically.

`plan_topk(shape, dev)` scores every legal tile against a cost model and returns
the best few. No GPU is touched and nothing is timed.

See docs/internals/fused-moe-16bit.md.
"""

import dataclasses
import math

from kohakuwullm.kernels.gemm.device import Device
from kohakuwullm.kernels.gemm.plan import _ldsm_per_mma, _occupancy, _warp_tile

BLOCK_M = (32, 64, 128, 256)
BLOCK_N = (32, 64, 128, 256)
BLOCK_K = (32, 64, 128)
WARPS = (4, 8)
STAGES = (2, 3, 4)


@dataclasses.dataclass(frozen=True)
class GroupedShape:
    """One grouped GEMM: ``rows`` rows over ``experts`` groups, ``n`` by ``k`` each.

    ``accs`` is how many ``(BLOCK_M, BLOCK_N)`` accumulators the kernel carries --
    two for GEMM1, which computes the gate and value halves together.
    ``epilogue_bytes`` is DRAM streamed per output element. A scatter-add into a
    token-sized buffer is L2-resident and counts zero. ``live_tiles`` is how many
    ``(BLOCK_M, BLOCK_N)`` fp32 tiles are simultaneously live, which the epilogue
    can raise above ``accs``; zero means it equals ``accs``.
    """

    rows: int
    n: int
    k: int
    experts: int
    accs: int = 1
    epilogue_bytes: int = 0
    live_tiles: int = 0


@dataclasses.dataclass(frozen=True)
class MoePlan:
    """A tile, a launch shape and the model's prediction for it."""

    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    cta_per_sm: int
    tiles: int
    iters: int
    wave_eff: float
    group_eff: float
    pipe_busy: float
    predicted_tflops: float
    limiter: str

    @property
    def tile(self) -> dict[str, int]:
        return {"BLOCK_M": self.bm, "BLOCK_N": self.bn, "BLOCK_K": self.bk}


def _candidates():
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    for stages in STAGES:
                        yield bm, bn, bk, warps, stages


def score(
    shape: GroupedShape,
    dev: Device,
    elem_bytes: int,
    bm: int,
    bn: int,
    bk: int,
    warps: int,
    stages: int,
) -> MoePlan | None:
    """Predicted rate for one candidate, or None if the candidate is illegal."""
    wm, wn = _warp_tile(bm, bn, warps)
    if not wm:
        return None

    acc_regs = (shape.live_tiles or shape.accs) * bm * bn / (32 * warps)
    if acc_regs > 168:
        return None
    est_regs = int(acc_regs) + dev.reg_overhead
    smem = (bm + shape.accs * bn) * bk * elem_bytes * (stages - 1)
    if smem > dev.smem_per_cta:
        return None
    cta_per_sm = _occupancy(dev, est_regs, smem, warps)
    if cta_per_sm < 1:
        return None

    n_tiles = -(-shape.n // bn)
    iters = -(-shape.k // bk)
    # Every expert's row range ends in a partial tile, half wasted on average.
    row_tiles = shape.rows / bm + shape.experts / 2
    tiles = row_tiles * n_tiles
    # The grid is a bound, so no-op programs occupy launch slots the waves see.
    launched = (-(-shape.rows // bm) + shape.experts) * n_tiles

    pipe = dev.moe_cta_tax[min(cta_per_sm, len(dev.moe_cta_tax)) - 1]
    pipe *= 1.0 - dev.ldsm_slope * max(_ldsm_per_mma(wm, wn) - dev.ldsm_ref, 0.0)
    # MACs per byte staged through shared memory; below saturation the loads bind.
    intensity = 2.0 * bm * bn * shape.accs / ((bm + shape.accs * bn) * elem_bytes)
    pipe *= min(1.0, intensity / dev.moe_intensity_sat)
    if shape.k % bk:
        pipe *= dev.uneven_k
    pipe = max(pipe, 0.05)

    waves = launched / (dev.sms * cta_per_sm)
    wave_eff = waves / max(math.ceil(waves), 1)
    group_eff = (shape.rows / bm) / row_tiles

    peak = dev.mma_peak * 1e12
    flops = 2.0 * shape.rows * shape.n * shape.k * shape.accs
    refill = (iters + stages) / iters
    best_s = flops * refill / (peak * pipe * wave_eff * group_eff)
    limiter = "mma"

    weights = shape.experts * shape.n * shape.k * shape.accs * elem_bytes
    acts = shape.rows * shape.k * elem_bytes
    outs = shape.rows * shape.n * shape.epilogue_bytes
    dram_s = (weights + acts + outs) / (dev.dram_bw * 1e9)
    if dram_s > best_s:
        best_s, limiter = dram_s, "dram"
    l2_s = (
        2.0
        * shape.k
        * shape.rows
        * shape.n
        * shape.accs
        * (1.0 / bm + 1.0 / bn)
        / (dev.l2_bw * 1e9)
    )
    if l2_s > best_s:
        best_s, limiter = l2_s, "l2"

    return MoePlan(
        bm=bm,
        bn=bn,
        bk=bk,
        warps=warps,
        stages=stages,
        cta_per_sm=cta_per_sm,
        tiles=int(tiles),
        iters=iters,
        wave_eff=wave_eff,
        group_eff=group_eff,
        pipe_busy=pipe,
        predicted_tflops=flops / best_s * 1e-12,
        limiter=limiter,
    )


def plan_topk(
    shape: GroupedShape, dev: Device, elem_bytes: int = 2, topk: int = 1
) -> list[MoePlan]:
    """The ``topk`` highest-scoring legal tiles for this grouped shape, best first."""
    dev.validate()
    scored = [
        p
        for p in (score(shape, dev, elem_bytes, *c) for c in _candidates())
        if p is not None
    ]
    if not scored:
        raise ValueError(
            f"no legal tile for {shape.rows}x{shape.n}x{shape.k} on {dev.name}"
        )
    # Fewer warps breaks a tie: the model cannot separate them, and the smaller CTA
    # gives finer wave granularity.
    scored.sort(key=lambda p: (-p.predicted_tflops, p.warps))
    return scored[:topk]
