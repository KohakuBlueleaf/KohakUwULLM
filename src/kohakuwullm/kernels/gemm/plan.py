"""Choose a GEMM tile and schedule from the problem shape, analytically.

`plan(m, n, k, dev)` scores every legal tile against a cost model and returns the
best `Plan`. No GPU is touched and nothing is timed.

See docs/performance/gemm.md section 5e.
"""

import dataclasses
import math

from kohakuwullm.kernels.gemm.device import Device

BLOCK_M = (64, 128, 256)
BLOCK_N = (64, 128, 256)
BLOCK_K = (32, 64)
WARPS = (4, 8)
STAGES = (3, 4)
MMA_M, MMA_N, MMA_K = 16, 8, 16


@dataclasses.dataclass(frozen=True)
class Plan:
    """A tile, a schedule and the model's prediction for it."""

    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    ctas: int
    cta_per_sm: int
    maxnreg: int | None
    stream_k: bool
    sk_tiles: int
    sk_ctas: int
    tiles: int
    iters: int
    wave_eff: float
    pipe_busy: float
    predicted_tflops: float
    limiter: str

    @property
    def tile(self) -> tuple[int, int, int]:
        return self.bm, self.bn, self.bk


def _warp_tile(bm: int, bn: int, warps: int) -> tuple[int, int]:
    """Split the CTA tile over warps, keeping each warp tile as square as it can."""
    best = None
    for wm_count in (1, 2, 4, 8):
        wn_count = warps // wm_count
        if wn_count < 1 or wm_count * wn_count != warps:
            continue
        wm, wn = bm // wm_count, bn // wn_count
        if wm < MMA_M or wn < MMA_N:
            continue
        if best is None or abs(wm - wn) < abs(best[0] - best[1]):
            best = (wm, wn)
    return best or (0, 0)


def _ldsm_per_mma(wm: int, wn: int) -> float:
    """`ldmatrix` instructions issued per MMA at the given warp tile."""
    mmas = (wm / MMA_M) * (wn / MMA_N)
    return ((wm / MMA_M) + (wn / MMA_M)) / mmas if mmas else 1e9


def _occupancy(dev: Device, est_regs: int, smem: int, warps: int) -> int:
    """CTAs resident per SM under the register, shared memory and thread limits."""
    threads = warps * 32
    by_reg = dev.regs_per_sm // max(est_regs * threads, 1)
    by_smem = dev.smem_per_sm // max(smem, 1)
    by_thread = dev.max_threads_per_sm // threads
    return max(min(by_reg, by_smem, by_thread), 0)


def _candidates(elem_bytes: int):
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    for stages in STAGES:
                        yield bm, bn, bk, warps, stages


def score(
    m: int,
    n: int,
    k: int,
    dev: Device,
    elem_bytes: int,
    bm: int,
    bn: int,
    bk: int,
    warps: int,
    stages: int,
) -> Plan | None:
    """Predicted rate for one candidate, or None if the candidate is illegal."""
    wm, wn = _warp_tile(bm, bn, warps)
    if not wm:
        return None

    acc_regs = bm * bn / (32 * warps)
    if acc_regs > 168:
        return None
    est_regs = int(acc_regs) + dev.reg_overhead
    smem = (bm + bn) * bk * elem_bytes * (stages - 1)
    if smem > dev.smem_per_cta or warps * 32 > 1024:
        return None

    cta_per_sm = _occupancy(dev, est_regs, smem, warps)
    maxnreg = None
    # Trade registers for a second resident CTA when the cap alone would allow it.
    if cta_per_sm < 2:
        capped = dev.regs_per_sm // (2 * warps * 32)
        if capped >= acc_regs + 16 and _occupancy(dev, capped, smem, warps) >= 2:
            cta_per_sm, maxnreg = 2, capped
    if cta_per_sm < 1:
        return None

    tiles = -(-m // bm) * -(-n // bn)
    iters = -(-k // bk)
    ctas = min(dev.sms * cta_per_sm, tiles)

    pipe = dev.bar_tax[min(cta_per_sm, len(dev.bar_tax)) - 1]
    pipe *= 1.0 - dev.ldsm_slope * max(_ldsm_per_mma(wm, wn) - dev.ldsm_ref, 0.0)
    # A K that BLOCK_K does not divide masks every load and costs registers.
    if k % bk:
        pipe *= dev.uneven_k
    pipe = max(pipe, 0.05)

    peak = dev.mma_peak * 1e12
    flops = 2.0 * m * n * k

    wave_eff = (tiles / dev.sms) / max(-(-tiles // dev.sms), 1)
    # Each tile refills the software pipeline before its first MMA.
    refill = (iters + stages) / iters
    plain_s = flops * refill / (peak * pipe * wave_eff)

    dp_tiles = (tiles // ctas) * ctas
    sk_tiles = tiles - dp_tiles
    sk_ctas = min(max(round(math.sqrt(dev.sk_balance * sk_tiles * iters)), 1), ctas)
    compute_s = flops * refill / (peak * pipe)
    fixup_s = (sk_ctas + 2 * sk_tiles) * bm * bn * 4 / (dev.dram_bw * 1e9)
    sk_s = compute_s + fixup_s

    stream_k = sk_tiles > 0 and sk_s < plain_s
    best_s = min(sk_s, plain_s) if sk_tiles else plain_s

    # Neither schedule can beat the traffic the problem itself requires.
    dram_s = (m * k + k * n + m * n) * elem_bytes / (dev.dram_bw * 1e9)
    l2_s = 2.0 * k * m * n * (1.0 / bm + 1.0 / bn) / (dev.l2_bw * 1e9)
    limiter = "mma"
    if dram_s > best_s:
        best_s, limiter = dram_s, "dram"
    if l2_s > best_s:
        best_s, limiter = l2_s, "l2"

    return Plan(
        bm=bm,
        bn=bn,
        bk=bk,
        warps=warps,
        stages=stages,
        ctas=ctas,
        cta_per_sm=cta_per_sm,
        maxnreg=maxnreg,
        stream_k=stream_k,
        sk_tiles=sk_tiles if stream_k else 0,
        sk_ctas=sk_ctas if stream_k else 0,
        tiles=tiles,
        iters=iters,
        wave_eff=wave_eff,
        pipe_busy=pipe,
        predicted_tflops=flops / best_s * 1e-12,
        limiter=limiter,
    )


def plan_topk(
    m: int, n: int, k: int, dev: Device, elem_bytes: int = 2, topk: int = 1
) -> list[Plan]:
    """The ``topk`` highest-scoring legal tiles, best first."""
    dev.validate()
    scored = [
        p
        for p in (score(m, n, k, dev, elem_bytes, *c) for c in _candidates(elem_bytes))
        if p is not None
    ]
    if not scored:
        raise ValueError(f"no legal tile for {m}x{n}x{k} on {dev.name}")
    scored.sort(key=lambda p: -p.predicted_tflops)
    return scored[:topk]


def plan(m: int, n: int, k: int, dev: Device, elem_bytes: int = 2) -> Plan:
    """Best tile and schedule for this shape on this card."""
    return plan_topk(m, n, k, dev, elem_bytes, 1)[0]
