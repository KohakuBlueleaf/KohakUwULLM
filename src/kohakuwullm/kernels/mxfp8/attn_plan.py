"""Choose a flash-attention tile from the shape, analytically, then time a shortlist.

`plan_attn` scores every legal tile against the card's budgets without touching
the GPU. `tuned_attn_plan` times only the top few and caches by shape.

See docs/internals/mxfp8-attention.md.
"""

import dataclasses
import math

import torch

from kohakuwullm.kernels.gemm.device import RTX_5090, Device

BLOCK_M = (32, 64, 128)
BLOCK_N = (32, 64, 128, 256)
WARPS = (4, 8)
STAGES = (2, 3, 4)
MMA_M, MMA_N, MMA_K = 16, 8, 32


@dataclasses.dataclass(frozen=True)
class AttnPlan:
    """A flash-attention tile and the model's prediction for it."""

    block_m: int
    block_n: int
    warps: int
    stages: int
    programs: int
    cta_per_sm: int
    wave_eff: float
    smem: int
    acc_regs: float
    mma_per_iter: int
    score: float


def _smem(block_m, block_n, head, stages, q_bytes, kv_bytes):
    """Shared memory for the pipelined K and V tiles plus their scale blocks."""
    per_stage = block_n * head * (kv_bytes + 2) + block_n * (head // 32)
    return per_stage * max(stages - 1, 1) + block_m * head * q_bytes


def score_attn(
    t, head, n_heads, dev, block_m, block_n, warps, stages, q_bytes=1, kv_bytes=1
):
    """Predicted merit of one tile, or None when a budget rejects it."""
    if head % 32 or block_n % 32:
        return None
    acc_regs = block_m * head / (32 * warps)
    if acc_regs > 168:
        return None
    smem = _smem(block_m, block_n, head, stages, q_bytes, kv_bytes)
    if smem > dev.smem_per_cta or warps * 32 > 1024:
        return None

    est_regs = int(acc_regs) + dev.reg_overhead
    threads = warps * 32
    cta_per_sm = max(
        min(
            dev.regs_per_sm // max(est_regs * threads, 1),
            dev.smem_per_sm // max(smem, 1),
            dev.max_threads_per_sm // threads,
        ),
        0,
    )
    if cta_per_sm < 1:
        return None

    programs = -(-t // block_m) * n_heads
    waves = max(-(-programs // dev.sms), 1)
    wave_eff = (programs / dev.sms) / waves

    # MMAs per warp per K block, and the ldmatrix each one costs.
    mma = (block_m // MMA_M) * (block_n // MMA_N) * (head // MMA_K) // warps
    ldsm = ((block_m + block_n) / 16.0) / max(mma, 1)
    pipe = dev.bar_tax[min(cta_per_sm, len(dev.bar_tax)) - 1]
    pipe *= 1.0 - dev.ldsm_slope * max(ldsm - dev.ldsm_ref, 0.0)

    # Softmax cost scales with the score tile, MMA cost with its depth.
    softmax_ratio = (block_m * block_n) / max(block_m * block_n * head / 32.0, 1.0)
    return AttnPlan(
        block_m=block_m,
        block_n=block_n,
        warps=warps,
        stages=stages,
        programs=programs,
        cta_per_sm=cta_per_sm,
        wave_eff=wave_eff,
        smem=smem,
        acc_regs=acc_regs,
        mma_per_iter=mma,
        score=wave_eff * max(pipe, 0.05) / (1.0 + softmax_ratio),
    )


def plan_attn(
    t, head, n_heads=1, dev: Device = RTX_5090, topk=1, q_bytes=1, kv_bytes=1
):
    """The ``topk`` highest-scoring legal tiles for this attention shape."""
    dev.validate()
    out = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for w in WARPS:
                for s in STAGES:
                    p = score_attn(
                        t, head, n_heads, dev, bm, bn, w, s, q_bytes, kv_bytes
                    )
                    if p:
                        out.append(p)
    if not out:
        raise ValueError(f"no legal attention tile for T={t} head={head}")
    out.sort(key=lambda p: -p.score)
    return out[:topk]


_CACHE: dict[tuple, AttnPlan] = {}


def tuned_attn_plan(
    t,
    head,
    n_heads=1,
    dev: Device = RTX_5090,
    shortlist=4,
    runner=None,
    dtype=torch.bfloat16,
) -> AttnPlan:
    """Best of the model's top ``shortlist`` tiles, timed once and cached.

    ``runner(plan) -> callable`` builds the thing to time; without it the
    model's own ranking is returned unmeasured.
    """
    key = (t, head, n_heads, str(dtype))
    if key in _CACHE:
        return _CACHE[key]
    cands = plan_attn(t, head, n_heads, dev, shortlist)
    best = cands[0]
    if runner is not None:
        best_ms = math.inf
        for cand in cands:
            try:
                fn = runner(cand)
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                beg = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                beg.record()
                for _ in range(20):
                    fn()
                end.record()
                torch.cuda.synchronize()
                ms = beg.elapsed_time(end) / 20
            except Exception:
                continue
            if ms < best_ms:
                best, best_ms = cand, ms
    _CACHE[key] = best
    return best


def clear() -> None:
    _CACHE.clear()
