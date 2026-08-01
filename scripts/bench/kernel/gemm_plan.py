"""Validate the analytic GEMM planner against cuBLAS over a shape grid.

Each shape is timed order-controlled: arms run round-robin and each keeps its
best round, so clock drift hits every arm equally.
"""

import argparse
import json

import torch

from kohakuwullm.kernels.gemm import RTX_5090, StreamKGemm, TunedGemm, plan

SHAPES = [
    # square, the classic ladder
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (6144, 6144, 6144),
    (8192, 8192, 8192),
    (12288, 12288, 12288),
    # deep K
    (4096, 4096, 16384),
    (2048, 2048, 32768),
    (8192, 2048, 16384),
    # wide N, thin K: the LM head shape
    (16384, 32768, 2048),
    (8192, 65536, 1024),
    # tall M, thin N: MoE expert and projection shapes
    (16384, 1280, 5120),
    (16384, 5120, 1280),
    (8192, 1280, 1280),
    (32768, 2048, 2048),
    (65536, 1024, 1024),
    # awkward tile counts, where wave quantization bites hardest
    (5120, 5120, 4096),
    (7168, 7168, 4096),
    (10240, 3072, 4096),
    (3072, 10240, 4096),
    (11264, 11264, 2048),
    # non-multiples of the tile
    (4000, 4000, 4096),
    (6000, 3000, 5000),
]


def mamf(fn, warmup=15, iters=40):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tune", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    dev = RTX_5090

    print(
        f"{torch.cuda.get_device_name(0)}  {args.dtype}  "
        f"ISA {dev.mma_peak:.1f} TF/s  {args.rounds} interleaved rounds\n"
    )
    print(
        f"{'shape':>22} {'tile':>14} {'w':>2} {'R':>2} {'sk':>5} "
        f"{'pred':>7} {'ours':>7} {'cuBLAS':>7} {'ratio':>6} {'err':>8}"
    )
    rows = []
    for m, n, k in SHAPES:
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        c = torch.empty((m, n), device="cuda", dtype=dtype)
        flops = 2.0 * m * n * k
        p = plan(m, n, k, dev, a.element_size())
        try:
            if args.tune:
                g = TunedGemm(
                    m, n, k, dev, a.element_size(), shortlist=args.tune, dtype=dtype
                )
                p = g.plan
            else:
                g = StreamKGemm(m, n, k, dev, a.element_size(), p=p)
            g(a, b, c)
        except Exception as exc:
            print(f"{str((m, n, k)):>22} {type(exc).__name__}: {str(exc)[:40]}")
            continue
        ref = a.double() @ b.double()
        err = (c.double() - ref).abs().max().item() / ref.abs().max().item()
        del ref

        arms = {"ours": lambda: g(a, b, c), "cub": lambda: torch.matmul(a, b, out=c)}
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        best = {nm: 1e30 for nm in arms}
        for _ in range(args.rounds):
            for nm, fn in arms.items():
                best[nm] = min(best[nm], mamf(fn))
        tf = {nm: flops / t * 1e-9 for nm, t in best.items()}
        print(
            f"{str((m, n, k)):>22} {str(p.tile):>14} {p.warps:>2} "
            f"{p.cta_per_sm:>2} {p.sk_ctas:>5} {p.predicted_tflops:>7.1f} "
            f"{tf['ours']:>7.1f} {tf['cub']:>7.1f} "
            f"{tf['ours'] / tf['cub']:>6.3f} {err:>8.5f}",
            flush=True,
        )
        rows.append(
            {
                "shape": [m, n, k],
                "tile": list(p.tile),
                "warps": p.warps,
                "cta_per_sm": p.cta_per_sm,
                "stream_k": p.stream_k,
                "sk_ctas": p.sk_ctas,
                "predicted": p.predicted_tflops,
                "ours": tf["ours"],
                "cublas": tf["cub"],
                "ratio": tf["ours"] / tf["cub"],
                "rel_err": err,
                "limiter": p.limiter,
            }
        )
        del a, b, c
        torch.cuda.empty_cache()

    if rows:
        geo = 1.0
        for r in rows:
            geo *= r["ratio"]
        geo **= 1.0 / len(rows)
        wins = sum(1 for r in rows if r["ratio"] > 1.0)
        mape = sum(abs(r["predicted"] - r["ours"]) / r["ours"] for r in rows) / len(
            rows
        )
        print(
            f"\ngeomean vs cuBLAS {geo:.4f} over {len(rows)} shapes, "
            f"{wins} wins, worst {min(r['ratio'] for r in rows):.3f}"
        )
        print(f"model error against measured: {mape * 100:.1f}% mean absolute")
        print(f"max rel err vs fp64: {max(r['rel_err'] for r in rows):.6f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
