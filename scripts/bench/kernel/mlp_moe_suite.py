"""MLP and MoE at 16-bit and MXFP8, across the implementations we ship."""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from kohakuwullm.kernels.gemm import RTX_5090, TunedGemm
from kohakuwullm.kernels.mxfp8.fused_act import rmsnorm_mx, swiglu_mx
from kohakuwullm.kernels.mxfp8.interop import as_vendor_scales
from kohakuwullm.kernels.mxfp8.quantize import (
    mxfp8_matmul_pq,
    quantize_mx,
    quantize_mx_vendor,
)

SHAPES = [(4096, 2048, 8192), (8192, 2048, 8192), (4096, 4096, 14336)]


def bench(fn, warmup=8, iters=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    b = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        b[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return min(x.elapsed_time(y) for x, y in zip(b, e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/mlp_moe/mlp_moe.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for t, d, ffn in SHAPES:
        for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            torch.manual_seed(0)
            x = torch.randn(t, d, device="cuda", dtype=dtype)
            wg = torch.randn(ffn, d, device="cuda", dtype=dtype) * d**-0.5
            wu = torch.randn(ffn, d, device="cuda", dtype=dtype) * d**-0.5
            wd = torch.randn(d, ffn, device="cuda", dtype=dtype) * ffn**-0.5
            nw = torch.randn(d, device="cuda", dtype=dtype)
            row = {"T": t, "D": d, "FFN": ffn, "dtype": dtype_name}

            def torch_mlp():
                h = F.rms_norm(x, (d,), nw)
                return F.silu(h @ wg.T) * (h @ wu.T) @ wd.T

            row["torch_16bit"] = bench(torch_mlp)
            cm = torch.compile(torch_mlp)
            try:
                row["torch_16bit_compiled"] = bench(cm)
            except Exception:
                row["torch_16bit_compiled"] = None

            gg = TunedGemm(t, ffn, d, RTX_5090, x.element_size(), dtype=dtype)
            gd = TunedGemm(t, d, ffn, RTX_5090, x.element_size(), dtype=dtype)

            def ours_16bit():
                h = F.rms_norm(x, (d,), nw)
                g = gg(h, wg.T.contiguous())
                u = gg(h, wu.T.contiguous())
                return gd(F.silu(g) * u, wd.T.contiguous())

            try:
                row["ours_16bit"] = bench(ours_16bit)
            except Exception as exc:
                row["ours_16bit"] = None
                row["ours_16bit_err"] = type(exc).__name__

            if dtype is torch.bfloat16:
                wgq, wgs = quantize_mx(wg)
                wuq, wus = quantize_mx(wu)
                wdq, wds = quantize_mx(wd)

                def mx_unfused():
                    h = F.rms_norm(x, (d,), nw)
                    hq, hs = quantize_mx(h)
                    g = mxfp8_matmul_pq(hq, hs, wgq, wgs, dtype)
                    u = mxfp8_matmul_pq(hq, hs, wuq, wus, dtype)
                    aq, as_ = quantize_mx(F.silu(g) * u)
                    return mxfp8_matmul_pq(aq, as_, wdq, wds, dtype)

                def mx_fused():
                    hq, hs = rmsnorm_mx(x, nw)
                    g = mxfp8_matmul_pq(hq, hs, wgq, wgs, dtype)
                    u = mxfp8_matmul_pq(hq, hs, wuq, wus, dtype)
                    aq, as_ = swiglu_mx(g, u)
                    return mxfp8_matmul_pq(aq, as_, wdq, wds, dtype)

                row["mxfp8_unfused"] = bench(mx_unfused)
                row["mxfp8_fused"] = bench(mx_fused)

                wgv, wgvs = quantize_mx_vendor(wg)
                wuv, wuvs = quantize_mx_vendor(wu)
                wdv, wdvs = quantize_mx_vendor(wd)

                def mx_vendor():
                    h = F.rms_norm(x, (d,), nw)
                    hq, hs = quantize_mx_vendor(h)
                    g = torch.ops.kohakuwullm.mxfp8_mm_swizzled(hq, hs, wgv, wgvs)
                    u = torch.ops.kohakuwullm.mxfp8_mm_swizzled(hq, hs, wuv, wuvs)
                    aq, as_ = quantize_mx_vendor(F.silu(g) * u)
                    return torch.ops.kohakuwullm.mxfp8_mm_swizzled(aq, as_, wdv, wdvs)

                try:
                    row["mxfp8_vendor"] = bench(mx_vendor)
                    row["mxfp8_vendor_compiled"] = bench(torch.compile(mx_vendor))
                except Exception as exc:
                    row["mxfp8_vendor_err"] = f"{type(exc).__name__}: {exc}"[:80]
            rows.append(row)
            print(row, flush=True)
            del x, wg, wu, wd
            torch.cuda.empty_cache()
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
