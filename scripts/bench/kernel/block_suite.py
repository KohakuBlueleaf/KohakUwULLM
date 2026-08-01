"""Transformer block: pure PyTorch, PyTorch compiled, and our kernels compiled."""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from kohakuwullm.kernels.attention.flash_attn import triton_varlen_attn
from kohakuwullm.kernels.mxfp8.fused_act import rmsnorm_mx, swiglu_mx
from kohakuwullm.kernels.mxfp8.quantize import mxfp8_matmul_pq, quantize_mx

CONFIGS = [(4096, 2048, 8192, 16), (8192, 2048, 8192, 16), (4096, 4096, 14336, 32)]


def bench(fn, warmup=6, iters=20):
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


class Weights:
    """One block's parameters, shared by every arm so the comparison is fair."""

    def __init__(self, d, ffn, dtype):
        g = lambda *s, f: (torch.randn(*s, device="cuda", dtype=dtype) * f)
        self.wqkv = g(3 * d, d, f=d**-0.5)
        self.wo = g(d, d, f=d**-0.5)
        self.wg = g(ffn, d, f=d**-0.5)
        self.wu = g(ffn, d, f=d**-0.5)
        self.wd = g(d, ffn, f=ffn**-0.5)
        self.n1 = torch.randn(d, device="cuda", dtype=dtype)
        self.n2 = torch.randn(d, device="cuda", dtype=dtype)


def torch_block(x, w, d, heads, t):
    h = F.rms_norm(x, (d,), w.n1)
    qkv = h @ w.wqkv.T
    q, k, v = qkv.chunk(3, -1)
    hd = d // heads
    q, k, v = (z.view(t, heads, hd).transpose(0, 1)[None] for z in (q, k, v))
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = a[0].transpose(0, 1).reshape(t, d)
    x = x + a @ w.wo.T
    h2 = F.rms_norm(x, (d,), w.n2)
    return x + (F.silu(h2 @ w.wg.T) * (h2 @ w.wu.T)) @ w.wd.T


def ours_block(x, w, d, heads, t, cu, mxw):
    hq, hs = rmsnorm_mx(x, w.n1)
    qkv = mxfp8_matmul_pq(hq, hs, mxw["qkv"][0], mxw["qkv"][1], x.dtype)
    hd = d // heads
    q, k, v = (z.reshape(t, heads, hd) for z in qkv.chunk(3, -1))
    a = triton_varlen_attn(
        q.contiguous(), k.contiguous(), v.contiguous(), cu, t, causal=True
    ).reshape(t, d)
    aq, as_ = quantize_mx(a)
    x = x + mxfp8_matmul_pq(aq, as_, mxw["o"][0], mxw["o"][1], x.dtype)
    h2q, h2s = rmsnorm_mx(x, w.n2)
    g = mxfp8_matmul_pq(h2q, h2s, mxw["g"][0], mxw["g"][1], x.dtype)
    u = mxfp8_matmul_pq(h2q, h2s, mxw["u"][0], mxw["u"][1], x.dtype)
    actq, acts = swiglu_mx(g, u)
    return x + mxfp8_matmul_pq(actq, acts, mxw["d"][0], mxw["d"][1], x.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/block/block.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for t, d, ffn, heads in CONFIGS:
        torch.manual_seed(0)
        dtype = torch.bfloat16
        x = torch.randn(t, d, device="cuda", dtype=dtype)
        w = Weights(d, ffn, dtype)
        cu = torch.tensor([0, t], device="cuda", dtype=torch.int32)
        mxw = {
            k: quantize_mx(v)
            for k, v in (
                ("qkv", w.wqkv),
                ("o", w.wo),
                ("g", w.wg),
                ("u", w.wu),
                ("d", w.wd),
            )
        }
        row = {"T": t, "D": d, "FFN": ffn, "heads": heads}

        row["torch"] = bench(lambda: torch_block(x, w, d, heads, t))
        try:
            row["torch_compiled"] = bench(
                torch.compile(
                    lambda z: torch_block(z, w, d, heads, t)
                ).__call__.__get__(x)
                if False
                else (lambda: torch.compile(torch_block)(x, w, d, heads, t))
            )
        except Exception as exc:
            row["torch_compiled_err"] = type(exc).__name__
        try:
            row["ours"] = bench(lambda: ours_block(x, w, d, heads, t, cu, mxw))
            co = torch.compile(ours_block)
            row["ours_compiled"] = bench(lambda: co(x, w, d, heads, t, cu, mxw))
        except Exception as exc:
            row["ours_err"] = f"{type(exc).__name__}: {exc}"[:90]
        rows.append(row)
        print(row, flush=True)
        del x, w, mxw
        torch.cuda.empty_cache()
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
