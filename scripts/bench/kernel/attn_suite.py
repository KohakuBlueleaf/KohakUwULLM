"""MXFP8 vs bf16/fp16 attention: dtypes, SDPA and varlen baselines, windowing."""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from kohakuwullm.kernels.attention.flash_attn import triton_varlen_attn
from kohakuwullm.kernels.mxfp8.attention import (
    mxfp8_attention_q,
    quantize_rows,
    column_mean,
)

SHAPES = [(4096, 64), (8192, 64), (16384, 64), (8192, 128), (16384, 128), (32768, 64)]
WINDOWS = [None, 1024, 4096]


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


def rms_vs_fp64(out, q, k, v, causal, n=2048):
    """Relative RMS against an fp64 reference over the first ``n`` rows."""
    with torch.no_grad():
        t = q.shape[0]
        s = (q[:n].double() @ k.double().T) * (q.shape[-1] ** -0.5)
        if causal:
            m = (
                torch.arange(t, device=q.device)[None, :]
                > torch.arange(n, device=q.device)[:, None]
            )
            s = s.masked_fill(m, float("-inf"))
        ref = torch.softmax(s, -1) @ v.double()
        d = out[:n].double() - ref
        return (d**2).mean().sqrt().item() / (ref**2).mean().sqrt().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/attn/attn_suite.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for t, head in SHAPES:
            torch.manual_seed(0)
            q = torch.randn(t, head, device="cuda", dtype=dtype)
            k = (
                torch.randn(t, head, device="cuda")
                + 2.0 * torch.randn(1, head, device="cuda")
            ).to(dtype)
            v = torch.randn(t, head, device="cuda", dtype=dtype)
            q3, k3, v3 = q[:, None], k[:, None], v[:, None]
            cu = torch.tensor([0, t], device="cuda", dtype=torch.int32)
            mu = column_mean(k)
            qq, qs = quantize_rows(q)
            kq, ks = quantize_rows(k, mu)

            row = {"dtype": dtype_name, "T": t, "head": head}
            try:
                row["sdpa"] = bench(
                    lambda: F.scaled_dot_product_attention(
                        q3.transpose(0, 1)[None],
                        k3.transpose(0, 1)[None],
                        v3.transpose(0, 1)[None],
                        is_causal=True,
                    )
                )
            except Exception:
                row["sdpa"] = None
            row["varlen"] = bench(
                lambda: triton_varlen_attn(q3, k3, v3, cu, t, causal=True)
            )
            row["mxfp8"] = bench(
                lambda: mxfp8_attention_q(qq, qs, kq, ks, v, causal=True)
            )
            o_mx, _ = mxfp8_attention_q(qq, qs, kq, ks, v, causal=True)
            o_bf = triton_varlen_attn(q3, k3, v3, cu, t, causal=True)[:, 0]
            row["mxfp8_rms"] = rms_vs_fp64(o_mx, q, k, v, True)
            row["bf16_rms"] = rms_vs_fp64(o_bf, q, k, v, True)

            for w in WINDOWS[1:]:
                if w >= t:
                    continue
                try:
                    row[f"varlen_w{w}"] = bench(
                        lambda: triton_varlen_attn(
                            q3, k3, v3, cu, t, causal=True, window=w
                        )
                    )
                except Exception as exc:
                    row[f"varlen_w{w}"] = None
                    row[f"varlen_w{w}_err"] = type(exc).__name__
            rows.append(row)
            print(row, flush=True)
            del q, k, v, qq, kq
            torch.cuda.empty_cache()
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
