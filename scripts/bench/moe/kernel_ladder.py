"""What each rewrite actually bought, measured as the ladder we climbed.

A naive-versus-final comparison answers the wrong question. It says the final kernel is
fast without saying which of the four things we changed made it fast, and every one of
those steps cost days. So each subsystem here is measured as the **sequence of
implementations the repo actually went through**, and the interesting number is the
increment between adjacent rungs, not the product.

This matters most for the MoE experts, where the end-to-end sweep measures 1.53-2.31x and
cannot attribute it. Three candidate sources are folded together in that figure:

* fp8 arithmetic in the GEMMs,
* fused epilogues, which stop the ``(tokens*top_k, hidden)`` intermediates from ever
  being written,
* the bf16 path simply being worse.

The third is the one worth being careful about. The shipped bf16 path is **already
grouped** -- it calls ``grouped_gemm``, not a per-expert loop -- so "the old path was
naive" is not obviously true, and the eager rung here is what tests it. Measuring
eager -> grouped-bf16 -> grouped-fp8 -> fused-fp8 separates all three.

Every row reports the same numbers, and the accuracy column is not optional: a kernel that
is fast and wrong is not a result, and two of the four rungs were wrong in ways no speed
measurement would have shown.

Usage::

    .venv/bin/python scripts/bench/moe/kernel_ladder.py --out out/bench/moe/kernel_ladder
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench.core.timing import bench_ms, bench_peak_memory, rel_error
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, quantize_mx, quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.grouped import grouped_mxfp8_gemm
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts
from kohakuwullm.models.presets import get_preset

WARMUP = 20
ITERS = 60


def _routing(tokens: int, top_k: int, experts: int, device, seed: int = 0):
    """Expert-sorted routing for ``tokens * top_k`` assignments."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assign = torch.randint(0, experts, (tokens * top_k,), generator=generator).to(
        device
    )
    order = torch.argsort(assign)
    counts = torch.bincount(assign, minlength=experts)
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    token_of = (order // top_k).to(torch.int32)
    return offsets, order.to(torch.int32), token_of


def _eager_experts(x, w_in, w_out, token_of, offsets, experts):
    """One ``index_select`` plus one matmul pair per expert -- the obvious implementation.

    Present as the floor, not as a straw man: this is what a MoE layer looks like before
    anyone writes a grouped kernel, and the question is how much of the final speedup is
    simply having left it behind.
    """
    out = torch.zeros(x.shape[0], w_out.shape[1], device=x.device, dtype=x.dtype)
    bounds = offsets.tolist()
    for e in range(experts):
        lo, hi = bounds[e], bounds[e + 1]
        if hi <= lo:
            continue
        rows = token_of[lo:hi].long()
        xe = x.index_select(0, rows)
        h = xe @ w_in[e].t()
        gate, value = h.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * value
        out.index_add_(0, rows, h @ w_out[e].t())
    return out


def _grouped_bf16(x, w_in, w_out, token_of, offsets, experts):
    """Grouped GEMM with the epilogues still separate -- what the bf16 arm ships.

    ``grouped_gemm`` takes no gather, so the rows are materialised first. That copy is
    part of what the fused path removes and belongs in this rung's cost.
    """
    rows = token_of.long()
    xs = x.index_select(0, rows)
    h = grouped_gemm(xs, w_in, offsets)
    gate, value = h.chunk(2, dim=-1)
    h = torch.nn.functional.silu(gate) * value
    out = torch.zeros(x.shape[0], w_out.shape[1], device=x.device, dtype=x.dtype)
    return out.index_add_(0, rows, grouped_gemm(h, w_out, offsets))


def ladder_moe(preset: str, tokens: int, device, dtype) -> list[dict]:
    """eager -> grouped bf16 -> grouped fp8 -> fused fp8, forward only."""
    config = get_preset(preset)
    dim, hidden = config.dim, config.moe_hidden
    experts, top_k = config.moe_num_experts, config.moe_top_k
    offsets, order, token_of = _routing(tokens, top_k, experts, device)

    x = torch.randn(tokens, dim, device=device, dtype=dtype)
    w_in = torch.randn(experts, 2 * hidden, dim, device=device, dtype=dtype) * dim**-0.5
    w_out = torch.randn(experts, dim, hidden, device=device, dtype=dtype) * hidden**-0.5

    reference = _eager_experts(x, w_in, w_out, token_of, offsets, experts)
    rows = []

    def record(name, fn, note=""):
        try:
            out = fn()
            ms = bench_ms(fn, warmup=WARMUP, iters=ITERS)
            peak = bench_peak_memory(fn)
            rows.append(
                dict(
                    preset=preset,
                    tokens=tokens,
                    stage=name,
                    ms=ms,
                    peak_gib=peak,
                    rel_error=rel_error(out, reference),
                    note=note,
                )
            )
        except Exception as exc:
            rows.append(dict(preset=preset, tokens=tokens, stage=name, error=repr(exc)))

    record(
        "1-eager", lambda: _eager_experts(x, w_in, w_out, token_of, offsets, experts)
    )
    record(
        "2-grouped-bf16",
        lambda: _grouped_bf16(x, w_in, w_out, token_of, offsets, experts),
    )

    xq, xs = quantize_mx(x)
    wq_in, ws_in = quantize_mx(w_in.reshape(experts * 2 * hidden, dim))
    wq_in = wq_in.view(experts, 2 * hidden, dim)
    ws_in = ws_in.view(experts, 2 * hidden, dim // BLOCK_SCALE)

    def grouped_fp8():
        h = grouped_mxfp8_gemm(
            xq, xs, wq_in, ws_in, offsets, index=token_of, out_dtype=dtype
        )
        gate, value = h.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * value
        hq, hs = quantize_mx(h)
        wq_out, ws_out = quantize_mx(w_out.reshape(experts * w_out.shape[1], hidden))
        per_row = grouped_mxfp8_gemm(
            hq,
            hs,
            wq_out.view(experts, -1, hidden),
            ws_out.view(experts, -1, hidden // BLOCK_SCALE),
            offsets,
            out_dtype=dtype,
        )
        out = torch.zeros(x.shape[0], w_out.shape[1], device=x.device, dtype=dtype)
        return out.index_add_(0, token_of.long(), per_row)

    record("3-grouped-fp8", grouped_fp8, "epilogues still separate")

    gate_w = torch.ones(tokens * top_k, device=device, dtype=dtype)
    packed = MXFP8ExpertWeights(w_in, w_out)
    record(
        "4-fused-fp8",
        lambda: mxfp8_moe_experts(
            x, w_in, w_out, gate_w, token_of, order, offsets, packed
        ),
        "epilogues folded in",
    )
    return rows


def ladder_quantize(device, dtype) -> list[dict]:
    """torch reference -> Triton plain layout -> Triton writing the cuBLAS swizzle."""
    rows = []
    for shape in ((16384, 1536), (149553, 896)):
        x = torch.randn(*shape, device=device, dtype=dtype)

        def torch_ref():
            blocks = x.reshape(x.shape[0], -1, BLOCK_SCALE)
            amax = blocks.abs().amax(dim=-1, keepdim=True).float()
            exponent = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448.0))
            return (blocks / torch.exp2(exponent)).to(torch.float8_e4m3fn)

        for name, fn in (
            ("1-torch", torch_ref),
            ("2-triton-plain", lambda: quantize_mx(x)),
            ("3-triton-swizzle", lambda: quantize_mx_vendor(x)),
        ):
            try:
                ms = bench_ms(fn, warmup=WARMUP, iters=ITERS)
                gb = x.numel() * (x.element_size() + 1) / 1e9
                rows.append(
                    dict(shape=list(shape), stage=name, ms=ms, gbps=gb / (ms / 1e3))
                )
            except Exception as exc:
                rows.append(dict(shape=list(shape), stage=name, error=repr(exc)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/moe/kernel_ladder")
    ap.add_argument("--presets", nargs="+", default=["Kohaku-MoE-1B", "Kohaku-MoE-3B"])
    ap.add_argument("--tokens", type=int, default=8192)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.makedirs(args.out, exist_ok=True)

    result = {"quantize": ladder_quantize(device, dtype), "moe": []}
    for name in args.presets:
        result["moe"].extend(ladder_moe(name, args.tokens, device, dtype))

    for row in result["quantize"]:
        print(
            f"quantize {row.get('shape')} {row['stage']:18s} "
            f"{row.get('ms', float('nan')):8.3f} ms {row.get('gbps', 0):7.0f} GB/s"
            f"{'  ' + row['error'][:60] if 'error' in row else ''}"
        )
    for row in result["moe"]:
        if "error" in row:
            print(
                f"moe {row['preset']:15s} {row['stage']:16s} FAILED {row['error'][:70]}"
            )
            continue
        print(
            f"moe {row['preset']:15s} {row['stage']:16s} {row['ms']:8.3f} ms  "
            f"peak {row['peak_gib']:5.2f} GiB  rel {row['rel_error']:.2e}  {row['note']}"
        )

    path = os.path.join(args.out, "kernel_ladder.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
