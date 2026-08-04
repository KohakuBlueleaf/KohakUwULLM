"""Fused 16-bit MoE expert path against the eager one and against MXFP8.

Reports throughput, peak memory and fp64-referenced accuracy in one table, for
fp16 and bf16, forward and forward+backward.

    .venv/bin/python scripts/bench/kernel/moe_fused.py
"""

import argparse

import torch
import torch.nn.functional as F

from kohakuwullm.bench import bench_ms, bench_peak_memory, device_name, ulp_error
from kohakuwullm.kernels.moe.fused_moe import fused_moe_experts
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed
from kohakuwullm.kernels.mxfp8.moe import MXFP8ExpertWeights, mxfp8_moe_experts

# Kohaku-MoE-1B: dim 768, expert hidden 384, 64 experts, top-8.
DIM = 768
HIDDEN = 384
EXPERTS = 64
TOP_K = 8


def make_case(tokens, dtype, device, seed=0):
    """Balanced random routing over ``EXPERTS``; returns the whole kernel contract."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, DIM, device=device, dtype=dtype, generator=g) * 0.5
    w_in = (
        torch.randn(EXPERTS, 2 * HIDDEN, DIM, device=device, dtype=dtype, generator=g)
        * DIM**-0.5
    )
    w_out = (
        torch.randn(EXPERTS, DIM, HIDDEN, device=device, dtype=dtype, generator=g)
        * HIDDEN**-0.5
    )
    pairs = tokens * TOP_K
    gate = torch.rand(pairs, device=device, dtype=dtype, generator=g) * 0.5 + 0.25

    expert_of = torch.randint(0, EXPERTS, (pairs,), device=device, generator=g)
    order = torch.argsort(expert_of, stable=True).to(torch.int32)
    counts = torch.bincount(expert_of, minlength=EXPERTS)
    offsets = torch.zeros(EXPERTS + 1, device=device, dtype=torch.int32)
    offsets[1:] = counts.cumsum(0)
    token_of = (order.long() // TOP_K).to(torch.int32)
    return x, w_in, w_out, gate, token_of, order, offsets


def eager(x, w_in, w_out, gate, token_of, order, offsets):
    """``MoEMLP._routed_eager``: gather, two grouped GEMMs, eager swiglu, combine."""
    xs = x.index_select(0, token_of.long()).contiguous()
    h = grouped_gemm(xs, w_in, offsets)
    gate_h, value = h.chunk(2, dim=-1)
    out_sorted = grouped_gemm(F.silu(gate_h) * value, w_out, offsets)
    return combine_routed(out_sorted, gate, order, token_of, x.shape[0])


def fused(x, w_in, w_out, gate, token_of, order, offsets):
    return fused_moe_experts(x, w_in, w_out, gate, token_of, order, offsets)


def make_mxfp8(w_in, w_out):
    packed = MXFP8ExpertWeights(w_in.detach(), w_out.detach())

    def run(x, w_in, w_out, gate, token_of, order, offsets):
        return mxfp8_moe_experts(x, w_in, w_out, gate, token_of, order, offsets, packed)

    return run


def reference64(x, w_in, w_out, gate, token_of, order, offsets):
    """fp64 autograd oracle: the same routing arithmetic, no kernel of ours in it."""
    rows = x.index_select(0, token_of.long())
    off = offsets.tolist()
    outs = []
    for e in range(EXPERTS):
        block = rows[off[e] : off[e + 1]]
        gate_h, value = (block @ w_in[e].T).chunk(2, dim=-1)
        outs.append((F.silu(gate_h) * value) @ w_out[e].T)
    scaled = torch.cat(outs, dim=0) * gate.index_select(0, order.long()).reshape(-1, 1)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)
    return out.index_add(0, token_of.long(), scaled)


def leaves(case):
    x, w_in, w_out, gate = (t.detach().requires_grad_(True) for t in case[:4])
    return (x, w_in, w_out, gate) + case[4:]


def accuracy(fn, case, dtype):
    """Forward and gradient ULP against an fp64 run of the same routing."""
    args = leaves(case)
    ref_args = leaves(tuple(t.double() if t.is_floating_point() else t for t in case))

    out = fn(*args)
    if out.grad_fn is None:
        raise RuntimeError(f"{fn} returned a tensor with no grad_fn")
    ref = reference64(*ref_args)

    seed = torch.randn(
        out.shape,
        device=out.device,
        generator=torch.Generator(device=out.device).manual_seed(7),
    )
    out.backward(seed.to(out.dtype))
    ref.backward(seed.double())

    got = {
        "out": out,
        "dx": args[0].grad,
        "dw_in": args[1].grad,
        "dw_out": args[2].grad,
        "dgate": args[3].grad,
    }
    want = {
        "out": ref,
        "dx": ref_args[0].grad,
        "dw_in": ref_args[1].grad,
        "dw_out": ref_args[2].grad,
        "dgate": ref_args[3].grad,
    }
    for name, tensor in got.items():
        if tensor is None:
            raise RuntimeError(f"{fn} left {name} with no gradient")
        if not tensor.any():
            raise RuntimeError(f"{fn} left {name} all zero")
    return {k: ulp_error(got[k], want[k], dtype, mode="rms") for k in got}


def timings(fn, case, iters):
    args = leaves(case)
    fwd = bench_ms(lambda: fn(*args), warmup=10, iters=iters)
    grad_seed = torch.randn_like(fn(*args))

    def step():
        for t in args[:4]:
            t.grad = None
        fn(*args).backward(grad_seed)

    fwdbwd = bench_ms(step, warmup=10, iters=iters)
    peak = bench_peak_memory(step)
    return fwd, fwdbwd, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--acc-tokens", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = "cuda"
    print(f"{device_name()}  dim={DIM} hidden={HIDDEN} experts={EXPERTS} top_k={TOP_K}")

    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).split(".")[-1]
        acc_case = make_case(args.acc_tokens, dtype, device)
        arms = {
            "eager": eager,
            "fused": fused,
            "mxfp8": make_mxfp8(acc_case[1], acc_case[2]),
        }
        print(
            f"\n=== {tag}: accuracy, {args.acc_tokens} tokens, "
            f"{args.acc_tokens * TOP_K} routed rows (ULP vs fp64, rms) ==="
        )
        print(
            f"{'arm':<8}{'out':>10}{'dx':>10}{'dw_in':>10}{'dw_out':>10}{'dgate':>10}"
        )
        for name, fn in arms.items():
            e = accuracy(fn, acc_case, dtype)
            print(
                f"{name:<8}{e['out']:>10.1f}{e['dx']:>10.1f}{e['dw_in']:>10.1f}"
                f"{e['dw_out']:>10.1f}{e['dgate']:>10.1f}"
            )

        case = make_case(args.tokens, dtype, device)
        arms["mxfp8"] = make_mxfp8(case[1], case[2])
        flops = 2 * args.tokens * TOP_K * DIM * HIDDEN * 3
        print(
            f"\n=== {tag}: throughput, {args.tokens} tokens, "
            f"{args.tokens * TOP_K} routed rows ==="
        )
        print(
            f"{'arm':<8}{'fwd ms':>9}{'TF/s':>8}{'fwd+bwd ms':>12}{'TF/s':>8}"
            f"{'peak GiB':>10}{'vs eager':>10}"
        )
        base = None
        for name, fn in arms.items():
            fwd, fwdbwd, peak = timings(fn, case, args.iters)
            if fwdbwd < fwd * 1.5:
                raise RuntimeError(f"{name}: fwd+bwd {fwdbwd:.3f} ms is not a backward")
            base = base or fwdbwd
            print(
                f"{name:<8}{fwd:>9.3f}{flops / fwd / 1e9:>8.1f}{fwdbwd:>12.3f}"
                f"{3 * flops / fwdbwd / 1e9:>8.1f}{peak:>10.2f}"
                f"{base / fwdbwd:>9.2f}x"
            )


if __name__ == "__main__":
    main()
