"""Latent MoE: does compressing the expert input width pay for the dispatch?

Dispatch is a gather and a scatter of ``T * top_k`` rows, and its cost is linear
in the width of what is moved. :class:`LatentMoEMLP` runs the experts at
``dim / alpha`` while the router still reads full width, so alpha is a direct
lever on the two stages that dominate the non-GEMM path once the router is fused.

The measurement has to answer two things, and only one of them is the obvious one:

* **Does dispatch shrink as advertised?** gather + combine at alpha 1 / 2 / 4.
* **Do the expert GEMMs lose efficiency at narrow input width?** Their FLOP count
  drops by alpha too, so wall time falling is not evidence of anything. The honest
  metric is TFLOP/s: a GEMM whose K shrinks from 1280 to 320 can easily get
  *slower per FLOP*, and if it does, part of the dispatch saving is handed back.

Two extra full-width GEMMs (``down``, ``up``) are charged to the latent variants,
because they are the price of the compression and leaving them out would flatter
it.

Note the variants are **not capacity-equivalent**: alpha=2 has roughly half the
expert-bank parameters. This measures cost at a given alpha, not quality per
parameter, and the parameter counts are reported so the trade is visible.

Run::

    CUDA_VISIBLE_DEVICES=<n> .venv/bin/python scripts/bench/moe/moe_latent.py
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench.core.timing import (
    bench_ms,
    cpu_enqueue_ms,
    device_name,
    graph_ms,
    stream_bandwidths,
)
from kohakuwullm.kernels.moe.moe_dispatch import combine_routed, expert_sort
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.components.moe_variants import LatentMoEMLP

# (num_experts, top_k, dim) -- the shapes the router work was measured on.
SHAPES = [(48, 4, 1152), (64, 6, 1280), (96, 8, 1536)]
ALPHAS = [1, 2, 4]


def build(dim: int, experts: int, top_k: int, hidden: int, alpha: int):
    """``(module, inner)`` where ``inner`` is the MoEMLP holding the expert bank."""
    if alpha == 1:
        moe = MoEMLP(dim, num_experts=experts, top_k=top_k, hidden=hidden, num_shared=1)
        moe = moe.cuda().to(torch.bfloat16).train()
        return moe, moe
    latent = LatentMoEMLP(
        dim,
        hidden=hidden,
        alpha=alpha,
        num_experts=experts,
        top_k=top_k,
        num_shared=1,
    )
    latent = latent.cuda().to(torch.bfloat16).train()
    return latent, latent.inner


def bank_params(inner: MoEMLP) -> int:
    return inner.w_in.numel() + inner.w_out.numel()


def stage_rows(tokens: int, hidden: int) -> list[dict]:
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        for alpha in ALPHAS:
            module, inner = build(dim, experts, top_k, hidden, alpha)
            width = inner.dim
            # What the experts actually consume: full residual at alpha=1, the
            # compressed activation otherwise.
            payload = x if alpha == 1 else module.down(x)

            idx, weight, counts = inner.router.route(x)
            offsets, order, token_of, _ = expert_sort(
                idx.reshape(-1), counts, idx.shape[-1], experts
            )
            flat_weight = weight.reshape(-1)

            def gather(p=payload, t=token_of):
                return p.index_select(0, t).contiguous()

            sorted_rows = gather()

            def expert_gemm(i=inner, r=sorted_rows, o=offsets):
                return i._expert_compute(r, o)

            expert_out = expert_gemm()

            def combine(e=expert_out, w=flat_weight, o=order, t=token_of):
                return combine_routed(e, w, o, t, tokens)

            def projections(m=module):
                return m.up(m.down(x))

            def whole(m=module):
                return m(x)

            # FLOPs the expert bank does: two GEMMs over T*top_k rows, one
            # (width -> 2*hidden) and one (hidden -> width).
            routed = tokens * top_k
            gemm_flops = 2 * routed * width * 2 * hidden + 2 * routed * hidden * width
            proj_flops = 0 if alpha == 1 else 2 * tokens * dim * width * 2

            measured = {
                "gather": gather,
                "combine": combine,
                "expert_gemm": expert_gemm,
                # alpha=1 has no projections at all, so it is a true zero rather
                # than a timed no-op -- measuring the closure would report the
                # graph-replay noise floor (~11 us) as if it were work.
                "down_up_proj": projections if alpha > 1 else None,
                "whole_layer": whole,
            }
            for stage, fn in measured.items():
                if fn is None:
                    ms = 0.0
                elif stage == "whole_layer":
                    # Wall time, not a graph replay: this stage is what a training
                    # step actually pays, host dispatch included, and the other
                    # stages are graph-replayed device time by design.
                    ms = bench_ms(fn, warmup=5, iters=20)
                else:
                    ms = graph_ms(fn)
                flops = (
                    gemm_flops
                    if stage == "expert_gemm"
                    else proj_flops if stage == "down_up_proj" else 0
                )
                rows.append(
                    {
                        "kernel": stage,
                        "alpha": alpha,
                        "num_experts": experts,
                        "top_k": top_k,
                        "dim": dim,
                        "latent": width,
                        "hidden": hidden,
                        "tokens": tokens,
                        "graph_us": ms * 1e3,
                        "cpu_us": cpu_enqueue_ms(fn) * 1e3 if fn is not None else 0.0,
                        "tflops": (flops / (ms / 1e3) / 1e12) if ms and flops else 0.0,
                        "bank_params_m": bank_params(inner) / 1e6,
                        "total_params_m": sum(p.numel() for p in module.parameters())
                        / 1e6,
                    }
                )
            del module, inner, payload, sorted_rows, expert_out
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dir", default="out/bench/moe/moe_latent")
    args = parser.parse_args()

    stream = stream_bandwidths()
    hottest = max(stream, key=stream.get)
    print(f"device: {device_name()}")
    print(
        "  STREAM: "
        + "  ".join(f"{k} {stream[k]:.0f}" for k in ("copy", "read", "triad"))
        + f" GB/s (peak {max(stream.values()):.0f}, highest={hottest})"
    )
    if hottest != "triad":
        print("  WARNING: triad is not the highest pattern -- suspect contention")

    rows = stage_rows(args.tokens, args.hidden)
    for row in rows:
        extra = f" {row['tflops']:7.1f} TF/s" if row["tflops"] else ""
        print(
            f"  E={row['num_experts']:3d} k={row['top_k']} dim={row['dim']:5d} "
            f"a={row['alpha']} lat={row['latent']:5d} {row['kernel']:>13s}: "
            f"gpu {row['graph_us']:8.1f} us{extra}"
        )

    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, "moe_latent.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "device": device_name(),
                "stream_bandwidths": stream,
                "config": {
                    "tokens": args.tokens,
                    "hidden": args.hidden,
                    "shapes": SHAPES,
                    "alphas": ALPHAS,
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
