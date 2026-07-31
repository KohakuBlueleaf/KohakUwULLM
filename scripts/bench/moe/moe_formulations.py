"""Six frontier MoE formulations, measured against each other at preset shapes.

Writes ``moe_formulations.json``; ``moe_formulations_plot.py`` draws it. Split for
the same reason as ``kernels.py`` / ``kernels_plot.py``: a new formulation adds a
measurement, a rejected figure only changes the drawing.

The formulations differ along three axes that each cost something different, and
a single latency number confounds all three. Every row therefore carries what is
needed to separate them:

* ``router_path`` -- ``fused`` or ``eager``. A formulation on the eager path pays
  ~300 us of per-op dispatch that has nothing to do with its own merits, so a
  slow eager row is a statement about our kernel coverage, not about the
  formulation. The ``router`` stage is measured separately for exactly this.
* ``slots`` -- the dispatch width actually issued. ReMoE has no top-k, so it
  emits a padded ``min(E, 4k)`` slots and pays ``4k`` experts' FLOPs to activate
  ``k`` on average. Its latency is mostly this, not its gate.
* ``params`` / ``active_params`` -- capacity. Latent MoE runs its experts on a
  ``dim/alpha`` activation, which shrinks the expert bank by ``alpha``: it is a
  smaller model, and ranking it on latency alone would be reporting a capacity
  cut as a speed win.

Run::

    CUDA_VISIBLE_DEVICES=<n> .venv/bin/python scripts/bench/moe/moe_formulations.py
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench.core.timing import (
    cached_peak_bandwidth,
    cpu_enqueue_ms,
    device_name,
    format_metrics,
    graph_ms,
    module_metrics,
    stream_bandwidths,
)
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.components.moe_variants import LatentMoEMLP

# (num_experts, top_k, dim, expert_hidden) from the MoE presets: MoE-1B-A120M,
# MoE-2B-A370M, MoE-3B-A500M, MoE-8B-A1B.
SHAPES = [(64, 3, 640, 448), (48, 4, 1152, 512), (64, 6, 1280, 448), (96, 8, 1536, 576)]

# Each entry is (label, class, kwargs). Everything else is held equal -- same
# expert geometry, same expert count, same top_k -- so a difference in a row is
# the formulation's, not the shape's.
FORMULATIONS = [
    ("DeepSeek-V3", MoEMLP, dict(score_func="sigmoid", num_shared=1)),
    ("DeepSeek-V4", MoEMLP, dict(score_func="sqrtsoftplus", num_shared=1)),
    ("MiMo-V2.5", MoEMLP, dict(score_func="sigmoid", num_shared=0)),
    (
        "OLMoE/Switch",
        MoEMLP,
        dict(score_func="softmax", aux_loss_weight=0.01, num_shared=1),
    ),
    ("ReMoE", MoEMLP, dict(router="relu", num_shared=1)),
    ("Kimi-K3 latent", LatentMoEMLP, dict(alpha=2, num_shared=1)),
]


def build(
    cls, kwargs: dict, experts: int, top_k: int, dim: int, hidden: int, seed: int
):
    # Reseeded per build so every formulation at a shape gets the *same* gate
    # matrix. Without this the expert load distribution differs between rows, the
    # grouped GEMM's grid is sized from the largest expert, and the resulting ~2%
    # swing is the same size as the differences being measured. It also makes
    # V3-vs-V4 a controlled comparison: both gates are strictly increasing, so on
    # identical logits they select identical experts and only the weights differ.
    torch.manual_seed(seed)
    module = cls(
        dim, num_experts=experts, top_k=top_k, hidden=hidden, multiple_of=1, **kwargs
    )
    return module.cuda().to(torch.bfloat16).train()


def _geometry(module) -> dict:
    """Shapes the cost model needs, read off the module's own parameters.

    Derived from parameter tensors rather than from the config, so a formulation
    that changes the expert input width (latent MoE) is costed correctly without
    the cost model knowing which class it is looking at.
    """
    inner = getattr(module, "inner", module)
    per_expert = inner.w_in[0].numel() + inner.w_out[0].numel()
    expert_ids = {id(inner.w_in), id(inner.w_out), id(inner.router.weight)}
    dense = sum(p.numel() for p in module.parameters() if id(p) not in expert_ids)
    return {
        "per_expert": per_expert,
        "router_params": inner.router.weight.numel(),
        "dense_params": dense,
        "expert_dim": inner.dim,
        "expert_hidden": inner.hidden,
        "params": sum(p.numel() for p in module.parameters()),
    }


def _cost(geom: dict, tokens: int, dim: int, slots: int, width: int = 2):
    """``(flops, bytes)`` for one forward.

    FLOPs is exact: every matmul is ``2 * rows * weight_elements``, and the only
    thing that varies between formulations is how many rows reach the expert bank
    (``slots``) and how wide the bank's input is.

    Bytes is a documented floor. It charges every weight once and every
    activation hand-off between the five stages (gather, GEMM, swiglu, GEMM,
    combine) once in each direction. It does *not* charge the grouped GEMM's
    weight-tile re-reads across M-blocks, nor credit L2 hits on the gather, so a
    row above 100% of STREAM is reading out of cache rather than beating DRAM.
    """
    rows = tokens * slots
    flops = 2 * (
        rows * geom["per_expert"]
        + tokens * geom["router_params"]
        + tokens * geom["dense_params"]
    )
    d_e, h = geom["expert_dim"], geom["expert_hidden"]
    moved = geom["params"] + 2 * tokens * dim + rows * (4 * d_e + 6 * h)
    return float(flops), float(moved * width)


def _steps(module, x: torch.Tensor):
    """``(forward, forward+backward)`` closures.

    The backward includes ``router_losses()``: an auxiliary load-balance loss is
    part of what a formulation costs, and OLMoE's in particular backpropagates
    through the full ``(T, E)`` score matrix that its gate had to keep alive.
    """
    params = [p for p in module.parameters() if p.requires_grad]

    def forward():
        with torch.no_grad():
            return module(x)

    def forward_backward():
        x.grad = None
        for p in params:
            p.grad = None
        out = module(x)
        loss = out.float().sum()
        extra = module.router_losses()
        if extra is not None:
            loss = loss + extra.float()
        loss.backward()

    return forward, forward_backward


def _dispatch_width(module, x: torch.Tensor) -> int:
    """Slots actually routed, which is not ``top_k`` for a ReLU router.

    ``x`` at full width on both classes: a latent block routes from the residual,
    so its router's input dim is ``dim`` even though its experts see ``dim/alpha``.
    """
    router = getattr(module, "inner", module).router
    with torch.no_grad():
        idx, _, _ = router.route(x.detach().reshape(-1, x.shape[-1]))
    return int(idx.shape[-1])


def _router_path(module) -> str:
    router = getattr(module, "inner", module).router
    fused = getattr(router, "use_fused", None)
    if fused is None:
        return "eager (no fused path)"
    return "fused" if fused else "eager"


def rows_for_shape(
    experts: int, top_k: int, dim: int, hidden: int, tokens: int, args
) -> list[dict]:
    out = []
    # One input for every formulation at this shape, so the routing decisions are
    # driven by the same tokens and not by a different draw per row.
    torch.manual_seed(args.seed + 1)
    x = torch.randn(
        tokens, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    for label, cls, kwargs in FORMULATIONS:
        module = build(cls, kwargs, experts, top_k, dim, hidden, args.seed)
        slots = _dispatch_width(module, x)
        geom = _geometry(module)
        flops, moved = _cost(geom, tokens, dim, slots)
        forward, forward_backward = _steps(module, x)

        common = {
            "formulation": label,
            "num_experts": experts,
            "top_k": top_k,
            "dim": dim,
            "hidden": hidden,
            "tokens": tokens,
            "slots": slots,
            "router_path": _router_path(module),
            "params": geom["params"],
            # Computed from the geometry rather than `MoEMLP.active_parameters()`:
            # on a latent block that method belongs to the *inner* MoE and so
            # omits the two projections and the outer shared expert, which every
            # token also passes through. It also uses `slots`, so ReMoE's padded
            # dispatch is charged the parameters it actually activates.
            "active_params": (
                geom["per_expert"] * slots
                + geom["router_params"]
                + geom["dense_params"]
            ),
        }

        inner = getattr(module, "inner", module)
        router_x = x.detach().reshape(-1, dim)
        router_geom = {
            "per_expert": 0,
            "router_params": geom["router_params"],
            "dense_params": 0,
            "expert_dim": 0,
            "expert_hidden": 0,
            "params": geom["router_params"],
        }
        rf, rb = _cost(router_geom, tokens, 0, 0)
        # The router's own bytes are the score matrix it materializes, which is
        # the whole point of the fused path: eager writes (T, E) fp32 plus an
        # fp32 copy of x, fused writes neither.
        rb += 4.0 * tokens * (experts + dim)

        def route(r=inner.router, t=router_x):
            return r.route(t)

        # `graph_us` (device) and `cpu_us` (host issue) ride beside module_metrics'
        # wall-clock `ms` because only the first two are comparable across
        # formulations here: a router is one 15-34 us kernel behind ~155 us of
        # Python, so wall time reports the Python, bimodally depending on how warm
        # the Triton launcher is -- a measured 5x spread across rows doing identical
        # work. The layer stages are affected too at d=640, where host issue is
        # 833 us against 405 us of device work.
        out.append(
            {
                **common,
                "stage": "router",
                **module_metrics(
                    route, rf, rb, warmup=args.warmup, iters=args.iters, ceiling="tf32"
                ),
                "graph_us": graph_ms(route) * 1e3,
                "cpu_us": cpu_enqueue_ms(route) * 1e3,
            }
        )
        out.append(
            {
                **common,
                "stage": "layer_fwd",
                **module_metrics(
                    forward, flops, moved, warmup=args.warmup, iters=args.iters
                ),
                "graph_us": graph_ms(forward) * 1e3,
                "cpu_us": cpu_enqueue_ms(forward) * 1e3,
            }
        )
        out.append(
            {
                **common,
                "stage": "layer_fwd_bwd",
                **module_metrics(
                    forward_backward,
                    3.0 * flops,
                    3.0 * moved,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                # Deliberately not graph-captured, though it can be: a capture
                # per row holds its own memory pool, and 24 of them turned a
                # 2.49 GiB peak into 3.87 GiB and DeepSeek-V3 at d=1536 from
                # 9.8 ms into 19.9 ms -- the whole table degraded through
                # allocator pressure. `layer_fwd` carries the comparable device
                # time; this stage's `ms` is the wall clock a training step pays.
                "graph_us": float("nan"),
                "cpu_us": cpu_enqueue_ms(forward_backward) * 1e3,
            }
        )
        del module
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dir", default="out/bench/moe/moe_formulations")
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

    rows = []
    for experts, top_k, dim, hidden in SHAPES:
        print(f"\nE={experts} k={top_k} dim={dim} hidden={hidden} T={args.tokens}")
        shape_rows = rows_for_shape(experts, top_k, dim, hidden, args.tokens, args)
        for row in shape_rows:
            tag = f"{row['formulation']} [{row['router_path']}, {row['slots']} slots]"
            line = format_metrics(f"{row['stage']:>13s} {tag}", row, width=52)
            print(f"{line}  gpu {row['graph_us']:7.1f} us  cpu {row['cpu_us']:7.1f} us")
        rows.extend(shape_rows)

    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, "moe_formulations.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "device": device_name(),
                "stream_bandwidths": stream,
                "peak_bandwidth_gbps": cached_peak_bandwidth(),
                "config": {
                    "tokens": args.tokens,
                    "shapes": SHAPES,
                    "seed": args.seed,
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
