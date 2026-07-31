"""Measure the MoE router + dispatch path: eager vs torch.compile vs MSLK vs fused.

Writes ``moe_router.json``; ``moe_router_plot.py`` draws it. Split for the same
reason as ``kernels.py`` / ``kernels_plot.py``: a new implementation adds a
measurement, a rejected figure only changes the drawing.

Two measurements per row, because this path is the one place in the repo where
they disagree:

* ``graph_us`` -- a CUDA-graph replay, so pure device time with every launch and
  dispatch cost removed.
* ``cpu_us`` -- host time to *issue* the work, no sync inside the loop.

The eager router's two numbers are the same (~320 us each), which is the whole
finding: it is bound by Python issuing eight ops, not by the GPU running them,
and that is why its cost does not move when E goes 32 -> 96 or D goes 640 ->
1536. A graph replay is the only honest way to see what the device actually does.

Run::

    CUDA_VISIBLE_DEVICES=<n> .venv/bin/python scripts/bench/moe/moe_router.py
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from kohakuwullm.bench.core.timing import (
    cached_peak_bandwidth,
    device_name,
    stream_bandwidths,
    ulp_error,
)
from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.kernels.moe.moe_dispatch import (
    combine_routed,
    combine_routed_reference,
    expert_sort,
    expert_sort_reference,
)
from kohakuwullm.models.components.moe import MoEMLP

# (num_experts, top_k, dim) drawn from the MoE presets.
SHAPES = [(64, 3, 640), (48, 4, 1152), (64, 6, 1280), (96, 8, 1536)]
# Each preset's real per-expert hidden width, for the rows whose cost depends on
# it. The stage rows deliberately use a fixed narrow `--hidden` instead, so that
# the dispatch overhead they isolate is not buried under expert GEMM time.
PRESET_HIDDEN = {640: 448, 1152: 512, 1280: 448, 1536: 576}


def graph_us(fn, iters: int = 50) -> float:
    """Median device time of ``fn`` in us, captured as a CUDA graph.

    Returns ``nan`` when capture fails, which is itself a result: the eager
    router could not be captured at all while it called ``torch.bincount``,
    because that op sizes its output from ``input.max()`` and so copies to host.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1e3
    except Exception:
        return float("nan")


def cpu_us(fn, iters: int = 30) -> float:
    """Host time to issue the work, with no device sync inside the loop."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1e6 / iters
    torch.cuda.synchronize()
    return elapsed


def peak_gib(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def build(dim: int, experts: int, top_k: int, fused: bool, hidden: int = 256) -> MoEMLP:
    moe = MoEMLP(
        dim, num_experts=experts, top_k=top_k, hidden=hidden, num_shared=0, fused=fused
    )
    moe = moe.cuda().to(torch.bfloat16)
    moe.train()
    if moe.router.use_fused is not fused:
        raise RuntimeError(f"router fused={moe.router.use_fused}, wanted {fused}")
    return moe


def _row(kernel, impl, experts, top_k, dim, tokens, fn, **extra) -> dict:
    row = {
        "kernel": kernel,
        "impl": impl,
        "num_experts": experts,
        "top_k": top_k,
        "dim": dim,
        "tokens": tokens,
        "graph_us": graph_us(fn),
        "cpu_us": cpu_us(fn),
        "peak_gib": peak_gib(fn),
        "idx_exact": True,
        "ulp": 0.0,
    }
    row.update(extra)
    return row


def router_rows(tokens: int) -> list[dict]:
    """Router-only: the three implementations, with exactness against eager."""
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        eager = build(dim, experts, top_k, fused=False)
        fused = build(dim, experts, top_k, fused=True)
        fused.router.load_state_dict(eager.router.state_dict())
        compiled = torch.compile(build(dim, experts, top_k, fused=False).router)
        compiled(x)

        # The router makes a discrete choice, so the only acceptable comparison
        # is index-for-index identity, not a tolerance on the weights.
        ref_idx, ref_w, _ = eager.router.route(x)
        got_idx, got_w, _ = fused.router.route(x)
        exact = bool((ref_idx.int() == got_idx.int()).all().item())
        ulp = ulp_error(got_w.float(), ref_w.float(), torch.bfloat16, mode="rms")

        for name, fn, checked in (
            ("eager", lambda m=eager: m.router.route(x), False),
            ("torch.compile", lambda c=compiled: c(x), False),
            ("fused triton", lambda m=fused: m.router.route(x), True),
        ):
            rows.append(
                _row(
                    "router",
                    name,
                    experts,
                    top_k,
                    dim,
                    tokens,
                    fn,
                    idx_exact=exact if checked else True,
                    ulp=ulp if checked else 0.0,
                )
            )
    return rows


def stage_rows(tokens: int) -> list[dict]:
    """The two dispatch stages we replaced, each against its eager form."""
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        moe = build(dim, experts, top_k, fused=True)
        idx, weight, counts = moe.router.route(x)
        flat_expert = idx.reshape(-1)
        _, order, token_of, _ = expert_sort(flat_expert, counts, top_k, experts)
        expert_out = torch.randn(
            tokens * top_k, dim, device="cuda", dtype=torch.bfloat16
        )
        flat_weight = weight.reshape(-1)

        def eager_sort():
            offsets = torch.zeros(experts + 1, dtype=torch.int32, device="cuda")
            offsets[1:] = counts.cumsum(0).to(torch.int32)
            order_ = flat_expert.argsort()
            return offsets, order_, torch.div(order_, top_k, rounding_mode="floor")

        def fused_sort():
            return expert_sort(flat_expert, counts, top_k, experts)

        def eager_combine():
            return combine_routed_reference(
                expert_out, flat_weight, order, token_of, tokens
            )

        def fused_combine():
            return combine_routed(expert_out, flat_weight, order, token_of, tokens)

        ulp = ulp_error(
            fused_combine().float(),
            eager_combine().float(),
            torch.bfloat16,
            mode="rms",
        )
        for stage, name, fn in (
            ("expert-sort", "eager (argsort+cumsum+div)", eager_sort),
            ("expert-sort", "fused (counting sort)", fused_sort),
            ("combine", "eager (scale + index_add)", eager_combine),
            ("combine", "fused (scale+scatter)", fused_combine),
        ):
            rows.append(
                _row(
                    stage,
                    name,
                    experts,
                    top_k,
                    dim,
                    tokens,
                    fn,
                    ulp=ulp if stage == "combine" else 0.0,
                )
            )
        rows.extend(
            _mslk_rows(
                x,
                expert_out,
                flat_weight,
                order,
                token_of,
                experts,
                top_k,
                dim,
                tokens,
            )
        )
    return rows


def _mslk_rows(
    x, expert_out, flat_weight, order, token_of, experts, top_k, dim, tokens
) -> list[dict]:
    """MSLK's equivalents, and *why* each one does or does not apply here."""
    try:
        import mslk.moe as mslk
    except ImportError:
        return []

    rows = []

    def mslk_combine():
        out = torch.zeros_like(x)
        scaled = (
            expert_out * flat_weight.index_select(0, order.long()).reshape(-1, 1)
        ).contiguous()
        mslk.scatter_add_dense_tokens(out, scaled, token_of)
        return out

    note = ""
    try:
        mslk_combine()
    except Exception as exc:
        note = (str(exc).splitlines() or [type(exc).__name__])[0][:150]
    if note:
        rows.append(
            {
                "kernel": "combine",
                "impl": "mslk scatter_add",
                "num_experts": experts,
                "top_k": top_k,
                "dim": dim,
                "tokens": tokens,
                "graph_us": float("nan"),
                "cpu_us": float("nan"),
                "peak_gib": float("nan"),
                "idx_exact": True,
                "ulp": 0.0,
                "note": note,
            }
        )
    else:
        rows.append(
            _row(
                "combine", "mslk scatter_add", experts, top_k, dim, tokens, mslk_combine
            )
        )

    # index_shuffling would replace top-k + sort + histogram in one op, so its
    # shape support decides whether MSLK can serve this repo's presets at all.
    scores = torch.rand(tokens, experts, device="cuda")
    try:
        mslk.index_shuffling(scores, None, None, None, top_k)
        supported, why = True, ""
    except Exception as exc:
        supported = False
        why = (str(exc).splitlines() or [type(exc).__name__])[0][:150]
    rows.append(
        {
            "kernel": "index_shuffling",
            "impl": "mslk index_shuffling",
            "num_experts": experts,
            "top_k": top_k,
            "dim": dim,
            "tokens": tokens,
            "supported": supported,
            "note": why,
        }
    )
    return rows


def layer_rows(tokens: int, hidden: int) -> list[dict]:
    """Gather and the two expert GEMMs, so the non-GEMM *share* is computable.

    Without the GEMMs in the same run there is no denominator, and "the router got
    faster" says nothing about whether the layer did.
    """
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        moe = MoEMLP(
            dim,
            num_experts=experts,
            top_k=top_k,
            hidden=hidden,
            num_shared=0,
            fused=True,
        )
        moe = moe.cuda().to(torch.bfloat16).train()
        idx, weight, counts = moe.router.route(x)
        offsets, order, token_of, _ = expert_sort(
            idx.reshape(-1), counts, top_k, experts
        )

        def gather():
            return x.index_select(0, token_of).contiguous()

        sorted_rows = gather()

        def experts_fn():
            return moe._expert_compute(sorted_rows, offsets)

        rows.append(_row("gather", "index_select", experts, top_k, dim, tokens, gather))
        rows.append(
            _row(
                "expert_gemm",
                "triton grouped",
                experts,
                top_k,
                dim,
                tokens,
                experts_fn,
                hidden=hidden,
            )
        )
    return rows


def full_rows(tokens: int) -> list[dict]:
    """The whole non-GEMM path, which is what the layer actually pays."""
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        for name, fused in (("eager", False), ("fused triton", True)):
            moe = build(dim, experts, top_k, fused=fused)
            sort = expert_sort if fused else expert_sort_reference
            combine = combine_routed if fused else combine_routed_reference

            def fn(m=moe, s=sort, c=combine):
                idx, weight, counts = m.router.route(x)
                _, order, token_of, _ = s(idx.reshape(-1), counts, top_k, experts)
                gathered = x.index_select(0, token_of).contiguous()
                return c(gathered, weight.reshape(-1), order, token_of, tokens)

            rows.append(_row("non_gemm_path", name, experts, top_k, dim, tokens, fn))
    return rows


def swiglu_rows(tokens: int) -> list[dict]:
    """The activation between the two expert GEMMs, three ways.

    It reads two strided chunks of one ``(M, 2H)`` tensor, which is why the
    repo's own ``swiglu_mul`` is the *slowest* option here: it contiguifies its
    inputs, and for chunks that is a full copy of both halves.
    """
    rows = []
    for experts, top_k, dim in SHAPES:
        m = tokens * top_k
        hidden = PRESET_HIDDEN[dim]
        h = torch.randn(m, 2 * hidden, device="cuda", dtype=torch.bfloat16)

        def eager(h=h):
            gate, value = h.chunk(2, dim=-1)
            return F.silu(gate) * value

        compiled = torch.compile(eager)
        compiled()

        def fused(h=h):
            gate, value = h.chunk(2, dim=-1)
            return swiglu_mul(gate, value)

        for name, fn in (
            ("eager silu*mul", eager),
            ("torch.compile", compiled),
            ("swiglu_mul (copies chunks)", fused),
        ):
            rows.append(
                _row("swiglu", name, experts, top_k, dim, tokens, fn, hidden=hidden)
            )
        del h
        torch.cuda.empty_cache()
    return rows


def layer_fusion_rows(tokens: int) -> list[dict]:
    """The whole layer, wall / device / host time, eager and compiled.

    Three numbers because they answer different questions. ``graph_us`` was ``nan``
    for every row of this table until the grouped GEMM stopped sizing its grid
    from ``counts.max().item()``: a whole MoE layer could not be captured at all.
    ``cpu_us`` against ``graph_us`` says whether the host can run ahead, which is
    what the sync actually cost -- not wall time on an isolated layer, where the
    stall overlaps device work that was already issued.
    """
    rows = []
    for experts, top_k, dim in SHAPES:
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        hidden = PRESET_HIDDEN[dim]
        moe = build(dim, experts, top_k, fused=True, hidden=hidden)
        compiled = torch.compile(moe)
        compiled(x)
        for name, fn in (
            ("eager", lambda m=moe: m(x)),
            ("torch.compile", lambda c=compiled: c(x)),
        ):
            rows.append(
                _row(
                    "whole_layer", name, experts, top_k, dim, tokens, fn, hidden=hidden
                )
            )
        del moe, compiled
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dir", default="out/bench/moe/moe_router")
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

    rows = (
        router_rows(args.tokens)
        + stage_rows(args.tokens)
        + layer_rows(args.tokens, args.hidden)
        + full_rows(args.tokens)
        + swiglu_rows(args.tokens)
        + layer_fusion_rows(args.tokens)
    )
    for row in rows:
        if "graph_us" not in row:
            continue
        print(
            f"  {row['kernel']:>14s} {row['impl']:>28s} "
            f"E={row['num_experts']:3d} k={row['top_k']} D={row['dim']:5d}: "
            f"gpu {row['graph_us']:8.1f} us  cpu {row['cpu_us']:8.1f} us"
        )

    os.makedirs(args.dir, exist_ok=True)
    path = os.path.join(args.dir, "moe_router.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "device": device_name(),
                "stream_bandwidths": stream,
                "peak_bandwidth_gbps": cached_peak_bandwidth(),
                "config": {"tokens": args.tokens, "shapes": SHAPES},
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
