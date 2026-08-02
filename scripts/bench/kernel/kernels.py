"""Per-module kernels: which implementation should the model actually use?

Every module here has more than one implementation and the choice is decided by
measurement. Each row carries the same four numbers -- latency, effective
FLOP/s, effective GB/s, peak memory -- and each module is scored against the
ceiling that actually binds it:

* **RMSNorm / SwiGLU / RoPE** are bandwidth-bound: the deciding metric is the
  fraction of *measured* DRAM bandwidth reached, never TFLOP/s. Effective GB/s
  counts the bytes the op logically needs, so an implementation that
  materializes temporaries (eager RoPE concatenates twice) falls below the
  ceiling by moving data it never needed -- exactly the cost worth seeing.
* **MoE expert GEMM / SwiGLU MLP** are compute-bound: the fraction of MAMF
  (270 TFLOP/s, bf16 in / fp32 accumulate). An fp32-operand GEMM is not on that
  ceiling, so it is scored separately -- and *not* against fp32 FMA either,
  because Triton lowers an fp32 ``tl.dot`` to TF32 mma by default. The measured
  TF32 rate goes in the JSON for the figure; see :func:`measure_tf32_peak`.
* **The LM head** is bound by neither -- at vocab 65536 the binding constraint
  is memory capacity, so the deciding column is peak GiB.

MoE shapes come from ``models/presets.py``, so every measured point is a model
we intend to train. Routed rows are ``tokens * top_k``: a benchmark that sends
each token to one expert measures a fraction of the layer's work. Both expert
matrices are measured -- ``w_in`` is ``(2*hidden) x dim``, ``w_out`` is
``dim x hidden``, and they behave differently.

This half only measures; ``kernels_plot.py`` draws the JSON it writes. The
printed DRAM ceiling doubles as a contention check: on an idle RTX 5090 it
reads ~1780 GB/s, and anything materially below that means the card is shared
and every row in the run is worthless.

Usage:
    .venv/bin/python scripts/bench/kernel/kernels.py --out out/bench/kernel/kernels
    .venv/bin/python scripts/bench/kernel/kernels_plot.py --dir out/bench/kernel/kernels
"""

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

from kohakuwullm.bench import (
    bench_ms,
    device_name,
    format_metrics,
    module_metrics,
    rel_error,
    stream_bandwidths,
    theoretical_bandwidth_gbps,
    ulp_error,
)
from kohakuwullm.bench.core.timing import TENSOR_MAMF_TFLOPS
from kohakuwullm.kernels.attention.rope import _reference_rope, apply_rope
from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm as triton_rms_norm
from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul
from kohakuwullm.kernels.moe.grouped_gemm import grouped_gemm, grouped_gemm_reference
from kohakuwullm.models.components.mlp import SwiGLU, SwiGLUTriton, resolve_hidden
from kohakuwullm.models.components.moe import TopKRouter
from kohakuwullm.models.presets import get_preset

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}

# Presets whose (experts, top_k) pairs the MoE panels sweep. Read through
# get_preset rather than copied, so the benchmark cannot drift from the models.
MOE_PRESETS = ("MoE-1B-A280M", "MoE-2B-A370M", "MoE-3B-A500M", "MoE-8B-A1B")

ROUTER_NOTE = (
    "router: sigmoid gating, top-k of E routed experts + 1 shared expert always on, "
    "aux-loss-free balancing bias (steers selection only, never the output weight)"
)
ROUTER_VARIANTS = {
    "sigmoid (ours)": dict(score_func="sigmoid"),
    "softmax": dict(score_func="softmax"),
    "sigmoid + group-limited": dict(score_func="sigmoid", n_groups=8, topk_groups=4),
    "sigmoid, no balance bias": dict(score_func="sigmoid", bias_update_rate=0.0),
}
DISPATCH = "dispatch (sort+gather+scatter)"
CEIL = "#D55E00"

# An fp64 reference is 4x the fp16 footprint, so elementwise checks build it on
# a bounded slice. Those kernels are row-independent, so a slice measures the
# same error the full tensor would; only the timing needs the full size.
PRECISION_ROWS = 4096

# One compiled artifact per (callable, dtype), reused across shapes: rebuilding
# inside the sweep would charge a fresh compile to every point, while sharing one
# artifact across dtypes doubles its guard variants toward dynamo's recompile
# limit -- past which it falls back to eager without failing.
_COMPILED: dict[str, Callable] = {}


def _compiled(key: str, fn: Callable) -> Callable:
    if key not in _COMPILED:
        _COMPILED[key] = torch.compile(fn, dynamic=False)
    return _COMPILED[key]


def measure_tf32_peak() -> float:
    """Measured TF32 tensor-core rate: the real ceiling for an fp32 operand.

    Triton lowers an fp32 ``tl.dot`` to TF32 mma unless asked for ``"ieee"``, so
    the fp32 path never touches the vector cores. Scored against
    ``VECTOR_PEAK_TFLOPS`` (an fp32-FMA figure) it reads over 100% of a ceiling
    it never used -- measured here instead. True fp32 FMA on this card runs at
    roughly 60 TF/s, so the two are not interchangeable.
    """
    m = n = k = 4096
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(n, k, device="cuda")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    ms = bench_ms(lambda: a @ b.T, warmup=5, iters=20)
    torch.backends.cuda.matmul.allow_tf32 = previous
    a = b = None
    torch.cuda.empty_cache()
    return 2 * m * n * k / ms * 1e-9


def _fwd_bwd(fn: Callable, *inputs: torch.Tensor) -> Callable:
    def run():
        fn().sum().backward()
        for tensor in inputs:
            tensor.grad = None

    return run


def _measure(
    label: str,
    fn: Callable,
    *,
    flops: float,
    bytes_moved: float,
    tensor_cores: bool,
    peak_bw: float,
    warmup: int = 5,
    iters: int = 20,
    **fields: Any,
) -> dict | None:
    """One benchmark row: the four metrics plus whatever identifies it."""
    try:
        metrics = module_metrics(
            fn,
            flops=flops,
            bytes_moved=bytes_moved,
            warmup=warmup,
            iters=iters,
            tensor_cores=tensor_cores,
            peak_bandwidth_gbps=peak_bw,
        )
    except Exception as err:  # noqa: BLE001
        print(f"  skip {label}: {type(err).__name__}: {str(err)[:110]}", flush=True)
        torch.cuda.empty_cache()
        return None
    print(format_metrics(label, metrics, width=42), flush=True)
    return {**fields, **metrics}


def _both_phases(
    label: str,
    fn: Callable,
    grads: tuple[torch.Tensor, ...],
    *,
    flops: float,
    bytes_moved: float,
    tensor_cores: bool,
    peak_bw: float,
    fb_flops: float | None = None,
    fb_bytes: float | None = None,
    **tags: Any,
) -> list[dict]:
    """Forward and forward+backward rows for one implementation.

    A GEMM's backward is ``dx`` and ``dW``, each the forward's shape, so the 3x
    default is exact; elementwise kernels pass their own counts instead.
    """
    common = dict(tensor_cores=tensor_cores, peak_bw=peak_bw, **tags)
    fwd = _measure(
        f"{label} fwd", fn, flops=flops, bytes_moved=bytes_moved, phase="fwd", **common
    )
    back = _measure(
        f"{label} fwd+bwd",
        _fwd_bwd(fn, *grads),
        flops=3 * flops if fb_flops is None else fb_flops,
        bytes_moved=3 * bytes_moved if fb_bytes is None else fb_bytes,
        warmup=3,
        iters=10,
        phase="fwd+bwd",
        **common,
    )
    return [r for r in (fwd, back) if r is not None]


def _precision(fn, ref_fn, dtype, mode: str, rows: int = PRECISION_ROWS) -> dict:
    """Error against an fp64 reference, with ``fn`` called exactly as it is timed.

    Not wrapped in ``no_grad``: ``torch.compile`` guards on grad mode, so calling
    a compiled artifact both ways doubles its recompiles, and dynamo silently
    falls back to eager once the limit is hit -- which would leave the "compile"
    rows measuring eager while still labelled compile.
    """
    got = fn()[:rows].detach()
    with torch.no_grad():
        ref = ref_fn(rows)
    out = dict(rel_err=rel_error(got, ref), ulp=ulp_error(got, ref, dtype, mode))
    del got, ref
    return out


# --------------------------------------------------------------------------
# bandwidth-bound modules
# --------------------------------------------------------------------------


def _build_rmsnorm(args, device, dtype, tokens):
    dim = args.dim
    x = torch.randn(tokens, dim, device=device, dtype=dtype, requires_grad=True)
    w = torch.ones(dim, device=device, dtype=dtype, requires_grad=True)
    compiled = _compiled(
        f"rmsnorm-{dtype}", lambda a, b: F.rms_norm(a, (a.shape[-1],), b, 1e-6)
    )
    impls = {
        "aten (fused)": lambda: F.rms_norm(x, (dim,), w, 1e-6),
        "triton": lambda: triton_rms_norm(x, w, 1e-6),
        "compile": lambda: compiled(x, w),
    }

    def ref_fn(rows):
        return F.rms_norm(x[:rows].double(), (dim,), w.double(), 1e-6)

    elem = torch.finfo(dtype).bits // 8
    counts = dict(
        # x*x and the accumulate, then scale by rstd and by the weight.
        fwd_flops=4 * tokens * dim,
        fb_flops=12 * tokens * dim,
        # fwd reads x, writes out; bwd reads dout and x, writes dx. The cached
        # rstd is one float per row and the weight is one row -- both rounding
        # errors at these widths.
        fwd_bytes=2 * tokens * dim * elem,
        fb_bytes=5 * tokens * dim * elem,
        ulp_mode="elementwise",
        shape=f"D={dim}",
    )
    return (x, w), impls, ref_fn, counts


def _build_swiglu(args, device, dtype, tokens):
    hidden = args.hidden
    g = torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(tokens, hidden, device=device, dtype=dtype, requires_grad=True)
    compiled = _compiled(f"swiglu-{dtype}", lambda a, b: F.silu(a) * b)
    impls = {
        "eager (2 kernels)": lambda: F.silu(g) * v,
        "triton": lambda: swiglu_mul(g, v),
        "compile": lambda: compiled(g, v),
    }

    def ref_fn(rows):
        return F.silu(g[:rows].double()) * v[:rows].double()

    elem = torch.finfo(dtype).bits // 8
    counts = dict(
        # sigmoid ~4 flops, then two multiplies.
        fwd_flops=6 * tokens * hidden,
        fb_flops=16 * tokens * hidden,
        # fwd reads gate and value, writes out; bwd reads dout/gate/value and
        # writes both gradients.
        fwd_bytes=3 * tokens * hidden * elem,
        fb_bytes=8 * tokens * hidden * elem,
        ulp_mode="elementwise",
        shape=f"H={hidden}",
    )
    return (g, v), impls, ref_fn, counts


def _build_rope(args, device, dtype, tokens):
    head_dim = args.head_dim
    heads = args.dim // head_dim
    x = torch.randn(
        tokens, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    cos = torch.randn(tokens, head_dim // 2, device=device, dtype=dtype).cos()
    sin = torch.randn(tokens, head_dim // 2, device=device, dtype=dtype).sin()
    compiled = _compiled(f"rope-{dtype}", _reference_rope)
    impls = {
        "eager (rotate_half)": lambda: _reference_rope(x, cos, sin, head_dim),
        "triton": lambda: apply_rope(x, cos, sin, head_dim),
        "compile": lambda: compiled(x, cos, sin, head_dim),
    }

    def ref_fn(rows):
        return _reference_rope(
            x[:rows].double(), cos[:rows].double(), sin[:rows].double(), head_dim
        )

    elem = torch.finfo(dtype).bits // 8
    elems = tokens * heads * head_dim
    table = tokens * head_dim * elem
    counts = dict(
        # two multiplies and an add per output element.
        fwd_flops=3 * elems,
        fb_flops=6 * elems,
        fwd_bytes=2 * elems * elem + table,
        fb_bytes=4 * elems * elem + 2 * table,
        # An output is x1*cos - x2*sin: a two-term reduction, so a near-zero
        # element is cancellation between larger terms and deserves the RMS
        # scale, not its own magnitude.
        ulp_mode="rms",
        shape=f"{heads}h x {head_dim}",
    )
    return (x,), impls, ref_fn, counts


POINTWISE = (
    ("rmsnorm", _build_rmsnorm),
    ("swiglu", _build_swiglu),
    ("rope", _build_rope),
)


def bench_pointwise(args, device, peak_bw):
    """Norm / gate / RoPE, scored against measured DRAM bandwidth."""
    rows = []
    for kernel, build in POINTWISE:
        for dname, dtype in DTYPES.items():
            for tokens in args.token_counts:
                inputs, impls, ref_fn, counts = build(args, device, dtype, tokens)
                for impl, fn in impls.items():
                    tag = dict(
                        kernel=kernel,
                        impl=impl,
                        dtype=dname,
                        tokens=tokens,
                        shape=counts["shape"],
                    )
                    try:
                        acc = _precision(fn, ref_fn, dtype, counts["ulp_mode"])
                    except Exception as err:  # noqa: BLE001
                        print(f"  skip {kernel} {impl} {dname}: {err}", flush=True)
                        continue
                    rows += _both_phases(
                        f"{kernel} {impl} {dname} T={tokens}",
                        fn,
                        inputs,
                        flops=counts["fwd_flops"],
                        bytes_moved=counts["fwd_bytes"],
                        fb_flops=counts["fb_flops"],
                        fb_bytes=counts["fb_bytes"],
                        tensor_cores=False,
                        peak_bw=peak_bw,
                        **tag,
                        **acc,
                    )
                del inputs, impls, ref_fn
                torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# MoE: expert GEMMs and the router in front of them
# --------------------------------------------------------------------------


def moe_shapes(names):
    """``(label, experts, top_k, dim, hidden)`` per preset, read from presets.py."""
    out = []
    for name in names:
        cfg = get_preset(name)
        hidden = resolve_hidden(cfg.dim, cfg.moe_ratio, cfg.moe_hidden, 128, glu=True)
        out.append((name, cfg.moe_num_experts, cfg.moe_top_k, cfg.dim, hidden))
    return out


def _routing_offsets(experts: int, routed: int, seed: int, device):
    """Skewed per-expert row counts; a uniform histogram is not what runs.

    The aux-loss-free bias keeps the load near balanced but never exactly there,
    and a skewed histogram costs the grouped GEMM its tail: the last tile of each
    expert is partially masked, and how much is masked depends on the skew.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(experts, generator=generator) + 0.4
    counts = (weights / weights.sum() * routed).long()
    counts[-1] += routed - counts.sum()
    offsets = torch.zeros(experts + 1, dtype=torch.int32)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    return offsets.to(device), offsets


def _gemm_case(device, peak_bw, preset, experts, top_k, rows_m, k, n, tag):
    """Every implementation of one expert GEMM shape, precision then speed.

    Precision runs first so the fp64 reference can be freed before timing:
    ``bench_peak_memory`` reports *total* allocated bytes, and a 600 MB
    reference left resident would inflate every peak-memory column equally.
    """
    offsets, offsets_cpu = _routing_offsets(experts, rows_m, 0, device)
    xm = torch.randn(rows_m, k, device=device) * 0.5
    wm = torch.randn(experts, n, k, device=device) * 0.05
    ref = grouped_gemm_reference(xm.double(), wm.double(), offsets_cpu)

    tensors = {}
    for dname, dtype in DTYPES.items():
        tensors[dname] = (
            xm.to(dtype).requires_grad_(),
            wm.to(dtype).requires_grad_(),
            wm[0].to(dtype).clone().requires_grad_(),
        )
    del xm, wm
    torch.cuda.empty_cache()

    def impls_for(dname):
        """``impl -> (fn, tensor_cores, grad tensors, matrices read, comparable)``.

        ``matrices read`` is the byte denominator: the grouped paths stream all
        ``E`` expert matrices, the single-GEMM ceiling streams one, and charging
        it for ``E`` would credit it with bandwidth it never used.

        ``comparable`` is False for that ceiling because it runs every row
        through *one* expert: the same FLOPs on different data, so it bounds
        throughput but its error against the routed reference means nothing.
        """
        x, w, wflat = tensors[dname]

        def triton(compute_dtype=None):
            return lambda: grouped_gemm(x, w, offsets, compute_dtype=compute_dtype)

        return {
            "loop (E x cuBLAS mm)": (
                lambda: grouped_gemm_reference(x, w, offsets_cpu),
                True,
                (x, w),
                experts,
                True,
            ),
            "triton grouped": (triton(), True, (x, w), experts, True),
            # tensor_cores=False keeps this off the bf16 MAMF denominator.
            # Triton lowers an fp32 tl.dot to TF32 mma, so its real ceiling is
            # measure_tf32_peak(), which the figure draws.
            "triton grouped, fp32 operands": (
                triton(torch.float32),
                False,
                (x, w),
                experts,
                True,
            ),
            "cuBLAS 1 GEMM (equal work)": (
                lambda: x @ wflat.T,
                True,
                (x, wflat),
                1,
                False,
            ),
        }

    accuracy: dict[tuple[str, str], dict] = {}
    for dname, dtype in DTYPES.items():
        for impl, (fn, _, _, _, comparable) in impls_for(dname).items():
            if not comparable:
                accuracy[(dname, impl)] = dict(rel_err=float("nan"), ulp=float("nan"))
                continue
            with torch.no_grad():
                got = fn()
            accuracy[(dname, impl)] = dict(
                rel_err=rel_error(got, ref),
                ulp=ulp_error(got, ref, dtype, mode="rms"),
            )
            del got
    del ref
    torch.cuda.empty_cache()

    flops = 2 * rows_m * n * k
    out_rows = []
    for dname, dtype in DTYPES.items():
        elem = torch.finfo(dtype).bits // 8
        for impl, (fn, tensor_cores, grads, mats, _) in impls_for(dname).items():
            moved = (rows_m * k + mats * n * k + rows_m * n) * elem
            tags = dict(
                kernel="moe_gemm",
                impl=impl,
                dtype=dname,
                preset=preset,
                num_experts=experts,
                top_k=top_k,
                routed_rows=rows_m,
                matrix=tag,
                m=rows_m,
                n=n,
                k=k,
                **accuracy[(dname, impl)],
            )
            out_rows += _both_phases(
                f"{tag} {impl} {dname} E{experts}/k{top_k}",
                fn,
                grads,
                flops=flops,
                bytes_moved=moved,
                tensor_cores=tensor_cores,
                peak_bw=peak_bw,
                **tags,
            )
    tensors.clear()
    torch.cuda.empty_cache()
    return out_rows


def bench_moe_gemm(args, device, peak_bw):
    """Both expert matrices, at the routed row count the presets really produce."""
    rows = []
    for preset, experts, top_k, dim, hidden in moe_shapes(args.moe_presets):
        routed = args.moe_tokens * top_k
        print(
            f"{preset}: E={experts} top-{top_k} dim={dim} hidden={hidden} "
            f"-> {routed} routed rows from {args.moe_tokens} tokens",
            flush=True,
        )
        rows += _gemm_case(
            device, peak_bw, preset, experts, top_k, routed, dim, 2 * hidden, "w_in"
        )
        rows += _gemm_case(
            device, peak_bw, preset, experts, top_k, routed, hidden, dim, "w_out"
        )
    return rows


def _dispatch_fn(x, topk_idx, top_k, experts):
    """The permutation around the expert GEMMs, exactly as ``MoEMLP.forward`` does it."""

    def run():
        flat_expert = topk_idx.reshape(-1)
        order = flat_expert.argsort()
        token_of = order // top_k
        counts = torch.bincount(flat_expert, minlength=experts)
        offsets = torch.zeros(experts + 1, dtype=torch.int32, device=x.device)
        offsets[1:] = counts.cumsum(0).to(torch.int32)
        # The model calls .item() here to size the grid, which is a device sync
        # per MoE layer per step; it is timed because it is in the model.
        int(counts.max().item())
        gathered = x.index_select(0, token_of)
        out = torch.zeros_like(x)
        out.index_add_(0, token_of, gathered)
        return out

    return run


def bench_router(args, device, peak_bw):
    """Gating variants, and the dispatch permutation they feed.

    The router is the part of an MoE layer with real design freedom, and its
    cost is invisible until it sits next to the expert GEMMs it decides.
    """
    rows = []
    for preset, experts, top_k, dim, _hidden in moe_shapes(args.moe_presets):
        tokens = args.moe_tokens
        x = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
        base = dict(kernel="router", preset=preset, num_experts=experts, top_k=top_k)
        topk_idx = None
        for vname, kwargs in ROUTER_VARIANTS.items():
            router = TopKRouter(dim, experts, top_k, **kwargs).to(device)
            router.train()
            fn = lambda r=router: r(x)  # noqa: E731
            if topk_idx is None:
                with torch.no_grad():
                    topk_idx = fn()[0]
            row = _measure(
                f"router {vname} {preset}",
                fn,
                # The gate is deliberately an fp32 GEMM -- selection flips on
                # near-ties in bf16 -- so it is scored against the fp32 ceiling
                # rather than MAMF. It is tiny either way: the panel exists to
                # show that the router's cost is launch overhead, not arithmetic.
                flops=2 * tokens * dim * experts,
                bytes_moved=4 * (tokens * dim + experts * dim + tokens * experts),
                tensor_cores=False,
                peak_bw=peak_bw,
                impl=vname,
                **base,
            )
            if row is not None:
                rows.append(row)
            del router
        routed = tokens * top_k
        row = _measure(
            f"{DISPATCH} {preset}",
            _dispatch_fn(x, topk_idx, top_k, experts),
            flops=routed * dim,
            # gather reads and writes routed rows; the scatter reads them back
            # and writes T rows, over a zero-init of the same T rows.
            bytes_moved=(3 * routed + 2 * tokens) * dim * 2,
            tensor_cores=False,
            peak_bw=peak_bw,
            impl=DISPATCH,
            **base,
        )
        if row is not None:
            rows.append(row)
        x = topk_idx = None
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------
# compute-bound modules
# --------------------------------------------------------------------------


def bench_mlp(args, device, peak_bw):
    """The whole feed-forward, not just the elementwise part.

    The projections go to cuBLAS in every variant -- cuBLAS already beat our own
    fp32-accumulate GEMM in ``hgemm_acc.py`` -- so the only fusable part is
    ``silu(gate) * value``, and this measures what that fusion is worth once
    amortized over the two GEMMs it sits between.
    """
    rows = []
    hidden = args.hidden
    # One weight set shared by all three variants, so a throughput difference
    # cannot be a difference in the data.
    base = SwiGLU(args.dim, hidden=hidden).state_dict()
    for dname, dtype in DTYPES.items():
        variants = {
            "eager": SwiGLU(args.dim, hidden=hidden),
            "triton-gate": SwiGLUTriton(args.dim, hidden=hidden),
        }
        for module in variants.values():
            module.load_state_dict(base)
        variants = {k: v.to(device=device, dtype=dtype) for k, v in variants.items()}
        # Keyed by dtype: one compiled artifact per module instance, or the
        # cached one would still be bound to the previous dtype's weights.
        variants["compile"] = _compiled(f"mlp-{dname}", variants["eager"])
        for tokens in args.token_counts:
            x = torch.randn(
                tokens, args.dim, device=device, dtype=dtype, requires_grad=True
            )
            elem = torch.finfo(dtype).bits // 8
            flops = 6 * tokens * args.dim * hidden
            moved = (2 * tokens * args.dim + 3 * hidden * args.dim) * elem
            for impl, module in variants.items():
                # "compile" wraps eager, so reset the wrapped module's weights.
                inner = getattr(module, "_orig_mod", module)
                rows += _both_phases(
                    f"mlp {impl} {dname} T={tokens}",
                    lambda m=module: m(x),
                    (x, *inner.parameters()),
                    flops=flops,
                    bytes_moved=moved,
                    tensor_cores=True,
                    peak_bw=peak_bw,
                    kernel="mlp",
                    impl=impl,
                    dtype=dname,
                    tokens=tokens,
                    shape=f"H={hidden}",
                )
            x = None
            torch.cuda.empty_cache()
        variants.clear()
        torch.cuda.empty_cache()
    return rows


def _head_reference(x, w, t, chunk: int = 2048) -> torch.Tensor:
    """Summed fp32 cross-entropy, with the logits materialized a chunk at a time.

    A whole ``(tokens, vocab)`` fp32 logit tensor is 4.3 GiB at 16k tokens and
    34 GiB at 131k. Chunk sums accumulate in fp64.
    See docs/performance/benchmarking.md.
    """
    xf = x.detach().float()
    wf = w.detach().float()
    total = torch.zeros((), device=x.device, dtype=torch.float64)
    for start in range(0, xf.shape[0], chunk):
        stop = start + chunk
        logits = F.linear(xf[start:stop], wf)
        total += F.cross_entropy(logits, t[start:stop], reduction="sum").double()
        del logits
    return total


def bench_head(args, device, peak_bw):
    """LM head: the three cross-entropy paths, judged on memory first."""
    rows = []
    for dname, dtype in DTYPES.items():
        for tokens in args.head_tokens:
            # Both must be leaves: `randn(requires_grad=True) * 0.5` is a
            # non-leaf, so every timed backward would also traverse the scaling
            # node -- whose graph is freed after the first call.
            x = (
                torch.randn(tokens, args.dim, device=device, dtype=dtype) * 0.5
            ).requires_grad_()
            w = (
                torch.randn(args.vocab, args.dim, device=device, dtype=dtype) * 0.02
            ).requires_grad_()
            t = torch.randint(0, args.vocab, (tokens,), device=device)
            truth = _head_reference(x, w, t)

            def lce(options, x=x, w=w, t=t):
                # options=None is torch's *materializing* path, not a default.
                return (
                    lambda: F.linear_cross_entropy(
                        x, w, t, reduction="none", options=options
                    )
                    .float()
                    .sum()
                )

            impls = {
                "naive (materialize logits)": lambda: F.cross_entropy(
                    F.linear(x, w).float(), t, reduction="none"
                ).sum(),
                "lce-reference": lce(None),
                "lce-chunked": lce(LinearCrossEntropyOptions(allow_retain_graph=True)),
            }
            elem = torch.finfo(dtype).bits // 8
            flops = 6 * tokens * args.dim * args.vocab
            moved = 2 * (tokens * args.dim + args.vocab * args.dim) * elem
            for impl, fn in impls.items():
                try:
                    got = fn()
                except Exception as err:  # noqa: BLE001
                    print(f"  skip head {impl} {dname} @ {tokens}: {err}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                rel = abs(got.double().item() - truth.item()) / abs(truth.item())
                del got
                row = _measure(
                    f"head {impl} {dname} T={tokens} fwd+bwd",
                    _fwd_bwd(fn, x, w),
                    flops=flops,
                    bytes_moved=moved,
                    tensor_cores=True,
                    peak_bw=peak_bw,
                    warmup=2,
                    iters=8,
                    kernel="head",
                    impl=impl,
                    dtype=dname,
                    tokens=tokens,
                    phase="fwd+bwd",
                    rel_err=rel,
                )
                if row is not None:
                    rows.append(row)
            x = w = t = truth = None
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/kernels")
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument(
        "--token-counts", type=int, nargs="+", default=[2048, 8192, 32768, 131072]
    )
    ap.add_argument("--head-tokens", type=int, nargs="+", default=[2048, 8192, 16384])
    ap.add_argument(
        "--moe-presets",
        nargs="+",
        default=list(MOE_PRESETS),
        help="preset names; experts, top-k, dim and expert width come from each",
    )
    ap.add_argument("--moe-tokens", type=int, default=8192)
    args = ap.parse_args()
    if args.hidden is None:
        args.hidden = resolve_hidden(args.dim, 4.0, None, 128, glu=True)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    # One measurement feeds both the denominator and the caption's breakdown:
    # measuring twice would let the two disagree by a few percent.
    streams = stream_bandwidths()
    peak_bw = max(streams.values())
    tf32_peak = measure_tf32_peak()
    print(
        f"device: {device_name()}  |  measured ceilings: {peak_bw:.0f} GB/s DRAM "
        f"({', '.join(f'{k} {v:.0f}' for k, v in sorted(streams.items()))}), "
        f"{TENSOR_MAMF_TFLOPS:.0f}T bf16 MAMF, {tf32_peak:.0f}T TF32 mma  |  "
        f"dim={args.dim} hidden={args.hidden} vocab={args.vocab}"
    )
    rows = []
    for title, bench in (
        ("bandwidth-bound modules", bench_pointwise),
        ("SwiGLU MLP", bench_mlp),
        ("LM head", bench_head),
    ):
        print(f"\n== {title} ==", flush=True)
        rows += bench(args, device, peak_bw)
    print("\n== MoE router ==", flush=True)
    router = bench_router(args, device, peak_bw)
    print("\n== MoE expert GEMMs ==", flush=True)
    gemm = bench_moe_gemm(args, device, peak_bw)

    # Machine provenance travels with the rows so the figures can be redrawn on
    # any machine: without it the plotter has to ask a live CUDA device what it
    # is looking at, which makes redrawing a finished run need a free GPU.
    payload = dict(
        config=vars(args),
        device=device_name(),
        memory_bus_width=torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).memory_bus_width,
        theoretical_bandwidth_gbps=theoretical_bandwidth_gbps(),
        peak_bandwidth_gbps=peak_bw,
        stream_bandwidths=streams,
        tf32_peak_tflops=tf32_peak,
        rows=rows + router + gemm,
    )
    path = os.path.join(args.out, "kernels.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {path}")
    print(
        f"figures: .venv/bin/python scripts/bench/kernel/kernels_plot.py --dir {args.out}"
    )


if __name__ == "__main__":
    main()
