"""GPU timing and precision-check helpers shared by ``scripts/bench/*``.

See docs/performance/benchmarking.md.
"""

import re
import statistics
import time
import warnings

import torch

from kohakuwullm.bench.core.contention import (
    contended_fraction_limit,
    sample_contended_fraction,
    sample_spread,
    spread_has_power,
    spread_threshold,
)

_FLUSH_BYTES = 256 * 1024 * 1024
# Keyed by device: a buffer on device 0 does not evict device 1's L2.
_flush_buffers: dict[int, torch.Tensor] = {}


def _flush_l2() -> None:
    index = torch.cuda.current_device()
    buffer = _flush_buffers.get(index)
    if buffer is None:
        buffer = torch.empty(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
        _flush_buffers[index] = buffer
    buffer.zero_()


def bench_samples(
    fn, warmup: int = 10, iters: int = 50, flush_l2: bool = True
) -> list[float]:
    """Every per-iteration time in milliseconds, in order."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush_l2:
            _flush_l2()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_ms(fn, warmup: int = 10, iters: int = 50, flush_l2: bool = True) -> float:
    """Median **wall** time of ``fn`` in milliseconds -- device *plus* dispatch.

    Not device time; prefer :func:`timing_profile`. See docs/performance/benchmarking.md.
    """
    return statistics.median(bench_samples(fn, warmup, iters, flush_l2))


def timing_profile(
    fn, warmup: int = 5, iters: int = 20, device: bool = False
) -> dict[str, float | bool]:
    """Wall, host and estimated device time for one op, together.

    ``device_est_ms`` is ``wall - host``. ``device=True`` adds a ``device_ms`` from a
    CUDA-graph replay. See docs/performance/benchmarking.md.
    """
    wall = bench_ms(fn, warmup=warmup, iters=iters)
    host = cpu_enqueue_ms(fn, warmup=warmup)
    estimate = wall - host
    out: dict[str, float | bool] = {
        "wall_ms": wall,
        "host_ms": host,
        "device_est_ms": estimate,
    }
    if device:
        out["device_ms"] = graph_ms(fn, warmup=warmup, iters=iters)
    # Against the real device time when it was captured, the estimate otherwise.
    # `nan != nan` guards a failed capture, which would otherwise clear the flag.
    captured = out.get("device_ms", float("nan"))
    reference = captured if captured == captured else estimate
    out["host_bound"] = bool(host >= reference)
    return out


def graph_ms(fn, warmup: int = 5, iters: int = 50) -> float:
    """Median ms of ``fn`` replayed as a CUDA graph -- device time, no dispatch.

    Returns ``nan`` when capture fails. No L2 flush between replays, so this is not
    valid bandwidth evidence below L2 size. See docs/performance/benchmarking.md.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return statistics.median(times)
    except Exception:
        return float("nan")


def cpu_enqueue_ms(fn, warmup: int = 10, iters: int = 30) -> float:
    """Host time to *issue* ``fn``, with no device sync inside the loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1e3 / iters
    torch.cuda.synchronize()
    return elapsed


def _short_kernel_name(name: str) -> str:
    """A mangled CUDA kernel name reduced to the part that identifies it."""
    name = re.sub(r"<.*", "", name)
    for prefix in ("void ", "at::native::", "(anonymous namespace)::"):
        name = name.replace(prefix, "")
    return name.strip()[:48]


def device_op_counts(fn, warmup: int = 5) -> dict:
    """Device operations one call of ``fn`` issues, with the names that issued them.

    Keys: ``kernel``, ``memory`` (memcpy + memset), ``total``, ``names``. Raises if
    two profiled calls disagree. See docs/performance/benchmarking.md.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    counts = []
    for _ in range(2):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            fn()
        torch.cuda.synchronize()
        device = [
            event.name
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        memory = sum(name.startswith(("Memcpy", "Memset")) for name in device)
        counts.append(
            {
                "kernel": len(device) - memory,
                "memory": memory,
                "total": len(device),
                "names": [_short_kernel_name(name) for name in device],
            }
        )
    if counts[0] != counts[1]:
        raise RuntimeError(
            f"launch count not reproducible: {counts[0]['total']} then "
            f"{counts[1]['total']} device ops; the first call is still compiling "
            "or autotuning"
        )
    return counts[0]


def bench_peak_memory(fn) -> float:
    """Peak allocated GiB during one call to ``fn``."""
    # Materialise the lazily allocated flush buffer before the stats reset.
    _flush_l2()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def rel_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2 error ``||got - ref|| / ||ref||``, computed in fp64."""
    got = got.detach().double()
    ref = ref.detach().double()
    denom = ref.norm().clamp_min(1e-30)
    return ((got - ref).norm() / denom).item()


def ulp_error(
    got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, mode: str = "elementwise"
) -> float:
    """Max error in units of last place for ``dtype``; 1.0 is as exact as it gets.

    Args:
        mode: ``"elementwise"`` scales by each element's own magnitude, for
            elementwise kernels; ``"rms"`` scales by the reference's RMS magnitude,
            for GEMMs and reductions. See docs/performance/benchmarking.md.
    """
    eps = torch.finfo(dtype).eps
    got = got.detach().double()
    ref = ref.detach().double()
    if mode == "rms":
        scale = ref.pow(2).mean().sqrt().clamp_min(torch.finfo(dtype).tiny)
    elif mode == "elementwise":
        scale = ref.abs().clamp_min(torch.finfo(dtype).tiny)
    else:
        raise ValueError(f"unknown ulp mode {mode!r}")
    return ((got - ref).abs() / (scale * eps)).max().item()


def stream_bandwidths(bytes_moved: int = 1 << 30, iters: int = 20) -> dict[str, float]:
    """Achieved GB/s for ``copy`` / ``read`` / ``triad``, on the current device."""
    if not torch.cuda.is_available():
        return {"copy": 0.0, "read": 0.0, "triad": 0.0}
    n = bytes_moved // 8  # sized so copy (1 read + 1 write) moves bytes_moved
    a = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    b = torch.empty_like(a).uniform_()
    c = torch.empty_like(a)

    def timed(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    out = {
        "copy": (2 * n * 4) / timed(lambda: c.copy_(a)) * 1e-6,
        "read": (n * 4) / timed(lambda: a.sum()) * 1e-6,
        "triad": (3 * n * 4) / timed(lambda: torch.add(a, b, alpha=2.0, out=c)) * 1e-6,
    }
    a = b = c = None
    torch.cuda.empty_cache()
    return out


def measure_peak_bandwidth(bytes_moved: int = 1 << 30, iters: int = 20) -> float:
    """Empirically achievable DRAM bandwidth in GB/s -- best of the three patterns.

    Preferred over any clock-derived figure. See docs/performance/benchmarking.md.
    """
    return max(stream_bandwidths(bytes_moved, iters).values())


_measured_peak: dict[int, float] = {}


def cached_peak_bandwidth() -> float:
    """Cached :func:`measure_peak_bandwidth` for the current device.

    The denominator every "% of peak" in the suite divides by.
    """
    if not torch.cuda.is_available():
        return 0.0
    index = torch.cuda.current_device()
    if index not in _measured_peak:
        _measured_peak[index] = measure_peak_bandwidth()
    return _measured_peak[index]


def theoretical_bandwidth_gbps() -> float:
    """Clock-derived roofline. Reports the STOCK clock -- see docs/performance/benchmarking.md."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    clock = getattr(props, "memory_clock_rate", None)
    width = getattr(props, "memory_bus_width", None)
    if clock and width:
        return 2 * clock * 1e3 * (width / 8) / 1e9
    return 0.0


def device_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} (sm_{props.major}{props.minor})"


# Compute ceilings for sm_120. See docs/performance/benchmarking.md for which to score against.
TENSOR_PEAK_TFLOPS = 400.0
TENSOR_MAMF_TFLOPS = 270.0
TF32_MAMF_TFLOPS = 111.0
VECTOR_PEAK_TFLOPS = 120.0


def module_metrics(
    fn,
    flops: float,
    bytes_moved: float,
    warmup: int = 5,
    iters: int = 20,
    tensor_cores: bool | None = None,
    ceiling: str = "tensor",
    peak_bandwidth_gbps: float | None = None,
) -> dict:
    """Latency, effective FLOP/s, effective bandwidth and peak memory for one op.

    Every row also carries ``host_ms`` / ``host_bound``, the ``net_*`` figures net of
    dispatch, both contention statistics, and ``suspect`` -- non-empty when the row is
    physically impossible, which also raises a ``RuntimeWarning``.

    Args:
        ceiling: ``"tensor"``, ``"tf32"`` or ``"vector"``.
        tensor_cores: legacy selector; ``False`` maps to ``"vector"``.

    See docs/performance/benchmarking.md.
    """
    peak_gib = bench_peak_memory(fn)
    # One sample loop feeding the median and both contention statistics.
    samples = bench_samples(fn, warmup=warmup, iters=iters)
    ms = statistics.median(samples)
    host_ms = cpu_enqueue_ms(fn, warmup=warmup)
    seconds = ms / 1e3
    if tensor_cores is not None:
        ceiling = "tensor" if tensor_cores else "vector"
    ceilings = {
        "tensor": TENSOR_MAMF_TFLOPS,
        "tf32": TF32_MAMF_TFLOPS,
        "vector": VECTOR_PEAK_TFLOPS,
    }
    if ceiling not in ceilings:
        raise ValueError(f"unknown ceiling {ceiling!r}; have {sorted(ceilings)}")
    ceiling = ceilings[ceiling]
    tflops = flops / seconds / 1e12
    gbps = bytes_moved / seconds / 1e9
    bw_peak = peak_bandwidth_gbps or cached_peak_bandwidth()
    net_ms = ms - host_ms
    net_tflops = flops / (net_ms / 1e3) / 1e12 if net_ms > 0 else float("nan")
    net_gbps = bytes_moved / (net_ms / 1e3) / 1e9 if net_ms > 0 else float("nan")
    row = {
        "ms": ms,
        "host_ms": host_ms,
        "net_ms": net_ms,
        "host_share": host_ms / ms if ms else float("nan"),
        "host_bound": bool(host_ms >= ms / 2),
        "tflops": tflops,
        "gbps": gbps,
        "net_tflops": net_tflops,
        "net_gbps": net_gbps,
        "peak_gib": peak_gib,
        "compute_ceiling_tflops": ceiling,
        "compute_pct": 100.0 * tflops / ceiling if ceiling else 0.0,
        "bandwidth_pct": 100.0 * gbps / bw_peak if bw_peak else 0.0,
        # Recorded on every row so a later run can be re-judged without re-measuring.
        "spread": sample_spread(samples),
        "spread_threshold": spread_threshold(ms),
        "spread_has_power": spread_has_power(ms),
        "contended_fraction": sample_contended_fraction(samples),
        "contended_limit": contended_fraction_limit(),
    }
    row["suspect"] = _implausible(row, ceiling, bw_peak)
    if row["suspect"]:
        warnings.warn(f"module_metrics: {row['suspect']}", RuntimeWarning, stacklevel=2)
    return row


HOST_SHARE_LIMIT = 0.5
CEILING_SLACK = 1.02


def _implausible(row: dict, ceiling: float, bw_peak: float) -> str:
    """Why this row cannot be believed, or ``""`` if nothing is wrong."""
    reasons = []
    if row["host_share"] > HOST_SHARE_LIMIT:
        reasons.append(
            f"host dispatch is {100 * row['host_share']:.0f}% of wall, so both "
            "the wall figure and the net-of-dispatch estimate are unreliable"
        )
    if ceiling and row["tflops"] > ceiling * CEILING_SLACK:
        reasons.append(
            f"{row['tflops']:.0f} TFLOP/s exceeds the {ceiling:.0f} ceiling; "
            "check the FLOP count and the ceiling choice"
        )
    net = row["net_tflops"]
    if ceiling and net == net and net > ceiling * CEILING_SLACK:
        reasons.append(
            f"net-of-dispatch {net:.0f} TFLOP/s exceeds the {ceiling:.0f} "
            "ceiling -- wall minus host over-subtracted"
        )
    if bw_peak and row["gbps"] > bw_peak * CEILING_SLACK:
        reasons.append(
            f"{row['gbps']:.0f} GB/s exceeds the measured {bw_peak:.0f} peak; "
            "L2-resident, or the byte count is wrong"
        )
    return "; ".join(reasons)


def format_metrics(name: str, m: dict, width: int = 26) -> str:
    """One line carrying all four metrics, for a benchmark's stdout.

    A trailing ``HOST-BOUND`` marks a row whose rates are floors, not measurements.
    """
    line = (
        f"  {name:>{width}}: {m['ms']:8.3f} ms  "
        f"{m['tflops']:8.1f} TF/s ({m['compute_pct']:5.1f}% of "
        f"{m['compute_ceiling_tflops']:.0f}T)  "
        f"{m['gbps']:8.0f} GB/s ({m['bandwidth_pct']:5.1f}%)  "
        f"{m['peak_gib']:7.3f} GiB"
    )
    return line + "  HOST-BOUND" if m.get("host_bound") else line
