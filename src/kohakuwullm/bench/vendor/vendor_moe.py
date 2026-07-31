"""Shared machinery for the vendor-MXFP8 MoE benchmarks: timing, layout, concurrency.

Device timing is a capture replayed through the flushed sample loop, never
``wall - host``. See docs/performance/benchmarking.md.
"""

import torch

from kohakuwullm.bench.analysis.admissibility import median_and_contention
from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.interop import (
    expected_swizzled_numel,
    vendor_mxfp8_matmul_swizzled,
)

# cuBLAS addresses MX scales as 128-row x 4-scale-column tiles; both are the ABI's.
ROW_TILE = 128
SCALE_COLS_PER_TILE = 4


def capture(fn) -> torch.cuda.CUDAGraph | None:
    """``fn`` as a replayable CUDA graph, or ``None`` if it cannot be captured."""
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
        return graph
    except Exception:
        return None


def device_time(fn) -> tuple[float, float, bool]:
    """``(median device ms, contended fraction, verdict has power)`` with a cold L2."""
    graph = capture(fn)
    if graph is None:
        return float("nan"), float("nan"), False
    return median_and_contention(graph.replay)


def measure_replay_floor() -> float:
    """Median ms of an empty replay -- the floor every :func:`device_time` carries."""
    scratch = torch.zeros(1, device="cuda")
    graph = capture(lambda: scratch.add_(1.0))
    if graph is None:
        return float("nan")
    return median_and_contention(graph.replay)[0]


def dense_ceilings(size: int = 4096) -> dict[str, float]:
    """The bf16 and MXFP8 rates this card reaches on a dense square GEMM, right now.

    Measured per run, never stored. See docs/performance/benchmarking.md.
    """
    torch.manual_seed(0)
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    aq, a_scale = quantize_mx_vendor(a)
    bq, b_scale = quantize_mx_vendor(b)
    flops = 2.0 * size**3
    return {
        label: flops / (device_time(fn)[0] / 1e3) / 1e12
        for label, fn in (
            ("bf16", lambda: a @ b.T),
            ("fp8", lambda: vendor_mxfp8_matmul_swizzled(aq, a_scale, bq, b_scale)),
        )
    }


class StreamedLoop:
    """Per-expert jobs fanned across ``n_streams``, forked and joined on each call.

    Streams are members so a captured graph replays onto the ones it captured.
    See docs/performance/benchmarking.md.
    """

    def __init__(self, work, n_streams: int):
        self.work = work
        self.n_streams = n_streams
        self.streams = [torch.cuda.Stream() for _ in range(n_streams)]

    def __call__(self):
        main = torch.cuda.current_stream()
        for stream in self.streams:
            stream.wait_stream(main)
        out = [None] * len(self.work)
        for i, job in enumerate(self.work):
            with torch.cuda.stream(self.streams[i % self.n_streams]):
                out[i] = job()
        for stream in self.streams:
            main.wait_stream(stream)
        # Discharge the allocator's cross-stream contract for side-stream outputs.
        for tensor in out:
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(main)
        return out


def aligned_offsets(rows_per_expert: int, experts: int, device="cuda"):
    """``(starts, counts, padded_total, offsets)``, every start 128-aligned.

    The capacity-padded case, so counts are balanced and the padding is reportable.
    """
    stride = -(-rows_per_expert // ROW_TILE) * ROW_TILE
    starts = [e * stride for e in range(experts)]
    counts = [rows_per_expert] * experts
    offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.tensor(
        [s + rows_per_expert for s in starts], dtype=torch.int32, device=device
    )
    return starts, counts, stride * experts, offsets


def scale_views(swizzled: torch.Tensor, starts, counts, k: int) -> list[torch.Tensor]:
    """Per-expert views into one swizzled scale buffer; raises on an unaligned start.

    ``scaled_mm`` would not catch that. See docs/performance/benchmarking.md.
    """
    col_tiles = k // (SCALE_COLS_PER_TILE * BLOCK_SCALE)
    tile_bytes = ROW_TILE * SCALE_COLS_PER_TILE
    flat = swizzled.view(torch.uint8)
    views = []
    for start, count in zip(starts, counts):
        if start % ROW_TILE:
            raise ValueError(
                f"expert row start {start} is not {ROW_TILE}-aligned; its scale slice "
                "would address another expert's exponents"
            )
        lo = (start // ROW_TILE) * col_tiles * tile_bytes
        span = -(-count // ROW_TILE) * col_tiles * tile_bytes
        view = flat[lo : lo + span].view(torch.float8_e8m0fnu)
        if view.numel() != expected_swizzled_numel(count, k):
            raise ValueError(
                f"scale view for {count} rows x K={k} is {view.numel()} elements, "
                f"scaled_mm wants {expected_swizzled_numel(count, k)}"
            )
        views.append(view)
    return views
