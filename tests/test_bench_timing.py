"""Checks for the measurement harness every benchmark's numbers come from.

The contention statistics moved with their module to
``test_bench_contention.py``; what is here needs a device.

A benchmark result is only as trustworthy as its denominator, and two of the
denominators here were wrong: ``module_metrics`` divided bandwidth by the
clock-derived *stock* figure on a card whose memory is overclocked, and the L2
flush buffer lived on device 0 no matter which device was being timed. Both
produce plausible-looking numbers, which is exactly why they need pinning.
"""

import pytest
import torch

from kohakuwullm.bench.core.timing import (
    TENSOR_MAMF_TFLOPS,
    VECTOR_PEAK_TFLOPS,
    _flush_buffers,
    _flush_l2,
    cached_peak_bandwidth,
    device_op_counts,
    format_metrics,
    measure_peak_bandwidth,
    module_metrics,
    stream_bandwidths,
    timing_profile,
)

# Per-test rather than module-wide, kept that way after the pure-statistics tests
# moved to `test_bench_contention.py`: what is left is device-bound, but a marker
# per test says which measurement needs a card rather than blanketing the file.
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="bandwidth measurement requires CUDA"
)

# Generous: well above any shipping consumer part, so the assertion catches a
# broken measurement (the earlier device-mismatch bug reported 180x the limit)
# without encoding this particular card's clock.
ABSURD_GBPS = 10_000.0


@cuda_only
def test_stream_bandwidths_are_plausible_and_peak_is_the_best_of_them():
    result = stream_bandwidths(1 << 28)
    assert set(result) == {"copy", "read", "triad"}
    for name, gbps in result.items():
        assert 0.0 < gbps < ABSURD_GBPS, f"{name} measured {gbps} GB/s"
    # measure_peak_bandwidth must not quote copy alone: a read-heavy kernel can
    # legitimately beat the copy figure, and scoring it against copy would
    # report it above 100% of peak.
    assert measure_peak_bandwidth(1 << 28) >= max(result.values()) * 0.9


@cuda_only
def test_module_metrics_scores_bandwidth_against_the_measured_peak():
    n = 1 << 24
    src = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    dst = torch.empty_like(src)
    moved = 2 * n * 4

    default = module_metrics(lambda: dst.copy_(src), flops=0.0, bytes_moved=moved)
    assert default["bandwidth_pct"] == pytest.approx(
        100.0 * default["gbps"] / cached_peak_bandwidth(), rel=1e-6
    )

    # An explicit peak still wins, so a caller measuring its own denominator is
    # not silently overridden by the cached one.
    forced = module_metrics(
        lambda: dst.copy_(src),
        flops=0.0,
        bytes_moved=moved,
        peak_bandwidth_gbps=1000.0,
    )
    assert forced["bandwidth_pct"] == pytest.approx(
        100.0 * forced["gbps"] / 1000.0, rel=1e-6
    )


@cuda_only
@pytest.mark.parametrize(
    "tensor_cores,expected",
    [(True, TENSOR_MAMF_TFLOPS), (False, VECTOR_PEAK_TFLOPS)],
    ids=["tensor", "vector"],
)
def test_module_metrics_ceiling_follows_the_caller_not_the_device(
    tensor_cores, expected
):
    x = torch.randn(512, 512, device="cuda")
    metrics = module_metrics(
        lambda: x @ x,
        flops=2 * 512**3,
        bytes_moved=3 * 512**2 * 4,
        tensor_cores=tensor_cores,
    )
    assert metrics["compute_ceiling_tflops"] == expected
    assert metrics["compute_pct"] == pytest.approx(
        100.0 * metrics["tflops"] / expected, rel=1e-6
    )


@cuda_only
def test_flush_buffer_lands_on_the_device_being_timed():
    index = torch.cuda.current_device()
    _flush_l2()
    assert _flush_buffers[index].device.index == index


@cuda_only
def test_wall_time_of_a_dispatch_bound_op_says_so():
    """The artifact that made two separate benchmarks wrong by over 2x.

    ``bench_ms`` syncs and *then* records its start event, so the GPU is idle
    inside the event window while Python issues work, and that idle time is
    charged to the kernel. The number is real wall time but it is not the
    kernel's, and nothing about it looks wrong -- which is why the flag has to
    travel with it rather than being left to the reader to suspect.

    Many tiny ops rather than one: a single op is not reliably host-bound (a 256
    element ``add_`` measures 6 us of host against 13 us of device), and
    ``graph_ms`` has a ~11 us replay floor that would swamp the device side of a
    one-kernel comparison anyway.
    """
    tiny = torch.empty(256, device="cuda")

    def fn():
        for _ in range(64):
            tiny.add_(1.0)

    profile = timing_profile(fn, device=True)
    assert profile["host_ms"] > profile["device_ms"]
    assert profile["host_bound"] is True

    metrics = module_metrics(fn, flops=0.0, bytes_moved=64 * 2 * 256 * 4)
    assert metrics["host_bound"] is True
    assert "HOST-BOUND" in format_metrics("tiny", metrics)


@cuda_only
def test_a_large_op_is_not_flagged_and_carries_host_time_anyway():
    """The flag must discriminate, and the host column must always be present.

    A copy this size is firmly device-bound, so a harness that flagged
    everything would be as useless as one that flagged nothing.
    """
    n = 1 << 26
    src = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    dst = torch.empty_like(src)

    metrics = module_metrics(lambda: dst.copy_(src), flops=0.0, bytes_moved=2 * n * 4)
    assert metrics["host_bound"] is False
    assert metrics["host_ms"] < metrics["ms"]
    assert "HOST-BOUND" not in format_metrics("copy", metrics)


@cuda_only
def test_device_op_counts_counts_launches_and_separates_memory_ops():
    """Launches, not seconds -- and memcpy split out rather than folded in.

    The count is what a fusion claim rests on, so the two things that would make it
    lie are pinned: an op that issues no kernel must not add to it, and a device-to
    -device copy must land in ``memory`` rather than inflate a "GEMM count".
    """
    a = torch.randn(256, 256, device="cuda")

    # Elementwise ops, not a matmul: a fp32 GEMM this size goes through cuBLASLt
    # split-K and issues *two* kernels, so a matmul cannot pin an exact count without
    # also pinning a vendor heuristic. Three elementwise ops are three launches.
    three = device_op_counts(lambda: a.relu().add(1.0).mul(2.0))
    assert three["kernel"] == 3, three["names"]
    assert three["memory"] == 0
    assert three["total"] == three["kernel"] + three["memory"]
    # The names are what make the integer auditable, so they must actually be there.
    assert len(three["names"]) == 3 and all(three["names"])

    copied = device_op_counts(lambda: a.clone())
    assert copied["memory"] == 1, copied["names"]
    assert copied["kernel"] == 0

    # A view or a shape query issues nothing, and a counter that charged for one would
    # make every path look more expensive than it is.
    assert device_op_counts(lambda: a.t().shape)["total"] == 0


@cuda_only
def test_device_op_counts_rejects_a_count_it_cannot_reproduce():
    """A first-call-only launch is compilation, and must not be reported as the path.

    The guard matters because the failure is silent in the useful direction: a Triton
    autotune replay inflates the first profiled call, and a benchmark that trusted it
    would report a fused path as issuing more work than it does.
    """
    a = torch.randn(64, 64, device="cuda")
    extra = iter([True, False])

    def unstable():
        result = a + 1.0
        if next(extra, False):
            result = result * 2.0
        return result

    with pytest.raises(RuntimeError, match="not reproducible"):
        device_op_counts(unstable, warmup=0)


@cuda_only
def test_timing_profile_estimates_device_time_by_subtraction():
    """``wall - host`` is the device estimate, not a graph capture.

    Capturing across a sweep perturbs rows that are not even being captured, so
    the default path must not do it. The subtraction is meaningful because the
    timing loop leaves the queue empty at its start event: the window is host
    issue *then* device execution, not the two overlapping.
    """
    n = 1 << 26
    src = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    dst = torch.empty_like(src)

    profile = timing_profile(lambda: dst.copy_(src), device=True)
    assert "device_ms" in profile
    assert profile["device_est_ms"] == pytest.approx(
        profile["wall_ms"] - profile["host_ms"], rel=1e-9
    )
    # On a device-bound op the estimate must track the real replay closely.
    assert profile["device_est_ms"] == pytest.approx(profile["device_ms"], rel=0.15)

    # And it is present without asking for a capture at all.
    cheap = timing_profile(lambda: dst.copy_(src))
    assert "device_ms" not in cheap
    assert cheap["device_est_ms"] > 0.0


@cuda_only
def test_timing_profile_reports_host_bound_when_a_capture_fails():
    """A failed capture must not read as compute-bound.

    ``graph_ms`` returns nan there, and every comparison against nan is false, so
    the naive check would silently clear a host-bound op.
    """

    def syncs():
        for _ in range(16):
            torch.empty(8, device="cuda").sum().item()

    profile = timing_profile(syncs, device=True)
    assert profile["device_ms"] != profile["device_ms"], "capture should have failed"
    assert profile["host_bound"] is True
