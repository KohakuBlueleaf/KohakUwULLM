# Performance

Numbers, and the method that makes them numbers rather than anecdotes. Read
[benchmarking.md](benchmarking.md) before quoting anything from here.

| Doc | Covers |
|---|---|
| [performance.md](performance.md) | Measured throughput across the ladder, on 4x RTX 5090, and where the time goes |
| [gemm.md](gemm.md) | How a fast GEMM is built on sm_120: cache levels, warp and register budgets, what to fuse and where, MXFP8 vs NVFP4, why a naive Triton kernel loses |
| [benchmarking.md](benchmarking.md) | How this repo measures: wall vs device time, the L2 flush, which ceiling to divide by, ULP modes |
| [ab-testing.md](ab-testing.md) | Running a trustworthy A/B: noise floors, block bootstrap, admissibility |
| [upstream-cutlass-findings.md](upstream-cutlass-findings.md) | Why CUTLASS grouped block-scaled GEMM is unusable on sm_120 |

## The denominators

Every percentage in this repo divides by one of these. Getting the denominator
wrong is the single most common way a benchmark here has lied.

| ceiling | value | when |
|---|---|---|
| DRAM bandwidth | **1791 GB/s** measured | any memory-bound kernel |
| `TENSOR_MAMF_TFLOPS` | 270 TF/s | bf16 / fp32-accumulate matmul — the default |
| `TF32_MAMF_TFLOPS` | 111 TF/s | a Triton fp32 `tl.dot` |
| `VECTOR_PEAK_TFLOPS` | 120 TF/s | genuine FMA with no tensor-core path |

The stock-clock theoretical bandwidth is 1792 GB/s — within a rounding error of the
measured figure and therefore a very convincing decoy. The card actually clocks to
2035 GB/s theoretical. Use `cached_peak_bandwidth()`, never `memory_clock_rate`.

## Before quoting a number

1. Is `host_bound` false, and `host_share` under 50%?
2. Did warmup cover **every** shape the timed loop sees?
3. Working set larger than L2 — or, on a graph replay, are you quoting time rather
   than bandwidth?
4. Right ceiling, measured rather than clock-derived?
5. Accuracy panel beside the throughput panel, in ULP, in the right mode?
6. Does `suspect` come back empty?

Results live in [../../out/bench/README.md](../../out/bench/README.md); the docs
here explain the methods, that index holds the numbers.
