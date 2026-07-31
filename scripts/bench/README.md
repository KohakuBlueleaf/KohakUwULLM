# scripts/bench

Benchmarks are part of the deliverable, not a scratch area. Every figure shows
throughput and accuracy together: a kernel that is fast and wrong is not a result.

Directories are by **what a script measures**, not by which technique it uses. `fp8/`
is the exception and is named for its subject rather than its scope — MXFP8 kernels
live under `kernel/` and `moe/`, beside the bf16 paths they are compared against,
because a speedup only means something next to the thing it replaced.

| directory | holds |
|---|---|
| `kernel/` | one op or one module, microbenchmarked: attention backends, norms, SwiGLU, the LM head, GEMM arithmetic formats, the optimizer step, the low-precision writeback. `bandwidth.py` is here because it is the roofline the rest divide by. |
| `moe/` | the MoE layer: router and dispatch, expert formulations, the grouped and fused MXFP8 expert paths, and the rewrite ladder that attributes their speedup. |
| `model/` | one model doing one thing end to end on one card, where the question is about the model rather than a kernel: `generate.py` is KV-cached decoding against re-running the prefix. |
| `e2e/` | whole models: 4-card step throughput, pipeline stage balance and memory, the architecture sweeps that need a full stack, and the preset ladder's parameter census. |
| `fp8/` | the training-level MXFP8 A/B — does loss track bf16, and does the swap cost stability margin. Loss quality, not kernel speed. |
| `data/` | the input pipeline: record reads, rendering, tokenization, packing. Never touches CUDA, so it runs while the cards are busy. |
| `_archive/` | superseded, kept for method. See `_archive/README.md`. Nothing here is wired into `run_all.sh`. |

`run_all.sh` drives the single-GPU stages plus the Kohaku end-to-end sweep. The `.sh`
files inside each directory are sweep drivers for the `.py` beside them — they exist
because a fresh process per row is required, not as convenience wrappers.

## Conventions

**Measure and plot are separate files** wherever the measurement costs GPU minutes:
`X.py` writes `X.json`, `X_plot.py` draws it. A rejected layout must not cost the
measurement again. Plotters take `--dir` (read JSON there, write the figure beside
it); measurement scripts take `--out`.

**Every script's module docstring says what question it answers, and ends with how to
run it.** Two scripts on the same subject with different axes are not duplicates and
both stay — `head.py`, `head_options.py` and `chunked_ce.py` sweep token count,
precision regime and tile geometry over the same subsystem, and only the second can
see the `options=None` trap that motivated it. Each names its siblings.

**A denominator is measured, not carried.** Bandwidth comes from
`bench.timing.cached_peak_bandwidth()` and dense GEMM ceilings from a measurement in
the same process on the same card. A stored 227 TF/s figure once disqualified a
legitimate 228 TF/s row, which was measuring the constant's age rather than the kernel.
Where a plotter cannot re-measure, the caption says the ceiling was carried.
