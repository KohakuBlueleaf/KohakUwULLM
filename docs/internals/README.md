# Internals

How the pieces are built, and which plausible-looking change to each is a trap.
Read these when you are modifying the thing, not when you are using it.

| Doc | Covers |
|---|---|
| [data.md](data.md) | KohakuVault sources, TIPO rendering, tokenization, loss masking, packing, the loader and its resume contract |
| [optimizers.md](optimizers.md) | Muon, parameter grouping and weight decay, muP, low-bit state, stochastic rounding |
| [kernels.md](kernels.md) | Every Triton kernel: what it does, what constrains its numerics, and the trap in it |
| [kernel-dev.md](kernel-dev.md) | The method: how to derive tile, warp and pipeline budgets from what a card reports, on any sm_120 part |
| [kernel-dsls.md](kernel-dsls.md) | Triton against TileLang and CuTeDSL, and why the published gaps may not reach sm_120 |
| [mxfp8.md](mxfp8.md) | Block-scaled fp8 training: the format, what is converted, what is verified, what is still open |
| [pipeline.md](pipeline.md) | Pipeline parallelism: stage splitting, boundary dtype, the Lightning wiring, DDP vs pipeline |
| [moe-router-loss.md](moe-router-loss.md) | The two router auxiliary losses in the fused kernel, and the boundary stream that carries them across stages |

## The rules these all follow

- **Reduce in fp32.** Summing 16k bf16 terms loses several percent.
- **Judge error in ULP, not absolutely** — `"elementwise"` for elementwise kernels,
  `"rms"` for GEMMs and reductions.
- **Never put a varlen axis in an autotune key.** It has cost this repo 365 ms/step
  and 950 ms/step, twice, in two different kernels.
- **Bound every grid from host-known values.** An `.item()` is a pipeline stall and
  a CUDA-graph capture failure.
- **A reference that shares the kernel's assumptions proves nothing.** Both defects
  ever found in MXFP8 survived their tests exactly that way.

## Next

- How any of these numbers were obtained: [../performance/benchmarking.md](../performance/benchmarking.md)
- What they add up to: [../performance/performance.md](../performance/performance.md)
