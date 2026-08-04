# The fused 16-bit MoE expert path

Until this landed, `MoEMLP` had two routed paths: an **eager** one (gather, two
grouped GEMMs, `silu(gate)*value` in ATen, combine) and a **fused MXFP8** one.
There was no fused 16-bit path, so every measurement that credited MXFP8 with a
speedup was really measuring *fusion plus quantization together* and attributing
all of it to the fp8 arithmetic.

This document is the 16-bit sibling: same fusion, no quantization.

## 1. What the eager path spends

At the production shape -- dim 768, expert hidden 384, 64 experts, top-8, 16384
tokens per microbatch, so **131072 routed rows** -- the eager path materialises
four `(M, ·)` tensors and keeps three of them alive for the backward:

| tensor | shape | why it is alive |
|---|---|---|
| `x_sorted` | `(M, dim)` | saved by grouped GEMM 1 |
| `h` | `(M, 2H)` | `chunk` makes `gate`/`value` views of it |
| `silu(gate)` | `(M, H)` | saved by the multiply |
| `act` | `(M, H)` | saved by grouped GEMM 2 |

That is `2M(dim + 4H)` bytes of saved activation. The fused path saves `pre`
`(M, 2H)` and `h` `(M, H)` and nothing else -- `6MH` bytes -- because `x` is the
block input, which the graph already holds, and it is never gathered to `(M, dim)`
at all.

## 2. The four kernels

GEMM1 folds the gather, both halves of `x @ w_in.T`, SwiGLU and the store into one
pass. GEMM2 folds the gate scale and the scatter-add. The two DGRADs mirror them.

**The 16-bit path needs no derived weight copy.** MXFP8 keeps four quantized
copies per layer -- each expert matrix blocked along K for FPROP and along its
transpose for DGRAD -- because an MX scale block is one-dimensional and cannot be
transposed in place. With no scales there is nothing to re-block: the DGRADs read
the master `(E, N, K)` weight and simply index the contraction axis on the row
stride, `offs_k[:, None] * stride_wn + offs_n[None, :]`. That load is *more*
natural than the forward's, which needs `tl.trans`.

**`h` is rounded to storage before the activation.** `acc.to(OUT_DTYPE).to(f32)`
in the GEMM1 epilogue, so the value the backward differentiates is the value the
forward evaluated. `gemm2_dgrad` rebuilds `h` in-register under the same rounding
rather than reloading it, which costs nothing because it already loads `pre` for
the SwiGLU derivative.

WGRAD is not duplicated. `mxfp8.grouped.grouped_mxfp8_wgrad_kernel` already
carries a 16-bit arm under `B_FP8=False`, with the gather and gate epilogue every
routed path needs; that is what its module docstring means by "the WGRAD kernel
every routed-expert path shares".

## 3. Measured, RTX 5090, 131072 routed rows

Accuracy is ULP against an fp64 autograd oracle, `mode="rms"` because these are
GEMMs and reductions. Best-of-50, `bench_ms`, grads reset each iteration.

| arm | fwd ms | TF/s | fwd+bwd ms | TF/s | peak GiB | out ULP (fp16) | out ULP (bf16) |
|---|---|---|---|---|---|---|---|
| eager | 1.703 | 136.2 | 5.803 | 119.9 | 1.66 | 6.4 | 7.9 |
| **fused** | **1.145** | **202.6** | **3.355** | **207.4** | **1.31** | **6.1** | **5.5** |
| mxfp8 | 0.787 | 294.7 | 2.919 | 238.4 | 1.49 | 442.7 | 52.5 |

Three things this table settles:

1. **Fusion alone is worth 1.73x and 0.79x the memory.** The fused 16-bit path
   carries *no* quantization error -- it matches the eager path's ULP to within
   the run-to-run spread -- so everything here is layout and launch count.
2. **MXFP8's remaining edge over the fused path is 1.15x, not 2x.** The other
   1.73x was fusion that the 16-bit path simply did not have.
3. **MXFP8 costs 8.4x more error in fp16 than in bf16** (442.7 vs 52.5 ULP)
   while its speed is identical in both. ULP is relative to the storage dtype, so
   this is the measured form of the claim in `internal/mxfp8-vs-fp16-loss.md`:
   MXFP8 competes with bf16, and fp16's extra three mantissa bits are precisely
   what it gives up.

Note also that `peak GiB` puts fused **below** MXFP8 (1.31 vs 1.49). The fp8 path
pays for four packed weight copies and its `dpre` quantization planes; at this
expert count that outweighs holding activations in 8 bits.

## 4. The tiles come from a planner, not a constant

`moe/plan.py` scores every legal tile analytically from a `Device`; `moe/tune.py`
times the model's top 6 once per shape and caches. Shapes are bucketed to 4096
rows so a routing count that jitters does not retune. This is the section 8
discipline from [kernel-convention.md](kernel-convention.md), and it is not
optional here: `triton.autotune` on a varlen M cost 365 ms per step the last time
it was left on.

The grouped model differs from the dense GEMM model in three terms, and **all
three were wrong on the first pass**. Each was found by ranking the top 8 and
timing them, per section 2 of the convention.

| prediction | outcome | what the model was missing |
|---|---|---|
| `BLOCK_M=32` wins: least partial-tile waste at expert boundaries | rank 0 in three of four kernels, and **loses by 25-45%** | nothing priced shared-memory load intensity, so a tile with almost no MAC-per-byte scored top |
| DRAM binds GEMM2 and GEMM2-DGRAD | every candidate scored an identical 190.6 TF/s while measurement spanned **109 to 215** | the scatter-add was priced at `rows x n` when it lands in a `(T, dim)` buffer that stays in L2 -- a 12x over-count that made a non-binding term bind |
| `gemm2_dgrad` is MMA-bound like its siblings | candidates spanned **5.7 to 124 TF/s** with identical predictions | its epilogue holds ~2 extra accumulator-sized tiles live, so `acc_regs = accs*bm*bn` under-counted registers and the top-ranked tiles all spilled |

The fixes are `moe_intensity_sat` and `moe_cta_tax` on `Device`, `epilogue_bytes`
counting streamed DRAM only, and `GroupedShape.live_tiles`. After them the three
MMA-bound kernels predict within ±9% and the shortlist of 6 contains the measured
winner for all four. The `live_tiles` fix alone moved `gemm2_dgrad`'s best
candidate from 124 to 210 TF/s -- not by changing the kernel, but by no longer
recommending tiles that spill.

**What the model still cannot see**: it ties `num_warps=4` and `8` exactly, and
the true winner flips per kernel (w8 loses 1.45x in GEMM1-DGRAD, won 2.9x in
GEMM2-DGRAD before the register fix). Ties break toward fewer warps and the
shortlist timing settles it. That is the split the convention predicts -- the
tuner earns its keep exactly where the model cannot see register allocation.

## 5. Wiring

`fused_moe_experts(x, w_in, w_out, gate, token_of, order, offsets)` matches the
signature `MoEMLP._routed_mxfp8` uses, so it binds into `self._routed` beside the
other paths without a runtime branch.

`offsets` must be **truncated to the real experts**. Neither GEMM2 nor GEMM2's
DGRAD carries a `valid_rows` guard; correctness depends on a sentinel bucket's
rows falling outside every tile the grid resolves.
