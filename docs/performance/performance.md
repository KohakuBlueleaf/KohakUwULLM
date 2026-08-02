# Performance on RTX 5090 (sm_120)

Everything here is measured on this box (4x RTX 5090, torch 2.13+cu132, triton
3.7.1) by the scripts in `scripts/bench/`. Re-run them after any kernel change;
the conclusions below are only as current as the last run.

## The hardware fact that shapes everything

Consumer Blackwell is **sm_120**, not sm_100. It keeps the sm_80-era
`mma.sync` execution model and ~100 KB of shared memory per SM, and it has **no
TMEM**. That single omission decides the attention story:

**FlashAttention-4 does not run on this card.** FA4's whole design -- accumulating
in Tensor Memory, 2-CTA MMA pairing -- is built on TMEM, which is an sm_100
(B200/GB200) feature. FA4 is real, it is fast, and it is irrelevant here. The
practical ceiling on sm_120 is FA2-class.

That is fine, because PyTorch 2.13 ships an FA2 varlen kernel natively:
`torch.nn.attention.varlen.varlen_attn`, with `cu_seqlens`, a trainable
backward, GQA and sliding windows. No third-party `flash-attn` build, no
sm_120 wheel hunt.

## fp16 vs fp32 accumulation

`scripts/bench/kernel/hgemm_acc.py`, M=N=4096:

| dtype | accumulate | split-K | TFLOP/s @ K=16384 | rel. error vs fp64 |
|---|---|---|---|---|
| bf16 | fp32 (cuBLAS) | - | 237 | 2.87e-3 |
| fp16 | fp32 (cuBLAS) | - | 232 | 3.59e-4 |
| fp16 | fp32 (Triton) | - | 210 | 3.59e-4 |
| fp16 | **fp16** (Triton) | 1 | **325** | 4.71e-3 |
| fp16 | fp16 (Triton) | 4 | 308 | 2.37e-3 |
| fp16 | fp16 (Triton) | 8 | 267 | 1.69e-3 |

NVIDIA rate-limits fp32 *accumulation* on GeForce tensor cores; the fp16
accumulator runs at full rate. On sm_120 that is worth **1.55x over Triton
fp32-acc and 1.40x over cuBLAS**.

The catch is that fp16-accumulate error grows with reduction depth: at K=16384 it
is *worse* than bf16. Split-K fixes it -- slice the reduction, accumulate each
slice in fp16 over a shorter run, combine slices in fp32. At split-K=8 the error
(1.69e-3) beats bf16 (2.87e-3) while still running 13% faster than cuBLAS.

**Where we use it:** only the MoE grouped GEMM, where we own the kernel and an
expert's `K` is the model width (~1k, not 16k) so the reduction is naturally
short. Measured there at 20 ULP / 1.32e-3 relative error -- still better than
bf16's 1.66e-3. **Never in a backward:** gradient magnitudes have far less
predictable range than activations, and fp16 overflows at 65504.

Plain linear layers keep cuBLAS: Triton's fp32-acc path is consistently *slower*
than cuBLAS (210 vs 237), so there is nothing to gain by taking the kernel over.

## MXFP8 end to end

Measured 2026-07-31 on 4x RTX 5090 under 4-stage pipelining, 262144 tokens/step,
micro 8192, both arms of every pair on the same cards
(`scripts/bench/e2e/e2e_kohaku_mxfp8.sh`, results in `out/bench/train/kohaku_e2e/`).

| rung | bf16 tok/s | +MXFP8 | speedup |
|---|---|---|---|
| Kohaku-200M | 267,456 | 289,096 | 1.081x |
| Kohaku-500M | 129,956 | 147,979 | 1.139x |
| Kohaku-1B | 79,273 | 96,807 | 1.221x |
| Kohaku-1.5B (ckpt) | 46,090 | 56,504 | 1.226x |
| Kohaku-MoE-1B | 159,907 | 245,108 | 1.533x |
| Kohaku-MoE-2B | 105,788 | 175,667 | 1.661x |
| Kohaku-MoE-3B (ckpt) | 63,568 | 101,863 | 1.602x |
| Kohaku-MoE-5B (ckpt) | 41,504 | 78,639 | 1.895x |
| Kohaku-MoE-8B (ckpt) | 27,848 | 64,266 | **2.308x** |

Dense plateaus at ~1.22x and does not move with gradient checkpointing -- the recompute
forward is matmul-heavy, which favours fp8, but it also re-quantizes the activations,
which does not, and the two scale together. Sparse keeps climbing because the fused
expert path replaces more of the step as the experts grow.

**Where the speed came from** (`scripts/bench/moe/kernel_ladder.py`), measured as the
sequence actually built rather than naive-versus-final:

    eager experts -> grouped bf16    5.7x    already in the shipped baseline
    grouped bf16  -> grouped fp8     1.49x   the dtype
    grouped fp8   -> fully fused     1.65x   the fusion

The bf16 path is *already grouped*, so every end-to-end ratio above measures only the
last two rungs. Quoting the 14-20x eager-to-fused figure as the fp8 speedup would be
wrong by about 6x.

Correctness, memory, and the open questions are in [mxfp8.md](../internals/mxfp8.md).

## Pipelining vs DDP on this box

Pipelining wins, and the reason is the interconnect rather than the algorithm.
`nvidia-smi topo -m` reports `NODE` for every GPU pair -- no NVLink, every transfer
across the PCIe host bridge. A ring all-reduce of the whole gradient is the access
pattern that fabric handles worst, and it is a barrier: the slowest leg sets the step.
Pipeline boundary sends are 12-29 MB point-to-point between adjacent stages and overlap
with compute under 1F1B.

It shows up as variance rather than only as mean. DDP rows measure 4-16% step spread
where pipeline rows sit at 0.1-0.6%, and MXFP8 makes DDP *worse* precisely because it
works: faster compute shrinks the step, so the collective becomes a larger share of it.

Note that the sweep's DDP arm runs with gradient checkpointing and the dense pipeline
arm does not, so the two are not a clean throughput comparison -- only the mxfp8-vs-bf16
ratios within each strategy are.

## Attention

`scripts/bench/kernel/attention.py`, dim 1280, 20 heads x 64, bf16:

- **varlen and padded SDPA are the same speed at the same shape** -- both land on
  a flash kernel. The win from `varlen` is not the kernel, it is that the shape
  is smaller because nothing is padded.
- **Packing is worth up to 6.3x.** At the length spread rendered TIPO samples
  actually have (`pad_frac=0.8204` in the equivalent padded batch), varlen
  fwd+bwd is 6.0-6.3x faster than padded and forward alone is 5.4-5.8x. Lower
  spreads give less: 3.1-3.8x at `pad_frac=0.69`, 2.8-3.2x at 0.65. From
  `out/bench/kernel/attention/attention.json`, `sweep="ragged"`.
- **GQA buys cache, not throughput.** 1 KV head against 20 is 256 vs 5120 bytes
  of KV per token -- exactly 20x -- but fwd+bwd measures **167.7 against 178.9
  TFLOP/s**, so the 20-head arm is the *faster* one by 6%. `kv_heads=4` (165.8
  TFLOP/s) is the default for the cache, and it costs a little throughput rather
  than gaining any. Earlier revisions of this page reported "68 vs 49 TFLOP/s"
  with MQA ahead; no artifact in the tree contains those numbers and the sweep
  runs the other way.
- **Sliding window at 16k context, packed: 512 -> 3.4x, 1024 -> 2.7x, 2048 ->
  2.0x** forward (3.5x / 3.0x / 2.3x fwd+bwd), against the 3.854 ms global
  baseline. At 4k context the same windows are worth 1.13x or less, so
  interleaving (`global_layer_every`) only pays past ~4k. The window sweep runs
  512/1024/2048 at 4k and 16k; there is **no window-256 and no 8k row**, so the
  "256 at 8k context: 2.40x" this page used to quote has no artifact behind it.
- **A window on the padded layout falls off the fused kernel.** At 4k, padded
  windowed rows measure 10.0 ms against 0.579 ms global -- **17x slower** -- and
  the padded 16k windowed cells do not complete at all. The window flag is only
  cheap on the packed path.
- **The four 16-bit backends agree to 5.6e-3 vs fp64** (varlen, triton, sdpa and
  flex all land at 5.57e-3, i.e. bf16 rounding and no more). `mxfp8` is the fifth
  backend and is **3.15e-2**, 5.7x looser -- see
  [../internals/mxfp8-attention.md](../internals/mxfp8-attention.md).

> Every `fwd+bwd` row in `attention.json` predates the gradient-reset fix in
> `step_fn`, so those arms timed an accumulate rather than a write. The bias is
> toward *understating* the packed-vs-padded ratio -- a fixed per-iteration tax
> is a larger share of the faster arm -- but the figure needs a re-run before the
> fwd+bwd columns are quoted as exact. The `fwd` columns are unaffected.

FlexAttention is ~20% slower than varlen for plain causal work and **must be
`torch.compile`d** -- uncompiled it materializes the score matrix and OOMs
instantly at 16k tokens. It earns its place only for masks the flash kernel has
no flag for.

## Module kernels

`scripts/bench/kernel/kernels.py`, dim 1280. Figures below are fwd+bwd, bf16,
from `out/bench_old/kernel/kernels/kernels.json` — **there is no
`out/bench/kernel/` copy, so this table has not been regenerated for the current
tree.**

| kernel | verdict |
|---|---|
| RMSNorm | ATen wins at small token counts (0.233 vs 0.461 ms at 2k); Triton wins at large (1.435 vs 1.765 ms at 131k). Default stays `rmsnorm` (ATen); `rmsnorm_triton` is there for the large-token regime. The crossover is between 2k and 131k; the sweep does not sample finely enough to place it, so the "~32k" this page used to give is an interpolation, not a measurement. |
| SwiGLU | Triton ties `torch.compile`, both beat eager at scale (5.33 and 5.18 vs 6.66 ms at 131k). Triton is also *more accurate* than eager -- **0.50 vs 0.97 ULP**, because it reduces in fp32. (An earlier revision gave eager as 1.86 ULP; the artifact says 0.97.) |
| MoE grouped GEMM | **13.6-26.7x over a loop of per-expert GEMMs**, growing with expert count: 7.5-9.1x at 32 experts, 19.2x at 64, 26.7x at 96. Accuracy is bit-identical to the loop at every point (2.41-2.89 ULP for both). This is the single biggest kernel win in the repo. The sweep runs 32/48/64/96 experts -- there is **no 128-expert row**, and the "14x at 128 experts, 98 vs 7 TFLOP/s, 2.23 ULP" this page used to quote matches nothing in the artifact. |

## The LM head is a memory problem, not a speed problem

At vocab 65536, 16384 tokens, bf16:

| path | time | peak memory | rel. error |
|---|---|---|---|
| naive (materialize logits) | 58.5 ms | 12.61 GiB | 5.1e-7 |
| `linear_cross_entropy(options=None)` | 41.5 ms | 6.61 GiB | 6.0e-6 |
| `linear_cross_entropy(options=LinearCrossEntropyOptions())` | 77.6 ms | **0.93 GiB** | 1.2e-6 |

From `out/bench_old/kernel/kernels/kernels.json`, `kernel="head"`, bf16. The
peak-memory column is **measured**, and it is lower than the analytic 16.5 / 8.5
/ 0.83 GiB this page used to print, because the allocator overlaps the logits
with their gradient rather than holding both at full size. Note also that the
chunked path is the **most accurate** of the three, not the least: 1.2e-6
against the reference path's 6.0e-6.

Two traps, both easy to hit:

1. **`options=None` is the reference path and still materializes.** The chunked
   implementation only engages when an explicit `LinearCrossEntropyOptions()` is
   passed. Passing nothing gets you a 7x memory regression that looks like it
   is using the new API.
2. **The returned scalar carries the input dtype.** At bf16 and 16k tokens the
   internal `mean` reduction is ~6% off the true value -- fine as a gradient,
   useless as a logged number or as a token-weighted denominator. `LMHead` always
   requests `reduction="none"` and reduces in fp32.

The chunked path costs ~1.9x the time of the reference path and saves 13.6x the
memory against the naive one. On a 32 GB card that trade is not close: 12.6 GiB
of logits decides the batch size, and batch size is worth more than 36 ms.

## MFU, and why every figure reported before 2026-07-30 is wrong

Two independent errors, in opposite directions. Any MFU number recorded before
`kohakuwullm/models/flops.py` existed should be re-measured rather than adjusted.

1. **Attention was not counted at all.** The model was `6 * active_parameters`,
   which covers the GEMMs whose cost is per token and none of the score/AV
   matmuls, whose cost is per *(query, key) pair*. Every dense preset therefore
   under-reported MFU.
2. **The embedding was counted as a GEMM.** `6 * active_parameters` charges the
   lookup table a matmul nobody runs. Tied, this is harmless -- the one
   `vocab x dim` weight really is used once, by the head. **Untied it invents a
   second `vocab x dim` GEMM per token**, a fixed `6 * V * D` per token, and
   over-reports MFU. `MoE-8B-A1B` is untied, and the pipeline split unties any
   preset it separates the embedding and head across.

   `tie_embeddings` now defaults to **False** and every Kohaku rung is untied, so
   the untied row below is the general case rather than the exception it was when
   these were measured. The `tied` column records the state at measurement time.

How much, per token at vocab 65536 (`old / corrected`, so `+` means the old
number was too high):

| preset | tied | ctx 512 | ctx 2048 | ctx 8192 |
|---|---|---|---|---|
| `Nano-500M` | yes | -3.0% | -10.9% | -32.9% |
| `Nano-1B` | yes | -2.4% | -8.8% | -27.9% |
| `MoE-3B-A500M` | yes | -3.3% | -12.1% | -35.4% |
| `MoE-8B-A1B` | **no** | **+7.1%** | **-0.0%** | -21.0% |

The 8B row is a coincidence worth knowing about: its fixed embedding over-count
is `6 * V * D` = 0.604 GFLOP/token, and its attention term happens to reach
exactly that at ctx 2048. So the two errors cancel there, over-report at shorter
documents and under-report at longer ones -- which is the worst possible shape for
a bug, since it looks correct precisely where it was most often measured.

Context length matters this much because attention is the only term that depends
on it, so **the correct denominator is the batch's own document lengths, not a
nominal context.** Rendered TIPO samples run 50-600 tokens against a 2048
context; charging them 2048 would triple their attention term.
`FlopCounter.batch_flops` reads `cu_seqlens` (or the padded `(B, S)` shape, where
padding *is* computed on and so *is* charged) and applies the closed forms
`L(L+1)/2` full-causal and `Lw - w(w-1)/2` windowed.

Two numbers are logged, not one. `perf/mfu` is model FLOPs -- what the
architecture owes. `perf/hfu` adds the second forward that gradient
checkpointing runs through the blocks; the gap between them is what recompute
costs. Neither is clamped: `PEAK_TFLOPS` is the **fp32-accumulate** ceiling
(270 TFLOP/s here), and per the table above fp16 accumulation reaches 325, so a
kernel accumulating in fp16 can legitimately report MFU above 1.0. A clamp
would have hidden the earlier peak-rate bug, where 209.5 TFLOP/s -- below what a
plain cuBLAS bf16 GEMM achieves -- put MFU over 100% for trivial code.

## What to set

```python
PRECISION = "bf16-mixed"                        # no scaler, no overflow babysitting
COMPILE = {"mode": "module", "dynamic": True}   # varlen token counts vary per step
OPTIMIZER = "muon"                              # 1.51x dense / 1.22x MoE on the step
OPTIMIZER_KWARGS = {"muon_lr": 0.02}
ARCH_OVERRIDES = {"qk_norm": True, "z_loss_weight": 1e-4, "mxfp8": True}
```

`dynamic=True` is not optional for varlen training: every step packs a different
total token count, and a static graph re-specializes (and recompiles) on each new
length.

`mxfp8=True` is worth 1.08x at `Kohaku-200M` and 2.31x at `Kohaku-MoE-8B` per the
table above, at a loss gap inside run-to-run scatter -- but read
[mxfp8.md](../internals/mxfp8.md) before enabling it on a run you care about, because the
memory story is the opposite sign on dense and sparse and the optimal micro-batch
differs between the two arms.

`GRAD_CKPT=True` costs ~30% throughput and buys back most of the activation
memory. Needed from `Kohaku-1.5B` and `Kohaku-MoE-3B` up; not for the dense rungs
below that.
