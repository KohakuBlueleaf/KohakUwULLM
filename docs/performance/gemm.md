# How to write a fast GEMM on consumer Blackwell

A matrix multiply is easy to write and very hard to make fast. Three loops give
you a correct answer in ten minutes. The last 30% of the hardware takes months.

This document explains where the speed comes from. It uses the RTX 5090 as the
target, because that is the card this project trains on. The card reports itself
as `sm_120`. Most published Blackwell advice is for `sm_100`, which is a
different machine. The differences matter, so this document names them.

Kernel-level rates in this document come from `out/bench_old/kernel/`, unless the
text says otherwise. Hardware latencies come from published characterization
work, which is cited at the end.

**The ISA ceilings are the exception, and their harness is not in this repo.**
Every `mma.sync` peak below -- 276.8, 551.0, 551.3, 1096.7 TFLOP/s and the L2 and
DRAM rates -- was produced by standalone CUDA microbenchmarks (`mma_peak.cu`,
`l2_bw.cu`) that were written in a scratch directory and never checked in. There
is no `scratchpad/` in the tree and no JSON records them, so they cannot be
re-derived from this repository. They are quoted here as prior measurements on
this card; rebuild the microbenchmarks before treating any of them as fresh.

**The planner in `kernels/gemm/` is bench-only today.** `plan`, `TunedGemm`,
`StreamKGemm` and `RTX_5090` have exactly one consumer in the tree --
`scripts/bench/kernel/gemm_plan.py`. No training path calls any of them: the
GEMMs a step actually runs are cuBLAS through `torch`, the Triton grouped GEMM
in `kernels/moe/`, and `_scaled_mm` on the MXFP8 path. `Device.query`,
`Device.from_json` and `Device.to_json` have no callers at all, including from
the bench. Read what follows as a design record and a measurement harness, not
as a description of what a training step executes.

---

## 0. The two numbers that bound everything

A GEMM does `2*M*N*K` floating point operations. It moves at least
`M*K + K*N + M*N` elements. The ratio of these two numbers is the arithmetic
intensity. It tells you which limit you are close to.

The RTX 5090 gives us these ceilings. Every one is measured on this card, by
`mma_peak.cu` for the instruction rates and `l2_bw.cu` for the memory rates.

| resource | value | how |
|---|---|---|
| DRAM read | 1830 GB/s | streaming read, working set far above L2 |
| L2 read | 13300 GB/s | 32 MB working set, L1 bypassed with `ld.global.cg` |
| bf16 or fp16 matmul, fp32 accumulate | **276.8 TFLOP/s** | back to back `mma.sync`, operands in registers |
| fp16 matmul, fp16 accumulate | 551.0 TFLOP/s | same |
| e4m3 matmul, fp32 accumulate | 551.3 TFLOP/s | same |
| fp32 FMA on the CUDA cores | 132.9 TFLOP/s | same |

Sustained SM clock under a tensor-only load is 3.16 GHz. Divide the rates by
`170 SMs * 3.16 GHz` and the per-SM numbers come out as exact powers of two:
512 FLOP/cycle for an fp32 accumulator, 1024 for an fp16 accumulator or for
fp8, and 128 FMA lanes.

**276.8 TFLOP/s is the number to divide by for a bf16 training GEMM**, because
training accumulates in fp32. The 420 TFLOP/s figure that appears in vendor
material is the fp16-accumulate rate at stock boost clock, which is a different
instruction and not one a trainer may use. An earlier revision of this document
used it as the bf16 ceiling and so reported every kernel at roughly 60% of peak
when the true figure was about 90%.

### Two ways to get this number wrong, both of which happened here

**Do not take a rate from a spec sheet.** The widely quoted 420 TFLOP/s for this
card is the fp16-accumulate rate at stock boost. It is a different instruction
and a trainer may not use it.

**Do not take a rate from a peak-finder without reading its disassembly.** The
`mmapeak` tool on this machine reports 409.2 TFLOP/s for `bf16bf16f32`. Its
kernel is built on `wmma::fragment<16,16,16>` and it credits `2*M*N*K` per
`mma_sync`, which is 8192 FLOP. Disassembled, that kernel contains exactly 8192
`HMMA.16816` for its 8192 loop iterations, so it issues **one** `m16n8k16` per
call and does 4096 FLOP. Its bf16 figure is therefore exactly 2x too high, and
the honest reading of it is 204.6 TFLOP/s. It is lower than our 276.8 because
its loop is a single serial accumulator chain with a `__syncwarp()` per
iteration, so it measures dependent-issue latency, not throughput.

The same 2x applies to every shape it reports as `16_16_16` or `32_8_16`,
including `f16f16f16` and `s8s8s32`. Its `16_8_32` shapes match the hardware
instruction and **are** counted correctly.

### The fp8 rates, and why the block scaled one is not the plain one

From the same disassembly, three different SASS opcodes appear for fp8:

| mmapeak kernel | SASS | reported | counted correctly |
|---|---|---|---|
| `f8f8f32_16_8_32` | `QMMA.16832.F32.E4M3.E4M3` | 408.9 | yes |
| `f8f8f16_16_8_32` | `QMMA.16832.F16.E4M3.E4M3` | 817.9 | yes |
| `mxf8mxf8f32_16_8_32` | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | 816.8 | yes |

**The block scaled instruction runs at twice the plain one and keeps that rate
with an fp32 accumulator.** Plain e4m3 into fp32 is half rate, exactly as bf16
into fp32 is; the `QMMA.SF` form is not. So MXFP8's advantage over bf16 is not
2x from K depth alone, it is closer to 4x, and the 693 TFLOP/s recorded for the
vendor kernel in section 4 sits comfortably under that ceiling rather than
above it.

Measured here with four independent accumulator chains, which is the honest
throughput rather than mmapeak's dependent-issue latency:

| instruction | TFLOP/s | vs bf16 into fp32 |
|---|---|---|
| `m16n8k16 f32.bf16.bf16.f32` | 279.3 | 1.00x |
| `m16n8k32 f32.e4m3.e4m3.f32` | 555.8 | 1.99x |
| `m16n8k32 kind::mxf8f6f4.block_scale ... ue8m0` | **1096.7** | **3.93x** |

**The block scaled instruction is twice the plain e4m3 one and roughly four
times bf16.** It keeps full rate with an fp32 accumulator, which plain e4m3 does
not. Note that it needs `-gencode arch=compute_120a,code=sm_120a`; a plain
`-arch=sm_120` rejects `.kind::mxf8f6f4` outright.

Against that ceiling, every MXFP8 path we have is far from the machine:

| path | best | of 1096.7 |
|---|---|---|
| vendor `_scaled_mm`, pre-quantized | 696.9 | **63.5%** |
| our Triton pre-quantized | 596.5 | 54.4% |
| our Stream-K port | 528.8 | 48.2% |

> **Unsourced, and it disagrees with section 4.** No artifact in `out/` holds
> these three figures. The archived sweep,
> `out/bench_old/kernel/mxfp8/mxfp8.json`, records four shapes with `impl` in
> `bf16 / pq / fused / vendor` and no Stream-K arm at all; its best vendor rate is
> **693.2** and its best `pq` rate is **568.2**, which is what
> [section 4](#what-it-buys-measured) quotes. Prefer the section 4 table, which
> the JSON backs row for row. Treat 696.9 / 596.5 / 528.8 as a later run whose
> output was not kept, and re-measure before quoting them.

### Stream-K does not transfer to MXFP8

The scheduling fix that is worth 1.06x on bf16 measures **0.65 to 0.84x of the
vendor kernel** here, and is slower than our own pre-quantized kernel on every
shape tried. The reason is in the ceiling above. At four times the arithmetic
rate, a tile's compute time is a quarter of what it was, so every fixed cost
inflates fourfold against it:

- The fixup atomics move the same bytes against a quarter of the compute.
- L2 traffic per tile is unchanged, so its utilisation at a `128 x 128` tile
  rises from about 32% to about 65%.

**Wave quantization stops being the dominant term when the arithmetic gets
cheap enough.** The bf16 cost model's weights do not carry over, and an MXFP8
planner needs its own calibration with larger tiles to cut L2 traffic. Replacing
`_scaled_mm` is not yet possible: the gap is 1.17x at best and 1.5x at worst.

Take one real shape from our benchmark, `16384 x 5120 x 1280`. It needs 215
GFLOP. It moves 223 MB if the reuse is perfect. At the ceilings above, the
arithmetic needs 896 us and the memory needs 125 us.

The arithmetic is 7 times slower than the memory. This shape is compute bound.
That single fact decides which optimizations are worth the effort. It is the
first thing to compute for any new shape, and it takes one line of arithmetic.

---

## 1. Memory: L1, L2 and VRAM

### Every operand is read more than once

Split the output into tiles of `BM x BN`. Each CTA reads a full row panel of A
and a full column panel of B. Count the reads:

- A is read `N / BN` times.
- B is read `M / BM` times.

Neither operand is read once. For `16384 x 5120 x 1280` with 128-wide tiles, A is
read 40 times and B is read 128 times. If all of those reads went to DRAM, the
traffic would be 5.6 GB instead of 223 MB. The kernel would be 25 times slower.

The cache is what prevents this. The whole job of a GEMM memory design is to make
sure that the re-reads hit a cache and not DRAM.

### The 5090 has an unusually large L2, and it hides mistakes

| level | size | latency |
|---|---|---|
| L1 and shared memory | 99 KB per CTA | 33 cycles |
| L2, near slice | 96 MB total | 79 cycles |
| L2, far slice | same 96 MB | 180 cycles |
| DRAM (GDDR7) | 32 GB | 372 cycles |

96 MB of L2 is very large. A datacenter A100 has 40 MB. This changes the advice
you find in most tutorials.

The standard fix for cache behaviour is to reorder the CTAs. A GPU launches CTAs
in a roughly linear order. If consecutive CTAs walk along one axis of the output,
they all share one operand panel and each touches a different panel of the other
operand. The usual fix is to group the CTAs into blocks of `GROUP_M` rows. This
is the `group_m` trick from the Triton tutorial. It keeps the live working set
small.

Now do the arithmetic for our card. Take `16384 x 5120 x 1280` with 128-wide
tiles. The grid is 128 by 40, which is 5120 CTAs. The card runs 170 of them at
once. With no reordering at all, those 170 CTAs touch about 28 MB of A and one
164 KB panel of B.

28 MB fits in 96 MB. The naive order is already cache resident.

This explains a result that looks strange at first. Our naive Triton kernel
reaches 90% of cuBLAS on bf16, and it has no reordering of any kind. The large L2
absorbs the mistake. On a card with 40 MB of L2, the same kernel would lose much
more.

The lesson is not that reordering is useless. The lesson is to size the working
set before you spend a week on it. Compute `live_CTAs * panel_bytes` and compare
it to L2. If it fits, look somewhere else first.

### The L2 on this card is not uniform

There is a second effect that most kernels ignore. The L2 is split into slices.
A slice in the same GPC as the SM answers in 79 cycles. A slice in a different
GPC answers in 180 cycles. This is a 2.3 times penalty for the same cache.

So a hit rate of 100% can still be slow. The reorder should try to keep each
CTA's data in its own GPC slice, not only in L2 as a whole. Public Triton kernels
do not do this today. It is an open opportunity on this hardware.

### Shared memory is the real constraint

Consumer Blackwell gives 99 KB of shared memory per CTA. The datacenter part
gives 228 KB. This is the single most important number when you port a kernel.

Shared memory holds the pipeline. A GEMM loads tile `k+2` while it multiplies
tile `k`. Each stage of that pipeline costs shared memory. More stages hide more
latency. With DRAM at 372 cycles, you need several stages to hide one load.

Work out the budget. A `128 x 128 x 64` tile in fp8 costs about 16 KB per operand
per stage. Three stages of A and B is 96 KB. That fits in 99 KB with nothing to
spare.

Now do the same in bf16. Each stage costs twice as much. Three stages need 192
KB, which does not fit. You get one or two stages, and the pipeline stalls.

This gives a result that is easy to miss. **A smaller data type does not only
save bandwidth. It doubles the pipeline depth you can afford.** On a card with
99 KB of shared memory, that second effect can be larger than the first.

---

## 2. Tensor cores and CUDA cores

### The tensor core issue rate depends on the accumulator, not on the input

Measured on this card, one `mma.sync` occupies its sub-core for:

| instruction | cycles | FLOP/cycle/SM |
|---|---|---|
| `m16n8k16.f32.bf16.bf16.f32` | 32 | 512 |
| `m16n8k16.f32.f16.f16.f32` | 32 | 512 |
| `m16n8k16.f16.f16.f16.f16` | 16 | 1024 |
| `m16n8k32.f32.e4m3.e4m3.f32` | 32 | 1024 |

Two separate effects hide behind "low precision is faster", and they compose:

- **An fp16 accumulator issues twice as fast as an fp32 one**, for the same
  input type and the same instruction shape. This is a GeForce restriction. A
  trainer cannot spend it, because a 16k-deep fp16 reduction is not accurate
  enough for a loss curve.
- **A narrower input covers more K per instruction.** e4m3 keeps the 32 cycle
  issue rate but contracts 32 elements instead of 16, so it delivers exactly
  2x. Nothing about the pipeline is involved.

The practical result is that **precision is a bandwidth and capacity decision,
not a compute decision**. You pick fp8 to move fewer bytes and to fit more
pipeline stages, and the FLOP rate follows.

### Consumer Blackwell uses a different instruction from datacenter Blackwell

This is the most expensive thing to learn late.

| feature | sm_100 (B200) | sm_120 (RTX 50) |
|---|---|---|
| MMA instruction | `tcgen05.mma` | `mma.sync` |
| Tensor Memory (TMEM) | 256 KB per CTA | none |
| `wgmma` | yes | no |
| two-SM cooperative MMA | yes | no |
| TMA | yes, with multicast | yes, single CTA only |
| mbarrier, clusters, DSMEM | yes | yes |
| FP8, FP6, FP4 block scaling | yes | yes |
| shared memory per CTA | 228 KB | 99 KB |

An `sm_100` GEMM kernel does not run slowly on `sm_120`. It does not run at all.
Any tutorial that mentions TMEM, `tcgen05` or warp group MMA describes a machine
we do not have.

`mma.sync` keeps its accumulator in registers. `tcgen05.mma` keeps it in a
separate memory. Because registers are small, the `mma.sync` instruction tile is
small. One warp cannot fill the tensor core, so several warps issue MMA
together. Note that 8 warps per CTA is a common choice and not a rule: section 5b
measures 4 warps winning on square shapes.

### Warp count decides whether the card hides latency

Characterization of this architecture shows a sharp threshold. With 5 or more
warps per sub-core, latency hiding improves by about 6 times. With 4 or fewer
warps, there is almost no hiding at all.

**This threshold does not govern a GEMM.** Section 5b measures the fastest
variants running at one and two warps per sub-core, because `num_stages`
pipelining hides the latency inside the warp. Read the rest of this section as
arithmetic you should be able to do, not as a target to hit.

This threshold is easy to miss, and it interacts with register pressure.

Take a `128 x 128` output tile with 4 warps. That is 128 threads. Each thread
holds `128 * 128 / 128 = 128` accumulator values. In fp32 that is **128 registers
per thread**, before any address, operand or scale register.

The register file holds 65536 registers per SM, and a thread may use at most 255.
To reach 5 warps per sub-core you need 20 warps resident per SM, which is 640
threads, which allows only `65536 / 640` or about **102 registers per thread**.
At 128 accumulators plus operands and addresses we are well past that, so the
kernel lands at two or three warps per sub-core. It is the wrong side of the
threshold.

[../internals/kernel-dev.md](../internals/kernel-dev.md) derives this budget in
full, and gives the table for other tile shapes.

The same tile with 8 warps needs 64 accumulator registers per thread. The same
tile with `BLOCK_N = 64` and 8 warps needs 32.

**Accumulator registers per thread is `BM * BN / (32 * num_warps)`.** Compute it
before you tune anything else. If it is above about 64, the occupancy is already
lost.

### The CUDA cores are not idle, and you can use them for free

The SM can issue a tensor core instruction and four FP32 FMA instructions in the
same cycle at no cost. Eight integer adds cost one extra cycle.

So a small amount of elementwise arithmetic inside the main loop is free. This is
useful. It means a scale multiply, a bias add or a simple cast can hide inside
the MMA stream.

There is an important exception. A reduction across lanes is not an FMA. It needs
shuffles. Shuffles do not co-issue with the tensor core. So "cheap elementwise
work is free" is true, and "cheap reductions are free" is false. Section 3 shows
why that difference decides a real design question.

### Wave quantization wastes whole SMs

The 5090 has 170 SMs. A grid of 256 CTAs runs as 1.5 waves. The second wave uses
86 SMs and leaves 84 idle. The kernel takes two waves of time and does 1.5 waves
of work.

A reference blockscaled GEMM for this architecture family reached only about 60%
of peak, and wave quantization was the stated cause.

There are two standard answers.

- A persistent kernel launches exactly one CTA per SM. Each CTA then loops over
  output tiles. There is no partial wave.
- Stream-K splits the K loop across CTAs so that every SM finishes at the same
  time. It costs a reduction at the end.

On a 170 SM card with tile counts that are rarely a multiple of 170, this is one
of the larger single wins available.

---

## 3. Fusion: what to fuse, and where

Fusion means doing another operation inside the GEMM kernel. The question is not
"can we", it is "which side".

### The rule: fusion cost multiplies by the reuse count

The main loop reads each operand many times. Anything you compute inside the main
loop is computed once per read, not once per element.

This gives a simple test. Take an operand with `S` elements and reuse `R`.

- A separate cast pass moves about `3*S` bytes once. It reads bf16, writes fp8
  and writes the scales. Then the GEMM reads `S` bytes `R` times.
- A fused cast moves `2*S` bytes on every read, because the operand stays bf16 in
  memory. That is `2*R*S`.

Fusion wins when `2R < 3 + R`, which means **`R < 3`**.

Now recall that `R_A = N / BN` and `R_B = M / BM`. For our shapes, `R_A` is 40
and `R_B` is 128. Both are far above 3. By this test, fusing either cast into the
GEMM is a loss.

### The cache changes the answer for one operand

The test above assumes all reads go to DRAM. They do not.

If the re-reads hit L2, then the fused version moves `2*S` bytes of DRAM traffic
once, and the separate version moves `3*S`. The fused version is now better on
DRAM at any reuse count. It pays only in L2 bandwidth and in repeated arithmetic.

The repeated arithmetic is small. For a 128 row panel of A against `N = 5120` and
`K = 4096`, the extra cast work is 21 million operations against 5.4 GFLOP of
matrix work. That is **0.4%**.

So on this card, with a 96 MB L2 and a 1 MB A panel, fusing the A cast looks
correct on paper. The operand with low reuse and cache residency is the one to
fuse. The operand with high reuse is the one to leave alone.

### Three costs that the byte count does not show

Our fused kernel measures 213 TFLOP/s. The same kernel with a separate cast pass
measures 389 TFLOP/s. Fusion made it 1.8 times slower. The byte count did not
predict this.

**First, shared memory.** To cast inside the kernel, the operand must arrive in
bf16. That doubles the bytes per pipeline stage. Section 1 showed that this
halves the pipeline depth. On a card with 99 KB of shared memory, that is the
whole latency hiding budget.

**Second, the reduction.** MXFP8 needs the maximum absolute value of each group
of 32 values along K before it can encode them. That is a reduction across lanes.
Section 2 showed that reductions do not co-issue with the tensor core. So this
work lands directly in the MMA issue path.

**Third, the scale register layout is fixed by the hardware.** The block scaled
MMA instruction requires the scale values in specific lanes. For the A operand,
threads 0 and 1 of each group of four supply the scales, and the other two hold
copies. For the B operand, only thread 0 supplies them, and three threads hold
copies.

That last point is a second reason to prefer fusing A over B, and it has nothing
to do with reuse. Fusing the B scales wastes four times the work per warp. Fusing
the A scales wastes two times. The hardware itself favours the same choice.

### The real blocker: the fast path cannot transform data

TMA and `cp.async` are copy engines. They move bytes from global memory to shared
memory without using the SM. They cannot change the data on the way.

So a fused cast cannot use the fast path. The data must go global memory, then
shared memory as bf16, then registers, then convert, then shared memory again as
fp8, then the tensor core. That is an extra round trip through shared memory and
an extra barrier. It breaks the asynchronous pipeline that the whole design
depends on.

A separate cast pass has no such problem. It is a pure streaming kernel. We
measure it at 1706 GB/s, which is 95% of DRAM peak.

**This is the general answer. Do not fuse work into the prologue of a GEMM. The
prologue is a copy engine, and a copy engine cannot compute.**

### Where fusion belongs instead

Fuse into the **epilogue of the kernel that produced the data**.

At the end of a normalization, an activation function or a previous GEMM, the
values are already in registers. A cast there is close to free. The next GEMM
then reads fp8 directly, and no separate pass exists at all.

This gives a clear rule for a training step:

| operand | when to cast | why |
|---|---|---|
| weight | once per optimizer step | it does not change between calls, so the cost divides by every GEMM that uses it |
| activation | in the epilogue of the kernel that made it | it is already in registers there |
| never | inside the consumer GEMM prologue | the copy engine cannot transform, and the cast repeats `R` times |

The epilogue is also the right place for the usual output fusions. Bias,
activation functions, residual adds and the output cast all belong there. The
output tile is already on chip, so these are close to free.

---

## 4. MXFP8 and NVFP4 linear layers

### What the format is

MXFP8 stores each value in 8 bits, as E4M3. It stores one shared exponent for
each group of 32 values along the contraction axis. NVFP4 stores each value in 4
bits as E2M1, with one scale for each group of 16.

The scale group runs along K, which is the axis the matrix multiply contracts.
This is convenient. If your K block is a multiple of the group size, every CTA
owns whole groups. No CTA needs a scale from another CTA.

### What it buys, measured

| path | rate | note |
|---|---|---|
| bf16 cuBLAS, fp32 accumulate | 240 TFLOP/s | the baseline |
| our Triton kernel, pre-quantized | 568 TFLOP/s | 82% of vendor |
| vendor `scaled_mm`, pre-quantized | 693 TFLOP/s | 2.9 times the baseline |
| vendor, plus both casts online | 516 to 585 TFLOP/s | 2.2 times the baseline |
| our fused kernel, casts inside | 213 TFLOP/s | slower than bf16 |

The accuracy cost is a relative error of 3.75% against 0.17% for bf16. In ULP
terms it is about 1.6 against 0.15.

End to end, the same change is worth 1.08 to 1.23 times on dense models and 1.53
to 2.31 times on sparse models. A kernel ratio of 2.9 does not become a step
ratio of 2.9. Most of a training step is not the GEMM.

### The NVFP4 instruction on this card

There is exactly one blockscaled FP4 instruction on `sm_12x`:

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

The shape is fixed at 16 by 8 by 64. You cannot tune it. You build a larger tile
from many of these. A CTA tile of `128 x 128 x 128` needs 8 warps arranged as 4
along M and 2 along N.

The CUTLASS helper functions for scale factors support a maximum of 128 in M and
N today. So `128 x 128` is both the sensible tile and the largest supported tile.

A public reference kernel of this shape reaches about 60% of peak, limited by
wave quantization. A tuned CUTLASS 4.5 NVFP4 kernel reaches 975 TFLOP/s on this
architecture, which is about 80% of peak. So the format works well here. The gap
is scheduling, not arithmetic.

Note that this applies to a plain GEMM. The **grouped** block scaled path in
CUTLASS is a different story on this card, and
[upstream-cutlass-findings.md](upstream-cutlass-findings.md) records where it
breaks.

### One thing to verify before trusting any Triton FP8 number

Triton exposes `tl.dot_scaled` for block scaled inputs. There is a reported issue
where this lowers to bf16 MMA on `sm_120` instead of the block scaled FP8
instruction. If that happens, the kernel dequantizes to bf16 and runs at bf16
speed, and it still gives the right answer. The bug is silent.

Our own measurement of 568 TFLOP/s is far above any bf16 rate, so our
pre-quantized path is almost certainly using the real instruction. That is
evidence, not proof. **Read the PTX and look for `mma.sync` with `block_scale`.**
It costs one command and it protects every FP8 number you report.

### MXFP8 or NVFP4

MXFP8 is the right default for training. The error is already 22 times bf16.
NVFP4 halves the bytes again and roughly doubles the peak rate, but the error
grows further, and the group size of 16 gives less room to absorb an outlier.

NVFP4 is attractive for inference weights, where you quantize once and measure
the result. For training, prove that MXFP8 is the bottleneck before you go
lower. Our own data says the bottleneck is elsewhere. Our FP8 kernel runs at 82%
of the vendor kernel, and our fused variant runs below bf16. There is more to win
by fixing those than by halving the format.

---

## 5. Why a naive Triton kernel loses to cuBLAS

A naive Triton GEMM is about 30 lines. It is correct. It reaches 90% of cuBLAS on
bf16 for a friendly shape on this card, and much less elsewhere. Here is what it
does not do.

### It does not compute its register budget

A `128 x 128` tile with 4 warps gives 128 accumulator registers per thread, which
lands at one warp per sub-core.

**Measured, this is worth about 1%** (section 5b), and 16 warps is worth nothing
at all. It reads like the largest defect and it is not. Compute the budget so you
know where you are, then sweep the tile, which is worth up to 15%.

### It does not order its CTAs

The naive kernel uses the raw program ID for both output axes. There is no
grouping. As section 1 showed, the 96 MB L2 on this card hides most of the cost,
so this is a smaller defect here than on other hardware. It is still real, and it
becomes large when the working set grows past L2.

### It masks every load, on every iteration

The inner loop applies a bounds mask to A and B on each step, even when K divides
evenly by the block size. Masked loads block the vectorized path. The fix is a
compile time flag for the aligned case, and an unmasked fast loop inside it.

### It does not tell the compiler what it knows

There are no hints that pointers are contiguous or that sizes are multiples of
the block. Without them the compiler must assume the general case and it emits
slower addressing.

### It uses one tile shape for every problem

A good tile for a large square matrix is a bad tile for a tall thin one. cuBLAS
picks from many kernels at run time. A fixed tile cannot match that.

Run time autotuning is not the answer either. We measured it as a 365 ms per step
tax, because the tuner re-benchmarks whenever a dimension changes. The answer is
an offline tuning table, keyed by shape, loaded at build time.

### It ignores wave quantization

A plain grid launches whatever number of CTAs the tiling implies. On 170 SMs that
usually leaves a partial wave. A persistent kernel or Stream-K removes it.

### It does not use the asynchronous engines

The naive kernel loads through the normal path. It does not use TMA, mbarrier or
a producer and consumer split. On this card that is a smaller loss than you would
expect, and section 6 explains why.

### What cuBLAS actually does that is hard to copy

The gap is not one trick. A recent public effort to write an MXFP8 GEMM by hand
started at 35% to 48% of cuBLAS, and it already used a persistent warp
specialized design with TMA. It needed ten more changes to reach 99%. The list
included static array indexing to stop register spills, worth 17%. It included a
deeper K block, worth 13 points. It included a Hilbert curve schedule for cache
locality, vectorized 256 bit stores through inline PTX, and a cache hint to stop
output stores from evicting live data.

None of those are ideas you would guess. Each one is worth a few percent. Ninety
percent of cuBLAS is not hard. The last ten percent is a hundred small things.

---

## 5b. Measured: what each change is actually worth

The sections above were written from first principles and from published
characterization. Then they were measured, on a free RTX 5090, one variant at a
time. Two of the predictions were wrong and the table is the correction.

bf16, fp32 accumulate, best Triton config per shape against cuBLAS:

| shape | cuBLAS | best Triton | ratio |
|---|---|---|---|
| 4096 x 4096 x 4096 | 235.8 | 224.8 | 95% |
| 16384 x 1280 x 5120 | 236.6 | 243.5 | **103%** |
| 16384 x 5120 x 1280 | 241.5 | 237.3 | 98% |
| 8192 x 1280 x 1280 | 229.2 | 220.9 | 96% |
| 4096 x 4096 x 16384 | 235.0 | 225.5 | 96% |
| 8192 x 8192 x 8192 | 240.3 | 243.4 | **101%** |

fp16 tracks it within a point. **A Triton GEMM reaches 95 to 104% of cuBLAS
here.** The naive kernel's 90% was not a Triton ceiling, it was a configuration.

Per change, from the same run:

| change | worth |
|---|---|
| group-M rasterization | **~0%**, and +2% only at 8192 cubed |
| 4 warps to 8 warps | **~1%** |
| `max_contiguous` / `multiple_of` hints | **+2.5%**, the most reliable single change |
| tile shape, chosen per shape | **up to +15%**, and it is what decides the result |

### The register cliff does not apply to a pipelined GEMM

This is the correction that matters. Compiled register counts, all with **zero
spills**:

| config | regs/thread | shared KiB | warps per sub-core |
|---|---|---|---|
| 128x128x64, 4 warps | 232 | 64 | 1 |
| 128x128x64, 8 warps | 162 | 64 | 2 |
| 128x64x64, 4 warps | 164 | 72 | 1 |

Doubling the warp count doubled warps per sub-core from 1 to 2 and bought 1%.
And `128x64x64` at **one warp per sub-core** is the *fastest* variant on square
shapes. The `>= 5 warps per sub-core` threshold is a latency-hiding rule for
generic code; a GEMM main loop hides its latency inside the warp through
`num_stages` software pipelining, so it does not need warp-level occupancy to do
it. Shared memory, not registers, is what caps residency here: 64 KiB per CTA
against a 99 KiB budget means one CTA per SM whatever the register count says.

Treat `regs_per_thread <= 102` as a rule for kernels that rely on warp-level
latency hiding, and **not** as a rule for a software-pipelined GEMM.

## 5c. Where a bf16 GEMM's time actually goes

The ceiling is 276.8 TFLOP/s (section 0). A bf16 GEMM's distance from it is the
product of three factors, each measurable on its own. All three come from one
fixed shape, `4096 x 4096 x 4096`, on an idle RTX 5090.

### Factor 1: the instruction mix, worth 4.2 points

`mma_ldsm.cu` runs the main loop with no global traffic and no epilogue: HMMA
fed from shared memory by `ldmatrix`, at a chosen ratio, with a chosen number of
`bar.sync` per iteration. The ratios come from counting the real kernel's SASS,
which is 32 HMMA, 12 LDSM and 2 BAR per k-iteration.

| mix per 32 HMMA | 1 CTA/SM | 2 CTA/SM |
|---|---|---|
| 12 LDSM, 0 BAR | 98.7% | 99.4% |
| 12 LDSM, 1 BAR | 95.2% | 97.2% |
| **12 LDSM, 2 BAR** | 93.5% | **95.8%** |
| 12 LDSM, 4 BAR | 91.3% | 94.8% |
| 0 LDSM, 2 BAR | 99.5% | 99.5% |

Read the last row against the third. **Barriers alone are free and `ldmatrix`
alone is free; only the combination costs anything.** A `bar.sync` makes every
warp rendezvous, which exposes the `ldmatrix` latency that warps were otherwise
hiding for each other. This is the tax a shared-memory pipeline pays for being
synchronous, and it is why the achievable target for this instruction mix is
265.2 TFLOP/s, not 276.8.

### Factor 2: wave quantization, worth up to 25 points

170 SMs and whole output tiles. With `BM = BN = 128` on this shape the grid is
1024 tiles, some SM must run `ceil(1024/170) = 7` of them, and the average is
6.02. That caps the kernel at `6.02/7 = 86.1%` before any other consideration.

Tile choice moves it, and the two effects fight:

| tile | wave efficiency | of its wave bound | %ISA |
|---|---|---|---|
| 256x128x32 | 75.3% | 92.8% | 69.9% |
| 128x128x32 | 86.1% | 95.7% | 82.4% |
| 128x64x64 | 92.7% | 92.3% | 85.6% |
| 64x128x64 | 92.7% | 94.3% | 87.3% |
| 64x64x64 | 96.4% | 84.5% | 81.5% |

A smaller tile balances better and computes worse. The product peaks near
`64 x 128`. **Stream-K decouples them**, and section 5d is what that buys.

### Factor 3: occupancy, worth 5 points

One CTA per SM leaves nothing to cover a tile's prologue and epilogue. Two CTAs
per SM cover each other. The register file decides which you get: at 8 warps a
`128 x 128` fp32 accumulator is 64 registers per thread, and two CTAs of 256
threads need the total at or under 128.

That single constraint is why `128 x 128` is the largest useful bf16 tile here,
and it is a register argument, not a shared memory one.

### What this replaces

An earlier revision of this section claimed bf16 was "shared memory limited
before it is tensor core limited", and put both our kernel and cuBLAS at 58% of
peak. Both statements were artefacts of dividing by the wrong ceiling. Deeper
pipelines were then measured and do nothing: at `BLOCK_K = 16`, going from 3 to
4 to 5 buffers moves the rate by less than 1%, and every one of them is slower
than `BLOCK_K = 32`. Shared memory capacity is not what limits a bf16 GEMM on
this card.

## 5d. Stream-K, and beating cuBLAS on a fixed shape

Wave quantization is the largest single term and it is a scheduling problem, so
it has a scheduling answer. Launch exactly `SMs * CTAs_per_SM` CTAs. Give each
one a whole number of output tiles, then split the leftover tiles along K so
every CTA finishes at the same moment. Split tiles accumulate into an fp32
scratch that a fixup pass casts into C.

At `4096^3`, bf16, order-controlled and best-of-N:

| kernel | TFLOP/s | %ISA | vs cuBLAS |
|---|---|---|---|
| plain `128x128x32` | 228.0 | 82.4% | 0.946 |
| plain `64x128x64` | 241.1 | 87.1% | 1.000 |
| cuBLAS | 241.1 | 87.1% | 1.000 |
| Stream-K, all CTAs split | 251.6 | 90.9% | 1.043 |
| **Stream-K, 85 CTAs split** | **255.9** | **92.5%** | **1.061** |

Two details carry most of it.

**Two CTAs per SM, bought with `maxnreg`.** The persistent kernel naturally
compiles to 160 registers, which fits one CTA. Capping it at 128 fits two and
costs 16 spilled registers, and the trade is worth about 1.4%.

**Cap how many CTAs take part in the K split.** The leftover here is 4 tiles.
Letting all 340 CTAs share them balances perfectly but makes 340 CTAs each
atomically add a `128 x 128` fp32 tile, which is 22 MB of atomic traffic to
finish 4 tiles. Letting too few share them rebuilds the imbalance. The cost is
`sk_total/s` of imbalance against `s * BM * BN * 4` of traffic, so the optimum
is near `s = sqrt(c * sk_total)`; `c` fits at about 14 on this card.

| CTAs splitting | 340 | 170 | 85 | 32 | 8 |
|---|---|---|---|---|---|
| TFLOP/s | 253.7 | 256.3 | **256.7** | 254.5 | 240.3 |

### Stream-K is a selection, not a default

Across six shapes and both dtypes it is a geometric mean of 1.004 against
cuBLAS, winning on eight of twelve points and losing badly on one.

| shape | wave eff | vs cuBLAS bf16 | vs cuBLAS fp16 |
|---|---|---|---|
| 4096 x 4096 x 4096 | 86.1% | **1.053** | **1.066** |
| 4096 x 4096 x 16384 | 86.1% | **1.054** | **1.053** |
| 16384 x 1280 x 5120 | 94.1% | 1.018 | 1.026 |
| 16384 x 5120 x 1280 | 97.2% | 1.020 | 1.007 |
| 8192 x 8192 x 8192 | 96.4% | 0.988 | 0.987 |
| 8192 x 1280 x 1280 | 94.1% | **0.896** | **0.897** |

The pattern is exactly what factor 2 predicts. Stream-K pays where wave
efficiency is worst and does nothing where it is already high.

The failure is instructive. `8192 x 1280 x 1280` gives 640 tiles against 340
CTAs, so `640 // 340` is 1 and the leftover is **300 of the 640 tiles**. Nearly
half the output goes through the atomic path, for about 67 MB of fixup traffic
against only 101 us of arithmetic. Both terms are known before launch, so the
choice is a calculation:

```
gain = (1 / wave_efficiency) / (1 + fixup_seconds / compute_seconds)
```

That predicts 0.79 for this shape and above 1.02 for the four that win, which is
the right call in all six cases. **This is the offline selection table that
section 5 says a fixed tile cannot replace**, and it is arithmetic on shape, not
a benchmark sweep.

Accuracy is unchanged: against an fp64 reference, cuBLAS, the plain kernel and
Stream-K all give a maximum relative error of 0.002869 and an RMS of 0.001662,
which is bf16 output rounding and nothing else.

## 6. Conclusion

A GEMM is a memory schedule with some arithmetic attached. The arithmetic is
fixed by the problem. Everything you control is about moving data.

The order of work that follows from this document:

1. **Measure your own ceiling before you divide by anything.** One microbenchmark
   of back to back `mma.sync` with register-resident operands. Every efficiency
   claim in this document changed when that number changed.
2. **Compute the roofline for your shape.** One line of arithmetic tells you
   whether to work on memory or on compute. Our shapes are 7 times compute bound,
   which rules out most bandwidth work immediately.
3. **Count the waves.** `(tiles/SMs) / ceil(tiles/SMs)` is a hard cap and on a
   170 SM card it is routinely 86%. It is the largest single term in section 5c
   and it costs one line to compute.
4. **Fix the waves with Stream-K, and decide per shape whether to.** Section 5d.
   Worth 1.06x against cuBLAS where wave efficiency is poor, 0.90x where the
   fixup traffic outweighs it, and the two are comparable before launch.
5. **Get two CTAs per SM.** It covers the prologue and epilogue that one CTA
   cannot. On this card that means the register total at or under 128, which
   caps the accumulator and therefore the tile.
6. **Sweep the tile shape.** Measured at up to 15%, shape-dependent. No one tile
   wins everywhere, which is why cuBLAS ships many kernels and why an offline
   table beats a constant.
7. **Check that your low precision instruction is the one you think it is.** Read
   the PTX. A silent fallback to bf16 gives correct answers at half speed. Ours
   emits `mma.sync...kind::mxf8f6f4.block_scale`, verified.
8. **Add the contiguity hints and an aligned-K fast path.** Worth 2.5%, cheap,
   and the most reliable change measured.
9. **Do not reach for warp count or rasterization first.** Measured at 1% and 0%
   respectively on this card. See section 5b.
10. **Put fusion in the epilogue, never in the prologue.** The load path is a copy
    engine. It cannot compute. Cast where the data is already in registers.
11. **Use asynchronous loads last, and expect nothing from warp specialization.**
    TMA does cut register pressure, measured at 126 registers with no spills
    against 128 with 16, but Triton then allocates twice the shared memory and
    the occupancy is lost again. Warp specialization measured 0.93x on this
    card: consumer warps still hold the accumulator in registers, because
    `mma.sync` has nowhere else to put it, so the producer and consumer split
    buys nothing that this architecture can spend.

The pattern is the same across all eleven. Most of the wins come from arithmetic
you can do on paper, about waves, registers, tiles and cache sizes. The
instruction level work comes last, and it is worth less than it looks.

The most expensive mistake is to read advice for the wrong chip. Consumer
Blackwell and datacenter Blackwell share a name and a tensor core generation, and
they do not share the instruction that does the multiply. Check which one a
tutorial targets before you follow it.

---

## Sources

Measurements in this document come from `out/bench/kernel/hgemm/hgemm_acc.json`
and `out/bench_old/kernel/mxfp8/mxfp8.json`. End to end figures come from
[performance.md](performance.md).

External references:

- [Dissecting the SM_120 Microarchitecture](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html),
  for cycle latencies, cache levels, warp thresholds and co-issue rules.
- [NVFP4 Blockscaled GEMM on RTX Pro Blackwell (SM12x)](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/),
  for the block scaled instruction, tile shapes and scale register layouts.
- [Blackwell GPU Wiki: SM100 vs SM120](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/).
- [MXFP8 GEMM: up to 99% of cuBLAS with CUDA and PTX](https://danielvegamyhre.github.io/2026/03/29/mxfp8-gemm.html),
  for the list of optimizations and what each one was worth.
- [Triton issue 7550: `tl.dot_scaled` using fp16 MMA on the 5090](https://github.com/triton-lang/triton/issues/7550).
- [Accelerating MoE with a persistent cache aware grouped GEMM in Triton](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/).
- [CUTLASS: persistent kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/).
- [CUTLASS: efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html).
