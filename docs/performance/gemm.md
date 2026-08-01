# How to write a fast GEMM on consumer Blackwell

A matrix multiply is easy to write and very hard to make fast. Three loops give
you a correct answer in ten minutes. The last 30% of the hardware takes months.

This document explains where the speed comes from. It uses the RTX 5090 as the
target, because that is the card this project trains on. The card reports itself
as `sm_120`. Most published Blackwell advice is for `sm_100`, which is a
different machine. The differences matter, so this document names them.

All rates in this document come from measurements in `out/bench/kernel/`, unless
the text says otherwise. Hardware latencies come from published characterization
work, which is cited at the end.

---

## 0. The two numbers that bound everything

A GEMM does `2*M*N*K` floating point operations. It moves at least
`M*K + K*N + M*N` elements. The ratio of these two numbers is the arithmetic
intensity. It tells you which limit you are close to.

The RTX 5090 gives us these ceilings:

| resource | value | source |
|---|---|---|
| DRAM bandwidth | 1791 GB/s | measured, see [README.md](README.md) |
| bf16 matmul, fp32 accumulate | 270 TFLOP/s | measured ceiling |
| fp16 matmul, fp16 accumulate | 325 TFLOP/s | measured |
| MXFP8 matmul, vendor kernel | 693 TFLOP/s | measured |

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

### The tensor core is the same speed for every format

On `sm_120`, all non-FP64 tensor core instructions have the same latency of 29
cycles and the same throughput of 23 cycles. FP16, BF16, FP8, FP6, FP4 and INT8
all share one pipeline.

Low precision is faster for one reason only. One instruction covers more K. An
fp8 instruction multiplies twice the depth of an fp16 instruction in the same
number of cycles. So the speed comes from the instruction shape, not from a
faster circuit.

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
small. One warp cannot fill the tensor core. You need several warps to issue MMA
together. This is why every fast `sm_120` kernel uses 8 warps per CTA.

### Warp count decides whether the card hides latency

Characterization of this architecture shows a sharp threshold. With 5 or more
warps per sub-core, latency hiding improves by about 6 times. With 4 or fewer
warps, there is almost no hiding at all.

This threshold is easy to miss, and it interacts with register pressure.

Take a `128 x 128` output tile with 4 warps. That is 128 threads. Each thread
holds `128 * 128 / 128 = 128` accumulator values. In fp32 that is **128 registers
per thread**, before any address, operand or scale register.

The register file allows 255 registers per thread. At 128 accumulators plus
overhead, the compiler fits one CTA per SM, and that CTA has 4 warps. That is 1
warp per sub-core. It is the worst side of the threshold.

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

This is the largest single defect and it is invisible in the source.

A `128 x 128` tile with 4 warps gives 128 accumulator registers per thread. The
occupancy collapses to one warp per sub-core, which is below the threshold where
this card hides latency at all.

The fix is one number. Use 8 warps, or make the tile narrower. No profiling is
needed to find this. It is arithmetic on the tile shape.

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

## 6. Conclusion

A GEMM is a memory schedule with some arithmetic attached. The arithmetic is
fixed by the problem. Everything you control is about moving data.

The order of work that follows from this document:

1. **Compute the roofline for your shape.** One line of arithmetic tells you
   whether to work on memory or on compute. Our shapes are 7 times compute bound,
   which rules out most bandwidth work immediately.
2. **Compute the accumulator registers per thread.** It is
   `BM * BN / (32 * num_warps)`. If it is above about 64, fix that before
   anything else. This card gives no latency hiding below 5 warps per sub-core.
3. **Check that your low precision instruction is the one you think it is.** Read
   the PTX. A silent fallback to bf16 gives correct answers at half speed.
4. **Size the working set against L2 before you reorder anything.** On a 96 MB
   L2, the classic reorder may buy very little.
5. **Remove partial waves.** 170 SMs and a tile count that is not a multiple of
   170 wastes whole SMs.
6. **Put fusion in the epilogue, never in the prologue.** The load path is a copy
   engine. It cannot compute. Cast where the data is already in registers.
7. **Use asynchronous loads last.** TMA is present on this card but its load
   latency is 488 to 620 cycles, which is worse than a plain DRAM read. It pays
   only when you already have enough pipeline stages to hide it, and shared
   memory limits those to about three. Multicast TMA is worse than absent here,
   because it degrades by about four orders of magnitude.

The pattern across all seven is the same. Most of the wins come from arithmetic
you can do on paper, about tiles, registers, waves and cache sizes. The
instruction level work comes last, and it is worth less than it looks.

The most expensive mistake is to read advice for the wrong chip. Consumer
Blackwell and datacenter Blackwell share a name and a tensor core generation, and
they do not share the instruction that does the multiply. Check which one a
tutorial targets before you follow it.

---

## Sources

Measurements in this document come from `out/bench/kernel/hgemm/hgemm_acc.json`
and `out/bench/kernel/mxfp8/mxfp8.json`. End to end figures come from
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
