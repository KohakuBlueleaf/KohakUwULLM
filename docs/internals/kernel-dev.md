# Kernel development on sm_120

This document is the method. It tells you how to derive a kernel's tile shape,
warp count and pipeline depth from the numbers a card reports, instead of copying
constants from another kernel.

It is written so that somebody with a different `sm_120` card can start work
without access to the machine the rest of this repo was tuned on.
[../performance/gemm.md](../performance/gemm.md) is the companion document. It
covers why a GEMM is shaped the way it is. This one covers how to compute the
shape for your own card.

---

## 1. Read the machine before you write anything

Never hardcode a hardware number you did not query or measure. Run this first.

```python
import torch
p = torch.cuda.get_device_properties(0)
print(f"name                {p.name}")
print(f"capability          sm_{p.major}{p.minor}")
print(f"SMs                 {p.multi_processor_count}")
print(f"registers per SM    {p.regs_per_multiprocessor}")
print(f"max threads per SM  {p.max_threads_per_multi_processor}")
print(f"shared mem per SM   {p.shared_memory_per_multiprocessor} B")
print(f"shared mem opt-in   {p.shared_memory_per_block_optin} B")
print(f"L2 cache            {p.l2_cache_size / 2**20:.0f} MB")
print(f"VRAM                {p.total_memory / 2**30:.1f} GiB")
```

Two numbers are **not** in that list and must be measured, not computed.

- **Peak DRAM bandwidth.** Use `kohakuwullm.bench.core.timing.cached_peak_bandwidth()`.
  Do not derive it from `memory_clock_rate`. On the 5090 the stock-clock
  theoretical figure sits within a rounding error of the measured one and the
  card actually clocks higher, so the theoretical number is a convincing decoy.
- **Peak matmul rate.** Time a large square cuBLAS GEMM. Do not compute it from
  the clock and the core count. Use the measured value as the denominator for
  every efficiency claim you make.

---

## 2. The two cards this repo knows about

Both are `sm_120`. They share an SM design, an instruction set and a shared
memory size. They do not share anything that scales with the die.

| | RTX 5090 | RTX 5060 Ti 16G |
|---|---|---|
| chip | GB202 | GB206 |
| SMs | 170 | 36 |
| L2 | 96 MB | 32 MB |
| VRAM | 32 GB | 16 GB |
| memory bus | 512 bit GDDR7 | 128 bit GDDR7 |
| DRAM bandwidth | 1791 GB/s measured | 448 GB/s stated |
| registers per SM | 65536 | 65536 |
| shared memory per CTA | about 99 KB | about 99 KB |
| L1 per SM | 128 KB | 128 KB |
| bf16 matmul, fp32 accumulate | 270 TFLOP/s measured | **estimate only, measure it** |

The 5060 Ti compute figure is deliberately left blank. Scaling by SM count and
clock suggests roughly 60 TFLOP/s, but an estimate is not a denominator. Measure
it and write the result here.

### The derived numbers matter more than the raw ones

| | RTX 5090 | RTX 5060 Ti 16G |
|---|---|---|
| L2 per SM | 0.56 MB | **0.89 MB** |
| DRAM bandwidth per SM | 10.5 GB/s | **12.4 GB/s** |
| ridge point, FLOP per byte | 151 | about 136 |

Read that table twice. **The small card has more cache per SM and more bandwidth
per SM than the big one.** It is not a scaled-down 5090. It is a slightly
memory-richer machine with the same SM.

The ridge point is the arithmetic intensity where a kernel stops being memory
bound and starts being compute bound. It is `peak_flops / peak_bandwidth`. Since
it differs between the two cards, **a kernel can be compute bound on one and
memory bound on the other.** That single fact governs which results transfer.

---

## 3. What transfers between the two cards, and what does not

### Transfers

- **Correctness and numerics.** Same instruction set. A ULP result on one card is
  a ULP result on the other.
- **Instruction selection.** Whether `tl.dot_scaled` emits a real block scaled
  MMA or falls back to bf16 is a compiler and architecture question, not a die
  question. Check it on either card.
- **Register and occupancy arithmetic.** Same register file, same 255 register
  cap, same four sub-cores per SM.
- **Shared memory budget and pipeline depth.** Same 99 KB.
- **The warps per sub-core threshold.** Same SM.

### Does not transfer

- **Absolute throughput.** Obviously.
- **Which side of the roofline you are on.** The ridge point differs by about
  10%. A kernel near the ridge can flip.
- **Wave quantization severity.** This one is the trap. See below.
- **L2 rasterization severity.** Also understated on the small card. See below.
- **Anything about VRAM capacity**, since 16 GB against 32 GB changes what fits.
- **Anything multi-GPU.**

### The trap: the small card hides scheduling problems

Take a grid of 1024 CTAs, one CTA resident per SM.

- On 36 SMs that is 29 waves. The last wave wastes at most 35 SMs out of 1044
  slots, which is **1.9%**.
- On 170 SMs that is 7 waves. The last wave wastes 166 slots out of 1190, which
  is **14%**.

The same kernel loses seven times more to wave quantization on the big card.

L2 behaves the same way. Live working set scales with concurrent CTAs, so the
small card holds about 21% of the big card's working set in 33% of the cache. It
has more headroom, so a bad rasterization hurts it less.

**The rule that follows: a positive result on the 5060 Ti transfers. A negative
one does not.** If a scheduling change helps on the small card, it will help at
least as much on the big one. If it does not help on the small card, that is not
evidence, because the small card did not have the problem.

Say which card produced a number, every time you report one.

---

## 4. The five budgets

Every kernel shape decision comes from one of five budgets. Compute all five
before you write code. Four of them are arithmetic on numbers from section 1.

### Budget 1: registers per thread, which sets occupancy

The register file holds 65536 32-bit registers per SM. A thread may use at most
255. Work backwards from the occupancy you want:

```
regs_per_thread_max = 65536 / (warps_per_subcore * 4 * 32)
```

The 4 is the sub-core count and the 32 is the warp size. This gives:

| warps per sub-core | max registers per thread |
|---|---|
| 4 | 128 |
| **5** | **102** |
| 6 | 85 |
| 8 | 64 |
| 12 (the maximum) | 42 |

Five is the number that matters **for a kernel that relies on warp-level latency
hiding**. Characterization of this architecture reports that hiding improves by
roughly 6 times at 5 or more warps per sub-core, and is nearly absent at 4 or
fewer.

**It does not apply to a software-pipelined GEMM.** Measured on this card, the
fastest bf16 GEMM configs run at one to two warps per sub-core with 160 to 232
registers per thread and zero spills, because `num_stages` hides the latency
inside the warp instead. See ../performance/gemm.md section 5b.

So: use the table to know where you are, and apply the 102-register rule to
elementwise and reduction kernels. For a GEMM, sweep the tile and check spills.

For a GEMM, the accumulator alone is:

```
accum_regs_per_thread = BM * BN * acc_words / (32 * num_warps)
```

`acc_words` is 1 for an fp32 accumulator and 0.5 for fp16, because two fp16
values pack into one register. Budget the accumulator at no more than about 60%
of the total, since operands, addresses and the pipeline need the rest.

| tile | warps | fp32 accumulator registers | verdict |
|---|---|---|---|
| 128 x 128 | 4 | 128 | over budget before anything else is counted |
| 128 x 128 | 8 | 64 | workable |
| 128 x 64 | 8 | 32 | comfortable |
| 64 x 64 | 4 | 32 | comfortable |
| 256 x 128 | 8 | 128 | over budget |

This repo ships `BLOCK_M 128, BLOCK_N 128` with `num_warps 4` in
`kernels/mxfp8/grouped.py`. That is the first row. Measure before assuming it is
wrong: the same shape in a dense bf16 GEMM costs only about 1%.

Verify the real number rather than trusting the estimate. Triton reports it:

```python
compiled = my_kernel[grid](...)
print(compiled.n_regs, compiled.n_spills, compiled.metadata.shared)
```

**Any value of `n_spills` above zero means you have already lost.** A spill goes
to local memory, which is DRAM.

### Budget 2: shared memory, which sets pipeline depth

```
smem_per_stage = (BM + BN) * BK * bytes_per_element
stages         = floor(99_000 / smem_per_stage)
```

With `BM = BN = 128` and `BK = 64`:

| operand dtype | bytes per stage | stages that fit |
|---|---|---|
| fp8 | 16 KB | 6 |
| bf16 or fp16 | 32 KB | 3 |
| fp32 | 64 KB | 1 |

You need at least 3 stages to hide a DRAM load, and at least 2 for an L2 hit.
DRAM is 372 cycles on this architecture, L2 far is 180 and L2 near is 79.

Two consequences that are easy to miss:

- **A smaller operand dtype does not only halve bandwidth. It doubles the
  pipeline depth you can afford.** On a card with 99 KB of shared memory that
  second effect is often larger than the first.
- **Anything that forces you to stage a wider dtype costs you stages.** This is
  the arithmetic behind the rule in gemm.md that a cast must not live in the
  GEMM prologue.

### Budget 3: warps per sub-core, which sets latency hiding

```
warps_per_sm      = ctas_per_sm * num_warps
warps_per_subcore = warps_per_sm / 4
```

`ctas_per_sm` is the smallest of the register limit, the shared memory limit and
the hardware maximum. Target 5 or more. See budget 1 for the register cost.

### Budget 4: waves, which sets SM utilization

```
ctas       = ceil(M / BM) * ceil(N / BN)
waves      = ceil(ctas / (SMs * ctas_per_sm))
efficiency = ctas / (waves * SMs * ctas_per_sm)
```

If efficiency is below about 0.9, use a persistent kernel. Launch exactly
`SMs * ctas_per_sm` CTAs and loop over output tiles inside. There is then no
partial wave by construction.

Compute this for the card you are on **and** for the 5090, because of the trap in
section 3.

### Budget 5: bytes against FLOPs, which tells you where to look

```
flops     = 2 * M * N * K
bytes     = (M*K + K*N + M*N) * bytes_per_element
intensity = flops / bytes
```

Compare `intensity` to the ridge point. If it is well above, the kernel is
compute bound and memory optimizations will not pay. If it is well below, tile
sizes and warps will not pay and you should work on traffic.

Do this before anything else. It costs one line and it decides which of the other
four budgets you care about.

---

## 5. A worked example, both cards

Take `M = 8192, N = 4096, K = 4096` in bf16, with `BM = BN = 128, BK = 64`,
8 warps, fp32 accumulator.

**Shared budgets, identical on both cards.**
Accumulator is `128 * 128 * 1 / (32 * 8)` = 64 registers per thread. Assume about
100 total after operands. Registers allow `65536 / (100 * 256)` = 2 CTAs per SM.
That is 16 warps per SM, so 4 per sub-core. Shared memory at 32 KB per stage
allows 3 stages, and 2 CTAs need 2 x 3 x 32 KB = 192 KB, which does not fit in
99 KB. So shared memory limits it to 1 CTA per SM with 3 stages, giving 8 warps
per SM and **2 per sub-core**. Below the threshold.

That is the real answer, and it shows why you compute all five. The register
budget said 2 CTAs. Shared memory said 1. The smaller wins.

The fix is to reduce `BK` to 32, which halves the stage cost and lets 2 CTAs
resident with 3 stages each, or to accept 1 CTA and raise `num_warps` to 16.

**Budgets that differ.**
The grid is `64 * 32` = 2048 CTAs.

| | RTX 5090 | RTX 5060 Ti 16G |
|---|---|---|
| waves at 1 CTA per SM | ceil(2048/170) = 13 | ceil(2048/36) = 57 |
| efficiency | 2048/2210 = **92.7%** | 2048/2052 = **99.8%** |
| intensity, FLOP per byte | 273 | 273 |
| ridge point | 151 | about 136 |
| verdict | compute bound | compute bound |

Both are compute bound here, so budgets 1 to 3 are what matter. The wave
efficiency differs as section 3 predicts, and the small card would not show you
the 7% the big card loses.

---

## 6. Repo conventions for a new kernel

### Where it goes

Kernels live in `src/kohakuwullm/kernels/<family>/`. A kernel that a config can
select is registered in `src/kohakuwullm/registry.py` and resolved once at build
time by `build(spec, REGISTRY)`. There is no runtime branch on a config value.

Benchmarks live in `scripts/bench/kernel/`. They are part of the deliverable, not
a scratch area.

### Numerics rules

This repo trains in low precision, so these are not optional.

- **Every Triton kernel needs a precision test against an fp64 reference**, in
  both fp16 and bf16, forward and backward. Put it in `tests/test_kernels.py`.
- **Judge error in ULP, not absolutely.** Use `bench.timing.ulp_error` and pick
  its mode. Use `elementwise` for an elementwise kernel. Use `rms` for a GEMM or
  a reduction, where a near-zero output is cancellation rather than a small true
  value.
- **Never trust a low-precision scalar reduction.** Summing 16k bf16 terms loses
  several percent. Reduce in fp32.
- **Token counts are int64.** A run passes 2^31 tokens in under an hour.

### Autotune

Do not use `triton.autotune` on a shape that changes at run time. It re-benchmarks
whenever a dimension moves, and with varlen data that measured **365 ms per step**
of `do_bench` L2 flushes. The tell in a profile is `FillFunctor<int>` running at
exactly DRAM peak.

Use fixed constants, or an offline tuning table keyed by shape and loaded at build
time.

### Verifying without a GPU

`TRITON_INTERPRET=1` runs a Triton kernel bit-exactly on the CPU. It has caught a
`uint32` randint and a lost sign bit in this repo.

**It latches when the kernel is decorated, not when it runs.** Never set it inside
a test file. One test file that set it caused the whole suite's kernels to run on
the CPU and pass.

### Benchmarking

- **Warm up every shape.** A partial warmup over a varlen stream charges Triton
  compilation as throughput.
- **Use graph timing for small kernels.** `graph_ms` gives device time without
  launch dispatch. Wall time and device time differ by tens of microseconds, which
  is the whole runtime of a small kernel.
- **A bandwidth figure above DRAM peak means you measured L2, not DRAM.** Report
  `l2_resident` alongside any bandwidth claim. A rate of 200% of peak is a cache
  hit reported as a fast kernel.
- **Every figure shows throughput and accuracy together.** A kernel that is fast
  and wrong is not a result.
- **Audit the benchmark as carefully as the kernel.** Three real bugs in this
  repo's history were benchmark bugs that made correct code look wrong. A
  non-leaf input tensor that made every timed backward fail. A per-element ULP
  metric that reported 24000 ULP for a numerically perfect GEMM. An fp64
  reference that ran out of memory at 131k tokens.

### The audit loop

Do not stop at "the tests pass".

1. Implement the slice.
2. Write tests that pin it. A negative case is worth more than a positive one.
3. Run the suite and the linters.
4. Audit the diff for typos, broken invariants and code that does what is typed
   but the wrong thing for the specification.
5. If a bug got past the tests, **fix the test first** and confirm it fails on the
   unfixed code. Then fix the bug.
6. Repeat.

---

## 7. Checklist for starting a kernel

Work down this list. Stop at the first item that fails.

1. Run the query in section 1. Write the numbers down.
2. Measure peak bandwidth and peak matmul rate. Do not compute them.
3. Compute budget 5. Decide whether you are memory bound or compute bound.
4. Compute budgets 1 to 4 for your first guess at a tile.
5. Compile the kernel and read `n_regs`, `n_spills` and `shared`. Compare them to
   your prediction. If they disagree, the prediction was wrong and you should
   understand why before continuing.
6. Confirm `n_spills == 0`.
7. Confirm the PTX contains the instruction you intended. For a block scaled fp8
   kernel, look for `mma.sync` with `block_scale`. A silent fall back to bf16
   gives correct answers at half speed.
8. Write the precision test against fp64, in fp16 and bf16, forward and backward.
9. Benchmark against the vendor kernel at the same shapes, and report accuracy in
   the same figure.
10. Recompute budget 4 for the RTX 5090 as well as your own card, and say in the
    result which card produced the numbers.

---

## Sources

Cycle latencies, the warps per sub-core threshold, shared memory capacity and the
co-issue rules come from
[Dissecting the SM_120 Microarchitecture](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html).
Block scaled MMA shapes and scale factor layouts come from
[Colfax on NVFP4 blockscaled GEMM for SM12x](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/).
RTX 5060 Ti 16 GB board figures come from
[TechPowerUp](https://www.techpowerup.com/gpu-specs/msi-rtx-5060-ti-shadow-2x-plus-16-gb.b12792).
Everything measured comes from `out/bench/kernel/`.
