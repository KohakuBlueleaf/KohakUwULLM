# Triton, TileLang and CuTeDSL

This repo writes its kernels in Triton. Two other Python DSLs now carry
production kernels. DeepSeek's FlashMLA work is associated with TileLang, and
FlashAttention 4 is written in CuTeDSL. This document compares the three and says
what an experiment on our hardware should look like.

**Read section 3 before you plan any work.** The published comparisons were all
measured on hardware that has features `sm_120` does not have, and that changes
what we should expect.

---

## 1. What each one is

### Triton

A Python DSL where you write **tile-level** code. You describe what one program
does with a block of data. The compiler owns the thread mapping, the shared
memory layout, the software pipeline and the register allocation.

You control the block sizes, `num_warps` and `num_stages`. You do not control how
warps divide the tile, how shared memory is swizzled, or where a barrier goes.

Strengths: shortest path from an idea to a working kernel, mature, already the
whole of `src/kohakuwullm/kernels/`, and understood by `torch.compile`.

Weakness: when the compiler's choice is wrong you have very little recourse. You
can change a block size and hope. There is no way to say "put these four warps on
the MMA and those two on the loads".

### TileLang

A tile-level DSL built on TVM. It sits one level lower than Triton. You still
write tile code, but you can also specify the memory scope of a buffer, the
pipeline structure and the layout, and you can drop to a lower level for one
part of a kernel while leaving the rest high level.

The published claim is that this middle position is the point. You keep most of
Triton's brevity, and you gain the few controls that decide the last 30%.

Reported results: an MLA kernel in about 70 lines of Python reaching 98% of the
hand-written FlashMLA, and GEMM at 1.03x to 1.25x of Triton across 4090, A100,
H100 and MI300X.

### CuTeDSL

A Python front end over the CUTLASS CuTe abstractions. This is the lowest level
of the three. You work with layouts, tiled copies and tiled MMA atoms directly.
It is the same mental model as CUTLASS C++, with Python compile times instead of
`nvcc` compile times.

NVIDIA maintains it, and it is now a TorchInductor GEMM backend named `NVGEMM`,
alongside ATen, Triton and CUTLASS C++.

Strengths: full control over the thread and memory hierarchy, and vendor
templates that are kept current with each architecture.

Weakness: you must know the hardware. There is no compiler making the layout
decisions for you, which is the point and also the cost.

---

## 2. The comparison as published

| | Triton | TileLang | CuTeDSL |
|---|---|---|---|
| level | tile, compiler owns layout | tile, you may pin layout and pipeline | layout and MMA atoms explicit |
| lines for a hard kernel | moderate | about 70 for MLA | most |
| GEMM vs Triton | baseline | 1.03x to 1.25x | comparable to CUTLASS |
| GEMM vs vendor | close on friendly shapes | 0.97x to 1.10x | up to 1.78x on MXFP8, B200 |
| compile time | fast | moderate, TVM based | fast, close to Triton |
| in `torch.compile` | native backend | no | yes, as `NVGEMM` |
| epilogue fusion | yes, via Inductor templates | manual | **not yet supported** |
| production users | very many | FlashMLA line of work | FlashAttention 4 |

Two entries in that table matter more than the speed columns.

**CuTeDSL has no epilogue fusion in the Inductor backend yet.** It is listed as
future work. Our main reason for wanting a non-vendor GEMM is that
`torch._scaled_mm` cannot absorb an epilogue, and the arithmetic in
[../performance/gemm.md](../performance/gemm.md) shows a fusable kernel at 82% of
vendor speed beating an unfusable one at 100%. A CuTeDSL GEMM that cannot fuse
its epilogue does not solve that problem. It is a faster black box.

**TileLang is not a `torch.compile` backend.** A TileLang kernel is a custom op,
so it is a fusion boundary, same as our Triton kernels are today.

---

## 3. Why the published numbers may not transfer

Every headline comparison above was measured on H100, B200, A100, 4090 or MI300X.

Look at what the two fastest of those have that we do not. H100 has `wgmma` and
warp group MMA. B200 has `tcgen05`, Tensor Memory and two-SM cooperative MMA. A
large part of what separates a good kernel from a poor one on those chips is how
well it uses exactly those features: asynchronous MMA, a dedicated accumulator
memory, warp specialization into producer and consumer groups, and cluster wide
data sharing.

`sm_120` has none of them. It has `mma.sync`, which is the synchronous, warp
collective, register accumulator instruction that every one of these DSLs has
supported since well before any of this work existed.

So the mechanism behind most of the published gap is absent on our card. On
`sm_120` all three DSLs are emitting the same family of instruction, against the
same 99 KB of shared memory, with the same register file. **The room for one DSL
to beat another is much smaller here than the published numbers suggest.**

That is a prediction, not a measurement. It is falsifiable and worth testing. But
it should set the expected value before anyone spends a week.

There is a second reason for caution. Our own experience with CUTLASS on `sm_120`
is poor. See [../performance/upstream-cutlass-findings.md](../performance/upstream-cutlass-findings.md):
the grouped block scaled GEMM is correct within one wave and then breaks, and
CUTLASS 4.x did not fix it. CuTeDSL is built on the same abstractions and the
same `sm_120` support surface.

Support status for all three on `sm_120`, as of writing:

- **Triton.** Works. One known defect worth checking, where `tl.dot_scaled` may
  lower to bf16 MMA instead of a block scaled FP8 MMA.
- **TileLang.** Runs, but several kernel paths carry capability guards written
  for `sm_90` and `sm_100` and need widening for `sm_120`.
- **CuTeDSL.** The Inductor backend documents H100 and B200. `sm_120` is not
  stated. Given that `sm_100` and `sm_120` use different MMA instructions,
  "supports B200" does not imply "supports the 5090". **Verify before planning.**

---

## 4. What an experiment should look like

Do not port a kernel first. Answer the cheap questions in order and stop at the
first failure.

**Step 1. Does it run at all on `sm_120`?**
Install it and compile the simplest possible GEMM. Confirm the output is correct.
This is an afternoon, and it is the question with the highest chance of ending
the investigation.

**Step 2. Does the PTX contain the instruction you want?**
For an FP8 kernel, look for `mma.sync` with `block_scale`. Every DSL can fall
back to bf16 and give correct answers at half the speed. This applies to all
three equally.

**Step 3. Fixed shapes, one GEMM, against three baselines.**
Use our real shapes from `out/bench_old/kernel/mxfp8/mxfp8.json`, not square
matrices. Compare against cuBLAS, against `torch._scaled_mm` and against our own
Triton kernel. Report accuracy in the same figure as throughput.

**Step 4. Only if step 3 wins by more than 10%, test the epilogue.**
The decision is not which GEMM is fastest. It is which GEMM plus its epilogue is
fastest, because a fusable kernel at 82% beats an unfusable one at 100% for us.
A DSL that wins step 3 and cannot fuse may still lose the real comparison.

**Step 5. Only then consider a port.**
And scope it to one kernel, not the directory.

Before any of this, fix the four cheap items in our existing Triton kernels that
[kernel-dev.md](kernel-dev.md) identifies. It is not a fair comparison to measure
a tuned TileLang kernel against a Triton kernel that is running at two warps per
sub-core. The first fix is a one-line constant.

---

## 5. A reasonable expectation

Ranked by expected value on our hardware, highest first:

1. **Fix our Triton kernels.** The register budget, the wave quantization and the
   instruction check. Cheap, and it is a prerequisite for any honest comparison.
2. **Test whether `NVGEMM` runs on `sm_120`.** If it does, it is a config line in
   Inductor rather than a port, and it costs almost nothing to find out.
3. **Try TileLang on one kernel.** It is the DSL whose design most directly
   targets the gap we have, which is that Triton will not let us pin a pipeline.
   It is also the only one of the three where a win means writing and maintaining
   kernels in a second language, so the bar should be high.
4. **CuTeDSL by hand.** Only if 1 to 3 fail and the profile says the remaining
   loss is in layout and scheduling decisions that Triton will not expose.

The honest summary is that we are probably not DSL limited. We are limited by
four constants in our own kernels and by not having checked which instruction the
compiler emits. Those should be exhausted first.

---

## Sources

- [TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/pdf/2504.17577)
  and the [tile-ai/tilelang repository](https://github.com/tile-ai/tilelang).
- [Generating state of the art GEMMs with TorchInductor's CuteDSL backend](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/),
  for the `NVGEMM` backend, its feature matrix and the epilogue fusion limitation.
- [FlashAttention-4: algorithm and kernel pipelining co-design](https://arxiv.org/html/2603.05451v1).
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA).
- [Triton issue 7550, `tl.dot_scaled` on the 5090](https://github.com/triton-lang/triton/issues/7550).
