# Why an MXFP8 GEMM is hard to write on sm_120

Our bf16 GEMM beats cuBLAS. Our MXFP8 GEMM sits at 48% of the machine while the
vendor reaches 63%. This document is the root cause, measured here and confirmed
against outside sources, and the decision that follows.

---

## 0. The numbers

Measured on one idle RTX 5090, `4096^3`, best-of-N.

| | TFLOP/s | of ISA |
|---|---|---|
| block-scaled MXFP8 ISA ceiling | **1096.7** | 100% |
| vendor `_scaled_mm` | 694.6 | 63.3% |
| our Triton pre-quantized | 529.2 | 48.3% |
| our Stream-K port | 528.8 | 48.2% |

The ceiling is `mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X...ue8m0` run
back to back with register-resident operands, four independent chains. It is
**3.93x** bf16 into fp32. An independent register-resident microbenchmark
published elsewhere reports **3.95x**, which agrees.

---

## 1. The hardware wants the scales somewhere specific

The instruction computes `D = C + (A x SFA) * (B x SFB)`. Each scale byte must
arrive in a particular lane and a particular byte of a particular register.

A tile is 128 rows. In a plain row-major `[M, ceil(K/32)]` scale array, the
128x4 slice a tile needs is spread over **128 separate cache lines**, one per
row. NVIDIA's answer is the 128x4 tiled layout, physically
`[ceil(M/128)*128, ceil(ceil(K/32)/4)*4]`, with the in-tile mapping

```
offset = (outer % 32) * 16 + (outer / 32) * 4 + inner
```

Co-locating the 512 scales of a 128x4 tile turns 128 cache-line accesses into
one coalesced transaction **and** lands each byte where the tensor core expects
it, with no gather and no lane shuffle. That is the whole purpose of the
swizzle, and it is a one-time cost paid outside the loop.

---

## 2. Triton cannot be given the swizzled layout

`tl.dot_scaled` accepts scales only as 2D `[M, K // 32]`. There is no parameter
for a pre-swizzled tensor. Triton's own block-scaled tutorial starts from the 5D
blocked layout and then, in its words, logically transposes and reshapes it
"into the 2D layout expected by `tl.dot_scaled`" — it **un-swizzles**, and the
compiler then rebuilds the lane placement inside the main loop. That tutorial
also targets compute capability 10 and 11, not 12.

So the one optimisation the hardware was designed around is unreachable through
the API.

---

## 3. What that costs, measured

Same kernel, same tile, same operands, with and without scales:

| variant | TFLOP/s | registers | loop insts | MMA | other/MMA |
|---|---|---|---|---|---|
| `128x128x128` scaled | 463.4 | **220** | 183 | 64 | 1.86 |
| `128x128x128` plain `tl.dot` | 423.2 | **158** | 132 | 64 | 1.06 |

Scale handling costs **+51 instructions, 36 of them `PRMT`, and +62 registers
per thread**.

Note what it does *not* cost: 1.86 non-MMA instructions per MMA is next door to
bf16's 1.41. **The kernel is not instruction-issue bound.** An earlier revision
of this analysis claimed 10.09 and was wrong; that figure came from a Stream-K
kernel whose loop extraction caught surrounding code.

---

## 4. The deadlock

Registers are the binding constraint, and they close a loop:

1. Scale staging costs 62 of 220 registers per thread, 28% of the budget.
2. 220 registers x 256 threads is 56 KB, so **one CTA per SM**. Two would need
   128 or fewer.
3. A `128x128` fp8 tile puts L2 at **67%** of the 24.8 B/cycle/SM budget. The
   same tile in bf16 is at 32%, because the bytes per tile are identical while
   the arithmetic is 3.93x faster, so the time to consume them is quartered.
4. Cutting L2 traffic needs 256-wide tiles. Those spill 76 to 82 registers,
   because the larger accumulator plus the scale fragments do not fit.

**Scale staging forbids the large tile that the arithmetic rate demands.** That
is the whole difficulty, and it is specific to MXFP8: at bf16 rates none of
these terms bind.

Two independent reports agree on where the pressure is. A published CUDA and PTX
MXFP8 GEMM reaching 99% of cuBLAS lists "static array indexing to stop register
spills" as worth 17 points. A production MoE MXFP8 kernel reports its wins came
from a manual swizzling pattern and from "minimizing SMEM and register usage to
increase SM occupancy".

---

## 5. Inline PTX cannot rescue this inside Triton

`tl.inline_asm_elementwise` is documented as "essentially `map` where the
function is inline assembly". It is elementwise: fixed lanes in, fixed lanes
out. It cannot express an MMA over distributed register fragments, cannot take
an accumulator in a chosen layout, and — decisively — cannot pin a scale byte to
a chosen lane and byte. Triton owns register allocation and fragment layouts,
which is exactly the thing that has to be controlled.

Gluon, Triton's low-level dialect, exposes `TensorMemoryScalesLayout`, which is
TMEM. TMEM is `tcgen05`, an sm_100 feature this card does not have.

**There is no in-Triton path.** Hand-written PTX for this kernel means leaving
Triton, not annotating it.

---

## 6. This is an ecosystem gap, not a Triton trap

Checked against the alternatives, and Triton is not the laggard:

| path | sm_120 MXFP8 block-scaled |
|---|---|
| **Triton `tl.dot_scaled`** | **works** — emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`, verified in SASS and PTX |
| CuTe DSL | **not reachable**: only the FP4 block-scaled ops are wired to the SM120 warp path; `MmaMXF8Op` sits on the SM100 tcgen05 path |
| CUTLASS Python DSL | `BlockScaledMmaOp` restricted to sm_100a |
| CUTLASS C++ | SM120 kernels exist from 4.2, but block-scaled examples crash with misaligned address on RTX 5090 |
| flashinfer | open RFC, does not exist yet |
| TileLang | MXFP8 block-scaled GEMM added for Blackwell; sm_120 reach unverified here |

Triton is the only high-level tool that reliably issues the right instruction on
this card. Its limitation is the scale layout, not the instruction.

---

## 6b. `torch.compile` is not actually blocked, and the fix is three lines

An earlier revision of this document, and of the session that produced it,
claimed `torch._scaled_mm` cannot be used under `torch.compile`. **That is
wrong, and it was wrong in two ways.**

Measured on this machine:

| path | `torch.compile(fullgraph=True)` |
|---|---|
| per-tensor fp8 `_scaled_mm` | **works** |
| rowwise fp8 `_scaled_mm` | **works** |
| MXFP8 block-scaled with swizzled scales, `_scaled_mm_v2` | fails |

Inductor registers lowerings for both `aten._scaled_mm.default` and
`aten._scaled_mm_v2.default`, and TorchAO uses the op under `torch.compile`
routinely. The failure is narrow and its message says so:

```
LoweringException: AssertionError: Inductor _scaled_mm_v2 lowering does not yet
support non-trivial swizzles (got swizzle_a=[1], swizzle_b=[1])
```

Only the swizzled MXFP8 path is affected -- which is, unhappily, exactly the
path that makes the vendor kernel fast (section 1).

**The fix is to make the op opaque to Inductor.** Wrap it in
`torch.library.custom_op` with a `register_fake`, and the graph compiles past
it: no lowering is attempted, no graph break, the kernel runs as the vendor
kernel. `kohakuwullm::mxfp8_mm_swizzled` in `kernels/mxfp8/interop.py` is that
wrapper.

Measured on a transformer block, `D=2048`, `FFN=8192`, 4096 tokens:

| arm | ms |
|---|---|
| vendor eager | 3.060 |
| vendor through the custom op, eager | 3.056 |
| **vendor through the custom op, compiled** | **3.033** |
| our Triton kernel plus fused epilogues, eager | 3.174 |

So the vendor kernel keeps its speed **and** compiles, and it beats our best
Triton path by about 4.6%. What you give up is fusion *into* the GEMM, which
nobody wants anyway -- epilogue fusion belongs in the producing kernel, which is
what `kernels/mxfp8/fused_act.py` does.

**The general lesson: a lowering gap is not a compilation gap.** Before
concluding that an op cannot be compiled, check whether it merely cannot be
*lowered*, and wrap it if so.

---

## 7. Decision

**Use `torch._scaled_mm` for MXFP8, through the custom-op wrapper in section
6b.** It is the fastest path and it compiles. Reasons, in order:

- The kernel gap is 1.31x, but MXFP8 is worth 1.08 to 1.23x end to end on our
  dense models. Closing the kernel gap completely is worth a few percent of a
  step, not a rewrite.
- Every alternative that could close it means leaving Python. The published
  CUDA and PTX kernel that reaches 99% needed ten distinct optimisations.
- bf16 and fp16 already beat cuBLAS and are shipped.

**Next, if this is revisited: spike TileLang for this one kernel.** Mixing DSLs
is fine — the GEMM is a leaf. TileLang exposes explicit layouts and inline PTX,
which is precisely what `tl.dot_scaled` withholds, and it is the cheapest escape
hatch that keeps the rest in Python. Verify first that its MXFP8 block-scaled
path reaches sm_120 and not only sm_100.

**Do not** attempt inline PTX inside Triton for this. Section 5 is why.

---

## Sources

- [The 128x4 Tiled Layout for Block Scaling Factors](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)
- [Block Scaled Matrix Multiplication, Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [MXFP8 GEMM: up to 99% of cuBLAS with CUDA and PTX](https://danielvegamyhre.github.io/2026/03/29/mxfp8-gemm.html)
- [1.5x faster MoE training with custom MXFP8 kernels, Cursor](https://cursor.com/blog/kernels)
- [CUTLASS issue 2800: Python DSL BlockScaledMmaOp restricted to sm_100a](https://github.com/NVIDIA/cutlass/issues/2800)
- [CUTLASS issue 2906: SM120 NVF4 GEMM misaligned address on RTX 5090](https://github.com/NVIDIA/cutlass/issues/2906)
- [CUTLASS issue 2867: block-scaled data formats for SM120 in CuTe DSL](https://github.com/NVIDIA/cutlass/issues/2867)
- [flashinfer RFC 3628: MXFP8 block-scaled for SM120a](https://github.com/flashinfer-ai/flashinfer/issues/3628)
- [Dissecting the SM_120 microarchitecture](https://zartbot.github.io/micro_arch/nvidia/sm_120/paper.html)
