# MXFP8

Native FP8 training: e4m3 values with a UE8M0 power-of-two scale per 32 elements along
the contraction axis. This document says what the format is, how the kernels implement
it, what is converted, what has been verified and how, and what is still open — the
last section is the one that decides whether you should turn it on for a run you care
about.

## Turning it on

```python
PRESET = "Kohaku-MoE-2B"
ARCH_OVERRIDES = {"mxfp8": True}
```

or, on a model you already built:

```python
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights, swap_mxfp8

report = swap_mxfp8(model)              # or scope=("attn.",) for attention only
assert not report.blocking, report.summary()
```

**`refresh_mxfp8_weights(model)` must be called after every optimizer step.** The
quantized copies are derived from the 16-bit masters and are built lazily on the first
forward, never invalidated. Omitting the refresh trains on initialization-time weights
for the whole run with no symptom in the loss — the trainer does this at
`training/trainer.py`, and any custom loop must too. The function returns the number of
modules refreshed so a caller can assert it rather than trust it.

---

# The format

## Why block scales, and why 32 along K

Three fp8 scaling granularities were measured on sm_120 at 4096³, against bf16 cuBLAS
at 224.8 TF/s:

| scaling | TF/s |
|---|---|
| bf16 cuBLAS | 224.8 |
| fp8 per-row | 404.9 |
| fp8 per-tensor | 498.8 |
| **MXFP8 block-32** | **537.8** |

Per-row is the *slowest* fp8 variant, not the fastest, and the reason is where the
scale is applied: a per-row scale applies in the epilogue, while a block scale is
consumed inside the MMA datapath for free. sm_120 lowers `tl.dot_scaled` to a native
`mma.sync...kind::mxf8f6f4.block_scale` instruction, so the scaling costs nothing at
all.

The second property that matters is that an MXFP8 block runs along **K, the contraction
axis**, which a GEMM already tiles. A block therefore never straddles two tiles, which
is what makes several of the fusions in this document free rather than merely possible.

## The quantizer, and the round-up rule

A block of 32 values is scaled so its `amax` lands at e4m3's maximum, 448.0, rather than
at 1.0 — the format's exponent range is what the shared scale exists to exploit. Because
the scale is a power of two by construction, `ceil(log2(amax / 448))` is the whole
quantizer; no division and no float scale is ever materialized, and the exponent is
stored directly as the biased byte the MMA consumes.

```python
import torch
from kohakuwullm.kernels.mxfp8 import quantize_mx, mxfp8_matmul_pq

x = torch.randn(16384, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 1536, device="cuda", dtype=torch.bfloat16)   # (N, K)

xq, xs = quantize_mx(x)          # e4m3 values, ue8m0 scales, one per 32 along K
wq, ws = quantize_mx(w)
y = mxfp8_matmul_pq(xq, xs, wq, ws, out_dtype=torch.bfloat16)      # (16384, 4096)
```

Both operands are blocked along **K** because that is what this product contracts.
Get that wrong and the GEMM still runs; it just returns the wrong numbers. The table
below is the whole rule.

**Scale exponents round up, not to nearest, and this is a correctness requirement.**
Round-to-nearest maps a block's `amax` *above* the format's max and clips it. NVIDIA
measured an 843M model diverging at 300B tokens under OCP-standard rounding and matching
bf16 with round-up. This repo reproduces the same failure at 1/500th scale: under
round-to-nearest the error grows at 24.7σ against the round-up arm, and the test is on
the *slope*, not the offset.

The exponent is floored at −127. A wholly-zero block gets its `amax` clamped to 1e-30 and
its exponent floored, which is a finite scale over exact zeros — the one arrangement
where `scale * 0` cannot become `NaN * 0`.

`quantize_mx` runs at 1.53–1.66 TB/s on the weight shapes, 86–93% of this card's measured
peak, and at 96% of the 1776 GB/s measured triad on the activation shapes. **It is
bandwidth-saturated with nothing left to tune.** Its apparent microbenchmark cost is
about 63 us of Python dispatch, flat across tensor size; that disappears once the cast is
fused into the producing norm's epilogue (see `rms_norm_mx` in
[kernels.md](kernels.md)).

The quantizer's tile is **fixed at `BLOCK_M=8, BLOCK_K=1024` with 8 warps, not
autotuned**, and this is load-bearing rather than a preference. It was
`@triton.autotune(key=["M", "K"])`, and `M` is the *varlen token count*: every step of a
packed stream is a new key, so every step re-ran the search. 174 distinct token counts in
400 steps at roughly 840 ms per autotune — almost all of it `do_bench`'s 256 MB L2 flush
— is 365 ms/step, which was the entire measured MoE fp8 regression, to the digit.

One config rather than a table keyed on shape regime: over all 35 shapes the training
path quantizes, every config lands within 1.12x of the per-shape best, and this one wins
28 of the 30 weight shapes at a mean regret of 1.01x. Its worst case is an activation
quantize worth 0.07% of a step.

## Which axis is the block axis: the rule that shapes everything

**An MX block must run along the contraction axis.** The three products of a linear layer
contract different axes:

| product | contracts | block axis |
|---|---|---|
| FPROP `y = x @ W.T` | K = `in_features` | K of both operands |
| DGRAD `dx = dy @ W` | N = `out_features` | N of `dy` and of `W` |
| WGRAD `dW = dy.T @ x` | M = tokens | the token axis |

Two consequences follow, and almost every design decision in this subsystem is one of
them.

**The weight needs two fp8 copies, not one.** `(out, in)` blocked along K for FPROP and
`(in, out)` blocked along N for DGRAD. There is no deriving one blocking from the other:
the scales are different numbers, not a permutation of the same ones. That is 2 bytes per
parameter of fp8 on top of the 16-bit master. It is worth it because a weight is
quantized once per optimizer step and read by every microbatch — quantizing per call
would pay a per-step cost `grad_accum` times over. The MoE expert path needs **four**
copies for the same reason, two matrices at two blockings each.

**Keeping WGRAD 16-bit collapses the layout requirement** to: activations blocked along K
only, `dout` blocked along N only. That is the second reason for the choice below, on top
of the numerical one.

## Why the weight gradient stays 16-bit

FPROP contracts the model width and DGRAD the expert width — both a few thousand terms.
WGRAD contracts the **token** axis, the longest one, so its quantization error
accumulates over the most terms; and unlike an activation error, which is resampled
every micro-batch, a weight-gradient error integrates into optimizer state as systematic
bias.

That reasoning is sound but its *evidence* was not. A 0.21-nat gap at `MoE-1B-A280M` was
long attributed to fp8 in WGRAD; it was actually a hard-coded `tl.float16` multiply
inside the kernel flushing 1.26e-7 gradients to zero (100% below fp16's min normal).
Fixing the multiply took the gap from 72x the noise floor to 1.2x with the operands
unchanged. So the 16-bit *operand* choice has never actually been tested on its own — see
the open questions below.

---

# What gets converted

| path | kernel | notes |
|---|---|---|
| attention q/k/v/o | `MXFP8Linear` | vendor `torch._scaled_mm` |
| feed-forward pair | `MXFP8Linear` | vendor `torch._scaled_mm` |
| routed experts | fused Triton | GEMM1+SwiGLU, GEMM2+gate+combine |
| **weight gradients** | **not fp8** | see above |
| attention scores / AV | **not fp8** | out of scope |
| norms, router, embedding | **never** | declared `never` in the swap census |

`swap_mxfp8` returns an *accounting*, not a list of successes: anything it could not
convert is charged to a bucket, and `report.blocking` is true if the model would train
as a silent bf16/fp8 mixture. A preset whose shapes do not satisfy the block
constraints fails loudly rather than quietly running half converted.

---

# The dense path

## `MXFP8Linear`

```python
from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear

layer = MXFP8Linear(1536, 4096).cuda()      # no bias; in_features must be % 128
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = layer(x)                            # output dtype comes from autocast

loss.backward()
optimizer.step()
layer.refresh_quantized_weight()            # or refresh_mxfp8_weights(model)
```

An `nn.Linear` replacement (no bias) with fp8 FPROP and DGRAD on vendor
`torch._scaled_mm`, and a bf16 WGRAD. It holds the two quantized weight copies and
`refresh_quantized_weight()` rebuilds them; that is called after `optimizer.step()`
rather than lazily on a dirty flag inside `forward`, because under gradient accumulation
the weight is unchanged across microbatches and a per-call flag is a branch on state the
caller already knows the answer to.

### Padding, and why it is exact

The vendor's swizzled scale layout needs the contraction axis aligned to 128. Only the
two contraction axes need it, and **only one of them is the caller's problem**:

- `in_features` is FPROP's K, shared by the activation cast, so it is a hard requirement.
  Padding it would mean padding every activation that reaches the layer, so the
  constructor raises instead.
- `out_features` is merely DGRAD's K, so it is **zero-padded here** rather than rejected.
  That is what lets `MoE-2B-A370M` (kv_out=192) and `Nano-200M-wide` use this module
  without reshaping their GQA ratios to suit a kernel.

Zero-padding a contraction axis is **exact, not approximate**, and the quantizer is why.
The scale is `ceil(log2(amax / 448))` over each 32-wide block, and a zero never raises
`amax` — so a block straddling real and padded columns keeps the scale it would have had,
and every real value quantizes to the same byte. A wholly-padded block gets the
clamped-and-floored treatment described above, i.e. a finite scale over exact zeros. Both
halves matter, and the second is the one that would have been a silent corruption rather
than a wrong number.

The padding function is resolved once in `__init__` — `_unpadded` or a `_PadLastAxis`
instance — rather than branched on per call, because the width is a property of the
layer's shape. `F.pad` with a zero width still issues a copy, which is why the
alternative is an identity function rather than a width of 0. It is a class rather than a
closure so the module stays picklable.

WGRAD is **unpadded on purpose**: it contracts tokens, so the padded axis is this
product's N, and padding it would return a gradient wider than the parameter. cuBLAS
accumulates in fp32 and writes the operand dtype directly, so unlike the MoE WGRAD there
is no fp32 buffer here to fold away.

### The output dtype comes from autocast, not from the weight

That is what `nn.Linear` does — via ATen's autocast registration — and keying off the
weight instead is wrong for the only configuration that matters. A backbone holds **fp32
master** parameters and relies on autocast for its compute precision, so `weight.dtype`
is fp32 and a projection fed by a norm would hand attention an fp32 `v`. `varlen_attn`
refuses that dtype and attention drops **silently** to SDPA with an explicit `(T, T)`
mask — a different kernel with a quadratic allocation, so any A/B downstream would be
comparing two attention implementations without knowing.

Consulting global autocast state inside a module is correct even though it would be wrong
inside a kernel. A kernel takes precision as a caller argument because it has no business
knowing the surrounding context; a module *replicating* `nn.Linear`'s contract has to
consult exactly what `nn.Linear` consults. Hardcoding bf16 fails the other way: under
fp16 autocast it silently downcasts, and fp16 here measures 8x more accurate than bf16 at
the same speed, so that variant is live.

The activation is cast *before* being saved, so WGRAD's two operands cannot disagree —
`dout` arrives in the output's dtype and `d2d.t() @ x2d` raises outright on mismatched
operands, which is the loud half of the same bug whose quiet half is the SDPA fallback.
`dx` comes back in the dtype of `x` as the caller passed it, whatever autocast did in
between, or autograd hands the upstream graph the wrong type; since `out_dtype` is a GEMM
parameter that is free rather than a following cast.

### The quantized cache is dropped on `_apply`, and is not a buffer

`_apply`'s dtype transforms are gated on `is_floating_point`, which is **True for e4m3
and e8m0**, so `.float()` would rewrite a registered cache to fp32 and hand the vendor
GEMM something it cannot consume. `persistent=False` avoids the 3x checkpoint but not
that. A plain attribute cannot suffer the corruption at all. The cache is a pure function
of `weight` and is rebuilt after every optimizer step regardless, so carrying it across a
device or dtype move buys nothing and `forward` rebuilds it on the next call.

The DGRAD copy is quantized from `self.weight.t()` with **no `.contiguous()`**: the
quantizer takes both strides, and the copy cost 2.00 ms of a 15.3 ms refresh for a
bit-identical result.

## The vendor scale layout

cuBLAS reads MX scale factors as **128-row x 4-scale-column tiles**, each tile further
interleaved 32x4x4 — `SWIZZLE_32_4_4`. Anything else is rejected outright rather than
silently mis-scaled, which is the one mercy of this ABI.

```python
from kohakuwullm.kernels.mxfp8 import quantize_mx, quantize_mx_vendor
from kohakuwullm.kernels.mxfp8.interop import (
    as_vendor_scales, vendor_mxfp8_matmul_swizzled,
)

# Either emit the swizzled layout directly from the cast...
xq, xs_swz = quantize_mx_vendor(x)     # scales FLAT and e8m0-typed, not (rows, K//32)
wq, ws_swz = quantize_mx_vendor(w)
y = vendor_mxfp8_matmul_swizzled(xq, xs_swz, wq, ws_swz)   # xq @ wq.T

# ...or convert natural-layout scales, at 14-18 us of device time per operand.
xq, xs = quantize_mx(x)
xs_swz = as_vendor_scales(xs)
```

The wrapper exists so the swizzle stays *outside* the timed call: doing it inside
charges the GEMM for two full layout copies, which is pure overhead on the vendor's
side of any comparison.

`quantize_mx_vendor` needs `K % 128 == 0` to fill a 4-scale-column swizzle tile, where
`mxfp8_matmul_pq` needs only `K % 32 == 0`. A K between the two is served correctly,
but only through our kernel.

`interop.py` holds that contract in one place so both the precision test and the
benchmark compare against the vendor path from one definition of it. The comparison
earns its place as a **test**, not just a benchmark baseline: `mxfp8_matmul_pq` is
bit-identical to the vendor path on every shape measured, so any divergence means our
scale layout drifted — a bug no fp64 reference catches on its own, since both would
simply be "MXFP8-accurate" and disagree.

`swizzle_mx_scales` zero-pads to whole 128x4 tiles and returns its result **flat**,
because the padded extent is not the logical shape and an accidental 2-D read of it would
silently address the padding.

**The swizzle is fused into the cast wherever a kernel already writes scales.**
`quantize_mx_vendor` and `rms_norm_mx_vendor` emit the swizzled layout directly. A
separate `swizzle_mx_scales` pass measures 14–18 us of device time per operand, which is
pure overhead on the vendor's side of any comparison. Inside those kernels the contiguous
run is only the 4 scale columns of one row (stride 1), then rows stride by 16 within a
32-row group; a whole 128x4 tile is 512 bytes, so a program covering one writes it as a
single dense span even though the row-major order inside it is permuted. That is why
`_quantize_mx_vendor_kernel` uses fixed 128x128 tiles, unlike the natural-layout kernel —
a program covering less than one swizzle tile would have to write partial tiles from
several programs.

The swizzled scale buffer is allocated with `zeros`, not `empty`: a partial final row tile
leaves padding that cuBLAS still reads, and stale bytes there are scales for rows that do
not exist.

`quantize_mx_vendor` copies **only** a layout the kernel cannot coalesce. It tiles in 2D
and takes both strides, so it reads along whichever axis is unit-stride: 0.398 ms strided
against 0.389 ms contiguous on a 149553x896 fp32 read, i.e. free. An unconditional
`.contiguous()` bought nothing and cost 1.22 ms on `weight.t()`, 8% of a 96-module weight
refresh, for a bit-identical result. Only a doubly-strided slice needs it.

Note the alignment asymmetry: `mxfp8_matmul_pq` needs only `K % 32 == 0`, the natural
layout's own requirement, while `quantize_mx_vendor` imposes `K % 128 == 0` to fill a
4-scale-column swizzle tile. A K between the two is served correctly, but only through
our kernel.

## The standalone matmuls

`quantize.py` carries two, and only one of them is for training.

**`mxfp8_matmul_pq`** takes operands already in e4m3 + ue8m0 and is the shape a training
step actually wants: weights quantized once per optimizer step, activations once by their
producer's epilogue. Against bf16 it measures 1.60–1.94x.

**`mxfp8_matmul`** quantizes both operands *inside* the tile loop. That is possible only
because the block axis is the contraction axis, and it is kept — but it **loses**.
Program `(pid_m, pid_n)` quantizes an `A` tile that every other `pid_n` also quantizes
(32x redundant at 4096³), and the fp32 widening for the amax halves the shared-memory
budget. Against bf16 it is 0.75–0.99x and never wins. It survives as the correctness
reference: the one place the whole pipeline is visible in a single function.

Accumulation is fp32 always; the output dtype is the caller's. fp32 output is worth
having, because a norm or activation on the vector cores would otherwise re-widen a bf16
result immediately.

### The pre-quantized tiling, and two schedules that did not pay

Pre-quantized operands are one byte wide and never widened to fp32, so the shared-memory
budget admits tiles the fused kernel cannot. From a 200-config search at 4096³, a
**narrow-M, deep-K tile wins** in a way the shape reasoning does not suggest:
`64x128x256` at 3 stages and 4 warps reaches 461 TF/s against 425 for the widest tile, so
the binding resource is K-depth per stage rather than tile area.

Two schedule variants were measured against the plain 2-D grid at 4096³ and neither is
taken: a grouped-M tile ordering for L2 reuse of the B tile moved 461 → 464 TF/s, inside
run-to-run noise, and a persistent one-program-per-SM form *lost*, 461 → 453. Since
`tl.dot_scaled` already lowers to the native block-scaled MMA, what is left between us
and cuBLAS is not the tile schedule at this level.

**The transpose stays in the loop.** Storing the weight's fp8 copy already transposed to
`(K, N)` is free — it is quantized once per optimizer step — and measures **25% slower**,
339 against 461 TF/s at 4096³: `(N, K)` storage reads K contiguously per row, while
`(K, N)` strides every load across K. The register-level transpose is the layout the MMA
wants anyway. Every kernel in this family inherits that decision.

---

# The grouped path

## `grouped_mxfp8_gemm`

Block-scaled fp8 with a variable row count per expert. **There is no vendor kernel for
this shape**, which is what makes it worth owning. `torch._scaled_mm` does one dense
block-scaled GEMM and the dense path above already feeds it bit-identically at 601–702
TF/s, so Triton is off the critical path there. CUTLASS 3.9.2 ships an sm_120 grouped
block-scaled kernel (`sm120_blockscaled_mma_array_tma.hpp`) whose own example segfaults
in host verification, so it is not even a trustworthy oracle. Whatever fraction of peak
this reaches *is* the ceiling for every MoE model this repo trains.

```python
from kohakuwullm.kernels.mxfp8 import quantize_mx
from kohakuwullm.kernels.mxfp8.grouped import grouped_mxfp8_gemm

xq, xs = quantize_mx(x)                     # (T, K) quantized ONCE per token
wq, ws = quantize_mx(w.view(-1, k))         # (E, N, K) -> flat, then reshaped back

out = grouped_mxfp8_gemm(
    xq, xs, wq, ws,
    offsets=offsets,     # (E+1,) exclusive prefix sum of per-expert row counts
    index=token_of,      # optional: the gather becomes the A-index, for free
    rows=int(valid),     # optional row bound, kept on device by the caller
)
```

`index` is what makes the MoE dispatch's row gather stop being a kernel of its own —
a row of fp8 is contiguous in K, so an indexed row is the same coalesced transaction
at a different address. Pass `out=` a **zeroed** buffer when a sentinel bucket owns
rows no tile will visit.

Measured on the 96-expert / dim-1536 / hidden-576 preset's first expert GEMM
(E=96, N=1152, K=1536), CUDA-graph timed so no dispatch is charged:

| rows/expert | TF/s | against |
|---|---|---|
| 384 | 535 | 73% of a 732 TF/s **bandwidth** ceiling |
| 682 | 519 | 75% of the 692 TF/s this card demonstrates on *dense* vendor MXFP8 at 4096³ |

**Which ceiling binds flips with the row count, and quoting the wrong one is how a tuned
kernel looks broken.** At 384 rows/expert the whole working set is 304 MiB of which 123
MiB is the weight bank read once, so bandwidth binds; by 682 rows the arithmetic per
weight byte has doubled and compute binds.

### The tiling does not transfer from the dense path

Dense wins with `(64, 128, 256)` — narrow M, deep K, because `m16n8k32` wants about eight
K-steps per tile for ILP. Here that config reaches 462 TF/s and `(128, 128, 64)` reaches
519: the search over the tile owner plus the strided weight index costs enough registers
that a deep-K tile spills, and `BLOCK_K=256` at 4 stages does not fit in shared memory at
all.

The lever here is **weight reuse via a wider `BLOCK_M`**, since the weight bank is read
once per pass no matter how the rows are routed. Going further does not help:
`BLOCK_M=256` needs 8 warps to avoid spilling and then loses to 128 anyway, 392 against
519. The shipped tile is `(128, 128, 64)` at 3 stages and 4 warps, and it won at every row
count measured from 384 to 682.

**These are constants, not an autotune set.** `triton.autotune` re-benchmarks on each new
key, and these kernels are called with a *device-dependent* row distribution behind a
host-known grid bound — so the key would be stable while the work behind it is not, and
the tuning would be for whichever batch arrived first. The shape that matters (K = model
width, N = expert width) is fixed for a given model.

### The grid, and the tile owner

The grid is sized entirely from host-known values, so a whole MoE layer stays
CUDA-graph capturable — the same constraint and the same solution as the bf16
`grouped_gemm` described in [kernels.md](kernels.md). The row axis is flattened into one
tile index and each program resolves `(expert, local tile)` from `offsets` in registers,
bounding the grid at `cdiv(M, BLOCK_M) + E`. Reading the largest expert's row count would
cost an `.item()`.

**N goes on grid dimension 0**; putting the flat row tile there cost 18% by destroying
the reuse of `x` in L2.

`GATHER` reads each row's source index from `index_ptr` instead of using the sorted
position directly, which is how the MoE dispatch's row gather stops being a kernel of its
own. **It is free**: the A-load is one row per lane pair either way, and a row of fp8 is
contiguous in K, so an indexed row is the same coalesced transaction at a different
address.

`dequant_mx` is the inverse of the block quantizer, needed wherever a *gradient* has to
read an operand the forward consumed in fp8. It broadcasts through a `(ROWS, groups,
SUB)` reshape rather than `repeat_interleave` on the scale: Triton has no interleaving
primitive, and the reshape is free because it only renames the layout the values already
have.

`out` may be passed in, and the one reason to do so is to hand in a *zeroed* buffer. A
ReMoE sentinel bucket's rows lie outside every tile the grid resolves, so an
`empty`-allocated output leaves them holding garbage that a later scatter would spread
into real tokens. Zeroing here rather than masking the scatter keeps the row bound on the
device, which is the `.item()` this path exists without.

### The scale-column mask, and why it was silent

**The MX scale load carries its own column mask, and it is not optional.** Every kernel
in this family steps a `tl.cdiv`-rounded K loop, so a contraction length that is not a
multiple of `BLOCK_K` gives a final iteration whose scale column is past the end of the
scale tensor — while the paired *value* load is masked to zero by `mask_k`.

That asymmetry is what made the read silent. A garbage exponent times zero is zero, so no
correctness test could see it, and `compute-sanitizer` cannot either, because the overrun
is a byte or two inside PyTorch's 512-byte-rounded allocation block and is therefore a
legal access. The one probe that sees it is a padded tensor behind a narrow view poisoned
with `0xFF`: that is NaN in e8m0, and `NaN * 0` is NaN. `tests/test_kernels.py` does
exactly that for the bare GEMM and for the fused expert path, each against an executed
unmasked twin.

`K_EXACT` is resolved on the host from `contraction % BLOCK_K`, so the common case
compiles the mask away rather than testing a predicate every iteration — the guard is free
where it does not apply, not a cost traded against safety. `_experts_gemm1_dgrad` is the
one kernel that can never need it: it contracts `2 * hidden`, always a multiple of 64 when
`hidden` is a multiple of the 32-wide MX block, so its loop never rounds up.

## The WGRAD kernel

`dW[e] = A[rows(e)].T @ B[rows(e)]` — a 16-bit multiply with an fp32 sum, one kernel with
two gather flags rather than two kernels, because both expert matrices need this product
and in each case exactly one operand is indexed by `token_of` while the other is already
in sorted order.

**Its tile inverts the forward's.** WGRAD contracts the token axis, so its tile is `(N,
K)` *of the weight* and the loop steps over rows — which inverts what `BLOCK_M` is for,
and with it the tuning. The `(128, 128)` fp32 accumulator already costs 128 registers per
thread at 4 warps, so the row step wants to be **small**: 32 beats 64 and 128, measured
197 TF/s for `dw_in` at the 8B preset against 150 at `(64, 64, 64)`. Deep-K is a trap
here — `(64, 128, 256)` at 4 warps spills so badly it runs 85 ms, **70x slower**.

### `MUL_DTYPE` is the caller's dtype, never a fixed one

This is the defect described above, in full. The multiply was an unconditional
`tl.float16`, chosen on a measurement taken at `randn` scale (2.07e-04 against bf16's
1.66e-03) that training never reproduces. fp16's smallest *normal* is 6.1e-5 and a weight
gradient is 1e-6 or below, so a bf16 caller's operand was cast into fp16's subnormal range
and lost most of its mantissa — 5.2x more error at 2^-20 than at 1.0, where an 8-bit
exponent is exactly scale-invariant. **A weight gradient needs range first and mantissa
second.** That gap was 0.21 nats at `MoE-1B-A280M`.

The launcher takes `MUL_DTYPE` from the `A` operand, not from `dw`: the accumulator's
rounding is an epilogue and this is the multiply's operand precision. The two coincide for
every caller today, but the one that matters is the operand.

### `B_FP8` is a separate question from the multiply

Whether the **operand** is fp8 is not the same as what precision the multiply runs in.
Reading B as fp8 and dequantizing is defensible on its own terms — the forward multiplied
the fp8 operand, so it is the exact gradient of the function that ran, and it carries no
second tensor from the forward. Every shipping caller takes the other branch for the bias
reason above: `MXFP8Linear`'s WGRAD multiplies the 16-bit `x2d`, and both MoE paths pass
`bs=None`.

The flag is kept because it makes an A/B over the operand's quantization possible with the
multiply held fixed — which is exactly how the 0.21 nats were attributed to `MUL_DTYPE`
and not to the operand. To keep that A/B honest, the 16-bit branch is widened to fp32 and
narrowed again by the `tl.dot` exactly as the dequantized path is, so the flag varies
quantization and not two arithmetic pipelines.

When `bs is None` the scale pointer still has to be *something* Triton can bind, so `bq`
stands in for it and the compiled kernel drops the load: `B_FP8` is a constexpr, so the
unused argument costs a pointer in the signature and nothing at run time. Passing a
one-element dummy instead would allocate per call for a tensor no branch reads.

---

# The fused expert path

## What is being removed

The eager expert path is `quantize → gather → GEMM1 → swiglu → quantize → GEMM2 →
combine`. Every arrow is a full round trip through HBM of a tensor that is `top_k` times
the size of the activation. At (E=96, top_k=8, dim=1536, hidden=576, T=8192) the non-GEMM
half of that costs **515 us out of 2117 — 24%** — of which the gather is already at 99% of
STREAM and a `torch.compile`d swiglu at 99.9%.

**Those cannot be made faster, only made to not exist.** So they are folded into the two
GEMMs that have to touch the same bytes anyway:

- **`x` is quantized once per token, not once per routed row.** The MX block runs along K
  and the gather runs along rows, so gathering quantized rows and quantizing gathered rows
  give bit-identical operands — and the first is `top_k` times less quantizer work over a
  tensor `top_k` times smaller.
- **The gather becomes GEMM1's A-index**, for the reason given above: a row of fp8 is
  contiguous in K either way.
- **SwiGLU and the requantize become GEMM1's epilogue.** This is the fusion the MX format
  makes free: GEMM2 contracts the expert width, so its scale blocks run along the axis
  GEMM1's output tile already spans. A tile owns whole blocks, and nothing has to be seen
  twice to compute a scale.
- **The combine becomes GEMM2's epilogue** — gate scale and scatter-add straight from the
  accumulator.

Measured, the fusion is worth **1.65x** on top of the 1.49x the dtype alone buys.

## `h` is rounded before the activation, on purpose

The GEMM1 accumulator holds more precision than the storage dtype, and rounding it before
applying SwiGLU is *less* accurate. It is still correct and the alternative is not: the
backward can only read what was stored, so keeping fp32 there would evaluate the
backward's SwiGLU derivative at a point the forward never used. **A gradient consistent
with the function is worth more than a function three mantissa bits better than the one
being differentiated.** The dtype arrives as `PRE_DTYPE` rather than being read off
`pre_ptr`, because inference passes `None`.

## Recovering the gate-weight gradient

Fusing the combine into GEMM2 destroys `out_sorted`, which `dL/dg` would otherwise need.
It comes back without recomputing anything from

    dL/dg_p = <dout[t_p], h_p @ W_out.T> = <dout[t_p] @ W_out, h_p>

whose left factor is exactly the unscaled DGRAD tile the backward already holds. So the
combine's backward costs one extra reduction in an epilogue rather than a second grouped
GEMM.

In `_experts_gemm2_dgrad` the accumulator `dout[t] @ W_out` is consumed three ways in
registers — dotted with `h` for the gate gradient, scaled by the gate for the SwiGLU
backward — because writing `dh` out and reading it back twice is two round trips over a
`top_k`-sized tensor. `h` is read back from its **fp8** copy, not a 16-bit one, and that
is deliberate and exact: the forward multiplied the fp8 values, so their dequantization is
the operand the true derivative contains.

## Precision decisions in the epilogues

**Both gradient scatters — `out` in the forward and `dx` in the backward — add atomically
in the caller's 16-bit dtype rather than fp32.** Depth is `top_k`, and the sum is rounded
to that same dtype at the end regardless, so an fp32 accumulator would shrink a 0.55%
error to 0.2% *inside* a container whose own quantum is 0.4% — while doubling 201 MiB of
atomic traffic.

**Which gradients get an fp32 buffer is decided by who writes each element, not by how
much precision it deserves.** `dw_in` and `dw_out` are written by exactly one program per
element — the WGRAD grid gives each `(expert, n-tile, k-tile)` its own program and loops
the token axis inside it — so their fp32 reduction happens in registers and the epilogue
can round straight to the caller's dtype. Allocating them fp32 and casting afterwards is
bit-identical and costs 1620 MiB per layer against 324.

`dgate` cannot do this. Its reduction crosses programs through an `atomic_add`, so its
accumulator has to be fp32 *in memory*, and its cast on the host is the one that stays.
It is also the only reduction in the backward whose depth is not bounded by `top_k`: each
pair's gate gradient is a dot product over the expert width, split across `BLOCK_N` tiles.

## The tiles, and the one cliff

`GEMM1_TILE`'s **`BLOCK_N` is per half.** GEMM1 computes the gate and value columns of one
hidden index in the same program, so its effective N tile is `2*BLOCK_N` and the 64 in the
config is the searched 128. Splitting the pair across two programs cannot work — the
epilogue needs both halves to apply SwiGLU, so one of them would have to go to memory,
which is the round trip this kernel exists to remove.

**Forgetting the per-half factor is a cliff, not a slowdown.** At `BLOCK_N=128` the two
accumulators plus the quantizer's fp32 widening spill hard enough to run **31.2 ms against
0.61**, while every neighbouring config is within 10%.

The backward tiles do not inherit the forward's, and the gap is not small:

| kernel | tile | stages | measured |
|---|---|---|---|
| GEMM2 DGRAD | `(64, 128, 128)` | 2 | 277 TF/s, ~81% of the ~340 its own traffic allows |
| GEMM2 DGRAD at the forward tile | `(128, 128, 64)`, 4 warps | 3 | spills, 99 TF/s |
| GEMM1 DGRAD | `(128, 128, 64)` | 2 | 400 TF/s |
| GEMM1 DGRAD at 3 stages | `(128, 128, 64)` | 3 | 384 TF/s |

Both DGRAD kernels want **two** pipeline stages where every forward kernel wants three: a
forward epilogue writes its tile and stops, while these either scatter atomically into
`(T, dim)` or run a dequantize, a full-row reduction, a SwiGLU backward and two quantizes
over the same registers. The stages lose to the epilogue. GEMM2's DGRAD is bound by
registers and epilogue bandwidth rather than by the MMA, which is the whole 2.8x.

GEMM1's DGRAD contracts `2H` and scatters `(M, dim)` atomically, so its N tile is the
model width and `BLOCK_N` is *not* per-half the way the forward's is. A `BLOCK_M` of 256
measured 404 TF/s, marginally ahead, and is not taken: at 682 rows per expert it wastes
11% of each tile on mask and at the gradient-accumulation ~85 rows it wastes two thirds,
for 1% at the tuning point.

The scatter-add in GEMM1's DGRAD is the price of fusing the gather into the forward: a
token routed to `top_k` experts collects `top_k` contributions, which the eager path
bought with `index_select`'s backward over a materialized `(M, K)` gradient. Adding
straight from the accumulator skips that tensor entirely.

## Wiring details that bite

**`MXFP8ExpertWeights` is not an `nn.Module`.** It owns no parameters, only buffers
derived from them, and holding them in a module would put them in `state_dict` and make a
checkpoint 3x the size of the weights it represents. Its `refresh` rebuilds all four
copies from the 16-bit masters and is called after each optimizer step.

**FPROP, DGRAD and `dgate` read the fp8 activations; the two weight gradients do not.**
WGRAD multiplies the 16-bit `x` saved from the forward and an `h` rebuilt from the stored
pre-activation. `pre` holds the rounded values GEMM1's epilogue used, so rebuilding `h` is
the same function at the same point for one elementwise pass (~1.1 ms across the stack)
against carrying another `(M, hidden)` tensor. The cost of saving `x` in 16 bits is about
0.97 bytes per element on the **ungathered** activation, `top_k` times smaller than the
routed rows.

**`ctx.needs_input_grad`, never `torch.is_grad_enabled()`.** Autograd runs `forward` under
`no_grad`, so the latter is always False there — asking it silently turned off the
pre-activation save that the backward then unpacked.

**`empty`, not `zeros`, for the rows a sentinel bucket owns** in the fused path:
`offsets` is truncated to the real experts, so no tile the grid resolves covers them and
nothing downstream reads them. Zeroing `2H` columns per routed row to guard a read that
cannot happen would cost a full pass over the largest tensor there. `_experts_gemm2`
needs no `valid_rows` guard for the same reason — a row it does not visit is a row it
cannot contribute garbage from. The *unfused* path does need the zeroed buffers, because
its GEMMs write through `grouped_mxfp8_gemm` into tensors a later `index_add_` reads;
`_maybe_zeros` returns `None` for every router without a sentinel bucket, where the memset
would be a full pass over the largest tensor bought for a case that cannot arise.

**`x`'s last stride is checked, not repaired.** WGRAD reads `x` as its B operand and
indexes the contraction axis with no stride of its own, so a non-unit last stride is read
as if it were unit and gives a silently wrong gradient — the forward's `quantize_mx` takes
both strides and would not notice. Every real caller arrives through `reshape` + `to` and
is already contiguous, so a `.contiguous()` here would be dead code hiding a caller that
meant something else.

**`x.dtype` is refused rather than lifted**, because `h`'s storage dtype has to be 16-bit
for the forward and the backward to differentiate the same SwiGLU. fp32 is the dtype that
actually arrives, on a 16-bit model too: normalization is on autocast's fp32 list, so a
caller that does not consult autocast reaches here with the norm's fp32 output whatever
its weights are.

`_validate_expert_call` is shared by the fused and unfused entry points so the two stay
exchangeable in an A/B — a path that accepted an input its twin refuses would not be a
control for it.

## The unfused path, and why it exists

`moe_unfused.py` composes the same arithmetic out of verified pieces instead of fusing it.
Every GEMM in it is `grouped_mxfp8_gemm`, the one grouped kernel checked **bit-for-bit
against a loop of vendor `torch._scaled_mm` calls**. The fused path has four kernels of
its own whose only oracle is fp64, and an fp64 oracle cannot see a scale layout that
drifted: a consistently wrong scale is still "MXFP8-accurate" against a reference derived
the same way.

So this path buys its correctness argument with round trips. Two of the fused path's four
savings are free and are kept, because they are properties of the shared primitive rather
than of the fusion: `x` is quantized once per token, and the gather is GEMM1's A-index.
What is given up is everything with an epilogue — SwiGLU, the requantize, the gate scale
and the scatter each become their own launch, and each of those already has a reference of
its own in `swiglu_mul`, `combine_routed` and `quantize_mx`.

The gate-weight gradient is the clearest illustration of the trade. Not fusing the combine
keeps `out_sorted`, so the gradient is `combine_routed`'s own backward — a kernel that was
already pinned against an autograd-composed reference before any of this existed.

`_GroupedExpertMatmul` is the only autograd node this path adds. Everything between the
two expert GEMMs is ordinary autograd over kernels that already existed, which is what
makes the composition auditable one piece at a time. It uses `swiglu_mul` rather than
`F.silu(a) * b` because `_as_2d` takes the chunks' row stride so nothing is copied, and
the eager pair would keep a third `(M, H)` tensor alive for the multiply's backward.

---

# What has been verified, and how

Verification is layered, because each layer catches what the one below cannot.

1. **Bit-equality against vendor cuBLAS.** `grouped_mxfp8_gemm` is `torch.equal` to a
   per-expert loop of `torch._scaled_mm`, and every fused expert GEMM is bit-identical
   to `grouped_mxfp8_gemm`, across ragged / empty / single-row experts and contraction
   lengths both divisible and indivisible by the tile.

   ```python
   from kohakuwullm.kernels.mxfp8.interop import vendor_mxfp8_matmul

   got = grouped_mxfp8_gemm(xq, xs, wq, ws, offsets, out_dtype=dtype)
   off = offsets.tolist()
   for e in range(experts):
       if off[e + 1] <= off[e]:                   # an empty expert is a real case
           continue
       vendor = vendor_mxfp8_matmul(
           xq[off[e] : off[e + 1]], xs[off[e] : off[e + 1]], wq[e], ws[e], dtype
       )
       assert torch.equal(got[off[e] : off[e + 1]], vendor)   # equal, not allclose
   ```

   Bit-equal rather than close: both consume the same e4m3 values and the same ue8m0
   exponents through the same MMA, so any difference at all is a layout bug.
2. **fp64 oracles in ULP**, both fp16 and bf16, forward and backward.
   `grouped_mxfp8_reference` is an fp64 matmul on dequantized operands rather than a loop
   of `_scaled_mm` calls, precisely so it shares no assumption with the vendor check
   above; the two layers are separate on purpose.
3. **Realistic gradient magnitudes.** Tests parametrize `dout_scale` over unit and
   training scale (2^-20). This is not decoration: every unit-scale case passed on the
   kernel that was flushing real gradients to zero.
4. **A structural reference for the whole path.** `mxfp8_moe_experts_reference` is
   autograd-composed in the input dtype with **no quantization**, deliberately — comparing
   two fp8 implementations only shows that they agree. It pins the routing, the gather,
   the SwiGLU and the combine; the precision oracle is fp64 over explicitly quantized
   operands, in `tests/test_kernels.py`.
5. **A 400-step loss A/B** against bf16 *and* a bf16-with-another-seed control, because
   an absolute loss difference is not a result. The fp8 arm sits at 1.2x the control's
   own deviation.

**A reference that shares the kernel's assumption proves nothing.** Both defects found
in this subsystem survived their tests that way — the WGRAD fp64 oracle mirrored the
kernel's own fp16 cast, and a MoE microbenchmark held the token count fixed, hiding an
autotune that re-benchmarked every step. The vendor bit-equality check is the only layer
here that shares no assumptions with the thing it checks, which is why it is the one
that cleared the GEMMs cleanly.

---

# Performance

Measured on 4x RTX 5090 under pipelining at micro 8192, against bf16 on the same cards
(`out/bench/train/kohaku_e2e/`):

| rung | speedup | peak memory |
|---|---|---|
| Kohaku-200M | 1.081x | ~unchanged |
| Kohaku-500M | 1.139x | ~unchanged |
| Kohaku-1B | 1.221x | +0.3 GiB |
| Kohaku-MoE-1B | 1.533x | **-44%** |
| Kohaku-MoE-2B | 1.661x | **-43%** |
| Kohaku-MoE-5B | 1.895x | +1.3 GiB (checkpointed) |
| Kohaku-MoE-8B | **2.308x** | +2.1 GiB (checkpointed) |

Two things that do not generalize and are easy to misquote:

- **Memory.** The sparse saving is the fused epilogues not materializing
  `(tokens*top_k, hidden)` intermediates — an *activation* saving. With gradient
  checkpointing there are few stored activations left to save, so the two cached fp8
  weight layouts show in full and the sign flips. Never plan capacity on "switch to fp8
  to fit a bigger model".
- **Micro-batch.** At 4096 the GEMMs are small enough that quantization overhead is not
  amortized and fp8 *loses* below 1B. The optimal micro-batch differs between the bf16
  and fp8 arms, so tuning one for both hands the advantage to whichever it suits.

`scripts/bench/moe/kernel_ladder.py` separates where the speed came from, measured as the
sequence actually built rather than naive-versus-final:

    eager experts -> grouped bf16    5.7x   (already in the shipped baseline)
    grouped bf16  -> grouped fp8     1.49x  (the dtype)
    grouped fp8   -> fully fused     1.65x  (the fusion)

# Hardware notes (sm_120)

- No TMEM, so FlashAttention-4-class techniques do not apply and Triton fp8 GEMM caps
  at roughly 70-85% of cuBLAS.
- TMA exists but **cannot carry ue8m0 scales**: its inner box must be a 16-byte
  multiple and scale tiles are 4-8 bytes.
- CUTLASS grouped block-scaled GEMM is unusable here through 4.3.4 — the `MX_F4F6F8`
  whitelist matches `KernelPtrArrayTmaWarpSpecializedCooperative` with
  `cute::is_same_v` rather than `is_base_of_v`, so the SM120 grouped tag falls between
  two matching conventions. See [upstream-cutlass-findings.md](../performance/upstream-cutlass-findings.md).
  Its correctness is also bounded: correct within one wave, trapping above the 170-tile
  SM count, and 4.x does not fix it.

# Open questions

- **fp8 WGRAD has never been tested on its own.** The experiment that appeared to rule
  it out was confounded by the fp16 multiply described above. WGRAD is 38-43% of the
  fused path at ~210 TF/s against ~520 for the fp8 products, so this is the largest
  remaining lever, and `B_FP8` exists to make the experiment a one-flag change.
- **The arm is host-bound**: 4050 device ops per step against bf16's 3249, and
  wall-minus-device 53 ms against 26. Device time says 1.26x where the step delivers
  1.078x, so the next dense gain is launch count, not kernel speed.
- **fp16 parameters need loss scaling.** `dpre` is allocated in the caller's dtype and
  gradients are ~1.26e-7 RMS: storing them in fp16 flushes 18.9% of entries to zero
  against bf16's none. No kernel change reaches this — it is a storage decision.
- **`combine_routed` is a nondeterministic scatter-add**, disagreeing with itself by
  0.0625 in bf16. Any A/B downstream of it needs that noise floor reported alongside.

---

## Where the Triton pre-quantized GEMM stops, and why

Measured on a free RTX 5090, bf16 out, against `F.scaled_mm` with pre-swizzled
scales. Four changes took the kernel from 63.5-83.5% of the vendor to
72.1-94.3%, mean 81.7%: a wider autotune space, group-M rasterization,
`max_contiguous`/`multiple_of` hints, and an aligned fast path gated on K depth.

The remaining gap on square shapes is about 26 points and it is **not** any of
these, all tested:

| hypothesis | result |
|---|---|
| tile selection | swept 12 tiles x 2 group modes x 2 warp counts x 3 stage depths |
| rasterization | added, worth ~0 on this card, as for dense bf16 |
| contiguity hints | added, worth a few points |
| mask arithmetic in the K loop | aligned path added, worth 10 points at K=4096 |
| `tl.trans(bq)` in the main loop | **refuted.** Removing it costs 17 points at 4096 cubed and 11 at 8192 cubed. Triton folds the transpose into the shared-memory layout, and storing B as `(N, K)` keeps K contiguous for the load. Storing `(K, N)` to avoid the transpose is strictly worse. |

What is left is the **scale feed**. The block-scaled MMA takes its scale factors
in a hardware-mandated per-thread register layout: for the A operand threads 0
and 1 of each quad supply them and the other two hold copies, and for B only
thread 0 does, a 4x replication. cuBLAS is handed scales already in the
`SWIZZLE_32_4_4` layout, so it loads them straight into that arrangement.
`tl.dot_scaled` takes the natural `(rows, K/32)` layout and the compiler must
shuffle into the fragment layout inside the loop.

Note also that the vendor arm is timed with the swizzle hoisted, which is fair
for training, where `quantize_mx_vendor` writes the swizzled layout straight out
of the cast once per optimizer step. It does mean the comparison hands the
vendor a layout our kernel is not allowed to consume: Triton exposes no way to
pass pre-swizzled scales to `tl.dot_scaled`.

So roughly 80% of the vendor is the practical ceiling for a Triton MXFP8 GEMM on
`sm_120` today, and closing the rest needs either a `tl.dot_scaled` that accepts
the swizzled layout, or dropping to inline PTX for the `mma.sync` and its scale
operands.
