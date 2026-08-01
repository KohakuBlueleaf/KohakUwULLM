# MXFP8: epilogue fusion, and a training-aware attention kernel

Three things, in the order the numbers say to do them. Measured on one RTX 5090,
a transformer block with `D=2048`, `FFN=8192`, 4096 tokens, forward only.

---

## 0. Where the block's time actually goes

| component | ms | share |
|---|---|---|
| block total, our Triton MXFP8 | 3.30 | 100% |
| the five GEMMs | ~1.04 | 32% |
| **activation quantization (5 calls)** | **0.368** | **11.2%** |
| SDPA, norms, SiLU, residuals | ~1.9 | 57% |

Two consequences that should govern effort.

**The GEMM is a third of the block.** Closing our entire 1.31x kernel gap to
CUTLASS is worth **7.7%** of the block. An ISA-perfect MXFP8 GEMM is worth 16%.
Those are the ceilings on all further GEMM work.

**Quantization costs more than the GEMM gap.** 11.2% against 7.7%. Fusing the
casts is the larger win and it is entirely within our control, requiring no
CUTLASS-class kernel and no Triton feature we lack.

---

## 1. Epilogue fusion: emit MXFP8 where the values are already in registers

[../performance/gemm.md](../performance/gemm.md) section 3 already states the
rule: **do not cast in the consumer's prologue, cast in the producer's
epilogue**, because the load path is a copy engine and cannot compute, while the
producing kernel already has the values in registers. Section 0 above is what
ignoring that rule costs.

### The free one: stop quantizing the same tensor twice

`gate` and `up` consume the **same** normed hidden state, and each quantizes it
independently. One of the four `[T, 2048]` quantizations is pure duplication:
**0.072 ms, 2.2% of the block, for a caller-side change.** Do this first.

### The three real fusions

| producer | consumer | what it saves |
|---|---|---|
| RMSNorm epilogue | qkv, gate, up | one `[T, D]` quantize, plus a bf16 round trip of the normed tensor |
| `SiLU(gate) * up` epilogue | down | one `[T, FFN]` quantize (0.082 ms) **and** a `[T, 8192]` bf16 write plus read, which is 134 MB, about 73 us at 1.83 TB/s |
| attention output | o projection | one `[T, D]` quantize |

The SiLU-times-up fusion is the biggest single item, because the intermediate is
`FFN`-wide: it is both the most expensive quantize and the largest bf16 tensor
that never needs to exist.

### Why this is a kernel change, not a graph change

`torch.compile` cannot do it. Inductor can fuse elementwise chains, but the MX
cast is a **blockwise reduction** — the per-32 absolute maximum along the
contraction axis — followed by a scaled round. That is a different fusion class,
and the output is a pair of tensors in a layout no Inductor pattern knows. It
has to be written into the epilogue of our own norm, SiLU and attention kernels.

### The MoE case is the same shape, more so

Every routed expert consumes a gathered activation. Quantizing after the gather
means quantizing the same rows once per expert they route to. Quantizing in the
**gather epilogue** does it once. With top-k routing the saving scales with k.

---

## 2. Attention is structurally harder than the MLP

In a linear layer one operand is a **weight**: quantized once per optimizer step,
so its cost divides across every GEMM that uses it. **In attention there are no
weights.** Both operands of every attention GEMM are activations, so each one
needs two online quantizations.

The five GEMMs of a training attention:

```
forward     S = Q K^T                    both activations
            O = P V                      both activations, P from softmax
backward    dV = P^T dO
            dP = dO V^T
            dQ = dS K,   dK = dS^T Q
```

That is up to ten online casts per attention call against five in the whole MLP.
**Section 1 is therefore a prerequisite for section 3, not a parallel task.**

---

## 3. A training-aware MXFP8 attention kernel

[SageAttention3](https://arxiv.org/pdf/2505.11594) is the reference. It does FP4
microscaling for inference and, more relevantly, **SageBwd**, the first 8-bit
*training* attention. It uses INT8 because it targets RTX 4090 class hardware.
On `sm_120` that choice inverts, for a reason we measured.

### Use MXFP8, not INT8, on this card

| instruction | measured | note |
|---|---|---|
| `mma.sync ... kind::mxf8f6f4.block_scale` | **1096.7 TFLOP/s** | ours, four independent chains |
| `IMMA` INT8 | about 414 TOP/s | mmapeak's 828 halved for its 2x shape over-count |
| bf16 into fp32 | 279.3 TFLOP/s | |

**MXFP8 block-scaled is about 2.6x INT8 here**, and 3.93x bf16. SageBwd's INT8
result transfers as *method*; its *format* choice does not.

MXFP8 is also the better fit numerically for attention. INT8 needs a per-tensor
or per-row scale and a calibration. MXFP8 carries a `ue8m0` exponent per 32
elements along the contraction axis, which adapts to exactly the thing attention
has: wide dynamic range within a row.

### The scale axis lands correctly, which is not obvious

MX scales run along the **contraction** axis. Check each GEMM:

- `S = Q K^T` contracts over `head_dim`. At 128 that is 4 scale groups per row,
  at 64 it is 2. Whole groups, no straddling.
- `O = P V` contracts over the **key sequence**. In a flash tiling the key axis
  is walked in `BLOCK_N` chunks, so as long as `BLOCK_N` is a multiple of 32 each
  tile owns whole groups.
- The backward GEMMs contract over sequence or head_dim likewise.

So the format is compatible with a flash tiling without reshaping anything.

### Softmax quantization: flash already computes the first level for free

SageAttention3's key numerical trick is **two-level quantization** of the
probabilities: row-normalize first, then block-quantize, which it reports cuts
data-range error by about 80%.

In a flash kernel that first level costs nothing, because the running maximum is
already there. The kernel computes

```
P_tilde = exp(S - m_i)          in (0, 1], already row-normalized by the max
O_i     = P_tilde V / l_i       the division by the running sum is deferred
```

`P_tilde` is exactly the row-normalized quantity SageAttention3 constructs
deliberately. Quantizing `P_tilde` to e4m3 with per-32 block scales, in
registers, immediately after the `exp` and before the `PV` MMA, is the natural
formulation rather than an extra step. **The deferred `1/l_i` must stay in
fp32** and be applied to the accumulator, never folded into the quantized
operand.

### The one GEMM that must stay high precision

SageAttention3 keeps `dP = dO V^T` in FP16, reporting that INT8 there distorts
gradients enough to impede learning. The mechanism is visible in the softmax
backward:

```
dS = P * (dP - rowsum(dP * P))
```

That is a **cancellation**. `dP` and `rowsum(dP * P)` are close in magnitude, so
absolute error in `dP` does not cancel with them; it survives into `dS` with the
relative error amplified by however much cancellation occurred. This is the same
rule as the repo's existing numerics guidance that a low-precision reduction is
never trustworthy, applied one GEMM upstream.

**Rule: any GEMM whose output feeds a subtraction of near-equal quantities stays
in bf16.** Keep `dO V^T` high precision, quantize the other four.

Expected budget, with `dP` excluded: four of five GEMMs at up to 3.93x, one at
1x. If the five were equal cost the ceiling would be about 2.4x on attention
arithmetic. They are not equal, so measure before promising anything.

### The hard part is not the math, it is the register file

`P` is never materialized in a flash kernel, so its quantization has to happen
**inside** the fused kernel, in registers, between the `exp` and the `PV` MMA. No
library GEMM can be used. That means `tl.dot_scaled` inside a flash loop, and
everything measured in [mxfp8-difficulty.md](mxfp8-difficulty.md) applies:

- scale staging cost **+62 registers per thread** in a plain GEMM (220 versus
  158), and a flash kernel already carries the accumulator plus the running `m_i`
  and `l_i`;
- forcing registers down to fit a second CTA spilled 102 to 432 registers and
  collapsed throughput to 0.08 to 0.15x;
- `warp_specialize` around `tl.dot_scaled` does not compile.

So an MXFP8 flash kernel is expected to be **register-critical from the first
line**. Budget it before writing it: accumulator `BLOCK_M x head_dim` fp32, plus
`m_i` and `l_i`, plus two operand fragments, plus scale staging for two
quantized operands.

### Order of work

1. **Section 1 first.** It is 11.2% of a block, needs no new capability, and its
   epilogue-cast machinery is exactly what attention will need ten times over.
2. **Forward-only MXFP8 attention** next, since `S = Q K^T` and `O = P V` are the
   two GEMMs with the cleanest scale-axis story and no cancellation hazard.
3. **Backward** last, with `dO V^T` pinned to bf16 from the start, and an
   ablation that puts it back in MXFP8 to confirm the hazard on our data rather
   than inheriting it.

---

## Sources

- [SageAttention3: Microscaling FP4 Attention for Inference and an Exploration of 8-Bit Training](https://arxiv.org/pdf/2505.11594)
- [SageAttention repository](https://github.com/thu-ml/sageattention)
- [mxfp8-difficulty.md](mxfp8-difficulty.md) for the sm_120 register and layout limits
