# Training-aware MXFP8 attention

A varlen flash attention whose score GEMM runs in block-scaled e4m3. It is a
peer of the bf16 kernel in [kernels.md](kernels.md), not a variant: same layout,
same grid, same autograd node shape, same `cu_seqlens` and window semantics. Only
the operands of `QK^T` change.

- `kernels/mxfp8/attention_fwd.py` — the forward kernel
- `kernels/mxfp8/attention_bwd.py` — `dK`/`dV` and `dQ`, opposite axes
- `kernels/mxfp8/attention_quant.py` — `(T, H, D)` -> e4m3 + ue8m0
- `kernels/mxfp8/attention.py` — the autograd node and `mxfp8_varlen_attn`

`_bwd_preprocess` is imported from the bf16 backward unchanged; the `delta` term
does not depend on the score precision.

---

## 1. What is quantized, and what is not

| product | precision | reason |
|---|---|---|
| `QK^T` | e4m3 operands, ue8m0 per-32 block scales, fp32 accumulate | the only GEMM whose inputs are activations on both sides |
| `PV` | input dtype (bf16/fp16) | `P` is already a probability; quantizing it costs accuracy the softmax cannot absorb |
| `dO V^T` | input dtype | feeds `dp - delta`, a near-exact cancellation |
| `dS^T Q`, `dS K` | input dtype | `dS` has already lost the exponent range that made fp8 attractive |

This split is the SageAttention3 result: the backward's `dp - delta` subtracts two
nearly equal quantities, so any error in `dp` is amplified by the ratio of `dp` to
their difference. Quantizing `V` or `dO` is what makes 8-bit attention untrainable;
quantizing `Q` and `K` is not.

Scale blocks run along the head dimension, which is the contraction axis of
`QK^T`. That is the only axis a `tl.dot_scaled` can take them on.

## 2. K smoothing

`quantize_heads` subtracts K's per-channel mean before quantizing. The claim that
makes it free:

```
softmax(Q (K - 1 mu^T)^T) = softmax(Q K^T - (Q mu) 1^T) = softmax(Q K^T)
```

because the removed term is constant along a score row, and softmax is invariant
to a per-row shift. Nothing is added to the forward but a subtract in registers.

**Neither gradient needs a correction term.** Write `p = softmax(s)` and
`ds_j = p_j (dp_j - delta)`. Then

```
sum_j ds_j = sum_j p_j dp_j - delta sum_j p_j = delta - delta = 0
```

so every row of `dS` sums to zero. Two consequences:

- `dQ = dS (K - 1 mu^T) = dS K - (sum_j ds_j) mu^T = dS K`. Smoothed and
  unsmoothed K give the *same* `dQ` in exact arithmetic. We use the smoothed one:
  in bf16 the row sum is only approximately zero, and the residual is multiplied
  by `mu`, so the unsmoothed form loses accuracy in proportion to how much
  smoothing was needed in the first place.
- `mu` depends on K, so the chain rule adds `-(1/T) sum_a dKs_aj` to `dK`. That
  term is `sum_a (dS^T Q)_aj = sum_m Q_mj (sum_a ds_ma) = 0`. `dK = dS^T Q`
  stands unchanged.

`mu` is `(H_kv, D)` fp32, computed with `mean(dim=0, dtype=torch.float32)` — fp32
accumulation without materializing a widened copy — and saved for the backward,
where `_bwd_dq_kernel` subtracts it in registers. No shifted copy of K is ever
written to memory.

## 3. Measured error

RTX 5090, bf16 inputs, against a per-document fp64 reference. Relative L2, which
is the right metric here: non-causal attention averages over every key, so its
peak output is a cancellation and a max-relative metric reads about 2x high.

| case | forward | dQ | dK | dV |
|---|---|---|---|---|
| causal, 4 documents, GQA 8/2 | 0.034 | 0.024 | 0.032 | 0.013 |
| causal + window 256 | 0.033 | 0.038 | 0.038 | 0.021 |
| bf16 varlen control | 0.002 | | | |

Roughly a 15x error multiplier over bf16, which is what three mantissa bits on
both score operands buys. Smoothing is worth about 2.5x of it on a K with a large
channel mean.

> **The multiplier is set by the control, and the two controls in the tree
> disagree.** The table above has no JSON artifact. The one that does --
> `out/bench/kernel/attention/attention.json`, `precision` -- measures the same
> forward at **3.15e-2 for `mxfp8` against 5.57e-3 for `varlen`, a 5.7x
> multiplier**, not 15x. The `mxfp8` numerator agrees between the two (0.034 vs
> 0.0315); what differs is the bf16 denominator, 0.002 here against 0.0056
> there, so the discrepancy is in the control's shape and reference rather than
> in the kernel. Quote 5.7x, which is reproducible, and treat 15x as pending a
> harness. The ranking the section exists to establish -- `V` and `dO` in
> 16-bit, `Q`/`K` in e4m3 -- is unaffected either way.

## 4. Why the tiles are autotuned, not planned

An earlier version shipped an analytic tile planner (`attn_plan.py`) and a
hand-rolled backward tuner. Both are gone. `triton.autotune` keyed on
`["HEAD_DIM", "IS_CAUSAL", "HAS_WINDOW"]` is correct here and the planner was not
worth its surface area: every key is a `constexpr`, so the search runs once per
shape *class* and never re-runs when the varlen token count moves. That is the
distinction that matters — an autotune key containing `M` is what cost 365 ms per
step in the MoE grouped GEMM, and this is not that.

## 5. The bug that motivates `test_output_is_differentiable`

The first version of this kernel was reached through a plain function that
launched Triton directly on raw tensors. Its output had no `grad_fn`. Training
ran, the loss fell, and every gradient below `o_proj` was zero — the projections,
the norms and every layer beneath the attention received nothing.

Nothing failed. The forward was correct and pinned by tests, the loss curve was
plausible, and a `fwd+bwd` benchmark reported **612 TFLOP/s** for a backward that
did not exist. The kernel measured 2.8x faster than bf16 while its forward alone
was 1.1x *slower*; that inconsistency is what exposed it.

Two rules come out of it, both now in
[kernel-convention.md](kernel-convention.md):

- A benchmark arm named `fwdbwd` must assert that the backward it is timing
  actually runs. `fwdbwd_ms < fwd_ms * 1.5` is not a fast kernel, it is a missing
  one.
- A kernel entry point that is reachable from a module's `forward` gets a test
  that asserts `out.grad_fn is not None` and that every input receives a nonzero
  gradient.

## 6. Not supported

- **Padded batches.** `MXFP8Attention` falls back to SDPA, as `triton` does.
- **`head_dim` not a multiple of 32.** A scale block must be whole.
- **`torch.compile`.** Same limitation as the bf16 Triton kernel: `triton.autotune`
  inside a compiled region raises `InductorError`. The MXFP8 *linear* path does
  compile, through the `custom_op` wrapper in [mxfp8-difficulty.md](mxfp8-difficulty.md).
