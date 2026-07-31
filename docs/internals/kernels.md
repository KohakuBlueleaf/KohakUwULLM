# The Triton kernels

`src/kohakuwullm/kernels/` is where this repo stops composing PyTorch ops and starts
writing its own. Every kernel in it exists because a measurement said the composed
version was leaving something on the table on this card, and every one is pinned
against a reference implementation that does not share its assumptions. This document
explains what each one does, how it works, what constrains its numerics, what it
measures, and — usually the most valuable part — which plausible-looking change to it
is a trap.

[performance.md](../performance/performance.md) answers "how fast is the model"; this file answers
"how does the kernel work and why is it shaped like that". MXFP8 has a document of its
own, [mxfp8.md](mxfp8.md), because block-scaled fp8 has a layout contract that spans
half of these kernels and deserves to be explained once.

## Map of the directory

| Package | Kernels | Status |
|---|---|---|
| `attention/` | varlen FlashAttention (fwd, bwd), RoPE | alternates to the vendor path |
| `elementwise/` | RMSNorm, SwiGLU | registry entries; ATen is the default |
| `loss/` | chunked linear+CE, z-loss, register-resident CE | `chunked_ce` is production |
| `moe/` | grouped GEMM, fused router, dispatch and combine | production |
| `mxfp8/` | quantizer, grouped block-scaled GEMM, fused experts | production; see [mxfp8.md](mxfp8.md) |
| `optim/` | stochastic rounding, 16-bit AdamW | production |

## Calling them

Everything not fp8-specific is re-exported from the package root. Each is an
`autograd.Function` behind a plain function, so it differentiates normally.

```python
from kohakuwullm.kernels import (
    apply_rope,                       # (x, cos, sin, rotary_dim) -> x rotated
    chunked_linear_cross_entropy,     # (x, weight, target, ...) -> (T,) fp32 loss
    grouped_gemm,                     # (x, weight, offsets) -> per-expert matmul
    logsumexp_square,                 # (hidden, weight, labels) -> z-loss
    rms_norm,                         # (x, weight, eps) -> normed
    stochastic_round_update_,         # in-place SR weight writeback
    swiglu_mul,                       # (gate, value) -> silu(gate) * value
    triton_varlen_attn,               # packed attention over cu_seqlens
)
```

Nothing here is selected by calling it directly in the model — the registry
resolves a name to one of these **once**, at build time. `norm="rmsnorm_triton"`
in a config is what puts `rms_norm` in the block; the direct calls below are for
tests, benchmarks and kernel work.

## The card these are written for

Consumer Blackwell reports architecture 12.0 but is not a small B200. It keeps the
sm_80-era `mma.sync` execution model with no WGMMA, offers roughly 100 KB of shared
memory per SM against 163 KB on SM80 and 227 KB on SM100, and — the omission that
decides the most — has **no Tensor Memory**. Four consequences run through every
kernel here:

- **A tile tuned for Hopper or Blackwell-datacenter either spills or idles here.**
  The shared-memory budget is the binding constraint on almost every GEMM tile in
  this directory, which is why the config lists are small and hand-picked rather
  than inherited from an upstream kernel.
- **FlashAttention-4-class techniques do not apply.** FA4 accumulates in TMEM and
  pairs CTAs on the strength of it. FA2-class is the ceiling on sm_120.
- **fp16 accumulation runs about 1.5x faster than fp32 accumulation** on GeForce
  tensor cores — 325 against 210 TFLOP/s measured. Error grows with reduction depth,
  so it is only taken where we own the kernel and the contraction is short.
- **DRAM peak is 1791 GB/s measured.** Several kernels here are bandwidth-bound with
  nothing left to tune, and knowing the ceiling is what lets us say so rather than
  keep searching.

## Habits every kernel here shares

These are not style preferences. Each one was learned from a defect.

**Widen for arithmetic, narrow for storage, and never let the wide form touch
memory.** The MX quantizer takes its amax in fp32 registers; the norm emits fp8 from
the registers it already holds; 16-bit AdamW computes in fp32 and stores in 16 bits.
On memory-bound kernels the wide arithmetic is free — the instructions execute while
the loads that nobody is waiting on complete.

**Reduce in fp32, and judge the result in ULP.** Summing 16k bf16 terms loses several
percent of the value. Every accumulator here is fp32 unless a specific argument says
otherwise, and every precision test compares against an fp64 oracle in ULP, in both
fp16 and bf16, forward and backward.

**Never put a varlen axis in an autotune key.** `triton.autotune` re-runs its search
whenever the key changes, and each search is dominated by `do_bench`'s 256 MB L2
flush. The token count of a packed batch is different every step, so a key containing
it re-benchmarks the kernel on every step forever. This has cost this repo two
separate regressions — 365 ms/step in the MX quantizer and roughly 950 ms/step in
SwiGLU, the latter five times the entire step it was inside. The tell in a profile is
`FillFunctor<int>` running at exactly DRAM peak. Model widths are safe keys; token
counts are not.

**Bound the grid from host-known values.** Reading a device tensor to size a launch
costs an `.item()`, which is a full pipeline stall and an outright CUDA-graph capture
failure. Every MoE kernel here resolves per-expert geometry inside the kernel from
`offsets` instead, so a whole MoE layer stays capturable.

**A reference that shares the kernel's assumptions proves nothing.** Both defects
ever found in the MXFP8 subsystem survived their tests exactly that way. Where a
vendor implementation exists, the strongest check is bit equality against it, because
it is the only oracle that shares no assumption with the thing it checks.

---

# Attention

## Varlen FlashAttention

PyTorch 2.13 ships a varlen FA2 kernel (`torch.nn.attention.varlen.varlen_attn`) with
`cu_seqlens`, a trainable backward, GQA and sliding windows, and that is the default
backend. The Triton kernel in `attention/` exists because sm_120 is an awkward target
that the vendor kernel is not tuned for: tile sizes chosen for a 163 KB or 227 KB
shared-memory budget do not fit in 100 KB, and Triton lets us autotune for the budget
we actually have.

The layout matches the vendor kernel exactly, so the two are drop-in alternates: `q`
is `(T, H, D)`, `k` and `v` are `(T, H_kv, D)`, and `cu_seqlens` is the `(N+1,)`
exclusive prefix sum of document lengths. Causal masking anchors to the bottom-right
of each document, which is the convention for equal-length q/k.

```python
import torch
from torch.nn.attention.varlen import varlen_attn
from kohakuwullm.kernels import triton_varlen_attn

lengths = torch.tensor([512, 300, 1236], device="cuda")
cu = torch.zeros(4, dtype=torch.int32, device="cuda")
cu[1:] = lengths.cumsum(0)                      # (N+1,) exclusive prefix sum
T, H, H_kv, D = int(cu[-1]), 20, 4, 64

q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(T, H_kv, D, device="cuda", dtype=torch.bfloat16)

max_seqlen = int(lengths.max())
out = triton_varlen_attn(q, k, v, cu, max_seqlen, causal=True, window=1024)

# The vendor kernel over the same operands. It spells the window as a (left, right)
# pair and is causal by way of `right=0`, where ours takes `causal` and a width.
vendor = varlen_attn(q, k, v, cu, cu, max_seqlen, max_seqlen,
                     window_size=(1023, 0), enable_gqa=H != H_kv)
```

`window=None` is global attention; an integer is the sliding-window width, and the
kernel skips key blocks lying entirely outside it — which is the whole of the 20%.
`sm_scale` defaults to `D ** -0.5`. `return_lse=True` additionally hands back the
log-sum-exp, which is what attention sinks consume — the vendor spelling of the same
request is `return_aux=AuxRequest(lse=True)`, and it returns `(H, T)` where ours
returns the transpose.

**Measured outcome** (`scripts/bench/kernel/attention.py`, RTX 5090):

| workload | against vendor varlen |
|---|---|
| sliding window 1k–4k, forward | **up to 20% faster** |
| full causal, fwd+bwd | ~19% slower |

The window win is real work avoided: this kernel skips key blocks that lie entirely
outside the window, which the vendor kernel still walks. The causal loss is
structural rather than a tuning miss. The backward here runs two passes — `dk`/`dv`
first, then `dq` — and therefore reads q, k and v twice, where FA2 fuses both into one
pass with atomics on `dq`. Closing the gap means fusing `dq` into the `dk`/`dv` loop.
Until then the honest configuration is `attn_sliding="triton"` for windowed layers and
`varlen` for global ones, which is what the presets do.

On FlashAttention-4: upstream has no sm_120 path at all (Dao-AILab issue #2307 is
unanswered). Community PRs #2329, #2330 and #2333 add sm_120 forward, backward and
varlen subclasses, and a patched build measures roughly 25% over FA2 on a 5060 Ti.
That is an out-of-tree dependency with three known bug fixes layered on top, which is
why it is a note here and not a requirement in `pyproject.toml`.

### How the forward works

One program owns a block of queries and streams the keys that block attends to,
keeping the running softmax state (`m_i`, `l_i`, and the output accumulator) in
registers so the `(M, N)` probability matrix never reaches memory. That is the whole
of "flash". The log-sum-exp is written out because both backward kernels recompute `p`
from it rather than storing it.

Two details are load-bearing:

**`NEG_INF` is `-1e6`, a finite sentinel, not `-inf`.** A key block that is fully
masked — reachable with a sliding window — would make the online-softmax rescale
compute `-inf - (-inf)`, which is NaN. Real logits are order 10, so a finite sentinel
masks just as hard and cannot produce NaN.

**Window bounds are computed as block bounds, not as a mask.** Everything older than
the window is skipped at block granularity (`lo = max(0, start_m * BLOCK_M - window +
1)`, rounded down to a block), which is where the 20% comes from. The elementwise
`tl.where` still runs inside the surviving blocks for the partial one at the edge.

A fully-masked row would leave `l_i == 0`, so the divide is guarded and the stored LSE
falls back to the sentinel. For causal-plus-window that row cannot occur — a query
always sees itself — but the guard keeps a wrong `window` argument from writing NaN
instead of a wrong number.

### How the backward works

`_bwd_preprocess` computes `delta[t] = sum_d out[t,d] * dout[t,d]` once; both gradient
kernels read it. Then `dk`/`dv` grids over keys and streams queries, and `dq` grids
over queries and streams keys. Both recompute `p` from the saved `lse`, so nothing
`(M, N)`-shaped is ever stored.

Two traps live in this file, and both are the kind that produce plausible wrong
numbers rather than a crash.

**The GQA gridding decision.** `_bwd_dkdv_kernel` grids over *query* heads, not kv
heads, and pays for it with fp32 atomics into shared `dk`/`dv` buffers. Gridding over
kv heads would let the GQA group accumulate in registers instead — strictly cheaper
arithmetic — but with 20 query heads over 4 kv heads it launches five times fewer
programs, each doing five times the work, which left the GPU measurably idle at long
sequences. The occupancy is worth more than the atomics. When `GQA_GROUP == 1` the
program is the only writer and a plain store replaces the atomic.

**`reset_to_zero=["dk_ptr", "dv_ptr"]` on the autotune decorator is mandatory,
not decorative.** Autotune benchmarks a candidate config by re-running the kernel.
With an atomic accumulation and no reset, the buffers accumulate once per trial and
come out thousands of times too large — silently, and only for the shapes whose
configs were not already cached.

The two backward kernels get **separate config lists**. They parallelize over opposite
axes, so they want opposite tile shapes: `dk`/`dv` wants a tall `BLOCK_N` and a modest
`BLOCK_M`, `dq` the mirror. An earlier version shared one list, which forced one of
them onto the other's shape.

## Rotary position embedding

Eager RoPE is six elementwise kernels and a concatenate for each of `q` and `k`, per
attention layer, running on the largest tensors in the block. The fused kernel is one
launch per tensor, with the shared `cos`/`sin` tables read once from L2, and it is
written out-of-place so autograd can still recompute under gradient checkpointing.

The backward of a rotation is the inverse rotation, so negating `sin` is the entire
backward and nothing needs saving beyond the tables. Partial rotary is supported: the
leading `rotary_dim` channels rotate and the tail passes through.

Every implementation takes the **half** tables, `(..., rotary_dim // 2)`, because the
rotation pairs channel `i` with `i + half` directly. `RotaryCache` stores the doubled
tables that `prepare` builds and slices to the first half on the way in; the doubled
half is the same values again, so the slice is the whole difference between the two
conventions.

### Tiling

The kernel tiles over **flattened `(token, head)` rows**: a program owns `BLOCK_R`
rows by `BLOCK_D` channels, where `BLOCK_D` is `next_power_of_2(rotary_dim // 2)`.
Within a row the channels are contiguous and consecutive rows are `head_dim` apart, so
the tile is a strided 2-D read, which costs nothing once tiled. `cos`/`sin` are indexed
by `row // n_heads`, so a tile that spans several heads of one token reads the table
once.

The first version instead ran **one program per `(token, head)` pair**. At 131072
tokens and 20 heads that is 2.6M programs each moving ~128 bytes, with `half = 32`
lanes active inside a `BLOCK` autotuned up to 128 — three quarters masked off — and the
`cos`/`sin` row re-read once per head. It reached 600 GB/s on the forward and 240 GB/s
on fwd+bwd. Retiling was worth **2.3x forward and 3.1x fwd+bwd**.

### Three peer implementations

`torch.compile`d eager is a **kernel**, not a side effect of compiling something
else. `rope.py` therefore holds three implementations of one signature —
`(x, cos, sin, rotary_dim)`, half tables, out-of-place — and `resolve_rope(name)`
selects between them:

| name | what it is |
|---|---|
| `triton` | the fused kernel above. `DEFAULT_ROPE_IMPL`. |
| `compiled` | `torch.compile(_reference_rope, **ROPE_COMPILE_OPTIONS)`, built on first use and cached at module level |
| `eager` | `_reference_rope`. The CPU fallback for the other two, and the oracle |

```python
from kohakuwullm.kernels.attention.rope import resolve_rope

rope = resolve_rope("triton")           # or "compiled" / "eager"
q_rot = rope(q, cos, sin, rotary_dim)   # cos/sin are the HALF tables
```

`RotaryCache` resolves the name to a callable **once**, in `__init__`, and `apply`
calls it. An unknown name raises there rather than falling back, because both fast
paths are numerically identical to each other and a silent fallback to `eager` is
visible only in a throughput plot nobody reads. In a config it arrives as a
registry spec whose remaining keys are `RoPE`'s constructor arguments; `RoPE.prepare`
hands the resolved name to the `RotaryCache` it builds each step:

```python
ARCH_OVERRIDES = {"posenc": {"name": "rope", "impl": "compiled"}}
```

Before this, compiled eager was reachable *only* by turning on the trainer's
`COMPILE` knob — so switching trainer compile off silently changed which rotation
ran, in a run whose loss curve gives no sign of it.

`ROPE_COMPILE_OPTIONS` is `dynamic=None, fullgraph=True, mode="default"`:

- **`dynamic=None`** (automatic) rather than `True` or `False`. A static graph is
  worth **1.33x** on the forward: one sweep at 131072 tokens gives 1424 GB/s for
  `None`, 1399 for `False`, 1072 for `True`. But the training token axis is not
  constant — `collate_packed` sums 64 document lengths and `PAD_TO_MULTIPLE`
  defaults to 0, so every step is a new total (only the pipeline microbatch loader
  pins it, at exactly its budget `k`). Under a moving axis `dynamic=None` recompiles
  once and then holds at two graphs over five token counts, landing on the same
  dynamic kernel `dynamic=True` would have started with; where the axis *is* fixed —
  the pipeline loader, a padded eval width — it keeps the 1424 GB/s static one. It
  dominates `True`, and the cost is one extra compilation. `dynamic=False` is the
  one to avoid: five graphs over five token counts and no bound on distinct totals,
  so it walks into `recompile_limit`, after which Dynamo stops compiling the frame
  and the rotation silently runs at eager's 373 GB/s.
- **`fullgraph=True`** because the function is four elementwise ops and two
  concatenates. There is nothing here that *can* legitimately break the graph, so a
  break means someone changed the reference into something Dynamo cannot trace, and
  the failure mode without `fullgraph` is a silent drop back to eager — the exact
  regression this whole selector exists to prevent.
- **`mode="default"`**: `max-autotune-no-cudagraphs` measured 1403 vs 1399 GB/s, a
  wash, for a much longer compile. `reduce-overhead` is wrong outright — it captures
  CUDA graphs around a callable invoked on fresh activations every step.

### Measured

bf16, 20 heads x 64, on one RTX 5090, median of 50 L2-flushed iterations, three
repeats. `fwd+bwd` is `out.sum().backward()` with **`x.grad` cleared each
iteration**: leaving it populated adds a read-modify-write of the 320 MiB gradient
that the byte count does not charge for, and it does not tax the three
implementations equally — it reads 754 GB/s for `triton` against 1087, but 328 for
`eager` against 377. Cross-implementation fwd+bwd numbers measured that way are not
comparable.

| tokens | | eager | triton | compiled |
|---|---|---|---|---|
| 16384 | fwd | 478 GB/s | **603** | 517 |
| 16384 | fwd+bwd | **393** GB/s | 356 | 384 |
| 131072 | fwd | 373 GB/s | 1382 | **1389** |
| 131072 | fwd+bwd | 376 GB/s | 1088 | **1094** |

At 131072 tokens the kernel and the compiled path are at parity — 0.995x on both
phases, within the 0.4% repeat spread. At 16384 the problem is too small to reach
the roofline and the ordering is not stable: the kernel wins the forward by 1.17x
and loses fwd+bwd to plain eager.

Accuracy is where they separate, and the kernel and the compiled path land in the
same place: **0.5 ULP elementwise against fp64, in both fp16 and bf16, forward and
backward** — a single rounding, at the store — because both reduce in fp32.
Eager reduces in the storage dtype and is 3.7 (bf16) / 3.9 (fp16) ULP rms; its
elementwise figure is 16256 ULP, which is the cancellation in `x1*cos - x2*sin`
showing up, not a defect. That is why `tests/test_models_posenc.py` bounds the two
fp32-reducing paths elementwise and everything else on the RMS scale.

So `triton` is the default on grounds other than speed: identical accuracy,
identical throughput at the sizes that matter, no compile step, no shape guards, no
recompile budget. `compiled` is the one to pick when the rotation sits inside an
already-compiled region, where it can fuse into its neighbours and the Triton call
cannot — Inductor treats it as opaque. Nothing in production compiles at that level
today.

---

# Elementwise kernels

## RMSNorm

One row per program. The forward caches `rstd` so the backward re-reads a single float
per row instead of recomputing the reduction, and the weight gradient accumulates into
a per-program partial buffer (`min(n_rows, 512)` programs) that is summed once at the
end. That last part is what separates this from a naive Triton RMSNorm backward, which
pays a global atomic per element.

`F.rms_norm` (ATen) is the default in the norm registry, and per
[performance.md](../performance/performance.md) it wins below roughly 32k tokens while this kernel
wins above. That alone would not justify keeping it.

**What justifies it is the fp8 emission.** A standalone MX quantizer already runs at
96% of DRAM bandwidth, so the only remaining way to make the cast cheaper is to not
move the data twice. Emitting from the norm costs nothing: the normed row is already
in fp32 registers, and a row here is the K axis of the projection that consumes it, so
every 32-element MX block sits inside one column tile. No cross-iteration reduction, no
second pass. `rms_norm_mx` returns `(out, e4m3, ue8m0)`; `rms_norm_mx_vendor` returns
scales already in cuBLAS's `SWIZZLE_32_4_4` layout, which is described in
[mxfp8.md](mxfp8.md).

The swizzled variant returns its scales **flat**, not `(rows, K//32)`, because the
layout zero-pads to whole 128-row tiles and the padded extent is not the logical
shape. A 2-D read of it would silently address padding. For the same reason the
swizzled scale buffer is allocated with `zeros` rather than `empty`: a partial final
row tile leaves padding that cuBLAS still reads, and stale bytes there are scales for
rows that do not exist.

```python
from kohakuwullm.kernels import rms_norm
from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm_mx, rms_norm_mx_vendor

y = rms_norm(x, weight, eps=1e-6)                      # (T, D) -> (T, D)

# One pass: the bf16 output WGRAD needs, plus the fp8 operand the next GEMM takes.
y, e4m3, ue8m0 = rms_norm_mx(x, weight)                # natural scale layout
y, e4m3, swz = rms_norm_mx_vendor(x, weight)           # cuBLAS SWIZZLE_32_4_4, flat
```

`rms_norm_mx` deliberately has **no ATen fallback**, unlike `rms_norm`. A caller
asking for fp8 operands has no use for a bf16-only result, so a silent fallback would
hand the consumer a shape it cannot take.

**The `WRITE_OUT=False` trap.** Skipping the 16-bit store looks like a further saving
of `2 * n_cols` bytes per row, and *this backward* would allow it — it recomputes from
`x` and `rstd` and never reads the normed output. It is nonetheless unusable while
WGRAD stays 16-bit, because `WGRAD = dout^T @ y` is the one product that consumes the
normed activation itself. The flag exists for a future all-fp8 WGRAD. Today the store
is load-bearing and the free cast is the whole win.

The fp8 pair carries no gradient (`mark_non_differentiable`): dgrad reaches the norm
through the bf16 output, and WGRAD reads that same tensor.

## SwiGLU

`silu(gate) * value` as one kernel. The win is memory, not math — the eager version
materializes `silu(gate)` and keeps it alive for the backward, whereas this saves only
`gate` and `value`, which the surrounding GEMMs need anyway, and recomputes the
activation during the backward.

```python
from kohakuwullm.kernels import swiglu_mul

h = self.w_in(x)                        # (T, 2H), one fused projection
gate, value = h.chunk(2, dim=-1)        # two strided views, NOT contiguous
out = self.w_out(swiglu_mul(gate, value))
```

That `chunk` is the shape the default `GLUMLP` produces and the one the kernel is
tuned for. Do not "fix" it with `.contiguous()` — see below.

**The inputs are read through their own strides and never contiguified**, and this is
the single most important line in the file. Both callers that matter hand in two
halves of one `(rows, 2H)` tensor: a fused `w_in` projection followed by `chunk`, which
is `GLUMLP`'s default, and the MoE expert path between its two grouped GEMMs. An
earlier version called `.contiguous()` on each half, which for a chunk copies both,
and that made the "fused" kernel *slower and heavier than eager* in exactly the
configuration the default produces:

| input layout | fused | eager |
|---|---|---|
| chunked, 8192x3456 | 248 us / 0.619 GiB | 190 us / 0.566 GiB |
| contiguous, 8192x3456 | 165 us | 154 us |

A kernel that only wins when its input happens to be contiguous is worse than no
kernel, because nothing checks. `_as_2d` therefore collapses leading axes only when
they are contiguous *relative to each other* — which holds for a chunk of a contiguous
tensor even though the chunk itself is not contiguous — and falls back to `.contiguous()`
only for a genuinely non-collapsible layout.

The gradients are written contiguously, and there is no equivalent saving to be had:
a chunked input's two halves receive separate `(rows, H)` gradients from autograd
regardless, and the slice backward concatenates them either way.

**The autotune key is `cols` alone, deliberately not `rows`.** `rows` is the varlen
token count; keying on it makes every step of a packed stream a fresh key and re-runs
the ten-config search, measured at roughly 950 ms/step of `do_bench` L2 flushes at
MoE-1B-A280M — five times the entire step. Dropping `rows` keeps the tuning rather
than freezing it, because the tuning does not depend on it: over nine row counts from
8k to 130k at six model widths the per-row winners scatter with no pattern, and one
config per `cols` costs at most 1.16x forward and 1.52x backward against the per-shape
best, on a memory-bound op worth a few ms of a 240 ms step.

Two consequences are worth stating rather than discovering. That 1.16x/1.52x bound was
measured over rows 8k–130k and says nothing outside it, so a caller far below that
range gets an untested config. And `scripts/bench/kernel/kernels.py` sweeps *rows* at
fixed `hidden`, so its whole sweep shares the config picked at the first row count it
runs — which is the honest thing to plot, because it is the config training gets.

---

# The loss

## Chunked linear + cross-entropy

The LM head is a memory problem, not a speed problem: at vocab 65536 and 16k tokens
the logits alone are 16.5 GiB, and on a 32 GB card that number decides the batch size.
There are two obvious ways out and this kernel is the third.

The first is to keep every logit tile in registers so it never reaches memory at all.
That is `loss/fused_ce.py`, and it costs `O(T)` memory — but it replaces cuBLAS with a
hand-written Triton GEMM, measured 26% slower than `F.linear_cross_entropy` on sm_120,
and its backward does not compile at the widths this repo trains (below).

The second is to materialize and eat the memory.

`chunked_ce` takes the middle: logits are produced by **cuBLAS** into a scratch tile of
`chunk x vocab_block` that is reused, and a Triton epilogue consumes the tile in place.
Peak scratch is set by the tile rather than by `T`, and the GEMMs stay at cuBLAS
throughput. At `T=8192, D=1792, V=65536` in bf16 the forward is *faster* than the
materializing path — 8.5 against 8.6 ms — at a third of its memory, because the
epilogue re-reads a tile that is still in L2 while the materializing path re-reads
`T * V` from DRAM.

```python
from kohakuwullm.kernels import chunked_linear_cross_entropy

token_loss = chunked_linear_cross_entropy(
    hidden.view(-1, dim),      # (T, D)
    head_weight,               # (V, D)
    labels.view(-1),           # (T,), -100 ignored
    chunk=8192,                # rows per GEMM
    vocab_block=8192,          # vocabulary columns per tile
    retain=0.0,                # fraction of forward tiles cached for the backward
)
loss = token_loss.sum() / (labels != -100).sum()   # reduce in fp32, yourself
```

### The three knobs

Each moves along the time/memory curve for a different reason, which is why they are
three knobs and not one.

- **`chunk`** (token axis). Small chunks starve the GEMM: below roughly 1024 rows the
  `M` dimension no longer fills the tensor cores. This is a real trade, not a free
  dial.
- **`vocab_block`** (vocabulary axis). Also shrinks the fp32 `dW` accumulator, which
  at full vocabulary is 0.44 GiB and dominates everything else in the backward.
- **`retain`** (fraction of logit tiles cached from the forward). The backward needs
  each logit tile again, and recomputing one is a single cuBLAS GEMM. `retain=0` pays
  a fourth GEMM and keeps nothing; `retain=1` pays three GEMMs and keeps `T * V`;
  every value between is a reachable point on the curve.

### Algorithm details

The forward folds each tile into a running max / sumexp / target logit with an **online
softmax** rather than a two-pass max-then-sum. The tile is the largest thing in the
kernel and a second pass over it doubles the only traffic the GEMM has not already
paid for. `exp(-inf - -inf)` is NaN, so the rescale is guarded for the first tile,
where there is no accumulated mass to rescale.

Inside the target-logit search, the load mask is load-bearing: an out-of-range lane
holds `-inf`, and its global index can still collide with a target that lives in a
later tile.

Tiles are visited in **vocabulary-major** order — every token chunk for one block of
the vocabulary before moving on — so the fp32 `dW` accumulator only ever spans
`vocab_block` rows instead of the whole `(V, D)`. When there is exactly one token
chunk per vocabulary block (`dw_direct`) there is nothing to accumulate across: cuBLAS
already sums the whole `K=T` axis in fp32, so the `dW` GEMM writes the staging buffer
instead of adding to it, skipping a zero-fill and a read of `(vocab_block, D)`.

The backward rewrites each tile as `(softmax - onehot) * dloss` **in place**, so it
never allocates a second one. Its grid is two-dimensional where the forward's is
one-dimensional, because this step is elementwise: rows and vocabulary blocks are
independent, and a row-per-program launch would leave most of the machine idle
whenever `chunk` is small.

`dX` and `dW` accumulate in fp32 for the reason in `CLAUDE.md`'s numerics rules: each
is a sum over many blocks, and rounding a partial sum to bf16 costs a ULP of the final
value every time.

### Two contract details a caller can get wrong

**The returned loss is unreduced `(T,)` fp32.** Reduction is left to the caller so a
token-weighted mean can be taken in fp32. `LMHead` does exactly this.

**`dloss` is expected to be `O(1)`** — reduce with `.sum()` and divide by the token
count afterwards. Folding a `1/n_tokens` factor in first pushes the small tail of
`softmax * dloss` into fp16 subnormals.

### Why backward-twice raises

With `retain > 0` the epilogue overwrites cached tiles through a raw Triton pointer,
which does not bump autograd's version counter. A second backward would read `dlogits`
as if they were logits and return a plausible wrong gradient instead of raising, so the
function tracks consumption itself and raises with a pointer at `retain=0.0`.
Recomputed tiles carry no such state.

## The register-resident CE, and why it stayed forward-only

`loss/fused_ce.py` is kept as a **documented dead end**: nothing in `src`, `tests` or
`scripts/bench` imports it, and it is not exported from `kernels/__init__`. It is here
because deleting it would delete the reason.

The forward delivers on its premise — 178.8 TF/s at `O(T)` memory. The backward fails
for two independent and structural reasons:

- `_dx_kernel` and `_dw_kernel` index the full width with `tl.arange(0, D)`, and
  Triton requires a power-of-two bound. `D=1792` (Nano-1B) is `2**8 * 7`, so it fails
  to compile outright.
- Each holds a `(BLOCK_T, D)` or `(BLOCK_V, D)` fp32 accumulator in registers — 458 KB
  per program at `BLOCK_T=64, D=1792` against a 256 KB register file. Rounding `D` up
  to a power of two makes it compile and then spill, which is worse than the traffic
  the design set out to avoid.

Keeping the whole width resident is what makes register residency work in the forward
and what breaks it in the backward. Blocking `D` to fix the backward reintroduces a
partial-sum accumulator and turns the kernel into `chunked_ce` with a hand-written GEMM
instead of cuBLAS. That is why production went to `chunked_ce`, and the backward here
raises with the width in the message rather than emitting a Triton trace that points at
an `arange`.

## z-loss

The z-loss pins the softmax normalizer near 1. Without it nothing stops the logits
drifting up together: the softmax is shift-invariant so cross-entropy never objects,
but the raw magnitudes grow until bf16 rounding starts eating the differences between
them. PaLM introduced it and many recipes carried it forward.

**Nothing in `configs/lm/` sets it, and the kernel below is the reason it stays
anyway.** It is a second full pass over `dim x vocab` — at MoE-1B and 8192 tokens
the head goes from 12.83 ms to 54.82 ms, and since the head stage is the pipeline's
critical path that is 182.0k → 114.5k tok/s on 4 cards (1.59x). And DeepSeek-V3
trained 14.8T tokens with no auxiliary loss
of any kind. Keep the kernel, leave the weight at zero:
[moe-router-loss.md](moe-router-loss.md#the-head-z-loss-is-a-different-thing-and-it-is-off).

`F.linear_cross_entropy` has no z-loss option, so it is computed separately — and
computing it naively would defeat the point, since `logsumexp` over a materialized
`(N, V)` is exactly the tensor the fused CE exists to avoid. So the forward walks the
batch in chunks under `no_grad` keeping only the per-row scalar, and the backward
recomputes each chunk's softmax to build the gradient from

    d/dx [lse(xW)^2] = 2 * lse(xW) * softmax(xW) @ W

Peak memory is one chunk of logits and the whole thing is `O(1)` in `N`. Chunk rows
are sized so one chunk of logits matches the input's own footprint.

---

# Mixture of experts

Three kernels make the MoE path work, and they divide by what they are avoiding:
`router.py` avoids per-op dispatch, `moe_dispatch.py` avoids a sort and a host sync,
and `grouped_gemm.py` avoids `E` launches per expert matrix. The MXFP8 versions of
the expert GEMMs are in [mxfp8.md](mxfp8.md); everything here is dtype-agnostic.

## The grouped GEMM

One launch computes `out[i] = x[i] @ W[expert_of(i)].T` for every token `i`, given
tokens already sorted by expert. The alternative — a Python loop of `E` small GEMMs —
costs `E` launches per expert matrix per layer, which at 64 experts x 3 matrices x 24
layers is over 4600 launches per forward and is launch-bound long before it is
compute-bound. Measured against that loop it is **14x at 128 experts** (98 against 7
TFLOP/s) at bit-equivalent accuracy, which makes it the single biggest kernel win in
the repo.

Tokens must arrive grouped: `group_offsets` is the `(E+1,)` exclusive prefix sum of
per-expert token counts, so rows `[offsets[e], offsets[e+1])` all belong to expert `e`.
The backward reuses the same primitive — `dx` is a grouped GEMM against `W`, and `dW`
is a grouped GEMM of `dout.T @ x` accumulated per expert.

```python
from kohakuwullm.kernels import grouped_gemm

counts = torch.tensor([300, 0, 512, 212], device="cuda")   # an empty expert is fine
offsets = torch.zeros(5, dtype=torch.int32, device="cuda")
offsets[1:] = counts.cumsum(0)

x = torch.randn(int(offsets[-1]), 1536, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4, 1152, 1536, device="cuda", dtype=torch.bfloat16)  # (E, N, K)

out = grouped_gemm(x, w, offsets)                  # (M, N), fp32 accumulate

# ~1.5x, forward only, and fp16 operands only -- a bf16 caller is refused, not demoted.
out = grouped_gemm(x.half(), w.half(), offsets, acc_fp16=True)
```

`acc_fp16` is safe here only because an expert's `K` is the model width. Never in a
backward — see below. `compute_dtype=torch.float32` forces the vector cores and is a
measurement baseline, not a training setting.

### The flattened grid

The obvious grid is `(cdiv(max_rows_per_expert, BLOCK_M), N_blocks, E)`. It cannot be
used, because the largest expert's row count is a *device* value and reading it costs
an `.item()`: a full host stall in the middle of every MoE layer, and an outright
CUDA-graph capture failure.

So the row axis is flattened into a single tile index, and each program resolves its
own `(expert, local tile)` from `offsets` in registers. That bounds the grid at
`cdiv(M, BLOCK_M) + E`, known from the row count and the expert count alone. The
lookup, `_tile_owner`, is a `cumsum` over `E <= BLOCK_E` lanes — two reductions rather
than five, because `done` (the count of experts whose tiles all precede this one)
yields the expert id and the tile prefix at once.

The flattened grid is also *tighter* than the per-expert one whenever the load is
skewed, because the per-expert grid sizes every expert's row axis for the worst expert:
at E=96 with one expert holding half the rows it launches 196k programs for 2k tiles of
real work.

Two edge cases fall out of the bound. `expert == num_experts` means the program is past
the last real tile and returns; empty experts contribute zero tiles and cost nothing.
Past the end, every lane counts as done — including the `BLOCK_E` padding lanes — so
`expert` can reach `BLOCK_E`, and the *load* is clamped rather than the returned id: the
caller needs the unclamped value to detect the case, and an unclamped load would run off
the end of an `(E + 1)` tensor.

### Grid dimension order, and the 65535 limit

**N goes on grid dimension 0** and the flat row tile on dimension 1. CUDA walks the x
axis fastest, so every n-block of one row tile runs back to back and the tile's rows
stay in L2 across them. With the row tile fastest instead, each n-block re-reads the
whole of `x` — 201 MiB at `T=8192, top_k=8, dim=1536`, nine times over. Measured 18%.

The price is that the row-tile axis is now grid dimension 1, which CUDA caps at 65535,
and the smallest `BLOCK_M` in the autotune set decides the worst case. That cap is
checked rather than designed around: 4.2M routed rows is far past anything a 32 GB card
holds, and the alternative launch order costs 18%.

### fp16 accumulation

`ACC_FP16` selects an fp16 accumulator, worth roughly 1.5x on GeForce tensor cores. It
is only safe here because an expert's `K` is the model width — about 1k, not the 16k
where fp16-accumulate error overtakes bf16 — so the reduction is naturally short.
Measured at 20 ULP / 1.32e-3 relative error against bf16's 1.66e-3.

**The backward stays fp32-accumulate either way.** Gradient magnitudes have far less
predictable range than activations and fp16 tops out at 65504.

`compute_dtype=None` keeps a 16-bit input as it is and lifts fp32 to bf16, so an fp32
caller reaches the tensor cores without an fp16 caller being silently demoted.
`torch.float32` forces the vector cores and is a measurement baseline, not a training
setting.

## The fused router

The eager router is roughly six separate launches — `linear`, `sigmoid`, `+bias`,
`topk`, `gather`, `scatter_add` — over tensors that are tiny next to a GEMM: a
`(T, E)` score matrix is 3 MiB at E=96. Measured on an RTX 5090 the whole sequence
costs about 350 us at T=8192 and **does not move** when E goes 32 → 96 or D goes
640 → 1536. A cost that ignores its own problem size is not compute; it is per-op
dispatch, and the only way to remove it is to stop issuing eight ops.

So the entire router is one kernel. With `E <= 128` a token's whole score row lives in
registers, which means the `(T, E)` scores never reach HBM and the top-k is a
register-resident selection rather than a sort of a materialized matrix. The load
histogram folds in as a one-hot tile accumulated across the top-k rounds and reduced
once, so the counts cost one atomic per expert per program instead of a second pass
over the indices. Inside the loop the winning score is read with a `tl.gather` — a
shuffle — rather than `sum(where(onehot, scores, 0))`, which would be a second
full-width reduction per round.

```python
from kohakuwullm.kernels.moe.router import fused_router

indices, weights, counts, aux, z = fused_router(
    x,                       # (T, D)
    router_weight,           # (E, D)
    router_bias,             # (E,) aux-loss-free balancing bias, or None
    top_k=8,
    score_func="sigmoid",    # or "sqrtsoftplus"
    load_accum=self.load,    # (E,) buffer this call adds its histogram into
    aux_loss_weight=0.0,     # 0 compiles the term, and everything feeding it, away
    z_loss_weight=0.0,
)
```

`aux` and `z` are `None` unless their weight is non-zero, and each is a
differentiable fp32 scalar — see
[moe-router-loss.md](moe-router-loss.md).

`counts` is the per-call histogram, and it comes free from the same one-hot reduction
that feeds `load_accum` — which is what lets the dispatch below be a counting sort
rather than an `argsort`.

### Three things it deliberately does not do

**No `torch.bincount`.** It sizes its output from `input.max()`, which is a
device-to-host copy — a full pipeline stall in the middle of every MoE layer, and the
reason the eager path cannot be CUDA-graph captured at all. The per-call `counts` this
kernel returns come free from the same one-hot reduction that feeds `load_accum`.

**No `triton.autotune`.** The forward has a side effect: the load histogram accumulates
into a buffer that outlives the call. Autotune *replays* the kernel to time its
candidates, so every autotuned launch would multiply the counts by the number of
configs tried. A measured sweep put a fixed `BLOCK_T=64, BLOCK_D=64` within 2% of the
best config on every target shape, which is not worth a silently corrupted load
estimate.

**No softmax gate.** An *elementwise* gate's backward needs only the `top_k` selected
scores, which is why both `sigmoid` (DeepSeek-V3) and `sqrtsoftplus` (DeepSeek-V4)
fuse here. A softmax couples all `E` and would force the full score matrix to be saved,
so softmax routers keep the eager path, which `TopKRouter` selects at build time.

### The balancing bias steers selection only

The returned weight comes from the **unbiased** score. This is not an implementation
detail to preserve casually — it is what makes aux-loss-free load balancing work, and
`tests/test_models.py::test_moe_bias_steers_selection_not_weights` pins it.

Lanes past `E` are set to `-inf` so they can never win an argmax, and selected lanes
are set to `-inf` as each round completes; `E >= TOP_K` guarantees a real expert is
still available every round.

### The gate activations and their gradients

`sqrtsoftplus` is computed as `sqrt(max(z,0) + log1p(exp(-|z|)))`, not
`sqrt(log(1+exp(z)))`: the direct form overflows for `z > 88` in fp32, and these logits
are unbounded by construction — an unbounded gate is the whole reason DeepSeek-V4 uses
this activation.

Both activations admit a derivative expressed through the **score alone**, which is
what keeps the backward's saved state at `(T, top_k)` — unless an auxiliary loss is
on, which needs every expert's score and so saves the `(T, E)` logit tile too.
Sigmoid gives the familiar
`s(1-s)`. For `s = sqrt(softplus(z))`, note that `exp(-s^2) = 1/(1+e^z) = 1 -
sigmoid(z)`, so `sigmoid(z) = -expm1(-s^2)` exactly and `ds/dz = sigmoid(z) / (2s)`.
Saving the selected *logits* instead would also work but costs a second `(T, top_k)`
buffer and keeps the logit tile live across the top-k loop, which at `BLOCK_E=128` is
the register pressure that already spills.

`expm1`, not `1 - exp`: the argument is `-s^2`, so for a small score the subtraction
cancels to zero and the gradient would vanish spuriously.

### Warps, and the one shape that spills

A 128-lane score row at `BLOCK_T=64` carries four live `64x128` fp32 tiles — the
scores, the biased copy, the one-hot accumulator and the dot accumulator. Over four
warps that is roughly 64 registers per thread of tile alone, and it spills: measured
**50.8 us at four warps against 28.8 at eight**, where the same tiles spread over 256
threads. Narrower rows do not spill and prefer the deeper pipeline, so
`_launch_config` returns `(8, 3)` at `BLOCK_E >= 128` and `(4, 4)` below.

### The backward

`_router_bwd_logits` scatters the `top_k` gate-logit gradients into a **zeroed dense
`(T, E)` tile**, not a sparse structure. `E <= 128` makes that 3 MiB, and the two GEMMs
that consume it (`dx = G @ W`, `dW = G.T @ x`) are then plain dense matmuls rather than
a gather-scatter pair costing more than the matmul it feeds. The stores need no atomics
because a token's top-k indices are distinct.

## Dispatch: the counting sort

Grouping `(token, slot)` pairs by expert is a permutation, and the eager path buys it
with `argsort` — a general comparison sort over `T * top_k` keys. But the keys are
expert ids: at most 128 distinct values, and their histogram is *already known*, because
the fused router hands back `counts` as a by-product of its top-k. A counting sort
therefore needs no comparisons at all, just one pass that reads each pair's expert and
claims the next free slot in that expert's range.

At `T=8192, E=96, top_k=8` the eager triple costs 167 us — `argsort` 83, `counts.cumsum`
54, `order // top_k` 30 — against two launches over the same data here.

The scan of the per-expert counts runs in **one program**, not a multi-block scan:
`E <= 128` means the whole histogram is a single vector, and a launch that does the scan
in registers beats any tiled version that would have to round-trip partial sums through
HBM. The placement pass then uses `tl.atomic_add` on a per-expert cursor, which returns
a distinct old value per lane even when lanes collide on one expert — exactly the
running cursor this needs.

**Order within an expert is therefore not deterministic**, and that is safe here rather
than alarming. The rows of one expert's GEMM are independent, and the results are
recombined with an `index_add_`-shaped scatter whose float accumulation over a token's
`top_k` contributions is *already* order-dependent. A deterministic variant would need a
global stable scan over `T * top_k`, which costs more than the sort it replaces and buys
reproducibility the surrounding path does not have. The tests compare the grouping
invariant, not the permutation; `expert_sort_reference` uses a stable sort so the CPU
path is reproducible on its own terms.

`max_rows` is returned as a **device tensor and is a load-imbalance diagnostic only**.
It used to size the grouped GEMM's grid, which cost an `.item()` per MoE layer and made
the layer impossible to graph-capture. Nothing on the hot path reads it now.

## Combine: gate scale and scatter in one pass

`combine_routed` scales each routed row by its gate weight and sums it back onto its
token. The eager spelling is two passes over the routed rows — which are `top_k` times
the size of the activation — with a scaled copy in between that nothing else reads. At
`T=8192, top_k=8, dim=1536` that intermediate is 201 MiB written and read straight back,
and removing it takes the combine from **698 us to 140 us**.

The backward yields both gradients in one pass, since both read the same two rows.
`grad_weight` is a full-width dot product per routed row, so it accumulates through an
fp32 atomic: the row is split across `D / BLOCK_D` programs and each owns a partial sum.

**The forward's atomic runs in the output's own dtype**, which is what `index_add_` did
too. A token's `top_k` contributions land in a 25 MiB output that stays resident in L2,
so widening the accumulator to fp32 would double the output traffic to buy accuracy on
a depth-8 sum.

### The sentinel bucket

A ReMoE router parks inactive slots in a bucket with no expert matrix. Those rows are
never computed, so they hold whatever `torch.empty` left behind, and `valid_rows` — a
**one-element device tensor**, never an int, because knowing it on the host is the
`.item()` this path exists without — bounds the sorted positions a real expert owns.

Rows at or past that bound are **skipped outright, not scaled by their zero gate
weight.** Those look equivalent and are not: `nan * 0` and `inf * 0` are `nan`, so one
uninitialised row would poison its token's entire output. The reference path masks them
to zero before the sum for the same reason.

In the backward, skipped rows get an explicit zero rather than an early return. Nothing
downstream reads them — the expert GEMM's backward covers the same buckets the forward
did — but `grad_rows` comes from `empty_like`, and leaving a garbage region inside a
gradient tensor is one refactor away from a silent NaN.

`combine_routed` is a nondeterministic scatter-add and disagrees with itself by 0.0625
in bf16. Any A/B downstream of it needs that noise floor reported alongside.

---

# Optimizer kernels

## Stochastic rounding

### Why it exists

Round-to-nearest discards every update smaller than half an ULP of the weight it lands
on. Under a 16-bit master weight that is most updates once training has settled: the
coordinate freezes while its gradient is still nonzero, and nothing in the loss curve
says why. Stochastic rounding rounds up with probability equal to the distance to the
upper neighbour, so the update survives in expectation.
[Zamirai et al.](https://arxiv.org/abs/2010.06192) introduced it for bf16 training and
[Ozkara et al.](https://arxiv.org/abs/2502.20566) carry it to 6.7B.

**This is the enabler for full 16-bit training, which is the target here**: weight and
optimizer state both 16-bit, with no master copy anywhere. Under an fp32 master SR
would buy nothing — the fp32 copy accumulates sub-ULP updates by itself, and the Falcon
report found the no-SR trajectory rejoins the SR one once the optimizer state
equilibrates — but there is no such copy in this configuration, so the writeback *is*
the accumulator.

There is also a correctness argument, not merely a speed one. **Under bf16 with
round-to-nearest, decoupled weight decay silently does nothing.** The `lr * weight_decay`
product at this repo's 3e-4 and 0.1 is a 3e-5 relative shrink per step, against a bf16
half-ULP of `2^-9 = 2.0e-3` — so RTN discards it every step, forever. That is why
`stochastic_round_update_` folds the decay into the *same* rounding as the update
instead of applying it separately.

### The construction

Reinterpret the fp32 value as int32, add a uniform `k`-bit integer, clear the low `k`
bits. Truncating the low bits of an IEEE-754 significand rounds *toward zero* on both
signs because the format is sign-magnitude, so the discarded bits are exactly the
fractional position between the two neighbours, and a carry out of bit `k` lands on the
away-from-zero neighbour — via a mantissa overflow into the exponent, which is that same
increment.

**The count of random bits must equal the count of discarded bits.** Fitzgibbon and
Felix bound this construction's bias by `(2^-D - 2^-N)/2`, tight for `N <= D` and
therefore zero only at `N == D`
([arXiv:2504.20634](https://arxiv.org/abs/2504.20634) §III-E). Their few-bit variant
*diverged* on nanoGPT where the bias-corrected one converged, so reusing one 16-bit draw
for a 13-bit target is a bug, not a saving.

**bf16 only, by construction rather than by omission.** bf16 is the top 16 bits of fp32,
so `k = 16` for every input including fp32 subnormals — one draw width, no branch. fp16
is not a bit prefix (five exponent bits against eight, a different bias), so its `k`
varies with the exponent: 13 bits when normal, growing to 23 across the subnormals, and
below `2^-24` the mask reaches into the fp32 exponent field altogether. Supporting it
means a variable-width draw and a separate below-subnormal branch, which is real
machinery in service of a dtype this repo does not train in. `_format_of` rejects fp16
rather than carrying a second derivation nothing exercises, and `adamw16`'s stochastic
writeback refuses it rather than quietly reusing its mask.

The kernel keeps the machinery for a format *with* a subnormal gap (`SUBNORMAL_GAP`,
`TINY_BITS`, `TAIL_SCALE`) even though bf16 never takes that branch, because the
derivation is what makes the bf16 case obviously correct rather than accidentally so.
Inside it: a shift by `>= 32` is undefined and `k` reaches 126 on an fp32 subnormal, so
the shift is clamped — making the discarded lane deterministic rather than correct.
Below the smallest subnormal the neighbours are 0 and `±TINY`, so the round-up
probability is the value in units of `TINY`; it is compared against `TAIL_ONE - p`
rather than `p`, which keeps the carry path's convention that a zero draw rounds toward
zero. `_TAIL_ONE` is `1 << 30` rather than `1 << 31` precisely so `TAIL_ONE - p` cannot
overflow int32. The sign travels as a bit rather than through a float negation, so a
negative value rounding to zero gives `-0.0` on every backend instead of depending on
how the compiler folds it.

An fp32 subnormal reads as exponent −127 here rather than the −126 it means, and that is
harmless: every fp32 subnormal is far below any 16-bit format's smallest subnormal, so it
takes the tail branch on either reading. Clamping it to −126 was tried and no test could
tell.

`_sr_round` returns fp32 rather than the target dtype, which keeps the caller's `.to()`
lossless: every value it returns is already on the target grid, so that cast cannot round
a second time.

```python
from kohakuwullm.kernels import stochastic_round_, stochastic_round_update_

# A bare cast: fp32 master -> bf16 storage, dithered.
stochastic_round_(bf16_param, fp32_value, seed=step, rng_offset=0)

# The fused writeback, which is what an optimizer actually calls:
#   param = param * (1 - lr*wd) + alpha * update,  rounded once, stochastically.
stochastic_round_update_(
    param, update, seed=step, decay=lr * weight_decay, alpha=-lr, rng_offset=base
)
```

The seed must **advance every step** and be **identical across data-parallel
replicas**; `rng_offset` separates the draws of several tensors sharing one seed.
Both are argued below, and both have cost a run.

### Where to apply it, and where not

**Apply it to the weight writeback and to elementwise first moments. Not to Muon's
momentum buffer.** The positive evidence is all on the weight update, and the one direct
test of SR on a Muon momentum buffer found it *worse* than round-to-nearest — final PPL
42.99 against 40.93 on GPT-2 Small, with higher variance throughout
([MuonQ](https://arxiv.org/abs/2605.11396) App. E). The mechanism is specific and does
not depend on the precision: Muon's polar projection keeps only direction and discards
magnitude, so it is nonlinear, and a zero-mean perturbation of the momentum does *not*
give a zero-mean perturbation of `polar(momentum)`. SR's whole argument is that the
errors cancel in expectation, and there they do not — they accumulate as persistent
directional noise. Muon's *weight* writeback is elementwise and takes SR normally; the
orthogonalization has already happened by then.

### The learning rate, or why an A/B can produce a false negative

**SR needs a 2–4x higher learning rate than the mixed-precision default**, so anything
tested at the tuned MP learning rate gets a false negative. Ozkara et al. (§2.2, §5.1)
model SR as a random walk: below half an ULP the update lands only with probability
proportional to its size, so a small learning rate makes SR *itself* stagnate — 7700
steps against 4600 in their toy problem. They attribute both published BF16+SR failures
(a 7% BERT-base gap and 2% on a 420M decoder) to this rather than to SR, and find SR
*more* robust to the higher rate than mixed precision is, because the dither
decorrelates gradients across time. Any A/B against this repo's tuned 3e-4 must re-sweep
the learning rate for the SR arm or it is measuring the wrong thing.

### The seed is a caller argument

Never a device query, never an ambient generator, for two reasons that both cost a run.
A seed that does not advance per step applies the *same* dither to the same element
forever, which re-freezes every coordinate whose fractional position is stable — SR in
name only. And under data parallel every replica must draw the *same* numbers, or
replicas walk to different weights from one gradient and the aggregated gradient stops
estimating what the optimizer thinks it estimates (Ozkara et al. §3).

`rng_offset` separates the draws of several tensors rounded under one seed. The RNG
counter is int64 and the addressing is 32-bit: a whole-model offset passes `2^31` at
about 2.1B parameters, while no single tensor in this repo's presets is near `2^31`
elements — which is why the guard is per tensor.

### Implementation notes worth keeping

`tl.randint` is documented as int32 and returns uint32, and an unsigned block refuses
the negative truncation mask, so the draw is bitcast rather than converted.

**Tiles are fixed at `BLOCK=1024, num_warps=4` rather than autotuned.**
`triton.autotune` keys on the argument list, so a key including `n` would retune once
per distinct parameter shape — hundreds of them in one model — to pick between
configurations that a kernel reading 4 bytes and writing 2 cannot tell apart.

`_format_of` is cached because it builds a tensor to read `tiny`'s bit pattern and an
optimizer calls the wrapper once per parameter, a few hundred times per step. Uncached
it cost about 5% of the writeback's measured bandwidth, charged as host time inside the
launch. Where a format has no subnormal gap the tail branch is unreachable and its
scale would be `2^163` for bf16 — a constant no fp32 holds — so it is zeroed here rather
than left for a later refactor to find.

Two Triton-versus-torch frictions show up as duplicated constants. Triton reads a module
global from a jitted body only when it is a `tl.constexpr` *instance*, while torch ops
refuse the constexpr wrapper — so each constant exists twice, and the twin is *derived*
from the host value rather than retyped, because a pair of hand-written literals is a
pair that drifts.

`_check` validates shape, layout and the RNG arguments but deliberately **not** `other`'s
dtype, because the two entry points disagree about what `other` is: `stochastic_round_`
copies an fp32 master and must have fp32, while the fused update takes Muon's
orthogonalized factor in `ns_dtype` (bf16) and promotes it on load. Asserting fp32 here
once made the guard reject the only production caller the update path has.

In `stochastic_round_update_`, the parameter is widened *before* the scale: `alpha`
carries `-lr` for Muon, and multiplying a bf16 update by it in bf16 would round the step
before it is added. `alpha` exists at all so Muon can share this kernel — its writeback
is `param * (1 - lr*wd) + (-lr*scale) * update`, and pre-scaling the update instead would
cost a second full pass over the largest tensor in the step.

The fused update exists because the unfused writeback is launch- and bandwidth-bound,
not because the arithmetic is hard: a per-parameter cast kernel over a few hundred
tensors pays about 65 us of launch each, and the torch spelling of the same writeback
materializes a full-size noise tensor plus four intermediates where this reads 2 + 2
bytes and writes 2.

`stochastic_round_reference` takes its `draw` tensor from the caller rather than
sampling, so a test can hand the reference and the kernel the same bits and demand
equality instead of comparing two distributions. It uses `full_like`/`zeros_like` rather
than Python scalars because `torch.where` widens bare ints to int64 and the reinterpret
would then double the length.

## The grouped writeback

Calling `stochastic_round_update_` once per parameter costs `n_tensors * ~33 us` of host
issue, which for a small model dwarfs the device time and for a large one does not. One
launch over a chunk table runs at DRAM peak (~1770 GB/s measured), so **the gain is
entirely a function of parameters per tensor**, and it decays across this repo's own
ladder:

| preset | tensors | params | per-param | grouped |
|---|---|---|---|---|
| Kohaku-200M | 173 | 0.204B | 5.78 ms | 6.25x |
| Kohaku-500M | 223 | 0.546B | 7.35 ms | 2.97x |
| Kohaku-1B | 323 | 0.982B | 10.64 ms | 2.39x |
| Kohaku-1.5B | 393 | 1.514B | 13.25 ms | 1.93x |
| Kohaku-MoE-1B | 208 | 0.991B | 6.89 ms | 1.54x |
| Kohaku-MoE-2B | 273 | 1.953B | 9.37 ms | 1.06x |
| Kohaku-MoE-3B | 351 | 2.907B | 13.33 ms | 1.01x |

The crossover lands where the two scalings meet: grouping pays below roughly **7.3M
parameters per tensor** and is a rounding error above it. Do not quote the small-model
number as the speedup.

**The whole-step measurement is a null result.** At Kohaku-1.5B with 8192 tokens, over
seven rotated repetitions at 0.4–0.6% spread, a step costs 473.17 ms with no writeback,
480.82 ms per-parameter and 480.52 ms grouped — **1.001x**. Both arms add the writeback's
roughly 6.9 ms of *device* time and nothing else, because the forward and backward give
the host hundreds of milliseconds to run ahead of, so 393 launches issue for free. The
isolated 1.93x at this rung does not survive into a step.
`scripts/bench/kernel/sr_whole_step.py` has the method and the two confounds it had to
design out. Whether the 200M rung's 6.25x survives is **untested**: a smaller model has
proportionally less forward and backward to hide behind.

Keep it for the reasons that do not depend on a speedup. It is never slower; it reaches
the floor a flat parameter buffer would have — 1.133 ms against 1.145 ms for the same
elements in one contiguous allocation, so the chunk-table indirection costs nothing —
and it removes 173 to 494 launches from the step, which is a launch-budget argument
rather than a latency one.

```python
from kohakuwullm.kernels import GroupedWriteback

# Built once, from the shapes, which never change. Every param in one group shares
# a dtype; updates are fp32, which is what the optimizer already has.
writeback = GroupedWriteback(params, updates)

for step in range(max_steps):
    ...
    writeback(seed=step, decay=lr * weight_decay)   # one launch for every parameter

# Anything that reallocates a parameter or update buffer invalidates the table.
writeback.rebuild(params, updates)
```

There is no `alpha` here — the per-parameter call takes one because Muon shares that
kernel, and this path takes updates the caller has already scaled. `rng_offsets()`
returns the per-tensor offsets that make the two **bit-identical**, which is what
`tests/test_stochastic_round.py::test_grouped_matches_the_per_parameter_loop` asserts.

### Why a class, and why chunk descriptors

The table that makes one launch possible is built from the *shapes*, which never change,
so it is built once and reused every step. That is the whole reason this is a class and
not a function: the alternative is rebuilding a few tens of thousands of chunk
descriptors in Python on every step, which reintroduces the host cost the launch collapse
just removed.

Chunk descriptors rather than a binary search over a prefix sum. A grid of
`ceil(total / BLOCK)` with each program searching for its own tensor is the obvious
design and it is wrong here: a block that straddles a tensor boundary would have to serve
two base pointers, and a parameter list whose smallest tensor is 896 elements against an
embedding's 58.7M straddles constantly. One descriptor per (tensor, chunk) makes every
program's range lie inside exactly one tensor, at a cost of 16 bytes per chunk of table.

`_GROUP_BLOCK` is 4096, larger than the per-tensor kernel's 1024, because the table costs
16 bytes per chunk: a whole model at 1024 is 245k descriptors and at 4096 it is 61k, and
a streaming kernel cannot tell the two block sizes apart (measured flat across 256–8192).

The pointee dtype cannot be read off an int64 handle the way it can off a real pointer
argument, so the parameter dtype arrives as a constexpr — which is also why all
parameters in one group must share a dtype. Mixed lists belong in two groups, which is
what this repo's `KEEP_FP32_DEFAULT` policy produces anyway, since the fp32 tail needs no
rounding at all.

The table holds raw pointers, so anything that reallocates a parameter or an update
buffer invalidates it; `rebuild` exists for the cases that legitimately do (a resumed run,
a re-sharded optimizer) and is cheap enough to call whenever ownership is unclear.

**The RNG offset of element `i` of tensor `t` is `cu_numel[t] + i`**, which is exactly
what the per-parameter path produces when called with `rng_offset=cu_numel[t]`. That is
deliberate: it makes the grouped kernel *bit-identical* to the loop it replaces, so
`tests/test_stochastic_round.py::test_grouped_matches_the_per_parameter_loop` is an
equality test rather than a comparison of two distributions. `rng_offsets()` exists to
serve it.

## AdamW with 16-bit state

The point of this kernel is the second moment. AdamW's `v = grad**2` squares the
exponent, and fp16 carries only five exponent bits: a gradient of 1e-4 gives 1e-8, which
is subnormal, and below roughly 2.4e-4 the square underflows to zero outright — at which
point the update divides by `sqrt(0) + eps` and the adaptive scaling collapses. Storing
the **root** instead removes the problem: `sqrt(v)` for that same gradient is 1e-4,
comfortably normal. So the square exists only inside the kernel's fp32 arithmetic and
never reaches memory in either form.

This is what makes the fp32 arithmetic free rather than merely tolerable. Per element the
kernel reads param, grad, `exp_avg` and `exp_avg_rms` and writes three of them: 14 bytes
of 16-bit traffic against roughly a dozen flops. Against 1789 GB/s of measured bandwidth
and about 120 TF/s of vector throughput that is bandwidth-bound by three orders of
magnitude, so widening to fp32 costs instructions nobody is waiting on.

Squaring a 16-bit root doubles its relative error, which costs mantissa the divide does
not need — and buys the exponent headroom that fp16 does need.

Debiasing arrives as reciprocals already computed on the host, because they are functions
of the step count alone and recomputing `1 - beta**t` per element would spend a `pow` on
every lane to reach the same scalar. The decay is decoupled — applied to the parameter,
not folded into the gradient — so it does not enter the moments and stays independent of
the adaptive scale.

**Callers converting from a torch optimizer's `exp_avg_sq` must take the square root**,
or the first step divides by a squared quantity.

Rounding on the writeback is a caller argument rather than a build-time choice, because
the answer differs by dtype: bf16 keeps seven mantissa bits, so a decay-plus-accumulate
can lose an update smaller than half an ULP and stochastic rounding is the remedy; fp16
keeps ten, eight times finer, and may not need it. That is a measurement nobody here has
taken, so the kernel does not assume — and `stochastic=True` is bf16-only, since the
16 random bits are exactly the discarded ones for bf16 and the fp16 discard width varies
with the exponent, where the same construction degenerates to round-to-nearest.

---

# Recurring traps

Five failure modes have each cost this repo more than one incident. They are worth
recognizing by shape rather than by location.

**An autotune key containing a varlen axis.** Costs 365 ms/step (MX quantizer) or
950 ms/step (SwiGLU) of pure `do_bench` overhead, and looks like a slow kernel.

**An autotune over a kernel with a side effect.** Counts multiply by the number of
configs tried; the router avoids autotune entirely and the attention backward resets its
atomic buffers per trial.

**A masked value load paired with an unmasked scale or index load.** The masked side
zeroes the product, so garbage on the other side is invisible — except on `0xFF`, which
is NaN in e8m0, and `NaN * 0` is NaN. See [mxfp8.md](mxfp8.md).

**Scaling an uncomputed row by zero instead of skipping it.** `nan * 0` and `inf * 0`
are `nan`. Every sentinel-bucket path here skips rather than scales.

**A benchmark that shares the kernel's assumption.** Three real defects in this repo's
history were benchmark bugs: a non-leaf input tensor that made every timed backward
fail, a per-element ULP metric that reported 24000 ULP for a numerically perfect GEMM,
and an fp64 reference that OOMed at 131k tokens. A measurement you have not audited is
not evidence.
