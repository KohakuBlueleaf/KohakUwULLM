# Benchmarking

Everything in `scripts/bench/` is part of the deliverable, not a scratch area, and the
reason is that a benchmark is the only thing standing between a kernel change and a
claim about it. This document is the method: how a number is produced here, which
denominator it is scored against, and the specific ways a measurement in this repo has
lied in the past.

The harness lives in `src/kohakuwullm/bench/` so that a precision check in
`tests/test_kernels.py` and a benchmark row in `scripts/bench/` cannot disagree about
whether a kernel is accurate. Read this before adding a benchmark; the failures below
were all found after publishing a number, not before.

Results live in [../out/bench/README.md](../../out/bench/README.md) and the throughput
story is in [performance.md](performance.md). This document explains how those numbers
were obtained. For comparing two *training runs* rather than two kernels, see
[ab-testing.md](ab-testing.md).

## The shape of a benchmark here

Everything below is one import away. This is the skeleton every script in
`scripts/bench/` follows; the rest of this document is why each line is there.

```python
import torch
from kohakuwullm.bench import (
    cached_peak_bandwidth, device_op_counts, format_metrics,
    make_packs, make_tokens, module_metrics, timing_profile, ulp_error,
)

from kohakuwullm.kernels import rms_norm

x = torch.randn(16384, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.ones(1536, device="cuda", dtype=torch.bfloat16)

def run():
    return rms_norm(x, w, eps=1e-6)

run()                                             # warm up EVERY shape first

metrics = module_metrics(
    run,
    flops=0,                                      # a norm does no matmul
    bytes_moved=2 * x.numel() * x.element_size(),
    ceiling="vector",
)
print(format_metrics("rms_norm", metrics))
assert not metrics["suspect"], metrics["suspect"]

ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6) * w
print("ULP:", ulp_error(run(), ref, torch.bfloat16, mode="elementwise"))
```

That prints, on this box:

```
    rms_norm:    0.138 ms   0.0 TF/s (0.0% of 120T)   731 GB/s (40.9%)   1.088 GiB  HOST-BOUND
ULP: 0.498
AssertionError: host dispatch is 59% of wall, so both the wall figure and the
net-of-dispatch estimate are unreliable
```

**Read that as the harness working, not as a broken example.** 0.498 ULP says the
kernel is as exact as bf16 allows. The 731 GB/s does not say it reaches 41% of DRAM
peak — the `HOST-BOUND` marker and the assertion both say 16384x1536 is too small to
be a bandwidth measurement at all, and the rate is a floor. Grow the tensor until
`host_share` drops under 50%, then the number means something. Everything below is
some version of this one lesson.

`module_metrics` runs the timing profile, divides by both ceilings, and flags the
row. It is the call to reach for; `timing_profile` and `bench_ms` are the layers
underneath, for when you want the parts.

---

## 1. The one thing that goes wrong most: wall time is not device time

`bench_samples` synchronizes, records a CUDA start event, calls `fn`, records an end
event, and synchronizes again. Because the queue is empty when `fn` begins issuing, the
GPU sits idle inside the event window while Python marshals arguments — and that idle
time lands in the result. `bench_ms` is therefore **wall time: device plus dispatch**,
and for small work it is mostly *not* device time.

Two measurements in this repo were wrong by more than 2x from exactly this:

| what was measured | reported | actual | the gap |
|---|---|---|---|
| fused MoE router | ~180 us (bimodal, on Triton launcher warmth) | 34 us kernel | Python dispatch |
| MXFP8 quantize | 28% of DRAM peak | 96% of DRAM peak | a flat ~63 us of dispatch |

The tell in the second case is the useful one: **a cost that does not scale with its
problem size is dispatch, not compute.** The 63 us did not move across a 5x span of
tensor sizes.

### What to call instead

`timing_profile(fn)` returns the three numbers that make each other interpretable:

| key | meaning |
|---|---|
| `wall_ms` | the event-window median — device *plus* host issue |
| `host_ms` | host time to issue `fn` with no sync inside the loop (`cpu_enqueue_ms`) |
| `device_est_ms` | `wall - host` |
| `host_bound` | `host >= device`; the one-bit summary |

```python
from kohakuwullm.bench import timing_profile

p = timing_profile(lambda: model.step(batch))
print(f"{p['wall_ms']:.2f} wall = {p['host_ms']:.2f} host + {p['device_est_ms']:.2f} device")
if p["host_bound"]:
    print("improve the op COUNT, not the kernel")
```

When `host_bound` is true, the number to improve is the op *count*, not the kernel.
`device_op_counts` gives you that count.

`device_est_ms` is a subtraction rather than a graph replay, and the subtraction is
meaningful rather than a fudge precisely because the loop synchronizes before recording
its start event: the window is host issue *then* device execution, not the two
overlapping. It lands within **6–14%** of a real CUDA-graph replay on GEMM shapes.

Two reasons the estimate is preferred over capturing every row:

- A capture holds a memory pool, and doing it across a sweep perturbs rows that are not
  even being captured. Measured: peak **2.49 → 3.87 GiB**, wall times up to **20%**
  higher, and one spread reading inflated to 98%.
- `graph_ms` has a **~11 us replay floor**, so below roughly that scale it reports the
  floor instead of the kernel.

Use `timing_profile(fn, device=True)` for a handful of rows where the estimate needs
confirming. Never for a sweep.

### `wall - host` has a validity range, and outside it produces impossibilities

The two times are not strictly additive, so subtracting a host time that is itself most
of the measurement leaves a residual that is noise. At a **45%** host share the estimate
is already weak; past **50%** it has produced results above the hardware peak. Twice.

| published figure | ceiling it exceeded | cause |
|---|---|---|
| 3404 TF/s grouped GEMM | 270 | `wall - host` over-subtracting |
| 989 TF/s fp8 linear | 270 | same |
| 1401 TF/s bf16 vendor-MoE arm | 227 measured | 98% host share on a 64-iteration Python loop |

Both of the first two were caught by a human noticing the value was physically
impossible, not by the harness. That is why `module_metrics` now runs `_implausible`
on every row, sets `suspect`, and raises a `RuntimeWarning`:

| constant | value | what it gates |
|---|---|---|
| `HOST_SHARE_LIMIT` | 0.5 | above this, neither `ms` nor `net_ms` is evidence |
| `CEILING_SLACK` | 1.02 | a legitimately L2-resident kernel may slightly exceed a ceiling |

---

## 2. The L2 flush

A microbenchmark that repeats the same call re-reads its inputs out of L2 on every
iteration and reports a number the real model will never see. For small kernels this
alone is worth tens of percent. So `bench_samples` writes a **256 MiB** scratch buffer
between iterations — bigger than the L2 on any current card, so writing it evicts
whatever the previous iteration left behind.

The buffer is keyed by device. A buffer on device 0 does not evict device 1's L2, and a
multi-GPU scan that flushes the wrong card measures out of cache.

`bench_peak_memory` materialises the flush buffer before resetting the memory stats,
because it is allocated lazily by the timing path — without that, the first row of a
benchmark reports 256 MiB less peak than every row after it.

### CUDA graphs cannot be flushed, and that changes what they measure

A graph replays a fixed capture; a flush cannot be spliced into it. So `graph_ms`
measures device time honestly but **not device bandwidth**: a working set that fits in
L2 is re-read out of cache every replay and reports **187–215% of DRAM peak** on sub-L2
shapes. That is a cache hit wearing a fast kernel's clothes. Only DRAM-resident working
sets are valid bandwidth evidence from a replay.

This is not academic for MXFP8: the fp8 weight bank fits in this card's 96 MiB L2 at
every preset in the MoE sweep while the bf16 bank does not, so an unflushed replay hands
the fp8 arm a cache advantage the model never gets. `bench/vendor/vendor_moe.py`
therefore captures the callable *once* and times `graph.replay` through the ordinary
flushed sample loop — device time, no dispatch, cold L2 — which is the combination
neither `bench_ms` nor `graph_ms` provides alone.

`graph_ms` returns `nan` on a failed capture rather than raising, because failing to
capture is itself a finding: anything that reads device memory on the host cannot be
captured (`bincount` sizes its output from `input.max()`; `.item()` is explicit), and a
benchmark that crashed there would hide the reason. Note that `nan != nan`, so any code
comparing against a captured time must guard for it — a silent comparison against `nan`
is always false and would clear a `host_bound` flag without saying so.

The replay path has its own floor. The replay's host enqueue lands between the two GPU
timestamps, so every captured time is inflated by roughly one empty-graph replay.
`measure_replay_floor()` measures that per run through the same flushed loop the rows
use, rather than storing it as a constant: a threshold that disqualifies a row must not
be a guess.

---

## 3. Warmup must cover every shape

A partial warmup over a varlen stream charges Triton compilation as throughput. The
first profiled call of a Triton path also issues its autotune replays, and warmup alone
is a silent guard — it says nothing when it is too short.

`device_op_counts` therefore profiles **twice and requires the two to agree**. A launch
count that is not reproducible across two calls is measuring compilation, not the path,
and the function raises rather than returning it.

The related trap is autotuning on a varying `M`. A varlen benchmark that lets Triton
autotune per step pays `do_bench`'s L2 flushes on every new shape — measured at
**365 ms/step**. The signature in a profile is `FillFunctor<int>` running at exactly
DRAM peak, which is the flush buffer, not your kernel.

### Counting launches, not just timing them

```python
from collections import Counter
from kohakuwullm.bench import device_op_counts

ops = device_op_counts(run)          # raises if two profiled calls disagree
print(ops["kernel"], "kernels +", ops["memory"], "memcpy/memset")
for name, n in Counter(ops["names"]).most_common(8):
    print(f"  {n:4d}  {name}")
```

When a path is host-bound the unit that matters is the number of dispatches, because
that is the number a fusion is trying to move. `device_op_counts` returns the kernel
names alongside the totals, deliberately: a launch count is a claim about a *mechanism*
and the bare integer cannot be checked. Three of the ops in a fused MoE backward were
`dw.to(dtype)` casts rather than any of the work the fusion is about — a reader who saw
only "14" could not tell that from a count of GEMMs, nor see that those three were the
next thing to remove.

Memcpy and memset are counted separately rather than dropped. Each is a real dispatch
with a real host cost (a `torch.zeros` before a scatter-add is not free), but they are
not what a "GEMM count" claim is about, so a reader can add them or not.

---

## 4. Why a peak is not a result

Every "% of peak" divides by something, and the denominator decides the answer. Three
rules:

**Measure the bandwidth ceiling; do not derive it from the clock.**
`memory_clock_rate` reports the **stock** clock. On this factory-overclocked RTX 5090
that is 28 Gbps → 1792 GB/s, while the card actually runs 31.8 Gbps → 2035 GB/s. Every
percentage drawn against the wrong one is off by that ratio, and it is wrong in the
flattering direction. The measured achievable figure on this box is **1791 GB/s** — and
note that it sits within a rounding error of the stock-clock theoretical, which makes
the theoretical number a very convincing decoy. `cached_peak_bandwidth()` is the one
denominator every bandwidth percentage in the suite divides by; it is cached per device
because measuring costs ~100 ms and a fifty-row benchmark would otherwise spend more
time calibrating than measuring.

**Quote three access patterns, not one.** `stream_bandwidths()` returns all three
because no single microbenchmark is *the* ceiling: `copy` may dispatch to the DMA
engine, `read` carries no write traffic and so can legitimately beat it, and `triad` is
the closest analogue to a fused elementwise kernel. `measure_peak_bandwidth` takes the
best of the three; quoting any one alone understates the other two.

**Pick the compute ceiling that the kernel can actually reach.** The constants in
`bench/core/timing.py`:

| constant | TFLOP/s | what it is |
|---|---|---|
| `TENSOR_PEAK_TFLOPS` | 400 | mma peak from SM count and clock |
| `TENSOR_MAMF_TFLOPS` | 270 | achievable fp32-accumulate matmul rate — the default scoring ceiling |
| `TF32_MAMF_TFLOPS` | 111 | a Triton fp32 `tl.dot`, which lowers to TF32 tensor cores |
| `VECTOR_PEAK_TFLOPS` | 120 | genuine FMA work with no tensor-core path |

The TF32 row is the one that surprises people. An fp32 *matmul* does not fall to the
vector units — Triton lowers `tl.dot` on fp32 operands to TF32 tensor cores — so scoring
a Triton fp32 GEMM against `VECTOR` reports it above 100% of peak. Pass
`ceiling="tf32"`. The older `tensor_cores=False` argument predates this finding and
means "not a bf16 matmul"; it maps to the vector peak and is kept working for existing
callers.

`TENSOR_MAMF_TFLOPS` is also the fp32-accumulate ceiling, which matters because on
GeForce tensor cores fp16 accumulation is **not** rate-limited the way fp32 is:

| accumulate | TFLOP/s |
|---|---|
| fp32 (Triton) | 210 |
| fp32 (cuBLAS bf16) | 237 |
| **fp16 (Triton)** | **325** |

A kernel accumulating in fp16 can therefore legitimately report above 100% of the
scoring ceiling. Nothing is clamped, and that is deliberate: a clamp would have hidden
the earlier peak-rate bug, where a `PEAK_TFLOPS` of 209.5 — below what a plain cuBLAS
bf16 GEMM achieves — put MFU over 100% for trivial code. See
[performance.md](performance.md) for the accuracy cost of fp16 accumulation and where it
is used.

**And re-measure the ceiling per run where the comparison is against it.**
`dense_ceilings()` in the vendor-MoE benchmark measures this card's bf16 and MXFP8 rates
on a 4096-square GEMM at the start of every run. A stored 227 TF/s bf16 figure once
disqualified a legitimate **228** TF/s grouped row: that constant was a measurement from
another day, not a hardware bound, and a threshold firing on 0.4% is measuring its own
staleness.

---

## 5. Accuracy, judged in ULP

A kernel that is fast and wrong is not a result. Every figure in `scripts/bench/` shows
throughput and accuracy together, because a chart reporting only the fast half invites
exactly that mistake.

**Judge error in ULP, not absolutely.** An absolute tolerance that passes in fp32 is
meaningless in bf16. `ulp_error(got, ref, dtype, mode)` reports the max error in units
of last place for `dtype`; at or below 1.0 the kernel is as exact as the format allows.

**Choosing the mode is not cosmetic:**

| mode | scales by | use for |
|---|---|---|
| `"elementwise"` | each element's own magnitude | elementwise kernels — every output is an independent function of one input, and a small true value really does deserve a small absolute error |
| `"rms"` | RMS magnitude of the whole reference | GEMMs and reductions |

```python
from kohakuwullm.bench import rel_error, ulp_error

ref = (a.double() @ b.double())                 # fp64 oracle, chunked if large
got = my_gemm(a, b)

ulp_error(got, ref, torch.bfloat16, mode="rms")          # a GEMM: RMS-scaled
ulp_error(act, act_ref, torch.bfloat16, mode="elementwise")   # a norm: per-element
rel_error(got, ref)                                       # one number for a table
```

The `rms` mode exists because of a specific false alarm: a numerically perfect GEMM was
once reported at **24,000 ULP**. In a reduction, an output near zero is *cancellation*
between large terms, and the absolute error it inherits comes from the magnitude of
those terms, not from its own. Judging such an element against itself is what produced
the 24,000.

`rel_error` (relative L2, `||got - ref|| / ||ref||`) is the complement: one number for a
whole tensor, computed in fp64, useful where you want a single figure for a table.

**Reference in fp64, and chunk it.** Every Triton kernel needs a precision test against
an fp64 reference, in *both* fp16 and bf16, forward and backward — see
`tests/test_kernels.py`. Watch the reference's own memory: one fp64 reference in this
repo OOMed at 131k tokens, which looked like a kernel failure and was not.

**Verify kernels without a GPU where you can.** `TRITON_INTERPRET=1` runs a Triton
kernel bit-exactly on the CPU. It has caught a `uint32` `randint` and a lost sign bit in
this repo. It leaks, though: set in one test file it silently ran the whole suite's
kernels on CPU, so scope it and unset it.

---

## 6. The FLOP model is analytic, not counted

Every shape in this repo is known at build time, so the arithmetic is *derivable* — and
derivable is what a scaling denominator has to be, because a profiler's FLOP count
already contains the inefficiency you are trying to measure against.
`bench/model/flops_analytic.py` owns the closed forms.

**Parameters are the wrong denominator, and the error is large.** With untied
embeddings the vocabulary table is 50M parameters at Kohaku-200M — a quarter of the
model — and a gather does no multiply-accumulate at all. The LM head is the same size
and *is* a GEMM. A params-based x-axis charges arithmetic to a lookup and gets the small
rungs wrong by tens of percent, which is exactly where a scaling exponent is most
sensitive.

**Matrix and vector are separated because they bind on different hardware limits.**
Matrix work runs on tensor cores at ~227 TF/s (bf16, measured on this box). The vector
side — norms, SiLU, RoPE, softmax, residual adds — runs on the SIMT units at ~120 TF/s
and is bandwidth-bound long before it is FLOP-bound. Adding the two into one number and
dividing by a tensor-core peak reports a utilisation no kernel could reach. `FlopBudget`
exposes `matrix`, `vector`, `total` and `matrix_share` for this reason.

### Conventions

| convention | value |
|---|---|
| one multiply-accumulate | 2 FLOPs |
| GEMM backward | 2x forward (dgrad and wgrad, same shapes) |
| elementwise backward | ~2x forward |
| attention score/AV backward | charged 2x |
| gradient checkpointing | charges the block forward twice, not the head |

FlashAttention's backward recomputes the scores, which is nearer 2.5x than 2x. That is
recorded here rather than folded in, so the number stays a **lower bound with a known
direction**.

Two details that are easy to get wrong:

- Attention scores are charged against `q_dim`, not `kv_dim`. GQA shares keys to save
  traffic, but every query head still scores against its group's key, so the arithmetic
  does not divide.
- `causal_pairs_per_token` is **token-weighted, not document-weighted**. A 4096-token
  document contributes eight times the tokens of a 512-token one, so weighting documents
  equally understates the mean for exactly the long documents that dominate the cost.
  The sum is exact rather than a continuous integral — lengths are integers and the range
  is small.

### MFU and HFU

Two numbers are logged, not one. `perf/mfu` is model FLOPs — what the architecture owes.
`perf/hfu` adds the second forward that gradient checkpointing runs through the blocks;
the gap between them is what recompute costs.

**The correct denominator is the batch's own document lengths, not a nominal context.**
Attention is the only term that depends on length. Rendered TIPO samples run 50–600
tokens against a 2048 context; charging them 2048 would triple their attention term.
`FlopCounter.batch_flops` reads `cu_seqlens` (or the padded `(B, S)` shape, where padding
*is* computed on and so *is* charged).

[performance.md](performance.md) has the table of how wrong the old `6 * active_params`
model was, by preset and context length, and why the errors cancelled at exactly one
commonly-measured configuration.

---

## 7. Counting the model: which "active" you mean

`bench/model/ladder.py` builds every preset on `torch.device("meta")` and counts it.
Shapes are exact, no memory is allocated, and an 8B rung costs the same to census as a
200M one — so a test on a CPU-only box and a bench script that is holding a GPU for
something else can both call it. A closed-form solver is what produced the ladder, and
closed forms omit things: norms, the router matrix, the second embedding matrix. The
census exists to catch the omission rather than inherit it.

```python
from kohakuwullm.bench import census, ladder_census

rung = census("Kohaku-MoE-8B")               # meta device: no memory, exact shapes
print(rung.total, rung.active, rung.routed_active, rung.compute_active)

rung = census("Kohaku-1B", depth=12)         # overrides go straight to the preset
for r in ladder_census():                    # every rung, for a figure's x-axis
    print(r.name, r.compute_active)
```

There are three definitions of "active" in play, and they differ by more than any
architecture choice in the ladder:

| property | includes | why it exists |
|---|---|---|
| `active` | everything a token touches: embedding, head, router | the repo-wide definition, from `count_active_parameters` |
| `routed_active` | body only — no embedding, head or router | the convention the ladder's design targets were solved under |
| `compute_active` | `active` minus the embedding *table* | what actually does arithmetic |

`routed_active` and `active` differ by **41–69%** on a sparse rung, because untied
embed+head is 100–134M. Quoting one under the other's name misreads the ladder badly.

`compute_active` exists because `count_active_parameters` walks modules and charges the
embedding its whole `vocab x dim`, but a token reads one row — the lookup is a gather and
the other 65,535 rows multiply nothing. The LM head is the opposite case and stays in: it
is a full GEMM against every vocabulary row. The error this removes is large *and*
size-dependent, which is worse than merely large: the table is 50M of Kohaku-200M's 204M
"active" but only 100M of MoE-8B's 1371M, so charging it flattens the small end of a
scaling fit. With tied embeddings there is one matrix, the head makes it active, and
nothing is subtracted.

`capacity` is `sqrt(active * total)` under the design convention — a dense rung's
`active` is its total including embed and head, a sparse rung's is `routed_active`.
Mixing the two is not a choice made in the code; it is what the target sequence
(204, 380, 546, 981, …) was solved in, and reproducing it is the only way a figure can
show whether the ladder hits its targets. `capacity_full` uses `active` throughout and is
reported beside it because that is what a reader who looks up `count_active_parameters`
will compute — it runs 20–30% higher on every sparse rung.

---

## 8. Benchmark inputs must be as ragged as the real ones

A benchmark that feeds equal-length documents measures a padded workload wearing a
varlen API: every kernel sees the one shape it specializes best for, and the boundary
handling that dominates a real packed step never runs. `bench/core/batches.py` draws
document lengths uniformly from `[lo, hi]` and packs them to an exact token count,
trimming the document that straddles the boundary — which is what the real packer does,
and the case worth timing.

```python
import torch
from kohakuwullm.bench import make_packs, make_tokens

# 32768 tokens split into microbatches of 8192, documents 50-600 tokens each.
infos = make_packs(budget=32768, per_micro=8192, lo=50, hi=600,
                   device=torch.device("cuda"), seed=0)
# One flat (budget,) pair spanning every microbatch; slice it as you consume them.
ids, labels = make_tokens(infos, vocab=65536, device=torch.device("cuda"), seed=0)

start = 0
for info in infos:
    n = int(info.cu_seqlens[-1])
    out = model(ids[start : start + n], info)   # cu_seqlens carries the boundaries
    start += n
```

The token count per microbatch is **exact** rather than approximate because a pipeline
stage fixes its boundary activation shape at build time; a batch that came up short would
change the shape and rebuild the stage mid-measurement.

Two seeding details that are load-bearing: the generator is passed in rather than a seed,
so a caller building several microbatches draws one continuous stream (reseeding per
microbatch would make every one of them identical); and token ids are drawn from a seed
offset by one from the lengths, so document length is not correlated with token content.

---

## 9. Measuring a per-expert loop fairly

The MoE benchmarks hit a case general timing advice does not cover, and the traps are
worth naming because all three change the verdict.

**It is wave-bound, not launch-bound.** One expert at the 1B preset is 48 CTAs against
170 SMs, so 64 of them issued into one stream leave most of the card idle however cheaply
they are issued. Measured at 98% host time, CUDA graphs alone still leave it at 0.74x —
concurrency across streams is the fix, not fusion or cheaper launches. `StreamedLoop`
fans the jobs across streams with an explicit fork/join, because it has to survive graph
capture.

Round-robin rather than a work queue: a queue would balance ragged expert counts better,
but it needs the counts on the host to schedule, which is the device→host sync the design
exists to avoid — and with capacity padding the counts are equal anyway.

`record_stream` on the results discharges the allocator's cross-stream contract. Each
job's output is allocated on a side stream, so the caching allocator tags that block with
that stream and may reuse it as soon as the Python reference drops — which happens on the
*main* stream, after a consumer the allocator cannot see. It is there on the strength of
that contract, **not** because it fixed an observed corruption: it was added while chasing
a sequential-vs-streamed disagreement and did not change it. The real cause was that the
check sat on the output of `combine_routed`, a scatter-add whose float atomics do not
commute, so the bf16 path disagreed with *itself* by 0.0625 across two calls. On the
pre-combine expert outputs the two constructions are bit-identical at every stream count
from 1 to 64, with and without the call.

**The 128-row alignment is what makes the loop free to feed.** `quantize_mx_vendor`
writes swizzle tile `(row_tile, col_tile)` at `(row_tile * col_tiles + col_tile) * 512`,
so a span of whole row-tiles is contiguous in the flat buffer. An expert whose rows start
at a 128-aligned offset takes its scale operand as a plain **view**: no re-swizzle, no
copy, no extra launch. Only the start must be aligned, not the count. Without it the loop
pays E extra swizzle launches and the comparison changes.

`scale_views` raises on an unaligned start rather than trusting `scaled_mm` to catch it,
because the element-count check there would pass — an unaligned start yields a
*correctly sized* view of the wrong rows, so the GEMM runs and returns MXFP8-plausible
numbers scaled by another expert's exponents. `tests/test_kernels.py` pins this. The
alignment is not free: it wastes `E * 64` rows on average, about 6% at 1024 rows per
expert, and the caller reports that cost.

---

## 10. Figures

`bench/core/plotting.py` decides fonts, colors, grid and export settings in one place so
every benchmark in the repo produces figures that can sit side by side without reading as
separate projects. Headless (Agg), PNG plus SVG.

Two rules the layouts follow:

- **Throughput and accuracy are always shown together.** As above: a chart reporting only
  the fast half invites the mistake it should be catching.
- **Categorical color is stable across figures.** `Palette` assigns and remembers, so the
  color for `varlen` is the same in every plot in the suite and a reader who learned it
  once in the attention figure does not re-read the legend in the block figure. The ramp
  is Okabe-Ito, colorblind-safe, reordered so the first few are maximally distinct in both
  light and dark rendering.

A bar chart without value labels is a shape, so `bar_labels` prints each bar's value
above it.

---

## 11. Checklist

Before you quote a number:

1. Is `host_bound` false, and is `host_share` under 50%? If not, the rate is a floor, not
   a measurement.
2. Did the warmup cover **every** shape the timed loop will see?
3. Is the working set larger than L2 — or, if you used a graph replay, are you quoting
   time rather than bandwidth?
4. Is the ceiling the one this kernel can reach (`tensor` / `tf32` / `vector`), and is the
   bandwidth denominator measured rather than clock-derived?
5. Is there an accuracy panel next to the throughput panel, in ULP, with the right mode?
6. Does `suspect` come back empty?

And audit the benchmark itself the way you would audit the kernel. Three real bugs in
this repo's history were benchmark bugs that made working code look broken: a non-leaf
input tensor that made every timed backward fail, a per-element ULP metric that reported
24,000 ULP for a numerically perfect GEMM, and an fp64 reference that OOMed at 131k
tokens. **A measurement you have not audited is not evidence.**
