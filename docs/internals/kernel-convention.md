# The kernel development convention

This document is the method this repo follows when it writes or tunes a kernel.
It was derived while taking a bf16 GEMM from 0.90x of cuBLAS to 1.03x geometric
mean over 22 shapes, and every rule below is here because breaking it cost real
time in that session.

[kernel-dev.md](kernel-dev.md) is the arithmetic: the five budgets, the numbers
to query, the numbers to measure. This document is the process that surrounds it.
[../performance/gemm.md](../performance/gemm.md) is the worked example.

---

## 1. Establish the denominator before anything else

**Never report a percentage until you have measured what you are dividing by.**

Two wrong denominators were live in this repo at once, and each moved every
reported efficiency by more than 30 points.

- A **vendor spec figure**. The 420 TFLOP/s quoted for the RTX 5090 is the
  fp16-accumulate rate. Training accumulates in fp32, which is half rate on
  GeForce. Using it put every kernel at 58% of peak when the truth was 90%.
- A **library's achieved rate**. cuBLAS is not a ceiling. Dividing by it hides
  exactly the gap you are trying to see, and it can be beaten.

The denominator is a microbenchmark you write: back-to-back `mma.sync` with
register-resident operands and several independent accumulator chains.
`scratchpad/mma_peak.cu` is the one this repo uses.

**Report the accumulator type with every matmul rate.** "276.8 TFLOP/s" is
meaningless; "276.8 TFLOP/s bf16 into fp32" is a number.

### Audit the peak-finder too

A peak-finder is a benchmark and gets no special trust. `mmapeak` on this machine
reports 409.2 TFLOP/s for bf16. Its kernel is built on `wmma::fragment<16,16,16>`
and credits `2*M*N*K` per `mma_sync`, but it disassembles to exactly 8192
`HMMA.16816` for 8192 loop iterations, so it issues one `m16n8k16` per call and
does half the FLOPs it claims.

The check is one command:

```bash
cuobjdump -sass a.cubin | grep -oE '[HIQ]MMA[A-Z0-9._]*' | sort | uniq -c
```

Count the MMA opcodes, multiply by the FLOPs of the instruction that actually
appears, and compare against the tool's formula. Do this before you quote a tool.

---

## 2. Derive, then measure, then reconcile

The loop is:

1. **Predict from hardware numbers.** Compute the cost before running anything.
2. **Test a small derived set.** Not a grid. The candidates come from the
   prediction and there should be a handful.
3. **Compare against the prediction, not against the previous best.**
4. **If it matches, keep pushing in that direction.**
5. **If it does not, the model is wrong.** Find out why and change the model
   before changing the kernel.

Step 5 is the whole value. A grid sweep that finds a fast config teaches you
nothing and does not transfer to the next shape. In this session the model was
wrong four times, and each correction was worth more than the tuning:

| prediction | outcome | what the model was missing |
|---|---|---|
| L2 bandwidth limits the tile | refuted before running: L2 sits at 32% while compute is at 100% | nothing, the arithmetic answered it |
| flattening the k-loop recovers the per-tile prologue | **costs 4.3%** | the conditional epilogue lands inside the pipelined region |
| TMA cuts registers, so occupancy rises | registers **did** fall to 126 with zero spills, but shared memory doubled | Triton allocates stage buffers differently for TMA |
| deeper pipelines hide the load | flat across 3, 4 and 5 buffers | the load path was never the limit |

**A refutation you can explain is a result.** Write it down next to the number it
replaces, so the next person does not retry it.

---

## 3. Separate the factors

A single efficiency number invites the wrong fix. Factor it until each term is
independently measurable.

For a GEMM:

```
%ISA = wave_efficiency * pipe_busy
wave_efficiency = (tiles / SMs) / ceil(tiles / SMs)
```

`wave_efficiency` is pure arithmetic on the shape. `pipe_busy` is everything
else, obtained by dividing. At `4096^3` a plain kernel runs at 82.4% of the
ceiling, which is 86.1% waves times 95.7% pipe. **The tile was already almost
perfect and the schedule was the entire problem** — and the single number 82.4%
does not tell you that. Tile tuning was worth about 1% here; fixing the schedule
was worth 6%.

---

## 4. Build the ceiling for your instruction mix, not just for the ISA

The ISA peak assumes operands are already in registers. A real kernel issues
loads and barriers alongside its MMAs, and that mix has its own ceiling.

Count the mix in the SASS main loop, then reproduce it in a microbenchmark with
no global traffic and no epilogue. For this GEMM the mix is 32 HMMA, 12 LDSM and
2 BAR per k-iteration, and the result was the most useful measurement of the
session:

| mix per 32 HMMA | 1 CTA/SM | 2 CTA/SM |
|---|---|---|
| 12 LDSM, 0 BAR | 98.7% | 99.4% |
| 12 LDSM, 2 BAR | 93.5% | **95.8%** |
| 0 LDSM, 2 BAR | 99.5% | 99.5% |

**Barriers alone are free. `ldmatrix` alone is free. Only the combination costs
anything**, because the rendezvous exposes load latency the warps were otherwise
hiding for each other. That sets the honest target at 265.2 TFLOP/s rather than
276.8, and reframes a kernel at 257 as 96.9% of achievable rather than 93% of
peak.

Find the main loop by locating the backward branch:

```python
# addresses are hex in cuobjdump output; the loop is [target, branch]
target < addr and "HMMA" in body  ->  smallest such body is the inner loop
```

---

## 5. Treat every resource limit as a gradient, not a gate

`kernel-dev.md` used to say "any value of `n_spills` above zero means you have
already lost". Measured, on the same tile:

| registers | spills | TFLOP/s |
|---|---|---|
| 160 | 0 | 250.4 |
| 128 | 16 | **253.9** |
| 84 | 84 | 52.6 |

The 16-spill config is fastest, because capping registers at 128 is what fits a
**second CTA** on the SM. A zero-spill TMA control at the same occupancy measures
253.1, the same rate, which proves those spills cost under 0.5%. Above roughly a
quarter of the register count it is a genuine cliff.

**Always find a control that isolates the variable you are blaming.** "It has
spills and it is slow" is not evidence that the spills made it slow.

---

## 6. Measure like the result matters

- **Best-of-N, never the mean**, and say which you used. Warm up 50, time 50 to
  100 with CUDA events, keep the minimum.
- **Interleave the arms.** Sustained benchmarking heats the card. The same config
  timed first and timed fifth in one run differed by **1.8%** here, which is
  larger than most effects worth chasing. Run arms round-robin, keep each arm's
  best round.
- **State the noise floor.** It is 1 to 3% on an idle 5090. Do not report a 1%
  win without it.
- **Small kernels are launch-bound in isolation.** A 262 KB `zero_()` measured
  16 us alone and 1.8 us in place, because in isolation it was bound by Python
  launch overhead. Decompose with bulk timing, not per-call events.
- **Report accuracy in the same table as throughput.** A fast wrong kernel is not
  a result.

---

## 7. Correctness rules that caught real bugs here

- **Check against fp64, not against the vendor kernel.** Comparing to cuBLAS
  reported `0.00000` error for a kernel that happened to be bitwise identical,
  which told us nothing about accuracy. The fp64 reference showed all three
  kernels at the same 0.002869, which is bf16 output rounding.
- **Do not load from and store to the same pointer in one Triton kernel.**
  Clearing a scratch buffer inside the pass that reads it silently corrupted
  about 2% of elements. Zero it from the host instead.
- **A reduction with atomics is not deterministic.** Do not assert bitwise
  equality across calls on a kernel that uses `atomic_add`; assert that accuracy
  holds instead.
- **Localise before theorising.** When the GEMM was wrong, dumping the scratch
  buffer showed the accumulation was correct to 1.0000, which pointed straight at
  the fixup and skipped every wrong hypothesis about the K partition.

---

## 8. Ship the model, not the constant

A tuned constant is worth one shape. The deliverable is a planner:

- **Hardware facts live in a `Device` with explicit knobs.** Three groups:
  queried from the driver, measured on the card, calibrated from a fit. The
  measured group has **no default** — `validate()` raises rather than guessing.
- **The planner scores every legal tile analytically** and returns a ranked list.
  No GPU is touched.
- **Tuning times the shortlist, not the space.** The top few candidates run once
  and the winner is cached by shape. This is an offline table, not
  `triton.autotune`, which re-benchmarks whenever a dimension moves and cost
  365 ms per step when it was left on.

Measured value of the split, bf16 over 22 shapes against cuBLAS:

| | plan only | plan + shortlist tuning |
|---|---|---|
| geometric mean | 0.993 | **1.028** |
| shapes won | 15/22 | **17/22** |
| worst case | 0.714 | **0.950** |
| model error | 5.1% | 3.5% |

The tuner earns its keep exactly where the model cannot see: register allocation.
On a shape whose K does not divide `BLOCK_K`, every candidate compiles to 255
registers and only the compiler knows which one spills.

---

## 9. A model's weights do not survive a change of regime

The bf16 model says wave quantization dominates, and it does — at bf16 rates.
Ported unchanged to MXFP8, whose block scaled instruction runs at **3.93x** bf16,
the same Stream-K schedule measures 0.65 to 0.84x of the vendor kernel and loses
to our own simpler pre-quantized kernel on every shape.

The arithmetic says why. At four times the rate, a tile's compute time is a
quarter of what it was, so every fixed cost inflates fourfold against it. The
fixup atomics move the same bytes; L2 traffic per tile is unchanged, so its
utilisation at a `128 x 128` tile rises from about 32% to about 65%.

**When you change precision, re-derive which term dominates before you port the
optimisation.** A schedule fix worth 1.06x in one regime can be a 1.4x loss in
another, and nothing about the code changed.

---

## 10. The checklist

1. Measure the ISA ceiling for your accumulate dtype. Disassemble anything you
   take a peak number from.
2. Compute the roofline, the waves and the five budgets. Write the prediction
   down before you run.
3. Build the instruction-mix ceiling from the SASS opcode counts.
4. Run a small derived candidate set. Reconcile against the prediction and fix
   the model first when they disagree.
5. Confirm the compiled `n_regs`, `n_spills` and `shared` match what you assumed.
6. Confirm the PTX or SASS contains the instruction you intended.
7. Verify against fp64, in both dtypes, and report ULP alongside throughput.
8. Interleave arms, best-of-N, state the noise floor.
9. Land it as a planner with a `Device` of explicit knobs, and say which card
   produced every number.
