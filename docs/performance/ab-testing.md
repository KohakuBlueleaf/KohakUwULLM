# A/B testing a training change

The question this document answers is not "did the treatment arm end lower?" It is
**"could this experiment have told the difference?"** Those are different questions, and
in low precision the second one usually decides the answer.

Two runs of the *same* recipe on the *same* replayed tokens do not produce the same
curve. The embedding backward and the flash-attention backward both accumulate with
atomics, whose float addition does not commute; training amplifies the divergence. So
"fp8 is 0.005 worse" is not a statement until you know what bf16's gap against *itself*
is on the same setup. Everything below is machinery for establishing that floor and
refusing to conclude without it.

The statistics live in `src/kohakuwullm/bench/analysis/` rather than in the plotting
script, for the reason the repo splits `kernels.py` from `kernels_plot.py` plus one that
is specific here: **the verdict is the most consequential line of the whole experiment,
and in a plotting script it cannot be tested.** In a module it can, against synthetic
arms whose answer is known by construction.

For measuring a *kernel* rather than comparing two runs, see
[benchmarking.md](benchmarking.md).

---

## 1. The design: a replicate is not optional

The minimum admissible design has three arms, not two:

| arm | differs from baseline in | what it measures |
|---|---|---|
| `bf16` | — | the baseline |
| `bf16_ctrl` | **nothing at all** | the resolution of the experiment |
| treatment | the one thing under test | the effect |

`bf16_ctrl` is a second run of the identical configuration. Its gap against the baseline
*is* the noise floor, and every gap in this repo is reported as a **multiple of it**,
never as an absolute number.

Without that arm, `factorial_read` returns the verdict `"no-replicate"` and the sentence
"the dtype effect has no scale to be read against — inconclusive by construction". That
is a result about the design rather than about the treatment, and it is the honest one.
A block standard error says how precisely *this pair of runs* differs; it says nothing
about whether a second treatment run would differ the same way.

With two arms per family, `factorial_read` averages within each family before
differencing across. That is what makes it a 2x2 rather than four curves: averaging
within family drives the run-to-run component *down* in the effect, while the replicate
difference keeps it *isolated* as the scale to beat.

### The floor is a lower bound, not the resolution

Measured on this box at MoE-1B-A280M:

| pair | reproducibility |
|---|---|
| two bf16 arms, final-window mean | 1e-4 |
| two bf16 arms, windowed gap | 0.003 – 0.005 |
| **the same fp8 configuration, re-run across two sessions** | **0.0097** |

The last row is the one that matters. fp8 arms carry cuBLAS `scaled_mm`, whose split-K
reduction order is not guaranteed, on top of the atomics both arms share. That is a
single observed pair rather than a characterised distribution, so treat **0.01** as the
floor for any arm with fp8 in it. A gap of 0.2 survives either reading. A gap of 0.02
does not, and needs a second same-configuration fp8 arm as its own control before it
means anything.

---

## 2. Steps are not independent samples

Consecutive losses share a model. A standard error taken over 30,000 individual steps
understates the truth by roughly `sqrt(block)`. So every statistic here reduces
non-overlapping **block means** first.

```
BLOCK = 200
```

200 steps is long enough to decorrelate consecutive losses at this batch size, and short
enough to leave ~90 blocks in a 40k-step run's tail — which is where the standard errors
come from. `block_means` drops any short final block rather than weighting it.

`difference(left, right)` returns the mean paired difference of two arms' block means and
the standard error of that mean, as a `Difference` whose `sigma` property is
`|mean| / err`.

For curves rather than statistics, `trailing_mean` smooths with a **trailing** window,
not a centred one: a centred window leaves the final `window/2` points averaged over
fewer samples than the rest, and that is exactly the region a "does the gap grow late?"
reading depends on. Its head is padded with `values[0]`, so callers must skip the first
`window` points rather than plot the padding beside real curves.

---

## 3. Effect against noise, and where 3.0 comes from

`factorial_read` divides the effect by the replicate's own gap and reads the ratio:

| ratio | verdict | meaning |
|---|---|---|
| ≤ 1.0 | `inside` | indistinguishable from a bf16 replicate |
| ≤ 3.0 | `comparable` | needs more replicates to call |
| > 3.0 | `outside` | a real regression at this token count |

```
RESOLVABLE_RATIO = 3.0
```

**Where 3.0 comes from:** one replicate pair estimates the run-to-run scale with a single
degree of freedom, so that scale is itself uncertain by roughly a factor of two. Below a
factor of three, "1.2x the noise" and "0.8x the noise" are the same reading. Calling a
2x difference significant off one replicate pair is claiming a precision the design does
not have.

`FactorialRead.sentence()` produces the conclusion in words from the same data that set
the verdict, and lives beside the logic rather than in the plotting script so the wording
cannot drift from the threshold that selected it — the failure that makes a figure
caption say something its own numbers do not.

---

## 4. A slope and an offset are different results

"fp8 is 0.001 worse" and "fp8 is diverging" look identical in a final-loss table. This is
the single most important distinction in the whole experiment, because a constant offset
is tolerable — more tokens or a retuned learning rate absorbs it — while a **widening**
gap is not. Our runs reach 651.8M tokens; the published MX divergence appeared at 300B,
which is 460x further out, so a slope that is barely resolvable here is large there.

So the question is never "is the slope nonzero". Two identical runs already drift apart,
because of those same atomics. The question is **"is the slope larger than the one bf16
already has against itself"**.

```
TREND_SIGMA = 3.0
```

`gap_trend` fits `gap ~ tokens` and returns the slope with an OLS standard error.
`slope_vs_null` compares a treatment's drift against the replicate's and reports the
*excess*, whose `growing` property requires both a positive slope and `sigma > 3`.

**The excess must be fitted on `treatment - control` directly.** Do not subtract two
slopes that were each measured against the baseline: those two gaps share the baseline's
own noise, so their errors are correlated and combining them in quadrature is wrong.
Conservatively wrong, which is worse than obviously wrong, because it silently costs
sensitivity. Algebraically the baseline cancels —
`(treatment - base) - (control - base) = treatment - control` — so the direct fit is both
simpler and exact. `treatment_slope` and `null_slope` are carried for reporting only;
nothing is computed from their difference.

`gap_trend` expects `token_blocks` pre-scaled by the caller (the scripts divide by 1e8, so
the slope reads per 100M trained tokens). Scaling inside the statistics module would bury
the unit where the reader cannot see it next to the number.

### This test has found a real result

MXFP8 round-to-nearest versus round-up is the case that validates the whole approach. On
*offset* the two are close. On slope, RTN grows at **24.7 sigma** while round-up stays
flat — reproducing NVIDIA's published 843M-parameter divergence at roughly 1/500th of the
scale it was found at. A final-loss table would have called them equivalent.

---

## 5. Divergence, and what does not count as it

```
DIVERGENCE_FACTOR = 1.5
```

`diverged(loss)` is true if the loss contains a non-finite value, **or** if the mean of
the final 200 steps exceeds 1.5x the run's own trailing minimum.

Both halves are required and the second is the one that gets omitted. A NaN is
unambiguous. But a run can survive a violent transient and end healthy — the fp8 4x-lr
arm here hits `|g| = 122` at step 19 and converges normally — so the test is applied to
the **final** loss against the run's own best, never to the maximum. Testing the maximum
would call every warmup a divergence.

It is relative to the run's own trailing minimum rather than an absolute threshold,
because loss scale changes across an lr sweep and any fixed number would flag the high-lr
runs for being high-lr.

### What an lr sweep is allowed to conclude

`margin_verdict` derives the sentence rather than letting a caption go stale against its
data. The case that matters is the one where **neither** dtype breaks:

> neither dtype diverged through Nx base lr: the sweep bounds the margin cost, it does
> not show margin is intact

It is tempting to report that as "fp8 costs no stability margin", and it does not support
that. If the sweep never found the baseline's edge either, it never measured a margin at
all. It bounds how much the treatment *could* have cost, under the span swept — a weaker
and different claim.

---

## 6. Gradient spikes: the metric that misleads

`spike_overlap` returns two spike counts and the Jaccard overlap of the steps they hit.
The counts are the robust half; the Jaccard is the half that misleads. It falls as the
threshold rises largely because a **shared event straddling the line is scored as two
private ones**, and the rarer the event, the more one straddle costs.

`near_miss_fraction` is what stops a falling Jaccard being read as divergence. For every
step one arm flagged and the other did not, it asks how far below the threshold the other
arm actually sat. A high fraction near the line (within 20% by default) means the two arms
saw the same event and the metric split it.

Report both, always.

---

## 7. Was the card exclusive, and could the check have told me?

Two questions, two statistics, and they are **not** the same question:

- **the spread band** asks whether the number is trustworthy.
- **the contended fraction** asks whether the card was shared.

They disagree legitimately. At 3.4% contamination the median is untouched, so a
measurement can be perfectly usable *and* taken on a busy card.

`bench/core/contention.py` owns both. It is arithmetic on a list of floats and imports no
torch, so the rules that gate every benchmark in the repo stay testable on a box with no
GPU.

### The contended fraction

`sample_contended_fraction` is the share of samples more than **1.1x** above the median;
above a limit of **1%**, the card was shared.

Both numbers were chosen by measurement on 76 sampler-labelled windows (33 contended, 43
clean) of a 40k-step training run, after excluding three windows the sampler called dirty
in which not one sample was actually slowed:

| detector | threshold | missed dirty | false-fired clean |
|---|---|---|---|
| **`frac > 1.1x median`** | **0.01** | **0 / 30** | **0 / 43** |
| `(max - min) / median` | 0.10 | 0 / 30 | 0 / 43 |
| `(p90 - p10) / median` | 0.02 | 0 / 30 | 0 / 43 |
| `mean / median` | 1.02 | 4 / 30 | 0 / 43 |

At 1.1x, not one sample in 8,000 across the 43 clean windows exceeded the threshold — the
clean side of this statistic is exactly zero. Any limit in 0.002–0.02 separates perfectly;
0.01 is the middle of that plateau, two orders of margin above a clean card's 0.0000.

`mean / median` is the worst of the four and is deliberately **not** implemented. It was
proposed, and the data that arrived to calibrate it disqualified it. `max - min` ties on
this dataset but is fragile by construction: it is one sample wide, so it tightens as
iterations grow, and at threshold 0.05 it already false-fires on 7 of 43 clean windows.

**Limitation, and it is not small.** The threshold is relative to *this sample's own*
median, because a microbenchmark has no clean baseline to compare against. That works for
the case it exists to catch — intermittent slow samples leave the median alone — but a
*uniformly* slowed measurement moves its own median with it and reads 0.0. Nothing in a
ratio can see that. Only an absolute reference can, which is what gating against
`stream_bandwidths()` is for.

### A verdict needs enough samples to resolve its own limit

At `N` iterations the smallest non-zero fraction is `1/N`, so the verdict has no
resolution between "clean" and "disqualified" until `N >= 1/limit` — **100 iterations** at
the current limit. Below that, one jittery sample is automatically a contention verdict:
at the 30 iterations a GEMM row used, a single hiccup reads 3.3% against a 1% limit.

This is not hypothetical. A 0.29 ms grouped GEMM whose median was reproducible to
**0.87% across five separate processes** on a verified-idle card was flagged in two of the
five, landing on a different arm each time — the signature of a detector with no
resolution, not of a shared card. The calibration behind the 1% limit used windows of a
40k-step run, where the sample count is large; carrying the limit to a 20-iteration
microbenchmark without carrying the sample count is what broke it.

`contended_fraction_min_iters()` returns that number so a caller can size its loop from
the limit rather than pick a count and hope.

### The spread band, and where it has no power

`sample_spread` is `(p90 - p10) / median`. Percentiles, not `max - min`: the range gets
*stricter* as iterations grow, because more samples mean more chances to catch one
outlier, so a fixed threshold silently tightens with `iters`. Measured on a verified-idle
card, a single hiccup put a vendor GEMM at 18.4% of range over 100 iterations — a
contention verdict on a card with nothing else on it.

The threshold is duration-dependent, not flat:

```
spread_threshold(m) = 0.05 + 0.010 / m        # m in ms
```

Scheduling jitter costs roughly a fixed wall time per iteration regardless of how much
work that iteration does. It is a constant *absolute* term and therefore a
duration-dependent *fraction*. The 0.05 base keeps the established 5% for long
measurements so per-step throughput rows keep their meaning; the 0.010 ms floor is the
jitter allowance, calibrated on a verified-idle RTX 5090 against the worst reading a clean
card produced — 44.4% p90–p10 on a 0.021 ms elementwise op over 20 iterations — plus
margin. Measured on an idle card it passes every shape from 0.02 ms to 0.6 ms at both 20
and 100 iterations, where a flat 5% would have flagged five of seven.

A flat 5% is defensible for three samples of a three-second step and is *below the jitter
floor* for twenty samples of a 0.3 ms kernel. One number cannot serve both, and the
failure is in the dangerous direction: it discards clean kernel rows as contended, which
looks like diligence.

Above 25% the allowance exceeds any plausible contention signal, so the test has no power
and `spread_has_power` says so rather than passing everything silently. In closed form
that is `0.010 / (0.25 - 0.05)` = **0.05 ms**; `spread_no_power_below_ms()` computes it
from the three constants rather than restating it, because it was quoted wrong twice
before that function existed — once as 0.2 ms and once as 0.03 ms.

Note that the contamination need not come from *your* card. Another agent's training run
on a neighbouring GPU shows up as contention on an idle card's short kernels, and the test
suite itself perturbs a live run's step times — the host-bound step loop means even
CPU-only tests do it.

### Carrying the verdict with the number

`median_and_contention(fn)` returns median, contended fraction and `has_power` from
**one** sample loop. Two loops would give the verdict its own samples, which is the exact
failure it avoids: a card that was shared during the timing loop and idle during the check
reads clean.

`contention_notes` splits the verdict **by arm** rather than disqualifying a row whole. A
dirty *baseline* arm invalidates the speedup ratio and nothing else — the treatment's rate
and its percentage of a roofline came from different samples and are still good.
Discarding both together once threw away eight of twelve GEMM rows whose fp8 arm read
exactly 0.00%. `suspect_speedup` is kept separate from `suspect` for the same reason: a
figure can draw a throughput panel from a row whose baseline was contended and leave that
row out of the speedup panel only.

### Which card ran it

`visible_devices()` records `CUDA_VISIBLE_DEVICES` because nothing else can.
`bench.timing.device_name()` reads visible index 0 and returns a byte-identical string on
all four cards in this box — whose sustained clocks differ by **2.7%** (GPU0 sustains 3097
MHz against GPU1's 3015). The launcher's environment is the only thing that ever knows
which card ran, and nothing can recover it afterwards.

---

## 8. Admitting a training run

`bench/analysis/runlog.py` owns the per-step CSV and three judgements about it. The
schema is defined there so the reader and the writer cannot drift apart:

```
step, tokens, trained, loss, grad_norm, lr, elapsed
```

The three judgements:

**Loss and grad norm are arithmetic.** They do not care who else was on the card. A
contended run's curves are as good as an exclusive one's.

**Step time is a measurement, and it does care.** A co-tenant inflates it; a *different
card* inflates it by that fixed 2.7%, which is 25x the size of some effects under test.
`speedup()` refuses a pair it cannot certify, and takes `same_card` as a **mandatory**
argument because nothing in the CSV records it. Defaulting it to `True` would silently
bless the exact comparison this repo has already got wrong once — two arms on two cards,
read as a dtype effect — and defaulting it to `False` would make the honest call the
inconvenient one.

**Which statistic reduces the step times.** `elapsed` is written to three decimals, so
per-step differences quantise to 1 ms — 0.7% at 150 ms/step, larger than most effects
here. A median of quantised diffs inherits that floor; a difference of two endpoints over
a long span does not, because the rounding error is paid once over 30k steps instead of
once per step. So `step_ms` is an **endpoint mean** over the post-autotune span, and a
contended run gets nothing at all rather than a median that looks like a number.

| constant | value | meaning |
|---|---|---|
| `TIMING_HEAD` | 100 | rows excluded from *timing* — Triton autotune, cache warm, loss transient |
| `MIN_TIMING_STEPS` | 200 | below this there is no usable timing span |

The head is excluded from timing only. The loss curve keeps those rows, since step 0 is
where every arm starts from bit-identical weights and that is worth seeing.

### Never compare a run that has not finished

`load_arms(..., require_summary=True)` skips arms with no `.json` summary. It is
**mandatory for any comparison of final values**, because a live run's CSV ends wherever
it has reached and admitting one silently compares a partially trained arm against a
completed one.

That is not hypothetical: it briefly made an fp8 sweep read at 600 of 2000 steps look like
a **+0.77 loss regression** against a finished bf16 arm — a divergence-sized number
produced entirely by reading a run too early. Curves-over-time plots want the default;
anything that reduces a run to one number wants `require_summary=True`. Same rule for
`discover_lr_multipliers`, which would otherwise report a sweep's divergence edge as
whatever the in-progress run has reached so far.

### Prove the arms saw the same data

`replay_identical` compares `step`, `tokens`, `trained` and `lr` across arms. These come
from the memmap and the schedule, never from the model, so they are **exactly** equal
across arms or the replay is broken.

Grad-norm spike alignment is the weak version of this test and **cannot** be used for it.
Two arms with different arithmetic genuinely drift apart, so a spike that moves by a step
is evidence about the treatment, not about the loader.

### Use trained tokens, not tokens seen

About 11% of a packed batch is prompt that carries no label, and the loss is a mean over
the other 89%. `load_arms` accumulates `trained` into `cumulative_tokens` for the x-axis.
Note also that the `loss` column is *already* the fp32 per-token mean the head reduced —
dividing by `trained` again is wrong, and it is wrong in a way that **hides** the result:
the extra 1/7000 pushes a real 0.2-nat difference into the fifth decimal, where it reads
as rounding.

### Logging without paying for it

`write_step_rows` buffers rows and stacks every buffered step's loss and grad-norm into
**one** tensor moved once. Reading either scalar per step costs a sync, and a sync
serializes host issue against device execution — which penalises fp8 specifically, since
it issues ~39% more ops and so has more host work that could otherwise hide under the
device. The halves are split by position on the read side; a reader that mismatched the
offsets would pair each step's loss with another step's gradient norm, and the CSV would
still look entirely plausible.

Only every `log_every`-th row's `elapsed` is device-accurate: the final row of each batch
is stamped after the transfer, and earlier rows keep their append-time host timestamp.

---

## 9. Throughput sweeps

`bench/analysis/sweep.py` classifies each row of an end-to-end sweep into one of three
states, and the vocabulary is load-bearing:

| status | meaning |
|---|---|
| `clean` | completed, per-step spread within tolerance |
| `contended` | completed, but the steps disagreed enough that something else was on the card — kept and marked, never averaged in |
| `oom` | did not complete |

`oom` is distinct from *never run*, which is the absence of a row (`best_cell` returns
`None`) and must stay distinguishable from a zero. A missing or NaN spread counts as
contended, not clean: an unknown spread is not evidence of a quiet card.

`best_cell` prefers the fastest clean row, falls back to the least-contended completed
row, and only then reports OOM.

**Spread only sees contention that varied *between* steps.** A run slowed uniformly
across all its steps has a low spread and a wrong absolute number. Within one sweep,
`falling_throughput` catches what survives: throughput dropping as the microbatch *grows*
is the signature of a uniformly slowed run.

Two attribution rules worth knowing when reading a sweep:

- `strategy` is the parallelism only — mode plus checkpointing. Folding dtype into it
  would merge `pp4` and `pp4_bf16` into one series and pick a best across precisions,
  which is how a bf16 number ends up compared against fp32 numbers for every other preset.
- Recorded config beats the tag. `resolve_dtype` returns `None` — genuinely unrecoverable
  — rather than guessing for the one case where the driver forces bf16 silently and the
  tag omits it.

---

## 10. Building an arm that genuinely differs

An A/B is worthless if the two arms are secretly identical, and that failure is silent by
construction: the experiment reports "no effect", which is exactly the answer a broken arm
produces.

The MXFP8 rounding comparison is the sharp case. Round-up (`ceil(log2(amax / 448))`) is
what ships; round-to-nearest is the OCP standard's choice and the one NVIDIA measured an
843M model diverging under at 300B tokens, so it is the discriminator any "is MX training
safe here" experiment has to include. The difference is **one token** of Triton source —
which is exactly why it cannot be a runtime flag: the rounding lives inside a
`triton.jit` body, and a branch there would be a per-element select in the innermost loop
of every GEMM in the model.

`bench/vendor/mxfp8_rounding.py` therefore rewrites the shipped kernel's source and loads
the result under a second module name, leaving the shipped file untouched. Real files on
disk, not `exec` of a string, because `triton.jit` reads a kernel's source through
`inspect.getsource`, which needs it findable by filename.

**Both rewrites assert their substitution count.** A `sed` that silently matched nothing
would leave the "round-to-nearest" arm bit-identical to round-up, and the experiment would
then report that rounding does not matter — the one failure mode the module exists to make
impossible. The expected count is 1 because the lone `tl.ceil` lives in `_quantize_block`,
the `triton.jit` helper that both the ours-path and vendor-path kernels inline; a reader
who assumes one occurrence means one code path would raise that count and break the
rewrite.

The arm being genuinely distinct is checkable without a GPU, and is worth re-checking
after any move under `kernels/mxfp8/`:

```python
rtn = load_round_to_nearest()
assert rtn.linear.quantize_mx_vendor is rtn.quantizer.quantize_mx_vendor
```

---

## 11. Standing rules

| rule | number |
|---|---|
| Block length for standard errors | 200 steps |
| Effect-to-noise ratio callable off one replicate pair | 3.0 |
| Sigma for a gap slope to count as growth | 3.0 |
| Divergence: final-200 mean over the run's own best | 1.5x |
| Noise floor for any arm containing fp8 | ~0.01 nats |
| Contended-fraction limit | 1% of samples above 1.1x median |
| Iterations needed for that verdict to mean anything | 100 |
| Spread band with no power below | 0.05 ms |
| **Throughput difference inside the floor** | **under ~4%** |
| **Throughput difference this repo has acted on** | **8–14%** |

The last two are the summary of §7 and §8 together. Card-to-card sustained clock is 2.7%,
identical work reproduces to about 1%, and the `elapsed` column quantises at 0.7% of a
150 ms step; a difference under roughly 4% is inside that stack and is not a result. The
effects worth acting on have been 8–14%, comfortably outside it. If your measured
difference lands between those bands, the answer is another replicate, not a tighter
threshold.

And the standing procedural rule, because it has bitten repeatedly: **gate on the
`.json`, never on the CSV.** A partial run does not look partial in a plot.
