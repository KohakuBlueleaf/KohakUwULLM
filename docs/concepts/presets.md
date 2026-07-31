# Presets: the Kohaku ladder

A preset is a dict of `LMArchConfig` field overrides, resolved by
`get_preset(name, **overrides)`. It is not a class and not a code path — every
preset builds the same `LMBackbone` described in [architecture.md](architecture.md).

```python
PRESET = "Kohaku-MoE-2B"
ARCH_OVERRIDES = {"mxfp8": True}
```

The **Kohaku ladder** is nine rungs, dense and sparse interleaved. It is the set to
use for new runs. This document says what each rung is, what constraints it was
solved under, and why the two older ladders are retired.

---

## 1. Why a ladder, and what makes one walkable

The point of a ladder is hyperparameter transfer: you tune a learning rate on a
small rung, and the schedule you found is still meaningful three rungs up. That only
works if the rungs differ in *scale* and not in kind.

Three things are held fixed across every Kohaku rung so that they do:

- **vocab 65536, untied embeddings, `head_dim` 64, GQA ratio in [4, 8].**
- **Sparsity** `kappa = top_k / num_experts = 0.125`, *exactly*, on every sparse rung.
- **Expert width** `moe_hidden = 0.5 * dim`, exactly, on every sparse rung but one.

`head_dim` stays 64 everywhere so the attention GEMMs remain tensor-core friendly at
every scale, and every `dim`, `kv_heads * head_dim` and feed-forward width is a
multiple of 128 so `config.mxfp8` is eligible on *every* projection rather than most
of them. A rung you cannot run in fp8 is a rung whose measurements do not compare.

The ladder's rungs form one smooth sequence in **effective capacity**,
`sqrt(active * total)`:

```
204  380  546  756  981  1182  1514  1952  2901   (M)
```

Dense rungs sit on that curve trivially (active = total); sparse rungs are placed so
they interleave rather than duplicate.

---

## 2. The ladder

Counts are **measured** by `scripts/bench/e2e/presets.py` on the meta device, not
solved. A closed form omits the norms, the router matrix and — since untying — the
second embedding matrix.

| preset | dim | depth | heads/kv | ffn | experts | total | active |
|---|---|---|---|---|---|---|---|
| `Kohaku-200M` | 768 | 17 | 12/2 | 2048 | — | 204M | 204M |
| `Kohaku-MoE-1B` | 768 | 16 | 12/2 | 2048 | 8/64 x 384, +1 shared | 991M | 248M |
| `Kohaku-500M` | 1280 | 22 | 20/4 | 3456 | — | 546M | 546M |
| `Kohaku-MoE-2B` | 896 | 21 | 14/2 | 2432 | 8/64 x 512, +1 shared | 1953M | 411M |
| `Kohaku-1B` | 1536 | 32 | 24/4 | 4096 | — | 982M | 982M |
| `Kohaku-MoE-3B` | 1024 | 27 | 16/2 | 2688 | 8/64 x 512, +2 shared | 2907M | 617M |
| `Kohaku-1.5B` | 1792 | 39 | 28/4 | 4736 | — | 1514M | 1514M |
| `Kohaku-MoE-5B` | 1280 | 30 | 20/4 | 3456 | 8/64 x 640, +1 shared | 4934M | 943M |
| `Kohaku-MoE-8B` | 1536 | 33 | 24/4 | 4096 | 16/128 x 384, +1 shared | 7713M | 1371M |

`KOHAKU_LADDER` is that tuple in order, for sweeps and for any caller that wants "the
next size up" rather than a name.

Measured end-to-end throughput and the fp8 speedup per rung are in
[mxfp8.md](../internals/mxfp8.md) and [performance.md](../performance/performance.md).

---

## 3. The design constraints, one at a time

### Sparse rungs are deliberately narrower

Every sparse rung is *narrower* than the dense model of the same total size:
`Kohaku-MoE-2B` is 896 wide where `Kohaku-1B` is 1536. Capacity comes from expert
count, and a narrow model makes both attention and the pipeline seam cheaper.

### Feed-forward widths are given explicitly

`mlp_hidden` is the **nearest** multiple of 128 to the SwiGLU `8/3` ratio, written out
rather than left to `resolve_hidden`'s round-up. Rounding one way only biases every
rung above its solved total: at `dim=1792`, round-up gives 4864 instead of 4736, which
is +0.9M on a count solved to 0.01%.

### `moe_first_dense = 1`, not DeepSeek-V3's 3

Keeping leading layers dense is standard, but at depth 16 three dense layers is a fifth
of the stack — that is a different architecture, not a warm-up.

### `Kohaku-MoE-2B` is the one deviation

Its `moe_hidden` is 512, which is `0.571 * dim`, not 0.500. The reason is MXFP8
eligibility: `0.5 * 896 = 448`, and 448 is not a multiple of 128. It would land as the
shared expert's `w_out.in_features` — FPROP's contraction axis, which cannot be
zero-padded without padding the SwiGLU output on every forward. (That is exactly what
makes the old `MoE-3B-A500M` fp8-ineligible.) 512 keeps the rung eligible at the cost
of 14% granularity, and the deviation is documented rather than hidden.

### Expert width scales with the *active* expert count, not with `dim` alone

Every rung through `Kohaku-MoE-5B` runs E=64 / top-8 with `moe_hidden = 0.5 * dim`,
which puts 9 experts (8 routed + 1 shared) against each token. `Kohaku-MoE-8B` moves
granularity to E=128 / top-16 — `kappa` is still 0.125, only the expert count doubles —
and therefore runs **17** experts per token. Holding `0.5 * dim` there would double the
feed-forward work per layer while leaving attention untouched, so its width is
`0.25 * dim` instead.

The quantity that has to stay put is **active feed-forward parameters per layer,
divided by attention parameters per layer**:

| rung | dim | depth | d/L | active FFN : attention | attention % of active |
|---|---|---|---|---|---|
| Kohaku-MoE-1B | 768 | 16 | 48.0 | 5.8 | 15.1% |
| Kohaku-MoE-2B | 896 | 21 | 42.7 | 6.8 | 13.2% |
| Kohaku-MoE-3B | 1024 | 27 | 37.9 | 6.7 | 13.2% |
| Kohaku-MoE-5B | 1280 | 30 | 42.7 | 5.6 | 15.3% |
| Kohaku-MoE-8B | 1536 | 33 | 46.5 | 5.5 | 15.7% |

Dense rungs sit at 3.4 on the same ratio and rise from 11.5% to 18.9% attention share
with scale, so a sparse ladder holding 5–7 and 13–16% is tracking the dense trend
rather than drifting away from it.

An earlier `Kohaku-MoE-8B` was `dim 1024, depth 38, moe_hidden 512` — E=128 / top-16 at
`0.5 * dim`. That reached 8B by depth: the ratio went to **11.3** and attention fell to
**8.2%** of active parameters, while `dim` went *down* from the 5B's 1280 and `d/L`
collapsed to 26.9 against 38–48 everywhere else. Frontier models hold `d/L` between
roughly 85 and 130 and flat-to-rising with scale (Llama-3-8B 128, Llama-3-70B 102,
DeepSeek-V3 117); a ladder whose aspect ratio falls at the top is scaling the wrong
axis.

Widening alone does not fix it. At E=128 / top-16 with `0.5 * dim`, every width from
1280 to 1792 still lands at 7.0–8.5% attention share, because the cause is the 17
active experts in the numerator rather than the width in the denominator.

`test_sparse_rungs_do_not_starve_attention` pins the 5–7 band. Nothing else in
`tests/test_presets.py` caught the original: `kappa`, the 128-alignment and the
parameter totals were all satisfied by it.

**Do not "fix" `moe_hidden` back to `0.5 * dim` while `top_k` is 16.**

---

## 4. The retired ladders

`Nano-*` and `MoE-*-A*` are **retired**. They stay registered only so benchmark JSONs
already on disk keep resolving and `out/bench/` comparisons stay readable. Do not start
new runs on them.

Two reasons they cannot be walked with one hyperparameter set:

- They predate untied embeddings, so their totals do not absorb the second
  `vocab x dim` matrix.
- Their sparsity moves by 2x at every dense/sparse step: `MoE-1B-A120M` routes
  `3/64 = 0.047`, `MoE-3B-A500M` routes `6/64 = 0.094`. Hyperparameter transfer across a
  ladder is only valid at fixed sparsity, so a learning rate tuned on one rung says
  nothing about the next.

### Dense `Nano-*`

Each scale has `wide` and `deep` variants at matched parameter count, so width and
depth can be compared independently. `head_dim` is 64 or 128 throughout.

| preset | dim | depth | heads/kv | head_dim |
|---|---|---|---|---|
| `Nano-25M` | 384 | 8 | 6/2 | 64 |
| `Nano-100M` | 640 | 14 | 10/2 | 64 |
| `Nano-200M` | 896 | 16 | 14/2 | 64 |
| `Nano-200M-wide` | 1152 | 10 | 18/3 | 64 |
| `Nano-200M-deep` | 704 | 26 | 11/1 | 64 |
| `Nano-500M` | 1280 | 20 | 20/4 | 64 |
| `Nano-500M-wide` | 1664 | 12 | 26/2 | 64 |
| `Nano-500M-deep` | 1024 | 32 | 16/2 | 64 |
| `Nano-1B` | 1792 | 26 | 28/4 | 64 |
| `Nano-1B-wide` | 2304 | 16 | 18/3 | 128 |
| `Nano-1B-deep` | 1408 | 42 | 22/2 | 64 |

`Nano-200M-deep` is the one that cannot run in fp8 at all: `dim=704` blocks five of its
six projections, and `dim` is FPROP's contraction axis, so it cannot be padded away.

### Sparse `MoE-*-A*`

Named `<total>-A<active>`. `moe_hidden` was solved by binary search against real
parameter counts; `moe_top_k` sets active mass.

| preset | dim | depth | heads/kv | experts | top_k | moe_hidden | first dense |
|---|---|---|---|---|---|---|---|
| `MoE-1B-A120M` | 640 | 18 | 10/2 | 64 | 3 | 448 | 1 |
| `MoE-1B-A280M` | 1024 | 20 | 16/4 | 32 | 4 | 512 | 1 |
| `MoE-2B-A370M` | 1152 | 24 | 18/3 | 48 | 4 | 512 | 2 |
| `MoE-3B-A500M` | 1280 | 28 | 20/4 | 64 | 6 | 448 | 2 |
| `MoE-3B-A500M-wide` | 1792 | 16 | 28/4 | 48 | 4 | 704 | 1 |
| `MoE-3B-A500M-deep` | 1024 | 40 | 16/2 | 80 | 7 | 320 | 2 |
| `MoE-8B-A1B` | 1536 | 32 | 24/4 | 96 | 8 | 576 | 2 |

The three `MoE-3B-A500M` variants are at matched size so the dim-versus-depth
comparison is not confounded by parameter count.

`MoE-8B-A1B` is 8.1B total / 1.14B active, which is ~128 GB of AdamW state — DDP cannot
hold it on a 32 GB card, so it is pipeline-only (see [pipeline.md](../internals/pipeline.md)). It is
the one retired preset with `tie_embeddings=False` set explicitly, because the pipeline
split separates the embedding from the head.

---

## 5. Adding a rung

If you add one, hold the three fixed quantities in §1 and check three things:

1. `dim`, `kv_heads * head_dim`, `mlp_hidden` and `moe_hidden` are all multiples of
   128, or the rung is not fp8-eligible and its measurements will not compare against
   its neighbours'.
2. `top_k / num_experts` is 0.125, or the rung breaks transfer for everything above it.
3. The reported total comes from `param_summary()` on a built (meta-device) model, not
   from a formula. A closed form omits the norms, the router matrix and the second
   embedding.
