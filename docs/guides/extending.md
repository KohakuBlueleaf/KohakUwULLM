# Extending the framework

Everything swappable lives in a registry. Either **register** a class into it, or
reference it by **dotted path** (no registration needed). `build(spec, REGISTRY,
**kwargs)` resolves it at construction time.

```python
from kohakuwullm import MLP, build

build("swiglu", MLP, dim=768)                    # registry name
build("my_pkg.layers.MyMLP", None, dim=768)      # dotted path
build({"name": "moe", "num_experts": 64}, MLP, dim=768)  # name + kwargs
build(MyMLP, None, dim=768)                      # a class
build(my_instance)                               # already-built passthrough
```

## A norm / MLP / attention / position encoding

```python
import torch.nn as nn
from kohakuwullm import MLP

@MLP.register("my_mlp")
class MyMLP(nn.Module):
    def __init__(self, dim, ratio=4.0, **kwargs): ...
    def forward(self, x): ...   # (..., D) -> (..., D), last-dim op
```
```python
ARCH_OVERRIDES = {"mlp": "my_mlp"}   # or {"mlp": "my_pkg.MyMLP"}
```

Contracts:

| slot | constructor | forward |
|---|---|---|
| norm | `(dim, eps, affine)` | `(..., D) -> (..., D)` |
| MLP | `(dim, ratio, hidden, multiple_of, bias, fused_gate)` | `(..., D) -> (..., D)` |
| attention | `(dim, heads, kv_heads, head_dim, qk_norm, qk_norm_affine, bias, sliding_window, sink, softmax_scale, eps)` | `(x, seq_info, posenc) -> x` |
| posenc | `(head_dim, theta, scaling, factor, ...)` | `prepare(position_ids, device, dtype) -> carrier or None` |

Take `**kwargs` -- the backbone passes every slot the full set, and a component
that only wants two of them must tolerate the rest rather than raise.

A feed-forward must be a **last-dim op** -- that is what lets the same module
serve packed `(T, D)` and padded `(B, S, D)` without a reshape.

An attention module must **respect document boundaries** when
`seq_info.packed`. If your kernel cannot take `cu_seqlens`, build a
block-diagonal mask with `_doc_ids` + `_causal_mask` from
`models/components/attention.py`. Getting this wrong produces a model that still
trains, just worse.

A position-encoding carrier exposes `apply(q, k)` where `q`/`k` are
`(..., H, Dh)` with the position axis immediately left of the head axis.

## An MoE router

```python
from kohakuwullm import ROUTER

@ROUTER.register("my_router")
class MyRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k, **kwargs): ...
    def forward(self, x):          # (T, D) -> (topk_idx, topk_weight)
        return idx, weight         # (T, k) long, (T, k) float
    def update_bias(self):         # called once per optimizer step
        return imbalance_ratio     # or None
```
```python
ARCH_OVERRIDES = {"moe_router": "my_router", "moe_router_kwargs": {...}}
```

If your router keeps a balancing state, it must be a **buffer, not a parameter** --
it is updated by a rule, not by a gradient, so it must not receive weight decay
or optimizer state.

## A prompt renderer

```python
from kohakuwullm import RENDERER

@RENDERER.register("my_renderer")
class MyRenderer:
    def __call__(self, rec, rng=None, **kwargs):
        return user_text, output_text
```
```python
RENDERER = "my_renderer"
```

Take **all** randomness from the injected `rng`. The dataset seeds it from
`(seed, epoch, index)`, which is what makes a repeated record render differently
each pass while the run stays reproducible. A renderer that calls the global
`random` module breaks both properties.

## A data source

Any object with `__len__` and `__getitem__` returning a normalized record dict
(see `data/vault.py::empty_record`). Register it, or pass the instance directly.

If it holds a database handle, **reopen it on PID change** -- see
`_ForkSafeVaults`. SQLite handles inherited across a fork return corrupt rows
rather than raising.

## A Triton kernel

Two obligations, both non-negotiable in this repo:

1. **A precision test** in the matching `tests/test_kernels*.py` -- module kernels
   in `test_kernels.py`, head-side losses in `test_kernels_ce.py`, MXFP8 in
   `test_kernels_mxfp8_{quantize,linear,grouped,experts}.py` -- against an fp64
   reference, in *both* fp16 and bf16, forward and backward. Use `ulp_error` and
   pick its mode: `"elementwise"` for elementwise kernels, `"rms"` for GEMMs and
   reductions.
2. **A benchmark row** in `scripts/bench/kernel/kernels.py` against whatever it replaces.
   A kernel that does not beat the baseline should be deleted, not kept.

Provide a CPU fallback so tests can run without a GPU:

```python
def my_kernel(x):
    if not x.is_cuda:
        return _reference(x)
    return _MyKernel.apply(x)
```

Its test goes in `tests/test_kernels_cpu_fallback.py`, which is the one kernel
test file with **no** `requires CUDA` module mark. A CPU-fallback test that sits
under that mark is only ever exercised on a box that has a GPU -- that is, never
in the situation the fallback exists for.

If the kernel accumulates in fp16, gate it on `x.dtype is torch.float16` and
raise otherwise -- and do not use it in a backward, where gradient range is
unpredictable and fp16 overflows at 65504.

## A training script

Copy `scripts/train/lm.py`. The conventions the engine relies on:

- every knob is an `ALL_CAPS` module-level global with a working default;
- the script is runnable with no config at all (defaults form a debug recipe);
- a `main()` guarded by `if __name__ == "__main__"`.

Then `kogine run scripts/train/my_script.py --config configs/lm/mine.py`.
See [writing-scripts.md](writing-scripts.md) for the full skeleton.

## A worked example, end to end

Adding a new MLP and training with it, start to finish.

**1. Write it.** `src/kohakuwullm/models/components/mlp.py` already holds the GLU
family, so a new variant goes beside them:

```python
@MLP.register("squared_relu")
class SquaredReLUMLP(nn.Module):
    """Feed-forward with a squared-ReLU activation."""

    def __init__(self, dim, ratio=4.0, hidden=None, multiple_of=128, bias=False, **kwargs):
        super().__init__()
        self.hidden = resolve_hidden(dim, ratio, hidden, multiple_of, glu=False)
        self.w_in = nn.Linear(dim, self.hidden, bias=bias)
        self.w_out = nn.Linear(self.hidden, dim, bias=bias)

    def forward(self, x):
        return self.w_out(torch.relu(self.w_in(x)) ** 2)
```

Resolve the width with `resolve_hidden`, never by restating
`hidden or int(dim * ratio * 2 / 3)`. That fallback omits the `multiple_of`
rounding, and two cost models in this repo got it wrong that way.

**2. Point a config at it.**

```python
PRESET = "Kohaku-500M"
ARCH_OVERRIDES = {"mlp": "squared_relu"}
```

**3. Check the shapes before spending a GPU.** The census builds on the meta
device, so an 8B model costs nothing:

```python
from kohakuwullm.bench.model.ladder import census
print(census("Kohaku-500M"))
```

**4. Pin it with a test.** Extend an existing function rather than adding a new
one. The test that earns its keep is the negative case — for an MLP, that the
width is a multiple of `multiple_of` and that a `(..., D) -> (..., D)` shape
holds for both packed and padded layouts.

**5. Smoke it.** 40 steps on one card against the real corpus:

```bash
kogine run scripts/train/lm.py --config configs/lm/smoke_mxfp8.py --set ARCH_OVERRIDES='{"mlp": "squared_relu"}'
```

**6. If it will run in fp8**, declare its matmul so the swap can account for it.
A module holding a matmul as a bare `nn.Parameter` is invisible to a scan over
`nn.Linear` children; see [mxfp8.md](../internals/mxfp8.md).

## Debugging a run

**The loss is fine but throughput is bad.** `ThroughputCallback` logs MFU and HFU
against the analytic FLOP model. If MFU is low and device time is close to wall
time, you are kernel-bound; if wall time far exceeds device time, you are
host-bound and the fix is launch count, not a faster kernel.
[benchmarking.md](../performance/benchmarking.md) covers separating the two.

**fp8 is on but nothing got faster.** `swap_mxfp8` returns an accounting, not a
success flag. Print `report.summary()`: a model can report zero skipped modules
while a large share of its per-token matmul stays bf16, because matmul held as a
bare parameter is not an `nn.Linear` child.

**A kernel test hangs or a ULP check looks impossibly good.** Check whether
`TRITON_INTERPRET` is set. It is read when `@triton.jit` *decorates*, so setting
it anywhere before a kernel module is imported converts every kernel in the
process to a CPU interpreter — the tests still pass, but they are measuring
numpy rather than the tensor cores. The tell is 100% CPU with no Triton cache
activity.

**An A/B shows a difference and you want to know if it is real.** Get the noise
floor first. MoE output is a nondeterministic scatter-add, so two identical runs
differ; anything under roughly 4% is inside that. Run a same-config second-seed
control arm and express the effect as a multiple of the control's own deviation.
See [ab-testing.md](../performance/ab-testing.md).

**Throughput numbers disagree between runs.** Check for another process on the
card. A run slowed uniformly has a *low* step-spread and a wrong absolute number,
so spread alone will not catch it — falling throughput as the micro-batch grows
is the signature that survives.
