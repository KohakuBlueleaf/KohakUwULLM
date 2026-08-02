# Multi-stream stage boundaries

A pipeline boundary is usually one tensor: the hidden state. Some models need
more to cross it.

| kind | example | shape |
|---|---|---|
| hidden | the activation | `(T, D)` |
| accumulator | an MoE router's auxiliary loss, summed across stages | `(1,)` |
| constant | a text context every DiT block cross-attends | `(T, D)` |
| skip | a U-Net encoder activation handed to a later decoder stage | `(T, D)` |

A stage returns a tuple, `PipelineStage` freezes its arity and shapes at
construction, and the last stage's `loss_fn` unpacks it.

`scripts/kohakuwupipe/streams_demo.py` runs all three non-trivial cases on four
ranks and asserts the gradient each one is about:

```bash
.venv/bin/python scripts/kohakuwupipe/streams_demo.py --case aux
.venv/bin/python scripts/kohakuwupipe/streams_demo.py --case dit
.venv/bin/python scripts/kohakuwupipe/streams_demo.py --case unet
```

## The accumulator

The common case: every stage adds a scalar, one stage applies it.

```python
from kohakuwupipe.parallel.streams import accumulate, accumulator

def forward(self, hidden, aux=None):
    hidden = self.blocks(hidden)
    carried = accumulator(hidden.device) if aux is None else aux
    return hidden, accumulate(carried, self.my_terms())
```

`d(total)/d(acc) == 1` at every hop, so a term reaches the optimizer worth
exactly itself — no `1/num_stages`, no double counting.

Three details are load-bearing.

**The accumulator is `(1,)`, not `()`.** A scalar has no trailing axis, and the
loss reduction identifies an accumulator by exactly that.

**It requires grad from the start.** `accumulator()` returns a leaf with
`requires_grad=True`. A boundary tensor with no backward edge gets no gradient
send at all, so a stage with no terms of its own would break the chain for every
stage behind it. This is also why a **constant** stream needs `GradCarrier`: it
multiplies by a learnable 1.0 so the edge exists without changing the value.

**Whether the stream exists is a whole-model property.** Decide it from the
whole layer stack, never from this rank's slice. A model whose first stage
happens to hold no contributing layers would otherwise send one tensor while the
next rank expects two, and `PipelineStage` froze that arity at construction.

## The dense-gradient trap

This is the one that costs an afternoon.

```
RuntimeError: Tensors for P2P must be non-overlapping and dense
```

Reducing an accumulator with `.sum()` is what triggers it. `sum()` builds its
gradient by `expand`ing a scalar to stride 0, and NCCL rejects a non-dense
tensor for P2P. The offending tensor is the **gradient**, not the stream, which
is why the obvious fixes do not work:

| attempt | result |
|---|---|
| widen the stream to `(1,)`, `(2,)`, `(4,)`, `(8,)` | still fails |
| `.contiguous()` on the stream | still fails |
| `(stream * torch.ones_like(stream)).sum()` | **works** |

Use `reduce_accumulator`, which is that expression:

```python
from kohakuwupipe.parallel.streams import reduce_accumulator

loss = loss + reduce_accumulator(stream)
```

The trap hides well. A **constant** or **skip** stream can pass while the
accumulator fails, purely because nothing upstream of them required grad, so no
backward send existed to reject. Bisecting one stream at a time is what
separates them:

```
['hidden']                 OK
['hidden', 'ctx']          OK      <- passes for the wrong reason
['hidden', 'skip']         OK      <- passes for the wrong reason
['hidden', 'acc']          FAIL
['hidden', 'acc_dense']    FAIL    <- .contiguous() does not help
['hidden', 'accw1/2/4/8']  FAIL    <- widening does not help
```

`GradCarrier` is *not* required on current torch for the constant case —
`streams_demo.py --no-carrier` passes — but it is kept because the failure it
prevents is silent (a context that trains nothing) rather than loud.

## Normalizing what an accumulator carries

An accumulator holds a **per-microbatch mean**. The token loss is a sum
normalized by the step's trained tokens. Adding one to the other without
thinking rescales the auxiliary coefficient by the accumulation depth:

| what you write | `weight=1e-3` trains as |
|---|---|
| sum the micro-batches' means | `3.2e-2` at 32 micro-batches |
| fold into the sum-reduced loss, then divide by tokens | `1.5e-8` |
| average over micro-batches | `1e-3` |

`build_loss_fn(stage_module, denom, num_microbatches)` does the last one: the
accumulator joins **after** the token normalization and is scaled by
`1/num_microbatches`. Neither wrong version has a symptom — the run trains, at a
coefficient nobody chose.
