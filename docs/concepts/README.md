# Concepts

What the framework *is*, before any question of how to run it. Read
[architecture.md](architecture.md) once; everything else in `docs/` assumes it.

| Doc | Covers |
|---|---|
| [architecture.md](architecture.md) | The backbone as a pure function, `SeqInfo` and packed varlen, the decoder block, every swappable component, how presets compose them, the KV cache |
| [presets.md](presets.md) | The Kohaku ladder — each rung, the constraints it was solved under, measured parameter counts, and the three definitions of "active" |

## The one mental model

1. A **backbone** is `model(tokens, seq_info) -> hidden`. Norm / MLP / attention /
   position encoding are swappable; Llama, Gemma and DeepSeekMoE are **presets**
   over one backbone, not separate classes.
2. A **`SeqInfo`** says how the batch is laid out — packed (varlen, training) or
   padded (eval). Only attention reads it.
3. The objective lives in **`LMHead`** and the trainer, never in the trunk.
4. Everything swappable lives in a **registry**; `build(spec, REGISTRY)` resolves
   it at build time.

## Next

- To run one: [../guides/training.md](../guides/training.md)
- To change one: [../guides/extending.md](../guides/extending.md)
- To know what it costs: [../performance/performance.md](../performance/performance.md)
