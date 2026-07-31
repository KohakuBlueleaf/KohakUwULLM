# Guides

Task-shaped. Each one answers "how do I do X", and assumes
[../concepts/architecture.md](../concepts/architecture.md).

| Doc | Covers |
|---|---|
| [training.md](training.md) | The Lightning trainer, schedules and the optimizer defaults, token accounting, callbacks, previews, resume |
| [writing-configs.md](writing-configs.md) | KohakuEngine config files, the full knob catalogue, overrides and sweeps |
| [writing-scripts.md](writing-scripts.md) | Writing a new training, bench or inference script against the installed package |
| [generation.md](generation.md) | Sampling (top-k / top-p / min-p), the KV cache, pipelined decode, and where the speed is |
| [extending.md](extending.md) | Adding a component, router, MoE formulation, data source, renderer or optimizer without touching the core |

## Start here

```bash
uv pip install -e ".[dev,bench]"

# 40 steps of the full shipping path on one card: MXFP8, Muon, real corpus
kogine run scripts/train/lm.py --config configs/lm/smoke_mxfp8.py

# dense 500M on 4 cards
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py
```

A config is plain Python: it names a preset, overrides architecture fields, lists
data sources, and sets the training knobs.

```python
PRESET = "Kohaku-500M"
ARCH_OVERRIDES = {"max_position": 4096, "qk_norm": True, "mxfp8": True}
SOURCES = [{"name": "danbooru", "repeat": 3}]
OPTIMIZER = "muon"
MAX_STEPS = 100_000
```

## Next

- What the knobs cost: [../performance/performance.md](../performance/performance.md)
- How a component is actually implemented: [../internals/](../internals/README.md)
