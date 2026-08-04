# Writing a script

Scripts live in `scripts/` and run under [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine):

```bash
kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m.py
```

A script declares its knobs as **module-level constants**. A config file is a plain
Python module that reassigns some of them. KohakuEngine loads the script, applies the
config over its globals, then calls `main()`. There is no argument parser, no YAML
schema, and no config class to keep in sync — the constant *is* the schema, and its
default is the value you get if nobody sets it.

That is the whole mechanism. Everything below follows from it.

## Skeleton

```python
"""One line saying what this script trains or produces."""

import lightning.pytorch as pl
import torch

from kohakuwullm.data import build_dataset, collate_packed
from kohakuwullm.training import LMTrainer

# ---- knobs (a config may override any of these) ----
PRESET = "Nano-25M"
ARCH_OVERRIDES: dict = {}
SOURCES = [{"name": "danbooru", "repeat": 1}]
MAX_LENGTH = 2048
SAMPLES_PER_BATCH = 32
GPUS = 1
GRAD_ACC = 1
LR = 3e-4
MAX_STEPS = -1
PRECISION = "bf16-mixed"
NAME = "my-run"


def main():
    pl.seed_everything(SEED, workers=True)
    ...
```

Group the constants and keep a short trailing comment giving units or the accepted
values (`# int count or list of device ids`, `# -1 -> run the full epoch count`). That
comment is the only documentation a config author reads, so it states *what the value
means*, never why the default was chosen — that belongs in a doc.

## The knob groups in `scripts/train/lm.py`

Read the real file for the current set; this is the shape of it.

| Group | Knobs |
|---|---|
| Resume | `CHECKPOINT_PATH`, `TRAINER_RESUME`, `RESUME_STATE` |
| Tokenizer | `TOKENIZER`, `VOCAB_SIZE` |
| Data | `DATA_ROOT`, `SOURCES`, `RENDERER`, `MAX_LENGTH`, `SAMPLES_PER_BATCH`, `NUM_WORKERS`, `PREFETCH_FACTOR`, `PAD_TO_MULTIPLE` |
| Batch layout | `BATCH_LAYOUT`, `PAD_TOKEN_ID`, `VAL_FRACTION` |
| Model | `PRESET`, `ARCH_OVERRIDES`, `HEAD_KWARGS` |
| Parallelism | `GPUS`, `GRAD_ACC`, `LOADER_KIND`, `PRECISION`, `DDP_COMPRESS_HOOK` |
| Optimizer | `OPTIMIZER`, `OPTIMIZER_KWARGS`, `LR`, `BETAS`, `WEIGHT_DECAY`, `EPS`, `USE_MUP`, `BASE_DIM` |
| Schedule | `SCHEDULER_CONFIG`, `SCHED_WARMUP_RATIO`, `EPOCH`, `MAX_STEPS`, `GRAD_CLIP`, `SEED` |
| Compile | `COMPILE`, `GRAD_CKPT` |
| Logging | `WANDB_PROJECT`, `WANDB_OFFLINE`, `NAME`, `LOG_INTERVAL`, `CKPT_INTERVAL`, `SAMPLE_INTERVAL`, `SAMPLE_PROMPTS`, `THROUGHPUT_INTERVAL` |

`PRESET = None` means `ARCH_OVERRIDES` is the entire `LMArchConfig` — that is the escape
hatch for an architecture with no preset.

## Conventions

**Never use `sys.path` hacks.** Import from the installed package. The repo is installed
editable (`uv pip install -e .`), so `from kohakuwullm...` works from any directory.

**Resolve once, call directly.** Build the optimizer, the schedule, the compile wrapper
and the collate function before the loop. A script should contain no per-step branch on
a knob.

**A step count, not an epoch count.** The training dataset is iterative and has no
length, so progress is measured in optimizer steps. Set `MAX_STEPS`; `EPOCH` exists for
finite datasets.

**Write outputs under `out/`.** It is git-ignored. Benchmarks additionally follow the
grouping in [../out/bench_old/README.md](../../out/bench_old/README.md) and write a table or a
figure, never bare JSON.

## A benchmark script

Benchmarks import their measurement helpers from the package rather than restating them,
so a kernel's precision check in `tests/` and its benchmark row use the same metric:

```python
from kohakuwullm.bench.core.timing import bench_ms, ulp_error
from kohakuwullm.bench.core.plotting import new_figure, save_figure
```

Every figure shows **throughput and accuracy together**. A kernel that is fast and wrong
is not a result, and a chart reporting only the fast half invites exactly that mistake.
[benchmarking.md](../performance/benchmarking.md) covers how to measure so the number means something.

## An inference script

`LMTrainer.generate` runs the preview sampler, which is deliberately cache-free. For
anything larger, build the backbone directly and drive it with a padded `SeqInfo`:

```python
from kohakuwullm import LMBackbone, SeqInfo, get_preset

model = LMBackbone(get_preset("Kohaku-500M")).cuda().eval()
info = SeqInfo.padded(batch, length)
hidden = model(tokens, info)
logits = model.head.logits(hidden[:, -1])
```

Run it under the same autocast dtype as training. With `mxfp8=True` the expert path
accepts only bf16/fp16 activations and will refuse fp32.
