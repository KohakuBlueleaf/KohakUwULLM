# Writing configs

Configs are [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine)
bare-style Python files. The training scripts use `ALL_CAPS` module-level
globals, so a config is just a file that overrides them.

```bash
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py
kogine run scripts/train/lm.py --config configs/lm/debug.py --set LR=1e-3
kogine config check scripts/train/lm.py --config configs/lm/debug.py
```

Set **only** what differs from the script's defaults; everything else falls
through. `configs/lm/debug.py` is the CPU-cheap smoke test, `smoke_mxfp8.py` the
smoke test for the *shipping* configuration, and `tipo_500m.py` /
`tipo_moe_1b.py` are the production recipes.

**A config belongs to one script.** The globals below are `scripts/train/lm.py`'s.
A model split across cards runs `scripts/train/lm_pipe.py`, which declares its own
set — `MICRO_TOKENS`, `NUM_MICROBATCHES`, `LAYERS`, `MXFP8`, `DATA_KIND`,
`PARAM_DTYPE` / `AUTOCAST_DTYPE`, `COMPILE_STAGE`, and the preview selectors
`SAMPLE_LOCAL` / `SAMPLE_FORWARD_ONLY` (see
[generation.md](generation.md#pipelined-generation)) — and
spawns its own ranks, `GPUS` of them:

```bash
kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py
```

`--set` coerces from the script's declared default and does **not** parse a list
literal, so `LAYERS` has to be pinned in the file. Read the script's globals
block for the catalogue; see [pipeline.md](../internals/pipeline.md#running-one).

## 1. The smallest config that works

```python
PRESET = "Kohaku-200M"
SOURCES = [{"name": "danbooru", "repeat": 1}]
NAME = "my-first-run"
```

Three lines. Everything else — optimizer, schedule, precision, logging — comes
from the script's defaults. Run it, confirm the loss falls, then change one thing
at a time.

## 2. Choosing the model

`PRESET` names a rung of the Kohaku ladder (see [presets.md](../concepts/presets.md)).
`ARCH_OVERRIDES` patches individual fields of the resolved `LMArchConfig`:

```python
PRESET = "Kohaku-500M"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "qk_norm": True,
    "mxfp8": True,
}
```

`PRESET = None` makes `ARCH_OVERRIDES` the *entire* config, for an architecture
with no preset.

## 3. Selecting a component

Any swappable slot takes a registry name, a dotted path, or a dict carrying
constructor kwargs. All three resolve at build time:

```python
ARCH_OVERRIDES = {
    "norm": "rmsnorm",                                  # registry name
    "mlp": "my_package.layers.MyMLP",                   # dotted path, no registration
    "moe_router": {"name": "topk", "score_func": "sigmoid"}, # name + kwargs
}
```

The slots are `norm`, `mlp`, `attn`, `posenc`, `moe_mlp`, `moe_router`, plus
`RENDERER`, `LOADER` and `OPTIMIZER` at the script level. Adding your own is
[extending.md](extending.md).

## 4. Data

`SOURCES` is a list; `repeat` **re-renders** rather than duplicating, because the
renderer draws a different task, length bucket and tag split each pass:

```python
SOURCES = [
    {"name": "danbooru", "repeat": 3},
    {"name": "danbooru_tagger", "repeat": 2},
]
```

One weighted pass over the full corpus is 13.478B tokens — 51,413 steps at a
262144-token global batch. [data.md](../internals/data.md) covers the sources and the renderer.

## 5. Overrides and sweeps

`--set` patches any knob without editing a file, which is what a sweep script
drives:

```bash
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py --set LR=2e-4
kogine config check scripts/train/lm.py --config configs/lm/tipo_500m.py
```

`config check` resolves the config and prints the effective values without
starting a run — use it before committing a card to something.

## 6. Knob catalogue

### Tokenizer / data

| knob | default | notes |
|---|---|---|
| `TOKENIZER` | `"models/tokenizer"` | built by `scripts/tokenizer/build_tokenizer.py` |
| `VOCAB_SIZE` | `65536` | must match the tokenizer |
| `DATA_ROOT` | `/xg7/caption-datasets` | local NVMe; 5x faster than NFS |
| `SOURCES` | `[{"name": "danbooru", "repeat": 1}]` | `repeat` re-renders, it does not duplicate |
| `RENDERER` | `"tipo"` | registry name / dotted path / dict |
| `MAX_LENGTH` | `2048` | truncate one rendered sample |
| `SAMPLES_PER_BATCH` | `32` | documents per step (token count varies) |
| `NUM_WORKERS` / `PREFETCH_FACTOR` | `16` / `4` | 16-24 workers is the sweet spot |
| `PAD_TO_MULTIPLE` | `0` | round packed length up; helps `torch.compile` |
| `BATCH_LAYOUT` | `"packed"` | `"padded"` for `(B, S)`; same loss per token, more wasted compute |
| `PAD_TOKEN_ID` | `0` | padded layout only |
| `VAL_FRACTION` | `0.0` | hold-out fraction |
| `LOADER_KIND` | `"torch"` | `map` \| `iterative` \| `ddp` \| `pipeline`; see [data.md](../internals/data.md) |
| `LOADER_KWARGS` | `{}` | token budget `k`, slack `m`, `batches_per_epoch`, ... |

Sources available: `danbooru`, `danbooru_tagger`, `nozomi`, `cc12m`, `coyo11m`,
`laion_coco`, `imagenet`.

### Model

| knob | default | notes |
|---|---|---|
| `PRESET` | `"Nano-25M"` | `None` -> `ARCH_OVERRIDES` is the whole config |
| `ARCH_OVERRIDES` | `{}` | any `LMArchConfig` field |
| `HEAD_KWARGS` | `{}` | `LMHead` args the arch config does not cover: `kernel` (`chunked_ce` \| `torch`), `chunk`, `vocab_block`, `retain`, `label_smoothing` |

The script's `PRESET` default is a *debug* one. For a real run pick a rung of the
Kohaku ladder -- `Kohaku-200M` / `500M` / `1B` / `1.5B` dense, `Kohaku-MoE-1B` /
`2B` / `3B` / `5B` / `8B` sparse. The `Nano-*` and `MoE-*-A*` names still resolve
so old benchmark JSONs keep working, but they predate untied embeddings and vary
sparsity between rungs, so a hyperparameter tuned on one says nothing about the
next. See [../concepts/architecture.md](../concepts/architecture.md#presets).

`LMArchConfig` fields, at their **actual defaults** -- a preset overrides the
shape ones, and anything here is overridable per config:

```python
ARCH_OVERRIDES = {
    # shape
    "vocab_size": 65536,
    "dim": 1024, "depth": 24, "heads": 16, "kv_heads": 4,
    "head_dim": None,              # None -> dim // heads
    "mlp_ratio": 4.0,
    "mlp_hidden": None,            # None -> derived from mlp_ratio
    "mlp_multiple_of": 128,
    "max_position": 8192,

    # components (registry name / dotted path / dict / class)
    "norm": "rmsnorm",             # rmsnorm_triton | layernorm | gemma_rmsnorm | dyt
    "mlp": "swiglu",               # geglu | gelu | swiglu_triton | moe | latent_moe
    "attn": "varlen",              # triton | sdpa | flex
    "attn_sliding": None,          # backend for sliding layers only; None -> attn
    "posenc": "rope",              # ndrope | ggrope | none
    "norm_eps": 1e-6,

    # attention detail
    "qk_norm": True,
    "qk_norm_affine": True,
    "attn_bias": False,
    "attn_sink": False,
    "sliding_window": None,        # e.g. 1024
    "global_layer_every": 0,       # e.g. 6 -> Gemma-3-style 5:1 interleave
    "window_pattern": None,        # explicit per-layer widths, cycled; overrides both
    "rope_theta": 10000.0,         # production configs raise this to 1e5
    "rope_scaling": None,          # linear | ntk | yarn
    "rope_factor": 1.0,
    "rope_partial": 1.0,           # fraction of head_dim that is rotated

    # MoE (ignored when moe_every == 0)
    "moe_every": 0,                # 1 -> every layer sparse
    "moe_first_dense": 0,          # keep leading layers dense
    "moe_num_experts": 64,
    "moe_top_k": 8,
    "moe_num_shared": 1,
    "moe_hidden": None,            # None -> derived from moe_ratio
    "moe_ratio": 4.0,
    "moe_router": "topk",          # sinkhorn | expert_choice | relu
    "moe_router_kwargs": {},       # e.g. {"score_func": "sigmoid", "bias_update_rate": 1e-3}
    "moe_mlp": None,               # MLP registry key for the expert; None -> "moe"
    "moe_mlp_kwargs": {},

    # head / embedding
    "tie_embeddings": False,       # untied by default -- see concepts/architecture.md
    "embedding_scale": False,      # multiply embeddings by sqrt(dim), as Gemma does
    "logit_soft_cap": None,        # forces the materializing path; prefer qk_norm
    "z_loss_weight": 0.0,          # head z-loss: 4.3x the head, no config sets it
                                   # (docs/internals/moe-router-loss.md)

    # residual / init
    "post_norm": False,
    "scale_residual_by_depth": False,
    "init_std": 0.02,
    "mup_base_dim": None,          # muP reference width; scales every hidden matrix

    # runtime
    "grad_ckpt": False,
    "mxfp8": False,                # fp8 projections + routed experts; see mxfp8.md
}
```

### Compute

| knob | default | notes |
|---|---|---|
| `GPUS` | `1` | int count or list of ids |
| `GRAD_ACC` | `1` | intra-batch split; 1 dataloader batch == 1 optimizer step |
| `GRAD_CLIP` | `1.0` | |
| `EPOCH` / `MAX_STEPS` | `1` / `-1` | `-1` -> run the full epoch count |
| `SEED` | `20090220` | |
| `PRECISION` | `"bf16-mixed"` | `32-true` \| `16-mixed` \| `bf16-true` |
| `DDP_COMPRESS_HOOK` | `"bf16"` | `fp16` \| `None` |
| `COMPILE` | `None` | `{"mode": "module", "dynamic": True}` \| `{"mode": "model"}` |
| `GRAD_CKPT` | `False` | ~30% slower, most activation memory back |
| `PARALLEL` | `"ddp"` | `"pipeline"` splits the model one stage per card; needs `LOADER_KIND="pipeline"` |
| `PIPELINE_KWARGS` | `{"micro_tokens": 8192, "num_microbatches": 32, "schedule": "1f1b", "param_dtype": "bf16", "autocast_dtype": "bf16"}` | read only when `PARALLEL="pipeline"`; that path owns its own precision, so `PRECISION` does not apply |

### Optimizer / schedule

| knob | default |
|---|---|
| `OPTIMIZER` / `OPTIMIZER_KWARGS` | `"adamw"` / `{}` |
| `LR` / `BETAS` / `WEIGHT_DECAY` / `EPS` | `3e-4` / `(0.9, 0.95)` / `0.1` / `1e-8` |
| `USE_MUP` / `BASE_DIM` | `False` / `256` |
| `SCHEDULER_CONFIG` | `{"lr": {"mode": "cosine", "min_value": 0.1, "end": -1}}` |
| `SCHED_WARMUP_RATIO` | `0.01` |

Registered optimizers: `adamw`, `adam`, `sgd`, `muon`, `fused_adamw`,
`adamw8bit`, `adamw4bit`, `adamwfp8`. The production dense and MoE recipes both
use `muon` -- `{"muon_lr": 0.02}` for the 500M dense rung, `{"muon_lr": 2e-3,
"embed_lr": 2e-3}` for every MoE recipe. It runs Muon on the hidden matrices and
its internal AdamW on the 1-D parameters and on the embedding and head, since an
axis indexed by token id has no singular values to equalize.

`end: -1` is auto-filled from the computed step count at runtime. The scheduler
is [AnySchedule](https://github.com/KohakuBlueleaf/AnySchedule); `mode` accepts
`constant`, `cosine`, `linear`, `polynomial`, `power`, `step`.

Weight decay applies to matrices only -- norm scales, biases and the MoE
balancing bias are excluded automatically. You do not need to configure that.

### Resume

| knob | default | notes |
|---|---|---|
| `CHECKPOINT_PATH` | `None` | `.ckpt` to start from |
| `TRAINER_RESUME` | `False` | `False` loads weights into a fresh run; `True` continues the run |
| `RESUME_STATE` | `True` | on a `TRAINER_RESUME`, also restore the RNG stream and the dataloader position |

A `TRAINER_RESUME` restores weights, optimizer, schedule position, loop counters,
the cumulative token totals and (with `RESUME_STATE`) the RNG and the data
position. The data position needs a loader that implements
`state_dict()` / `load_state_dict()`; without one the run restarts its stream from
the beginning and says so at startup.

### Logging

| knob | default |
|---|---|
| `WANDB_PROJECT` / `WANDB_OFFLINE` / `NAME` | `"KohakUwULLM"` / `True` / `"lm-debug"` |
| `LOG_INTERVAL` / `CKPT_INTERVAL` | `10` / `5000` |
| `SAMPLE_INTERVAL` / `SAMPLE_PROMPTS` | `1000` / `None` |
| `SAMPLE_TOKENS` | `None` -- every row runs to EOS or to the context limit |
| `SAMPLE_TEMPERATURE` | `0.8` |
| `SAMPLE_TOP_P` / `SAMPLE_TOP_K` / `SAMPLE_MIN_P` | `0.95` / `0` (off) / `0.0` (off) |
| `THROUGHPUT_INTERVAL` | `50` |

`LOG_INTERVAL` drives the token counters and rates (`train/tokens_seen`,
`train/tokens_trained`, `train/b_tokens_per_day`, ...); `THROUGHPUT_INTERVAL`
drives MFU (`perf/mfu`, `perf/mfu_avg`, `perf/hfu`). MFU is not clamped: the peak
rate in `ThroughputCallback.PEAK_TFLOPS` is the fp32-accumulate ceiling, so a
kernel that accumulates in fp16 can legitimately exceed 1.0.

## Two settings that are easy to get wrong

**`COMPILE` needs `dynamic=True`.** Varlen packs a different token count every
step. A static graph re-specializes and recompiles on each new length, which
costs more than compilation saves.

**`GRAD_ACC` splits the batch, it does not multiply it.** One dataloader batch is
one optimizer step regardless of `GRAD_ACC`; the setting only controls how that
batch is chopped for the backward. To raise the effective batch size, raise
`SAMPLES_PER_BATCH`.
