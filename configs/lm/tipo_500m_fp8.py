"""SUPERSEDED -- kept for the A/B it documents, not as a recipe to run.

``tipo_500m.py`` is now itself an MXFP8 config on ``Kohaku-500M``, so the pair this
file was half of no longer exists: it sits on the retired ``Nano-500M`` with tied
embeddings, and the caveat below about the weight refresh is stale -- the refresh has
been wired into ``LMTrainer`` since. Use ``tipo_500m.py``. The measured numbers here
are the single-card ones; the current four-card ladder is in docs/performance/performance.md.

TIPO-500M in MXFP8: `tipo_500m.py` differing in exactly one field.

The point of this file is that ``"mxfp8": True`` is the *only* difference from
``tipo_500m.py``. Same preset, same corpus, same optimizer, same schedule, same
precision plugin -- so a run of each is an A/B on the dtype and nothing else, which is
what the throughput and loss numbers need in order to be attributable. If you change
anything here, change it in both files or the comparison stops being one.

What the swap covers at Nano-500M: 120 projections -- attention q/k/v/o and the
feed-forward pair -- which is 79.0% of matmul MAC/token. The tied LM head is the
remaining 19.3% and deliberately stays bf16: it contracts over the vocabulary, and its
share is why the measured end-to-end win sits well below the per-GEMM 1.59x.

**Do not run this until the per-optimizer-step weight refresh is wired into the
trainer.** ``MXFP8Linear.forward`` populates its quantized cache only when the cache is
empty, so a training loop that never calls the refresh quantizes once at initialization
and then trains every fp8 GEMM against initialization-time weights while the fp32
masters move underneath. The loss would be bad enough to notice with nothing pointing at
the cause. ``LMBackbone.refresh_mxfp8()`` is the hook; it belongs beside
``update_router_bias()`` after ``opt.step()``, inside the ``not empty_step`` guard --
an all-masked batch does not move the weights, and re-quantizing then costs ~15 ms for
nothing.

Expected throughput, measured on one card with both arms certified exclusive: **1.195x**
at Nano-500M at 16384 tokens per step. Note that this is operating-point dependent --
at 8192 tokens the same swap measures 1.032x, because the weight refresh's dispatch cost
stops being hidden behind device work below ~16k tokens. ``GRAD_ACC`` above 1 puts every
microbatch in that regime.

Run:
    kogine run scripts/train/lm.py --config configs/lm/tipo_500m_fp8.py
"""

PRESET = "Nano-500M"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    "tie_embeddings": True,
    # The one field that differs from tipo_500m.py. Raises at construction if the
    # preset has a projection MXFP8Linear cannot take, rather than training a
    # bf16/fp8 mixture that every log would call an fp8 run.
    "mxfp8": True,
}

# Danbooru is repeated because the renderer draws a different task, length
# bucket and tag split each pass -- a repeat is a re-render, not a duplicate.
# The tagger-tagged variant is a genuinely different view of the same images.
SOURCES = [
    {"name": "danbooru", "repeat": 3},
    {"name": "danbooru_tagger", "repeat": 2},
    {"name": "coyo11m", "repeat": 1},
    {"name": "laion_coco", "repeat": 1},
    {"name": "cc12m", "repeat": 1},
    {"name": "nozomi", "repeat": 1},
]
MAX_LENGTH = 2048
SAMPLES_PER_BATCH = 64
NUM_WORKERS = 24
PREFETCH_FACTOR = 6

GPUS = [0, 1, 2, 3]
GRAD_ACC = 1
GRAD_CLIP = 1.0
EPOCH = 1
PRECISION = "bf16-mixed"
DDP_COMPRESS_HOOK = "bf16"

LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO = 0.01

# Varlen batches change token count every step, so the compiled graph must be
# dynamic or Dynamo re-specializes on each new packed length.
COMPILE = None
GRAD_CKPT = False

NAME = "TIPO-500M-v2-fp8"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 20
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
