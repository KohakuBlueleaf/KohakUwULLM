"""TIPO-500M successor: dense Kohaku-500M on 4x RTX 5090, full caption corpus.

The production dense recipe. One weighted pass is ~19.3B trained tokens (13.5B raw at
`repeat: 1`, counted by `scripts/data/token_census.py`), which at this rung's measured
12.79 B tokens/day is a little over a day and a half.

Run:
    kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m.py
"""

PRESET = "Kohaku-500M"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    # Untied. Tying only holds up for a very large corpus and batch; below that one
    # matrix serves two objectives and the embedding can collapse.
    "tie_embeddings": False,
    # Measured 1.139x at this rung against bf16 on the same cards, at a 400-step loss
    # gap inside run-to-run scatter. docs/internals/mxfp8.md has what is verified and what is not.
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

# Muon on the hidden matrices, its internal AdamW on 1-D parameters and on the
# embedding and head -- an axis indexed by token id has no singular values to equalize.
# 1.51x over AdamW on the dense step, and it holds one momentum buffer against AdamW's
# two fp32 moments.
OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 0.02}

LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO = 0.01

# Varlen batches change token count every step, so the compiled graph must be
# dynamic or Dynamo re-specializes on each new packed length.
COMPILE = None
GRAD_CKPT = False

NAME = "TIPO-500M-v2"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 20
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
