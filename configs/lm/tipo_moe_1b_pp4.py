"""TIPO on Kohaku-MoE-1B over 4 pipeline stages: 991M total, 248M active.

The validation recipe on the pipeline path: MXFP8 routed experts, aux-loss-free
router balancing, Muon, packed varlen and in-training previews, all at once.
See docs/internals/pipeline.md.

Run:
    kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b_pp4.py
"""

PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    # A tied head cannot span two ranks.
    "tie_embeddings": False,
    "mxfp8": True,
}

SOURCES = [
    {"name": "danbooru", "repeat": 3},
    {"name": "danbooru_tagger", "repeat": 2},
    {"name": "coyo11m", "repeat": 1},
    {"name": "laion_coco", "repeat": 1},
    {"name": "cc12m", "repeat": 1},
    {"name": "nozomi", "repeat": 1},
]
MAX_LENGTH = 2048
NUM_WORKERS = 16
PREFETCH_FACTOR = 4

GPUS = [0, 1, 2, 3]
PARALLEL = "pipeline"
PIPELINE_KWARGS = {
    "micro_tokens": 8192,
    "num_microbatches": 32,
    "schedule": "1f1b",
    "param_dtype": "bf16",
    "autocast_dtype": "bf16",
}
LOADER_KIND = "pipeline"
LOADER_KWARGS = {
    "k": 8192,
    "num_microbatches": 32,
    "ctx_max": 2048,
    "batches_per_epoch": 100_000,
}

PRECISION = "32-true"
GRAD_ACC = 1
GRAD_CLIP = 1.0
EPOCH = 1
MAX_STEPS = 100_000

OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}

LR = 5e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {
    "lr": {
        "mode": "composer",
        "end": -1,
        "schedules": [
            {"mode": "power", "end": 0.9, "s0": 2500, "b": -0.5},
            {"mode": "cosine", "end": 1.0, "min_value": 0.01},
        ],
    }
}
SCHED_WARMUP_RATIO = 0.02

COMPILE = None
GRAD_CKPT = False

NAME = "TIPO-MoE-1B-pp4"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = True
LOG_INTERVAL = 1
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
