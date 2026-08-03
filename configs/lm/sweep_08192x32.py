"""Microbatch sweep: 8192 tokens x 32 microbatches = 262144 per step.

The shape the 400k run is currently using, as the sweep's baseline. Sixty
steps, no checkpoint, no preview.

    kogine run scripts/train/lm_pipe.py --config configs/lm/sweep_08192x32.py
"""

DATA_KIND = "corpus"
DATA_ROOT = "/Iolite/text-dataset/_vault"
RENDERER = "plain"

SOURCES = [
    {"name": "en/nemotron-cc-v2.1-hq-syn", "repeat": 0.616},
    {"name": "en/nemotron-cc-v2.1-hq", "repeat": 1.000},
    {"name": "ja/fineweb-2-edu-japanese-10bt", "repeat": 1.000},
    {"name": "ja/fineweb-2-edu-japanese-extra", "repeat": 0.430},
    {"name": "en/nemotron-cc-v2.1-hq-dqa", "repeat": 1.000},
    {"name": "zh-tw/finepdfs-zh-zhtw", "repeat": 1.000},
    {"name": "zh-tw/ultra-fineweb-l3", "repeat": 1.000},
    {"name": "en/nemotron-cc-v2.1-hq-trans", "repeat": 1.000},
    {"name": "stem/nemotron-specialized", "repeat": 1.000},
    {"name": "zh-tw/wiki-zhtw", "repeat": 1.000},
    {"name": "zh-tw/finepdfs-zh-zhtw-classical", "repeat": 1.000},
    {"name": "zh-tw/ptt-zhtw", "repeat": 1.000},
]

LOADER_KWARGS = {
    "doc_filter": "ngram",
    "val_frac": 0.002,
    "split": "train",
    "ctx_max": 2048,
    "num_workers": 16,
    "prefetch_factor": 4,
    "batches_per_epoch": 400_000,
}

PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    "tie_embeddings": False,
    "grad_ckpt": False,
}
AUX_LOSS_WEIGHT = 0.0
ROUTER_Z_LOSS_WEIGHT = 0.0

MICRO_TOKENS = 8192
NUM_MICROBATCHES = 32
LAYERS = []
SCHEDULE = "1f1b"
PARAM_DTYPE = "fp16"
AUTOCAST_DTYPE = "fp16"

OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}
LR = 5e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
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

MAX_STEPS = 60
SEED = 20090220

CKPT_DIR = ""
CKPT_INTERVAL = 0
SAMPLE_INTERVAL = 0
CONSOLE_INTERVAL = 20
NAME = "sweep-08192x32"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = True
