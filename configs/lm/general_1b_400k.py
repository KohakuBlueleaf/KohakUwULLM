"""General-language pretrain on Kohaku-MoE-1B: 400k steps, ~105B tokens.

The first general run. Code and math are deliberately absent -- both repos are
gated, and outside guidance caps them near 1% of a general corpus anyway. See
internal/general-pretrain-datasets.md for the mixture and
internal/training-health-monitoring.md for what the metrics mean.

Run::

    kogine run scripts/train/lm_pipe.py --config configs/lm/general_1b_400k.py

Smoke, ten steps, no network::

    kogine run scripts/train/lm_pipe.py --config configs/lm/general_1b_400k.py \\
        --set MAX_STEPS=10 --set CKPT_INTERVAL=0 --set SAMPLE_INTERVAL=0

DRAFT: `repeat` weights are placeholders until the vault conversion finishes and
the real document counts are known. Do not launch without re-deriving them.
"""

DATA_KIND = "corpus"
DATA_ROOT = "/Iolite/text-dataset/_vault"
RENDERER = "plain"

# 400k steps x 262144 tokens = 104.9B. Shares target the 100B row of
# internal/general-pretrain-datasets.md section 2; `repeat` is set so one pass
# over this list lands near that split.
SOURCES = [
    {"name": "en/nemotron-cc-v2.1-hq", "repeat": 1},
    {"name": "en/nemotron-cc-v2.1-hq-syn", "repeat": 1},
    {"name": "en/nemotron-cc-v2.1-hq-dqa", "repeat": 1},
    {"name": "multi/nemotron-cc-v2-trans-dqa", "repeat": 1},
    {"name": "ja/fineweb-2-edu-japanese-10bt", "repeat": 1},
    {"name": "zh-tw/finepdfs-zh-zhtw", "repeat": 1},
    {"name": "zh-tw/ultra-fineweb-l3", "repeat": 1},
    {"name": "zh-tw/wiki-zhtw", "repeat": 1},
    {"name": "zh-tw/ptt-zhtw", "repeat": 1},
    {"name": "ko/hplt3-kor-hang", "repeat": 1},
    {"name": "stem/nemotron-specialized", "repeat": 1},
]

LOADER_KWARGS = {
    # Skipped documents cost their read and nothing else: the packer draws the
    # next one to fill the same token budget.
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
    # A tied head cannot span two ranks; the stage would untie it and warn.
    "tie_embeddings": False,
}
# Aux-loss-free balancing carries the load; no head z-loss, which costs 1.59x
# end to end. See docs/internals/moe-router-loss.md.
AUX_LOSS_WEIGHT = 0.0
ROUTER_Z_LOSS_WEIGHT = 0.0

MICRO_TOKENS = 16384
NUM_MICROBATCHES = 16
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

MAX_STEPS = 400_000
SEED = 20090220

CKPT_DIR = "out/ckpt/general-1b-400k"
CKPT_INTERVAL = 2000
NAME = "General-1B-400k"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False

# Previews are cut from the batch being trained on, so every step shows
# different real documents against their true continuations.
SAMPLE_INTERVAL = 2000
SAMPLE_COUNT = 4
SAMPLE_FROM_BATCH = 4
SAMPLE_PREFIX_FRAC = 0.25
SAMPLE_TOKENS = 256
SAMPLE_TEMPERATURE = 1.0
SAMPLE_MIN_P = 0.1
SAMPLE_TOP_P = 1.0
SAMPLE_TOP_K = 0
SAMPLE_LOCAL = False
SAMPLE_FORWARD_ONLY = True
