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

Weights are derived by scripts/data/mixture.py from measured mean document
length, so one pass delivers 104.9B tokens: en 68.1%, ja 22.7%, zh-TW 8.0%,
stem 1.2%. Korean and the 15-language `multi` set are absent -- they had not
finished converting, and this run validates the corpus path rather than the
multilingual hypothesis. Re-derive after adding a source.
"""

DATA_KIND = "corpus"
DATA_ROOT = "/Iolite/text-dataset/_vault"
RENDERER = "plain"

# 400k steps x 262144 tokens = 104.9B; one pass over this list delivers it.
SOURCES = [
    {"name": "en/nemotron-cc-v2.1-hq-syn", "repeat": 0.560},
    {"name": "en/nemotron-cc-v2.1-hq", "repeat": 1.000},
    {"name": "ja/fineweb-2-edu-japanese-10bt", "repeat": 1.000},
    {"name": "ja/fineweb-2-edu-japanese-extra", "repeat": 0.355},
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

# Corpus documents average ~870 tokens against TIPO's ~195, and packed varlen
# activation scales with the sum of squared document lengths, so the same token
# budget costs several times the memory. 8192 x 32 keeps 262144 tokens/step.
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
