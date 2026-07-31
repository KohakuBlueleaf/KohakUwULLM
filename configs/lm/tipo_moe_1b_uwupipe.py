"""TIPO on Kohaku-MoE-1B, 4 pipeline stages, on the kohakuwupipe trainer.

Drives ``scripts/train/lm_pipe.py``, not ``lm.py``::

    torchrun --standalone --nproc_per_node=4 $(which kogine) run \
        scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py

Ten steps, no network, one checkpoint at the end::

    torchrun --standalone --nproc_per_node=4 $(which kogine) run \
        scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py \
        --set MAX_STEPS=10 --set CKPT_INTERVAL=10 --set SAMPLE_INTERVAL=0

The split is measured at startup rather than pinned -- see
docs/internals/pipeline.md. The micro-batch shape is the measured optimum for
this rung on 4x RTX 5090.
"""

DATA_KIND = "corpus"
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
DATA_ROOT = "/xg7/caption-datasets"
SOURCES = [
    {"name": "danbooru", "repeat": 3},
    {"name": "danbooru_tagger", "repeat": 2},
    {"name": "coyo11m", "repeat": 1},
    {"name": "laion_coco", "repeat": 1},
    {"name": "cc12m", "repeat": 1},
    {"name": "nozomi", "repeat": 1},
]
RENDERER = "tipo"
LOADER_KWARGS = {
    "ctx_max": 2048,
    "num_workers": 12,
    "prefetch_factor": 4,
    "batches_per_epoch": 100_000,
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
# Left to the autotuner: the optimum moves with MICRO_TOKENS, and a pinned
# [5,4,4,3] measured 13% slower here than the [5,5,5,1] measurement derives.
LAYERS = []
SCHEDULE = "1f1b"
PARAM_DTYPE = "fp16"
AUTOCAST_DTYPE = "fp16"
MXFP8 = True

# Muon on the hidden matrices; its non-matrix group is AdamW over the 16-bit
# kernel, which is what makes it fp16-safe. See docs/internals/optimizers.md.
OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}
LR = 5e-4
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
MAX_STEPS = 100_000
SEED = 20090220

CKPT_DIR = "out/ckpt/tipo-moe-1b-uwupipe"
CKPT_INTERVAL = 2000
NAME = "TIPO-MoE-1B-uwupipe"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 10
THROUGHPUT_INTERVAL = 50
SAMPLE_INTERVAL = 2000
SAMPLE_PROMPTS = ["1girl", "scenery, no humans"]
