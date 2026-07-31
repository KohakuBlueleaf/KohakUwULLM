"""TIPO on Kohaku-MoE-1B: 991M total, 248M active, 4x RTX 5090.

The validation recipe. One run exercises everything at once -- MXFP8 on the routed
experts, the aux-loss-free router balancing, Muon, packed varlen, in-training sample
previews and the step-based progress bar -- which is the point: integration bugs only
appear when the pieces run together.

At the measured 13.82 B tokens/day for this rung, 100k steps of 262144 tokens is 26.2B
tokens in a little under two days, or ~1.36 weighted passes of the corpus.

**bf16 parameters, not fp16.** Gradients here are ~1.26e-7 RMS and `dpre` is stored in
the parameter dtype, so fp16 flushes 18.9% of its entries to zero against bf16's none.
No kernel change reaches that -- it needs loss scaling, which is unmeasured. See
docs/internals/mxfp8.md.

Run:
    kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py
"""

PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    "tie_embeddings": False,
    # 1.533x against bf16 on the same cards at this rung, and 44% less peak memory --
    # the fused expert epilogues never materialize the (tokens*top_k, hidden)
    # intermediates. Loss gap over 400 steps is 1.2x the bf16-against-bf16 floor.
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
SAMPLES_PER_BATCH = 64
NUM_WORKERS = 24
PREFETCH_FACTOR = 6

GPUS = [0, 1, 2, 3]
GRAD_ACC = 1
GRAD_CLIP = 1.0
EPOCH = 1
# 100k steps, not an epoch count: the iterative loader has no length, so the schedule
# needs an explicit total to run against -- and it is what the progress bar reports.
MAX_STEPS = 100_000
PRECISION = "bf16-mixed"
DDP_COMPRESS_HOOK = "bf16"

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
# 2000 steps at MAX_STEPS=100_000. Warmup is a fixed step count, not a fraction of
# the run; see docs/guides/training.md.
SCHED_WARMUP_RATIO = 0.02

COMPILE = None
# Fits without it at this rung: the fp8 arm peaks at 8.6 GiB per rank against a 32 GiB
# card. Checkpointing would cost ~16% throughput and, on a sparse rung, would also give
# back most of the fp8 memory saving, since that saving is an activation saving.
GRAD_CKPT = False

NAME = "TIPO-MoE-1B"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 20
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
