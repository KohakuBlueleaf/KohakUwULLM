"""Smoke test for the production path: small model, real data, everything switched on.

`debug.py` proves the trainer runs. This proves the *shipping configuration* runs -- MXFP8
on the projections and the routed experts, Muon, untied embeddings, a step target rather
than an epoch count, sample previews and the throughput callback -- on the real corpus,
in a couple of minutes on one card.

It exists because every defect found in the MXFP8 work was an integration defect: the
kernels were bit-exact against vendor while the surrounding code trained on stale weights,
re-benchmarked an autotune every step, or flushed gradients to zero. None of those is
visible in a unit test, and all of them are visible in fifty real steps.

Run:
    kogine run scripts/train/lm.py --config configs/lm/smoke_mxfp8.py
"""

PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES = {
    "depth": 4,
    "max_position": 2048,
    "tie_embeddings": False,
    "mxfp8": True,
}

SOURCES = [{"name": "danbooru", "repeat": 1}]
MAX_LENGTH = 1024
SAMPLES_PER_BATCH = 8
NUM_WORKERS = 4

GPUS = 1
GRAD_ACC = 2
GRAD_CLIP = 1.0
EPOCH = 1
MAX_STEPS = 40
PRECISION = "bf16-mixed"

OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 0.02}

LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO = 0.1

# Left off deliberately: compilation is exercised by the production configs, and a
# 40-step smoke spends more time compiling than training.
COMPILE = None
GRAD_CKPT = False

NAME = "smoke-mxfp8"
WANDB_OFFLINE = True
LOG_INTERVAL = 5
CKPT_INTERVAL = 100000
SAMPLE_INTERVAL = 20
THROUGHPUT_INTERVAL = 10
