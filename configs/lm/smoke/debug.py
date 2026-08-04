"""Smoke test: tiny model, one GPU, 50 steps. Should finish in about a minute.

Run:
    kogine run scripts/train/lm.py --config configs/lm/smoke/debug.py
"""

PRESET = "Nano-25M"
SOURCES = [{"name": "danbooru", "repeat": 1}]
MAX_LENGTH = 512
SAMPLES_PER_BATCH = 16
NUM_WORKERS = 4

GPUS = 1
EPOCH = 1
MAX_STEPS = 50
PRECISION = "bf16-mixed"

LR = 3e-4
SCHED_WARMUP_RATIO = 0.1

NAME = "lm-debug"
WANDB_OFFLINE = True
LOG_INTERVAL = 5
CKPT_INTERVAL = 1000
SAMPLE_INTERVAL = 25
THROUGHPUT_INTERVAL = 10
