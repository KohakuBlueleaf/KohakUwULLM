"""Kohaku-MoE-1B, short run, with a pipelined preview. Every knob is a value.

Training throughput and the pipelined generation path in one run; the script
spawns its own ranks::

    kogine run scripts/train/lm_pipe.py --config configs/lm/smoke/pipegen_bench.py

See docs/guides/generation.md.
"""

DATA_KIND = "synthetic"
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
SYNTHETIC_LEN = (50, 600)

PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    "tie_embeddings": False,
}
AUX_LOSS_WEIGHT = 0.0
ROUTER_Z_LOSS_WEIGHT = 0.0

MICRO_TOKENS = 16384
NUM_MICROBATCHES = 8
LAYERS = []
AUTOTUNE = False
SCHEDULE = "1f1b"
PARAM_DTYPE = "fp16"
AUTOCAST_DTYPE = "fp16"
MXFP8 = True
COMPILE_STAGE = False

OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 2e-3, "embed_lr": 2e-3}
LR = 5e-4
GRAD_CLIP = 1.0
MAX_STEPS = 10
SEED = 20090220

CKPT_DIR = ""
CKPT_INTERVAL = 0
NAME = "pipegen-bench"
WANDB_PROJECT = ""
WANDB_OFFLINE = True
LOG_INTERVAL = 1
THROUGHPUT_INTERVAL = 1
CONSOLE_INTERVAL = 1
PROGRESS_BAR = False

SAMPLE_INTERVAL = 5
SAMPLE_COUNT = 4
SAMPLE_TOKENS = 64
SAMPLE_TEMPERATURE = 1.0
SAMPLE_MIN_P = 0.1
SAMPLE_LOCAL = False
SAMPLE_FORWARD_ONLY = True
GPUS = 4
SAMPLE_PROMPTS = [
    {
        "name": "tags-short",
        "tags": "1girl",
        "meta": {"quality": "masterpiece", "rating": "general"},
        "target_len": "short",
    },
]
