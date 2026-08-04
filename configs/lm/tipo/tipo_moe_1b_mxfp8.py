"""TIPO on Kohaku-MoE-1B (~200M active) with MXFP8 linears *and* MXFP8 attention.

The 100k-step production rung. Identical to
``tipo_moe_1b_uwupipe.py`` except that ``attn`` selects the training-aware
block-scaled e4m3 attention kernel, so both GEMM families run in MXFP8::

    kogine run scripts/train/lm_pipe.py \
        --config configs/lm/tipo/tipo_moe_1b_mxfp8.py

Ten steps, no network, one checkpoint at the end::

    kogine run scripts/train/lm_pipe.py \
        --config configs/lm/tipo/tipo_moe_1b_mxfp8.py \
        --set MAX_STEPS=10 --set CKPT_INTERVAL=10 --set SAMPLE_INTERVAL=0

See docs/internals/mxfp8-attention.md for what the attention kernel costs in
accuracy, and docs/internals/pipeline.md for the measured split.
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
    # QK^T in block-scaled e4m3; PV and both gradient GEMMs stay 16-bit.
    # head_dim 64 is a whole number of 32-element scale blocks.
    "attn": "mxfp8",
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
# No COMPILE here: reduce-overhead captures CUDA graphs, and a 1F1B
# schedule holds microbatch N's activations for backward while N+1 runs
# forward, which graph capture forbids. Measured: rank 1 dies with
# "accessing tensor output of CUDAGraphs that has been overwritten".
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

CKPT_DIR = "out/ckpt/tipo-moe-1b-mxfp8"
CKPT_INTERVAL = 2000
NAME = "TIPO-MoE-1B-mxfp8"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 1
THROUGHPUT_INTERVAL = 1
CONSOLE_INTERVAL = 200
PROGRESS_BAR = True
SAMPLE_INTERVAL = 2000
SAMPLE_COUNT = 8
# Each row still ends at its own EOS; this only bounds a row that never emits one.
SAMPLE_TOKENS = 512
SAMPLE_TEMPERATURE = 1.0
SAMPLE_MIN_P = 0.1
# kwargs for tipo.build_prompt: the format the renderer trains on, not a bare
# tag string. See docs/internals/data.md.
SAMPLE_PROMPTS = [
    {
        "name": "tags-short",
        "tags": "1girl",
        "meta": {"quality": "masterpiece", "rating": "general"},
        "target_len": "short",
    },
    {
        "name": "tags-long",
        "tags": "1girl, solo, long_hair, school_uniform",
        "meta": {"quality": "masterpiece", "rating": "general", "aspect ratio": "0.7"},
        "target_len": "long",
    },
    {
        "name": "scenery",
        "tags": "scenery, no_humans",
        "meta": {"quality": "best quality", "rating": "general"},
        "target_len": "long",
    },
    {
        "name": "tag-to-long",
        "tags": "1girl, cherry_blossoms, outdoors",
        "meta": {"quality": "masterpiece", "rating": "general"},
        "task": "tag_to_long",
        "target_len": "long",
    },
]
