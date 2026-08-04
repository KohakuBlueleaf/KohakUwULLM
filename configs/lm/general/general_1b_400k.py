"""General-language + TIPO pretrain on Kohaku-MoE-1B: 400k steps, ~104.6B tokens.

Run::

    kogine run scripts/train/lm_pipe.py --config configs/lm/general/general_1b_400k.py

Smoke, ten steps, no network::

    kogine run scripts/train/lm_pipe.py --config configs/lm/general/general_1b_400k.py \\
        --set MAX_STEPS=10 --set CKPT_INTERVAL=0 --set SAMPLE_INTERVAL=0

General weights are derived by scripts/data/mixture.py at ctx_max 4096; TIPO is
one pass over each of its three sources, 5.17B, and the general budget is the
99.69B that leaves. The DQA, STEM and SmolTalk sources render as ChatML with
every user turn masked, so 9.4% of the run carries turn structure -- see
internal/chat-template-design.md and internal/general-pretrain-datasets.md.
"""

DATA_KIND = "corpus"
DATA_ROOT = "/Iolite/text-dataset/_vault"
# Per-source: a spec's own "renderer" wins, everything else falls back to plain.
RENDERER = "per_source"
# The TIPO vault does not live beside the text corpus.
TIPO_ROOT = "/xg7/caption-datasets"

# TIPO is a continuation format, not a turn structure, and its renderer trains on
# the prompt half by design. See internal/chat-template-design.md.
TIPO_RENDER = "tipo"

# 400k steps x 262144 tokens = 104.86B; one pass over this list delivers 104.88B.
SOURCES = [
    {"name": "en/nemotron-cc-v2.1-hq", "repeat": 1.000},
    {"name": "en/nemotron-cc-v2.1-hq-syn", "repeat": 0.270},
    {"name": "en/nemotron-cc-v2.1-mhq", "repeat": 1.000},
    {"name": "ja/fineweb-2-edu-japanese-10bt", "repeat": 0.560},
    {"name": "zh-tw/finepdfs-zh-zhtw", "repeat": 0.576},
    {"name": "zh-tw/ultra-fineweb-l3", "repeat": 1.000},
    {"name": "zh-tw/wiki-zhtw", "repeat": 1.000},
    {"name": "zh-tw/ptt-zhtw", "repeat": 1.000},
    {"name": "en/nemotron-cc-v2.1-hq-dqa", "repeat": 0.966, "renderer": "qa_chatml"},
    {"name": "ko/hplt3-kor-hang", "repeat": 0.130},
    {"name": "stem/nemotron-specialized", "repeat": 1.000, "renderer": "qa_chatml"},
    # JSON conversation rows, not document text. See scripts/data/to_vault_chat.py.
    {"name": "sft/smoltalk", "repeat": 1.000, "renderer": "chat", "schema": "json"},
    {"name": "danbooru", "repeat": 1, "renderer": TIPO_RENDER, "root": TIPO_ROOT},
    {"name": "coyo11m", "repeat": 1, "renderer": TIPO_RENDER, "root": TIPO_ROOT},
    {"name": "laion_coco", "repeat": 1, "renderer": TIPO_RENDER, "root": TIPO_ROOT},
]

LOADER_KWARGS = {
    # Skipped documents cost their read and nothing else: the packer draws the
    # next one to fill the same token budget.
    "doc_filter": "ngram",
    "val_frac": 0.002,
    "split": "train",
    # The model's trained context. Truncating below it throws away real length.
    "ctx_max": 4096,
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
    # fp16 throughout: the fused 16-bit expert path beats MXFP8 end to end and
    # carries no quantization error. See docs/internals/fused-moe-16bit.md.
    "mxfp8": False,
}
# Aux-loss-free balancing carries the load; no head z-loss, which costs 1.59x
# end to end. See docs/internals/moe-router-loss.md.
AUX_LOSS_WEIGHT = 0.0
ROUTER_Z_LOSS_WEIGHT = 0.0

# 262144 tokens/step. Measured over 60 steps on 4x5090: 276.0k tok/s at 9.17 GiB
# of 31.4, against 263.3k for 16384 x 16 and 184.0k for 4096 x 64.
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
# Warmup is a fixed step count, not a fraction of the run; the 0.02 default would
# be 8000 here. See docs/guides/training.md.
SCHED_WARMUP_STEPS = 2000

MAX_STEPS = 400_000
SEED = 20090220

CKPT_DIR = "out/ckpt/general-1b-400k"
CKPT_INTERVAL = 2000
NAME = "KohakUwU-1B-A200M-general400k"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False

# Previews are cut from the batch being trained on, at a turn boundary read out
# of its own loss mask: the prompt ends at <|im_start|>assistant and the
# reference is that one reply.
SAMPLE_INTERVAL = 2000
SAMPLE_COUNT = 2
SAMPLE_FROM_BATCH = 16
SAMPLE_TURNS_ONLY = True
SAMPLE_PREFIX_TOKENS = 1024
SAMPLE_TOKENS = 256
SAMPLE_TEMPERATURE = 1.0
SAMPLE_MIN_P = 0.1
SAMPLE_TOP_P = 1.0
SAMPLE_TOP_K = 0
SAMPLE_LOCAL = False
SAMPLE_FORWARD_ONLY = True
