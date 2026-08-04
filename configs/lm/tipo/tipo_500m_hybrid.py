"""SUPERSEDED as a recipe; kept as the worked example of the per-layer attention split.

It sits on the retired ``Nano-500M`` with tied embeddings, and it is the one config
that exercises ``attn_sliding`` -- which is why it is kept rather than deleted. To run
the ablation it describes, move both arms to ``Kohaku-500M``: the window reasoning
below is about the 2048 context, not about the preset, so it transfers unchanged.

TIPO-500M, fully modern: GQA + mixed sliding-window / global attention.

The architecture exploration for the 500M scale. Three choices carry the design,
all measured rather than assumed (see docs/performance/performance.md):

**GQA at 20 query heads / 4 kv heads.** The attention benchmark shows kv_heads=4
runs at essentially the same throughput as kv_heads=1 while keeping 4x the KV
capacity, and both beat full MHA (kv=20) by ~25% because the K/V projections
shrink. 4 is the point where the throughput curve has already flattened, so
going lower only costs quality.

**3:1 local:global interleave, window 1024.** Every 4th layer is global (and the
last layer always is, since it feeds the head). At the 2048 context this corpus
uses, a 1024 window covers the majority of any single rendered sample while the
global layers carry cross-document-scale reach. OLMo 3 uses 3:1, Gemma 3 uses
5:1; 3:1 is the conservative end for a model this small, where there are only 20
layers and a 5:1 pattern would leave just 4 global ones.

**A different attention kernel per layer type.** ``attn_sliding="triton"`` sends
the local layers to our own kernel, which skips out-of-window key blocks more
aggressively than the vendor kernel and measures up to 20% faster on the forward
at windows 1k-4k. The global layers stay on ``varlen``, where the vendor
kernel's fused backward wins. This is the whole reason the per-layer override
exists.

Run:
    kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m_hybrid.py

To ablate against the all-global baseline:
    kogine run scripts/train/lm.py --config configs/lm/tipo/tipo_500m.py
"""

PRESET = "Nano-500M"
ARCH_OVERRIDES = {
    # -- shape: 1280 x 20 layers, 20 heads x 64, GQA 5:1
    "kv_heads": 4,
    "head_dim": 64,
    "max_position": 4096,
    # -- mixed local/global attention
    "sliding_window": 1024,
    "global_layer_every": 4,  # layers 3, 7, 11, 15, 19 are global
    "attn": "varlen",
    "attn_sliding": "triton",
    # -- stability: QK-norm replaced logit soft-capping in every model that
    #    tried both. No head z-loss: it costs 4.3x the head and DeepSeek-V3
    #    trained 14.8T tokens without one. See docs/internals/moe-router-loss.md.
    "qk_norm": True,
    "qk_norm_affine": True,
    # -- a high theta is cheap insurance for later context extension; at 2048
    #    training length it changes nothing, and it means a YaRN/NTK extension
    #    later starts from a base that was not already saturated.
    "rope_theta": 100000.0,
    "tie_embeddings": True,
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
NUM_WORKERS = 16
PREFETCH_FACTOR = 6

GPUS = [0, 1, 2, 3]
GRAD_ACC = 1
GRAD_CLIP = 1.0
EPOCH = 1
PRECISION = "bf16-mixed"
DDP_COMPRESS_HOOK = "bf16"

LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO = 0.01

COMPILE = None
GRAD_CKPT = False

NAME = "TIPO-500M-hybrid"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 20
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
