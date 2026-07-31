"""SUPERSEDED by ``tipo_moe_1b.py`` -- the sparse recipe now runs on the Kohaku ladder.

``MoE-3B-A500M`` is retired: it routes 6/64 where the ladder pins 0.094 vs 0.125, it
predates untied embeddings, and its ``moe_hidden=448`` is the exact width that makes
the preset MXFP8-ineligible (it lands as the shared expert's ``w_out.in_features``,
FPROP's contraction axis, which cannot be zero-padded). ``Kohaku-MoE-3B`` is the
replacement rung; this file is kept because benchmark JSONs on disk reference it.

MoE 3B total / ~500M active, 4x RTX 5090.

Why MoE here, and why the model is *narrower* than the dense one despite being
6x the parameters: capacity comes from expert count, not from width. A narrower
``dim`` makes both the attention and the pipeline seam cheaper, and the
parameters that grew (the experts) never cross the wire.

Memory is the reason this cannot be plain DDP. At bf16 weights plus fp32 AdamW
moments a 3B model needs ~48 GB of state, and a 5090 has 32 GB -- so the
parameters must be split, not replicated. Two ways to run it:

* ``GRAD_CKPT=True`` + this config as written: DDP with activation
  recomputation, which fits only because the *active* parameter count is ~500M
  while the optimizer state still is not. Expect to need ZeRO/FSDP sharding.
* Pipeline parallelism (``kohakuwullm.training.parallel.pipeline``), which is the
  intended path: 4 cost-balanced stages, ~1/4 of the parameters per card.
  See docs/internals/pipeline.md.

Run:
    kogine run scripts/train/lm.py --config configs/lm/moe_3b.py
"""

PRESET = "MoE-3B-A500M"
ARCH_OVERRIDES = {
    "max_position": 4096,
    "rope_theta": 100000.0,
    "qk_norm": True,
    "tie_embeddings": True,
    # Keep the first two layers dense: early layers learn generic features that
    # every expert would otherwise have to relearn independently.
    "moe_first_dense": 2,
    "moe_num_experts": 64,
    "moe_top_k": 6,
    "moe_num_shared": 1,
    "moe_router_kwargs": {
        "score_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        # Aux-loss-free balancing: the bias steers selection only, so there is
        # no auxiliary gradient competing with the language objective.
        "bias_update_rate": 1e-3,
        "aux_loss_weight": 0.0,
        "z_loss_weight": 1e-3,
    },
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
SAMPLES_PER_BATCH = 32
NUM_WORKERS = 24

GPUS = [0, 1, 2, 3]
GRAD_ACC = 2
GRAD_CLIP = 1.0
EPOCH = 1
PRECISION = "bf16-mixed"

LR = 2e-4
WEIGHT_DECAY = 0.1
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.05, "end": -1}}
SCHED_WARMUP_RATIO = 0.02

# MoE routing sends different tokens to different experts each step, so the
# expert GEMM shapes change every step -- dynamic compile is mandatory.
COMPILE = None
GRAD_CKPT = True

NAME = "TIPO-MoE-3B-A500M"
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = False
LOG_INTERVAL = 20
CKPT_INTERVAL = 2000
SAMPLE_INTERVAL = 2000
THROUGHPUT_INTERVAL = 100
