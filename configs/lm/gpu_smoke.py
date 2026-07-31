"""Short real run that exercises what CPU tests cannot.

Every path here is one a CPU test skips: pinned-memory transfer of a packed
step, the sampling generator under autocast, the FLOP model against a real
device, and a checkpoint/resume through an actual Lightning cycle. Small enough
to finish in a couple of minutes, and deliberately not a benchmark -- the point
is that the machinery runs, not how fast.
"""

PRESET = "Nano-25M"
GPUS = 1
SEED = 20090220
PRECISION = "bf16-mixed"

LOADER_KIND = "ddp"
LOADER_KWARGS = {
    "k": 4096,
    "m": 256,
    "ctx_max": 2048,
    "num_workers": 2,
    "batches_per_epoch": 24,
}

# The Triton head and grouped GEMM are the default; this run wants them on,
# since being on a GPU is the whole reason for it.
HEAD_KWARGS = {"kernel": "chunked_ce"}

OPTIMIZER = "muon"
OPTIMIZER_KWARGS = {"muon_lr": 0.02}
LR = 3e-4
GRAD_CLIP = 1.0

MAX_STEPS = 12
EPOCH = 1
LOG_INTERVAL = 2
SAMPLE_INTERVAL = 6
THROUGHPUT_INTERVAL = 4
CKPT_INTERVAL = 6

WANDB_OFFLINE = True
NAME = "gpu-smoke"
