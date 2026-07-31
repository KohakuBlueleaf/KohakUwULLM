"""Pipeline training on kohakuwupipe, over the KohakuVault corpus.

Launched by torchrun, configured by KohakuEngine::

    torchrun --standalone --nproc_per_node=4 $(which kogine) run \
        scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_pp4.py

    torchrun --standalone --nproc_per_node=4 $(which kogine) run \
        scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_pp4.py \
        --set MAX_STEPS=10 --set DATA_KIND=synthetic

Every UPPER_CASE global is a knob a config may override; the defaults form a
runnable synthetic smoke test. ``scripts/train/lm.py`` is the Lightning path and
is unchanged. See docs/internals/pipeline.md and docs/guides/writing-configs.md.
"""

import torch

from kohakuwullm.bench import make_info, make_tokens
from kohakuwullm.data import build_loader
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.autotune import broadcast_costs, measure_costs
from kohakuwullm.training.parallel.uwupipe import (
    LMPipelineModule,
    PipelineStep,
    corpus_steps,
    plan_for,
    with_router_losses,
)
from kohakuwullm.training.parallel.uwupipe_callbacks import SamplePreview
from kohakuwullm.utils import autofill_schedule_steps
from kohakuwupipe import (
    Checkpoint,
    LossLog,
    PipelineGradScaler,
    PipelineTrainer,
    Throughput,
    describe,
    get_logger,
    init_pipeline,
    shutdown,
)

torch.set_float32_matmul_precision("high")
log = get_logger("lm_pipe")

# ===================================================================== #
# Data. "corpus" reads KohakuVault; "synthetic" is the benchmark stream.
# ===================================================================== #
DATA_KIND = "synthetic"
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
DATA_ROOT = "/xg7/caption-datasets"
SOURCES = [{"name": "danbooru", "repeat": 1}]
RENDERER = "tipo"
LOADER_KWARGS: dict = {}
SYNTHETIC_LEN = (50, 600)

# ===================================================================== #
# Model
# ===================================================================== #
PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES: dict = {
    "tie_embeddings": False,
    "max_position": 4096,
    "qk_norm": True,
}
HEAD_KWARGS: dict = {}
# Both ride a second boundary stream. See docs/internals/moe-router-loss.md.
AUX_LOSS_WEIGHT = 0.0
ROUTER_Z_LOSS_WEIGHT = 0.0

# ===================================================================== #
# Pipeline. LAYERS pins the split; None lets the cost model choose.
# ===================================================================== #
MICRO_TOKENS = 16384
NUM_MICROBATCHES = 16
LAYERS: list = []
# Measure per-layer cost at startup instead of predicting it. Ignored when
# LAYERS pins the split. See docs/internals/pipeline.md.
AUTOTUNE = True
AUTOTUNE_DOC_LEN = 325
SCHEDULE = "1f1b"
PARAM_DTYPE = "bf16"
AUTOCAST_DTYPE = "bf16"
MXFP8 = False
MXFP8_MOE = "fused"
COMPILE: dict | None = None

# ===================================================================== #
# Optimizer / schedule
# ===================================================================== #
OPTIMIZER = "muon"
OPTIMIZER_KWARGS: dict = {"muon_lr": 2e-3, "embed_lr": 2e-3}
LR = 5e-4
GRAD_CLIP = 1.0
# fp16 gradients underflow without it. Auto: on for fp16 params or autocast.
# See docs/kohakuwupipe/scaler.md.
GRAD_SCALER = "auto"
GRAD_SCALER_INIT = 65536.0
SCHEDULER_CONFIG: dict = {"lr": {"mode": "cosine", "min_value": 0.1, "end": -1}}
SCHED_WARMUP_RATIO = 0.02
MAX_STEPS = 32
SEED = 20090220

# ===================================================================== #
# Resume / logging
# ===================================================================== #
CHECKPOINT_PATH: str | None = None
CKPT_DIR = ""
CKPT_INTERVAL = 5000
NAME = "lm-pipe"
WANDB_PROJECT = ""
WANDB_OFFLINE = True
LOG_INTERVAL = 10
THROUGHPUT_INTERVAL = 50
SAMPLE_INTERVAL = 0
SAMPLE_PROMPTS: list | None = None


def synthetic_stream(vocab, micro, num_micro, lo, hi, device, seed=0, fresh=False):
    """Ragged packed microbatches, the shape the real loader emits.

    ``fresh=False`` reuses one batch, which is what the e2e benchmark times:
    rebuilding costs Python packing plus an ``.item()`` sync per step, and that
    is loader cost, not step cost. See docs/internals/pipeline.md.
    """
    step = 0
    cached = None
    while True:
        if cached is None or fresh:
            infos = [
                make_info(micro, lo, hi, device, seed=seed + step * num_micro + i)
                for i in range(num_micro)
            ]
            ids, labels = make_tokens(infos, vocab, device, seed=seed + step)
            cached = PipelineStep(
                ids, labels, infos, int((labels != -100).sum().item())
            )
        yield cached
        step += 1


def build_stream(ranks, tokenizer):
    """``(steps, loader)`` for the configured data source; loader is None if synthetic."""
    if DATA_KIND == "synthetic":
        stream = synthetic_stream(
            VOCAB_SIZE,
            MICRO_TOKENS,
            NUM_MICROBATCHES,
            *SYNTHETIC_LEN,
            ranks.device,
            seed=SEED,
        )
        return stream, None
    if DATA_KIND != "corpus":
        raise ValueError(
            f"DATA_KIND must be 'corpus' or 'synthetic', got {DATA_KIND!r}"
        )
    # rank/world_size describe the *data-parallel* group; a pure pipeline has
    # none, so every rank iterates the identical stream.
    loader = build_loader(
        "pipeline",
        SOURCES,
        tokenizer,
        renderer=RENDERER,
        root=DATA_ROOT,
        **{
            "k": MICRO_TOKENS,
            "num_microbatches": NUM_MICROBATCHES,
            "seed": SEED,
            "rank": 0,
            "world_size": 1,
            **LOADER_KWARGS,
        },
    )
    return corpus_steps(loader, ranks.device), loader


def build_reporter(name: str):
    """A metrics sink for the interval callbacks: the log, plus W&B on rank 0."""
    run = None
    if WANDB_PROJECT:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT,
            name=name,
            mode="offline" if WANDB_OFFLINE else "online",
        )

    def report(row: dict) -> None:
        log.info("metrics", **row)
        if run is not None:
            run.log({k: v for k, v in row.items() if k != "step"}, step=row["step"])

    return report


def main() -> None:
    ranks = init_pipeline()
    torch.manual_seed(SEED)
    tokenizer = None
    if DATA_KIND == "corpus" or SAMPLE_INTERVAL > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    config = get_preset(PRESET, vocab_size=VOCAB_SIZE, **ARCH_OVERRIDES)
    config = with_router_losses(config, AUX_LOSS_WEIGHT, ROUTER_Z_LOSS_WEIGHT)
    costs = None
    if AUTOTUNE and not LAYERS:
        costs = broadcast_costs(
            measure_costs(
                config,
                ranks.device,
                MICRO_TOKENS,
                doc_len=AUTOTUNE_DOC_LEN,
                param_dtype=PARAM_DTYPE,
                autocast_dtype=AUTOCAST_DTYPE,
                mxfp8=MXFP8,
            ),
            ranks.device,
        )
        if ranks.rank == 0:
            log.info(costs.summary())
    plans = plan_for(
        config,
        ranks.world,
        seq_len=MICRO_TOKENS,
        layers=LAYERS or None,
        costs=costs,
    )
    if ranks.rank == 0:
        log.info("stage split\n%s", describe(plans))

    stream, loader = build_stream(ranks, tokenizer)
    schedule_config = {
        key: autofill_schedule_steps(dict(sub), MAX_STEPS, SCHED_WARMUP_RATIO)
        for key, sub in SCHEDULER_CONFIG.items()
    }
    module = LMPipelineModule(
        config,
        micro_tokens=MICRO_TOKENS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        learning_rate=LR,
        param_dtype=PARAM_DTYPE,
        autocast_dtype=AUTOCAST_DTYPE,
        head_kwargs=HEAD_KWARGS or None,
        mxfp8=MXFP8,
        mxfp8_moe=MXFP8_MOE,
        compile_spec=COMPILE,
        scheduler_config=schedule_config,
        loader=loader,
    )

    report = build_reporter(NAME) if ranks.rank == 0 else None
    callbacks = [
        Throughput(every_n_steps=THROUGHPUT_INTERVAL, warmup_steps=8, report=report),
        LossLog(every_n_steps=LOG_INTERVAL, report=report),
    ]
    if CKPT_DIR:
        callbacks.append(
            Checkpoint(
                CKPT_DIR,
                plans[ranks.rank].start_layer,
                CKPT_INTERVAL,
                block_attr=module.block_prefix,
            )
        )
    if SAMPLE_INTERVAL > 0:
        callbacks.append(
            SamplePreview(
                module,
                tokenizer,
                ranks,
                prompts=SAMPLE_PROMPTS,
                every_n_steps=SAMPLE_INTERVAL,
            )
        )
    scaling = (
        "fp16" in (PARAM_DTYPE, AUTOCAST_DTYPE)
        if GRAD_SCALER == "auto"
        else bool(GRAD_SCALER)
    )
    if ranks.rank == 0 and scaling:
        log.info("loss scaling on", init_scale=GRAD_SCALER_INIT)
    trainer = PipelineTrainer(
        module,
        ranks,
        plans,
        micro_tokens=MICRO_TOKENS,
        num_microbatches=NUM_MICROBATCHES,
        schedule=SCHEDULE,
        grad_clip=GRAD_CLIP,
        scaler=PipelineGradScaler(enabled=scaling, init_scale=GRAD_SCALER_INIT),
        callbacks=callbacks,
    )
    if CHECKPOINT_PATH:
        trainer.load_checkpoint(CHECKPOINT_PATH)

    trainer.fit(stream, max_steps=MAX_STEPS)
    if CKPT_DIR:
        trainer.save_checkpoint(f"{CKPT_DIR}/last.ckpt")
    log.info(
        "done",
        steps=trainer.loop.global_step,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    shutdown()


if __name__ == "__main__":
    main()
