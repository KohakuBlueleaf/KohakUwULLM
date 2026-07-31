"""Train a decoder-only LM on packed variable-length caption/tag data.

Run with KohakuEngine, pointing at a config that overrides the globals below::

    kogine run scripts/train/lm.py --config configs/lm/debug.py
    kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py --set LR=2e-4

Every UPPER_CASE global here is a knob a config may override. The defaults form
a runnable debug recipe (tiny model, danbooru only, one GPU) so the script works
out of the box. See ``docs/guides/writing-configs.md`` for the full catalogue.
"""

import os
import warnings

if os.environ.get("LOCAL_RANK", "0") != "0":
    warnings.filterwarnings("ignore", ".*")

import lightning.pytorch as pl
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_hooks

from kohakuwullm.data import (
    build_dataset,
    build_loader,
    collate_packed,
    collate_padded,
)
from kohakuwullm.training import (
    LMTrainer,
    PipelinedLMTrainer,
    PipelineOnlyStrategy,
    SampleLogCallback,
    StepProgressBar,
    ThroughputCallback,
)
from kohakuwullm.utils import autofill_schedule_steps

torch.set_float32_matmul_precision("high")

# ===================================================================== #
# Resume / checkpointing
# ===================================================================== #
CHECKPOINT_PATH = None  # str | None -- .ckpt to load weights from
TRAINER_RESUME = False  # also resume optimizer / scheduler / global_step
# Applies only on a TRAINER_RESUME: restore the RNG stream and the dataloader
# position as well, so the continuation trains on data it has not already seen.
# Turn off to resume the model and the schedule while re-drawing the data order.
RESUME_STATE = True

# ===================================================================== #
# Tokenizer
# ===================================================================== #
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536

# ===================================================================== #
# Dataset
# ===================================================================== #
DATA_ROOT = "/xg7/caption-datasets"
# Each entry: {"name": <source>, "repeat": N, **source kwargs}. A repeat is a
# re-render, not a duplicate -- the renderer draws a fresh task/length/tag split.
SOURCES = [{"name": "danbooru", "repeat": 1}]
RENDERER = "tipo"
MAX_LENGTH = 2048  # truncate one rendered sample here
SAMPLES_PER_BATCH = 32  # documents packed into one step
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PAD_TO_MULTIPLE = 0  # round packed length up; helps torch.compile stop re-specializing
# "packed" (varlen, the fast default -- no padding is computed on) or "padded"
# ((B, S), for reproducing a recipe specified in sequences x length, for
# ablating against kernels with no varlen path, or for readable debugging).
# Both produce identical loss per token; they differ only in wasted compute.
BATCH_LAYOUT = "packed"
PAD_TOKEN_ID = 0
VAL_FRACTION = 0.0  # hold out this fraction of the dataset for validation

# ===================================================================== #
# Model -- preset + per-field overrides
# ===================================================================== #
PRESET = "Nano-25M"
ARCH_OVERRIDES = {}  # PRESET=None -> ARCH_OVERRIDES is the whole LMArchConfig
# LMHead kwargs the arch config does not cover: {"kernel": "torch"|"chunked_ce",
# "chunk": ..., "vocab_block": ..., "retain": ..., "label_smoothing": ...}.
HEAD_KWARGS: dict = {}

# ===================================================================== #
# Compute / batch / steps
# ===================================================================== #
GPUS = 1  # int count or list of device ids
GRAD_ACC = 1  # intra-batch split; 1 dataloader batch == 1 optimizer step

# "torch" is the map-style DataLoader below; "map" is the fast map-style loader;
# "iterative" packs to a token budget in the worker and yields whole PackedBatch
# objects; "ddp" is that plus a per-rank shard and a resumable position;
# "pipeline" emits fixed-size microbatch steps for a pipeline schedule.
# Registered in kohakuwullm.registry.LOADER.
LOADER_KIND = "torch"
LOADER_KWARGS: dict = {}
GRAD_CLIP = 1.0
EPOCH = 1
MAX_STEPS = -1  # -1 -> run the full epoch count
SEED = 20090220
PRECISION = "bf16-mixed"  # "32-true" | "bf16-mixed" | "16-mixed" | "bf16-true"
DDP_COMPRESS_HOOK = "bf16"  # "fp16" | "bf16" | None

# "ddp" replicates the model per card; "pipeline" splits it into one stage per
# card. Pipeline needs LOADER_KIND="pipeline" and owns its own precision.
# See docs/internals/pipeline.md.
PARALLEL = "ddp"
PIPELINE_KWARGS: dict = {
    "micro_tokens": 8192,
    "num_microbatches": 32,
    "schedule": "1f1b",
    "param_dtype": "bf16",
    "autocast_dtype": "bf16",
}

# ===================================================================== #
# Optimizer
# ===================================================================== #
OPTIMIZER = "adamw"
OPTIMIZER_KWARGS = {}
LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
EPS = 1e-8
USE_MUP = False
BASE_DIM = 256

# ===================================================================== #
# Scheduler (AnySchedule)
# ===================================================================== #
SCHEDULER_CONFIG = {"lr": {"mode": "cosine", "min_value": 0.1, "end": -1}}
SCHED_WARMUP_RATIO = 0.01

# ===================================================================== #
# Runtime
# ===================================================================== #
COMPILE = None  # {"mode": "module", "dynamic": True} | {"mode": "model"} | None
GRAD_CKPT = False

# ===================================================================== #
# Logging / checkpoints
# ===================================================================== #
WANDB_PROJECT = "KohakUwULLM"
WANDB_OFFLINE = True
NAME = "lm-debug"
LOG_INTERVAL = 10
CKPT_INTERVAL = 5000
SAMPLE_INTERVAL = 1000  # generate preview completions every N steps (0 disables)
SAMPLE_PROMPTS = None
THROUGHPUT_INTERVAL = 50


def _check_pipeline_config(gpu_count: int) -> None:
    """Reject a pipeline config that would fail late or train something else.

    See docs/internals/pipeline.md.
    """
    problems = []
    if gpu_count < 2:
        problems.append(f"PARALLEL='pipeline' needs >1 GPU, got {gpu_count}")
    if LOADER_KIND != "pipeline":
        problems.append(
            f"PARALLEL='pipeline' needs LOADER_KIND='pipeline', got {LOADER_KIND!r}: "
            "every rank must see identical fixed-size microbatch steps"
        )
    if PRECISION != "32-true":
        problems.append(
            f"PARALLEL='pipeline' needs PRECISION='32-true', got {PRECISION!r}: "
            "the stage supplies its own autocast via PIPELINE_KWARGS"
        )
    if GRAD_ACC != 1:
        problems.append(
            f"PARALLEL='pipeline' needs GRAD_ACC=1, got {GRAD_ACC}: "
            "PIPELINE_KWARGS['num_microbatches'] is the accumulation"
        )
    # The loader's k IS the microbatch, and the boundary shape is frozen from it.
    for loader_key, stage_key in (("k", "micro_tokens"), ("num_microbatches",) * 2):
        loader_value = LOADER_KWARGS.get(loader_key)
        stage_value = PIPELINE_KWARGS.get(stage_key)
        if loader_value is not None and loader_value != stage_value:
            problems.append(
                f"LOADER_KWARGS[{loader_key!r}]={loader_value} != "
                f"PIPELINE_KWARGS[{stage_key!r}]={stage_value}"
            )
    if problems:
        raise SystemExit("[pipeline config]\n  " + "\n  ".join(problems))


def _build_loaders(tokenizer):
    if LOADER_KIND != "torch":
        # The iterative loader packs to a token budget itself, so it yields whole
        # PackedBatch objects and neither collate nor batch_size applies.
        train_loader = build_loader(
            LOADER_KIND,
            SOURCES,
            tokenizer,
            renderer=RENDERER,
            root=DATA_ROOT,
            **LOADER_KWARGS,
        )
        return train_loader, None

    dataset = build_dataset(
        SOURCES,
        tokenizer,
        renderer=RENDERER,
        root=DATA_ROOT,
        max_length=MAX_LENGTH,
        seed=SEED,
    )
    val_dataset = None
    if VAL_FRACTION > 0:
        n_val = max(1, int(len(dataset) * VAL_FRACTION))
        generator = torch.Generator().manual_seed(SEED)
        dataset, val_dataset = data.random_split(
            dataset, [len(dataset) - n_val, n_val], generator=generator
        )

    if BATCH_LAYOUT == "padded":

        def collate(samples):
            return collate_padded(
                samples,
                pad_token_id=PAD_TOKEN_ID,
                max_length=MAX_LENGTH,
                pad_to_multiple=max(PAD_TO_MULTIPLE, 8),
            )

    elif BATCH_LAYOUT == "packed":

        def collate(samples):
            return collate_packed(samples, pad_to_multiple=PAD_TO_MULTIPLE)

    else:
        raise ValueError(
            f"BATCH_LAYOUT must be 'packed' or 'padded', got {BATCH_LAYOUT!r}"
        )

    loader_kwargs = dict(
        batch_size=SAMPLES_PER_BATCH,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = data.DataLoader(dataset, shuffle=True, **loader_kwargs)
    val_loader = (
        data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def _steps_per_epoch(train_loader, gpu_count: int) -> int:
    """Optimizer steps in one epoch, for the schedule's total.

    GRAD_ACC is an intra-batch split (1 dataloader batch == 1 optimizer step),
    so the count is the loader's length, not that over GRAD_ACC.

    A token-budget loader shards per rank itself, so its length is already
    per-rank and must not be divided again; and it has no length at all until
    ``batches_per_epoch`` pins its data-dependent batch count.
    """
    per_rank = getattr(train_loader, "steps_per_epoch", None)
    if per_rank is not None:
        return max(1, per_rank)
    try:
        return max(1, len(train_loader) // gpu_count)
    except TypeError:
        if MAX_STEPS <= 0:
            raise ValueError(
                f"LOADER_KIND={LOADER_KIND!r} yields a data-dependent number of "
                "batches, so the schedule has no total to run against: set "
                "MAX_STEPS, or pin batches_per_epoch in LOADER_KWARGS"
            ) from None
        return MAX_STEPS


def main():
    from transformers import AutoTokenizer

    pl.seed_everything(SEED)
    gpu_count = len(GPUS) if isinstance(GPUS, list) else GPUS

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    train_loader, val_loader = _build_loaders(tokenizer)

    steps_per_epoch = _steps_per_epoch(train_loader, gpu_count)
    train_steps = steps_per_epoch * EPOCH if MAX_STEPS < 0 else MAX_STEPS
    autofill_schedule_steps(
        SCHEDULER_CONFIG["lr"], train_steps, warmup_ratio=SCHED_WARMUP_RATIO
    )

    preset = None if PRESET in (None, "None") else PRESET
    trainer_kwargs = dict(
        preset=preset,
        arch_overrides=ARCH_OVERRIDES,
        vocab_size=VOCAB_SIZE,
        head_kwargs=HEAD_KWARGS,
        optimizer=OPTIMIZER,
        optimizer_kwargs=OPTIMIZER_KWARGS,
        learning_rate=LR,
        betas=tuple(BETAS),
        weight_decay=WEIGHT_DECAY,
        eps=EPS,
        use_mup=USE_MUP,
        base_dim=BASE_DIM,
        scheduler_config=SCHEDULER_CONFIG,
        gradient_accumulation_steps=GRAD_ACC,
        gradient_clip=GRAD_CLIP,
        compile_config=COMPILE,
        grad_checkpointing=GRAD_CKPT,
        resume_state=RESUME_STATE,
        name=NAME,
        log_interval=LOG_INTERVAL,
    )
    pipelined = PARALLEL == "pipeline"
    if pipelined:
        _check_pipeline_config(gpu_count)
        # `seq_len` is the planner's attention term, and the benchmark plans at
        # the microbatch size. See docs/internals/pipeline.md.
        trainer_kwargs.update(PIPELINE_KWARGS, seq_len=PIPELINE_KWARGS["micro_tokens"])
    trainer_cls = PipelinedLMTrainer if pipelined else LMTrainer

    if CHECKPOINT_PATH is not None and not TRAINER_RESUME:
        model = trainer_cls.load_from_checkpoint(
            CHECKPOINT_PATH,
            map_location="cpu",
            strict=False,
            weights_only=False,
            **trainer_kwargs,
        )
    else:
        model = trainer_cls(**trainer_kwargs)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(every_n_train_steps=CKPT_INTERVAL, save_last=True),
        ThroughputCallback(every_n_steps=THROUGHPUT_INTERVAL),
        # `train_steps`, not MAX_STEPS: the target is the resolved one the LR schedule
        # was already filled from, so the bar and the schedule cannot disagree about
        # when the run ends.
        StepProgressBar(total_steps=train_steps),
    ]
    if SAMPLE_INTERVAL > 0:
        callbacks.append(
            SampleLogCallback(
                tokenizer, prompts=SAMPLE_PROMPTS, every_n_steps=SAMPLE_INTERVAL
            )
        )

    if pipelined:
        strategy = PipelineOnlyStrategy(process_group_backend="nccl")
    elif gpu_count > 1:
        comm_hook = {
            "fp16": ddp_hooks.fp16_compress_hook,
            "bf16": ddp_hooks.bf16_compress_hook,
            None: None,
        }[DDP_COMPRESS_HOOK]
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            ddp_comm_hook=comm_hook,
            # MoE routes different tokens to different experts each step, so an
            # expert can receive no tokens in a given backward.
            find_unused_parameters=bool(ARCH_OVERRIDES.get("moe_every"))
            or (preset or "").startswith("MoE"),
        )
    else:
        strategy = "auto"

    logger = WandbLogger(project=WANDB_PROJECT, name=NAME, offline=WANDB_OFFLINE)
    trainer = pl.Trainer(
        max_epochs=EPOCH,
        # The resolved target, so an iterable loader with no epoch length still stops
        # where the schedule says it should rather than running until the data does.
        max_steps=train_steps,
        accelerator="auto",
        devices=GPUS,
        strategy=strategy,
        precision=PRECISION,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=LOG_INTERVAL,
    )
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=CHECKPOINT_PATH if TRAINER_RESUME else None,
    )


if __name__ == "__main__":
    main()
