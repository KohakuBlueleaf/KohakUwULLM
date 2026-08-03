"""Pipeline training on kohakuwupipe, over the KohakuVault corpus.

Launched and configured by KohakuEngine; the script spawns its own ranks::

    kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py

    kogine run scripts/train/lm_pipe.py --config configs/lm/tipo_moe_1b_uwupipe.py \
        --set MAX_STEPS=10 --set DATA_KIND=synthetic

``GPUS`` sets the rank count; the launcher stands down when ``RANK`` is already
set, so an externally started rank group is used as-is.

Every UPPER_CASE global is a knob a config may override; the defaults form a
runnable synthetic smoke test. ``scripts/train/lm.py`` is the Lightning path and
is unchanged. See docs/internals/pipeline.md and docs/guides/writing-configs.md.
"""

import math
import os
import sys

import torch
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.bench import make_info, make_tokens
from kohakuwullm.data import build_loader
from kohakuwullm.data.renderers.tipo import build_prompt
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.autotune import broadcast_costs, measure_costs
from kohakuwullm.training.parallel.uwupipe import (
    LMPipelineModule,
    PipelineStep,
    corpus_steps,
    plan_for,
    with_router_losses,
)
from kohakuwullm.training.parallel.uwupipe_callbacks import (
    RouterBiasSchedule,
    SamplePreview,
)
from kohakuwullm.utils import autofill_schedule_steps
from kohakuwupipe import (
    Checkpoint,
    LossLog,
    PipelineGradScaler,
    PipelineTrainer,
    ProgressBar,
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
# AnySchedule config scaling the aux-loss-free balancer's step size; None holds
# it flat. See docs/internals/moe-router-loss.md.
BIAS_SCHEDULE: dict | None = None

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
# torch.compile each rank's whole stage module, in place. Default inductor mode:
# reduce-overhead captures CUDA graphs, which 1F1B forbids.
COMPILE_STAGE = False

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
# Metric cadence, in steps. 1 sends every step to W&B; the progress bar owns
# stdout, and CONSOLE_INTERVAL is how often a full line reaches the log file.
LOG_INTERVAL = 1
THROUGHPUT_INTERVAL = 1
CONSOLE_INTERVAL = 200
PROGRESS_BAR = True
SAMPLE_INTERVAL = 0
# Each is kwargs for `tipo.build_prompt`; a bare tag string is off-distribution.
SAMPLE_PROMPTS: list | None = None
# Cut previews from the batch under training instead of fixed prompts.
SAMPLE_FROM_BATCH = 0
SAMPLE_PREFIX_FRAC = 0.25
SAMPLE_COUNT = 16
# None runs every row to EOS or to the model's context limit.
SAMPLE_TOKENS: int | None = None
SAMPLE_TEMPERATURE = 0.35
SAMPLE_TOP_P = 0.95
SAMPLE_TOP_K = 0
SAMPLE_MIN_P = 0.0
# Gather the model and decode on one card; pipelined decode costs one
# pipeline traversal per token.
SAMPLE_LOCAL = True
# When SAMPLE_LOCAL is off: drive pipelined decode with a forward-only
# send/recv pass instead of a training schedule.
SAMPLE_FORWARD_ONLY = True
# Ranks to launch when the caller started none. 0 uses every GPU.
GPUS = 0


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
    # The corpus keeps growing while a run trains, so the shard view is pinned
    # once and reused on resume. See docs/internals/data.md.
    snapshot = None
    if any("/" in spec["name"] for spec in SOURCES):
        from kohakuwullm.data.sources.corpus import read_manifest, write_manifest

        path = os.path.join(CKPT_DIR, "corpus-manifest.json") if CKPT_DIR else ""
        if path and os.path.exists(path):
            snapshot = read_manifest(path)["sources"]
            log.info("corpus manifest reused", path=path, sources=len(snapshot))
        elif path and ranks.rank == 0:
            snapshot = write_manifest(SOURCES, path)["sources"]
            log.info("corpus manifest written", path=path, sources=len(snapshot))

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
            "snapshot": snapshot,
            **LOADER_KWARGS,
        },
    )
    return corpus_steps(loader, ranks.device), loader


def build_reporter(name: str):
    """A metrics sink for rank 0: W&B every call, the log on ``CONSOLE_INTERVAL``.

    The progress bar owns stdout, so the log line is what survives in the file
    and is deliberately rarer than the W&B point.
    """
    run = None
    if WANDB_PROJECT:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT,
            name=name,
            mode="offline" if WANDB_OFFLINE else "online",
        )

    def report(row: dict) -> None:
        step = int(row.get("step", 0))
        # Perplexity is the LM's own metric; kohakuwupipe reports the loss.
        if isinstance(row.get("loss"), float):
            row = dict(row, ppl=math.exp(min(row["loss"], 20.0)))
        if run is not None:
            run.log({k: v for k, v in row.items() if k != "step"}, step=step)
        if CONSOLE_INTERVAL > 0 and step % CONSOLE_INTERVAL == 0:
            log.info("metrics", **row)

    def report_samples(step: int, rows: list) -> None:
        """Generated text goes to a W&B table only; it never reaches the log."""
        if run is None:
            return
        import wandb

        table = wandb.Table(columns=["step", "name", "prompt", "sample", "text"])
        for name, prompt, index, text in rows:
            table.add_data(step, name, prompt, index, text)
        run.log({"samples": table}, step=step)

    return report, report_samples


def optional_int(value) -> int | None:
    """``--set`` hands strings for a knob whose default is ``None``."""
    if value is None or value == "" or value == "None":
        return None
    return int(value)


def build_sample_prompts() -> list[tuple[str, str]]:
    """``(name, text)`` for each entry of ``SAMPLE_PROMPTS``.

    Each entry is kwargs for :func:`kohakuwullm.data.renderers.tipo.build_prompt`,
    plus an optional ``name``. See docs/internals/data.md.
    """
    specs = SAMPLE_PROMPTS or [{"tags": "1girl"}]
    prompts = []
    for index, spec in enumerate(specs):
        spec = dict(spec)
        name = spec.pop("name", None) or f"prompt{index}"
        prompts.append((name, build_prompt(**spec)))
    return prompts


def launch() -> None:
    """Spawn ``GPUS`` ranks and run ``main`` in each when the caller started none.

    ``kogine run`` starts one process; the Lightning path gets its ranks from
    the Trainer, and this path has no Trainer. See docs/internals/pipeline.md.
    """
    if os.environ.get("RANK") is not None:
        main()
        return
    nproc = GPUS or torch.cuda.device_count()
    if nproc <= 1:
        main()
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="lm_pipe",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, _worker)(_config_globals())


def _config_globals() -> dict:
    """The resolved UPPER_CASE config, to re-apply inside each spawned rank."""
    module = sys.modules[__name__]
    return {
        key: value
        for key, value in vars(module).items()
        if key.isupper() and not key.startswith("_")
    }


def _worker(overrides: dict) -> None:
    """A spawned rank: re-apply the parent's config, then train."""
    module = sys.modules[__name__]
    for key, value in overrides.items():
        setattr(module, key, value)
    main()


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
        compile_spec=COMPILE or ({} if COMPILE_STAGE else None),
        scheduler_config=schedule_config,
        loader=loader,
    )

    report, report_samples = build_reporter(NAME) if ranks.rank == 0 else (None, None)
    callbacks = [
        Throughput(every_n_steps=THROUGHPUT_INTERVAL, warmup_steps=8, report=report),
        LossLog(every_n_steps=LOG_INTERVAL, report=report),
    ]
    if PROGRESS_BAR:
        callbacks.append(ProgressBar(postfix=("moe/load_imbalance_max", "scale")))
    if BIAS_SCHEDULE:
        callbacks.append(
            RouterBiasSchedule(
                module, autofill_schedule_steps(dict(BIAS_SCHEDULE), MAX_STEPS)
            )
        )
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
                prompts=build_sample_prompts(),
                from_batch=SAMPLE_FROM_BATCH,
                prefix_frac=SAMPLE_PREFIX_FRAC,
                every_n_steps=SAMPLE_INTERVAL,
                samples=SAMPLE_COUNT,
                report=report_samples,
                max_new_tokens=optional_int(SAMPLE_TOKENS),
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                top_k=SAMPLE_TOP_K,
                min_p=SAMPLE_MIN_P,
                local=SAMPLE_LOCAL,
                forward_only=SAMPLE_FORWARD_ONLY,
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
    launch()
