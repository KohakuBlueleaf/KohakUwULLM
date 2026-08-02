"""Does a pipeline run produce a checkpoint holding every stage's weights?

    .venv/bin/python scripts/dist/pp_checkpoint_smoke.py

Each rank owns a disjoint slice of the model, so a rank-0-only save keeps only
stage 0. This trains a few steps, then counts what actually landed in the file
against the whole model. See docs/internals/pipeline.md.
"""

import faulthandler
import os

import lightning.pytorch as pl
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.data.loader.microbatch import MicroBatchedStep
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.loop.pipeline_trainer import PipelinedLMTrainer
from kohakuwullm.training.parallel.strategy import PipelineOnlyStrategy

VOCAB = 2048
MICRO = 256
N_MICRO = 2
STEPS = 4
WATCHDOG_SECONDS = 180.0
PRESET = "Nano-25M"
OVERRIDES = {"tie_embeddings": False, "max_position": 1024}
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 2


class Steps(data.Dataset):
    """Synthetic MicroBatchedStep batches; every rank sees the same stream."""

    def __len__(self) -> int:
        return STEPS

    def __getitem__(self, idx: int) -> MicroBatchedStep:
        gen = torch.Generator().manual_seed(idx)
        total = MICRO * N_MICRO
        tokens = torch.randint(0, VOCAB, (total,), generator=gen)
        return MicroBatchedStep(
            tokens=tokens,
            labels=tokens.roll(-1),
            seq_infos=[
                SeqInfo.from_lengths(torch.tensor([MICRO], dtype=torch.int32))
                for _ in range(N_MICRO)
            ],
            microbatch_tokens=MICRO,
            trained=tuple([MICRO] * N_MICRO),
        )


def build_module() -> PipelinedLMTrainer:
    return PipelinedLMTrainer(
        preset=PRESET,
        vocab_size=VOCAB,
        arch_overrides=OVERRIDES,
        micro_tokens=MICRO,
        num_microbatches=N_MICRO,
        optimizer="adamw",
        learning_rate=1e-4,
        scheduler_config={"lr": {"mode": "constant", "end": -1}},
        name="pp-ckpt",
        log_interval=1,
    )


def build_trainer(world: int, out_dir: str, max_steps: int) -> pl.Trainer:
    return pl.Trainer(
        accelerator="cuda",
        devices=world,
        strategy=PipelineOnlyStrategy(process_group_backend="nccl"),
        precision="32-true",
        max_steps=max_steps,
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(dirpath=out_dir, every_n_train_steps=STEPS, save_top_k=-1)
        ],
    )


def main() -> None:
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
    world = int(os.environ.get("WORLD_SIZE", "2"))
    rank = int(os.environ["RANK"])
    out_dir = os.environ.get("PP_CKPT_DIR", "/tmp/pp_ckpt_smoke")

    module = build_module()
    trainer = build_trainer(world, out_dir, STEPS)
    trainer.fit(module, data.DataLoader(Steps(), batch_size=None, shuffle=False))

    ckpt = os.path.join(out_dir, f"epoch=0-step={STEPS}.ckpt")
    trained = {k: v.clone() for k, v in module.backbone.global_state_dict().items()}

    # A fresh module resumes from the file: every rank must recover ITS OWN slice.
    # max_steps is already spent, so fit restores and stops without training, and
    # the weights must come back bit-identical.
    resumed = build_module()
    build_trainer(world, out_dir + "-resume", STEPS).fit(
        resumed,
        data.DataLoader(Steps(), batch_size=None, shuffle=False),
        ckpt_path=ckpt,
    )
    got = resumed.backbone.global_state_dict()
    assert set(got) == set(trained), "resumed stage owns different tensors"
    bad = [k for k in trained if not torch.equal(got[k], trained[k])]
    if bad:
        raise SystemExit(
            f"RESUME MISMATCH on rank {rank}: {len(bad)} of {len(trained)} tensors "
            f"differ, first: {bad[:3]}"
        )
    names = sorted(trained)
    print(
        f"[resume] rank{rank} recovered {len(names)} tensors bit-exactly, "
        f"{names[0]} .. {names[-1]}",
        flush=True,
    )

    if rank != 0:
        return
    whole = LMBackbone(get_preset(PRESET, vocab_size=VOCAB, **OVERRIDES))
    want = dict(whole.state_dict())

    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".ckpt"))
    if not files:
        raise SystemExit(f"CHECKPOINT MISSING: nothing written to {out_dir}")
    state = torch.load(
        os.path.join(out_dir, files[-1]), map_location="cpu", weights_only=False
    )["state_dict"]

    # By name, not by count: at pp=2 a half-saved model has the whole model's
    # parameter count once `backbone` and `stage_module` are both written.
    missing = sorted(set(want) - set(state))
    extra = sorted(set(state) - set(want) - {"token_counts"})
    saved = sum(t.numel() for k, t in state.items() if k in want)
    total = sum(t.numel() for t in want.values())
    blocks = sorted({k.split("blocks.")[1].split(".")[0] for k in state if "blocks." in k}, key=int)
    print(
        f"[ckpt] {files[-1]}: {len(state)} keys, {saved / 1e6:.2f}M of "
        f"{total / 1e6:.2f}M params, blocks {blocks}",
        flush=True,
    )
    if missing:
        raise SystemExit(
            f"INCOMPLETE CHECKPOINT: {len(missing)} of {len(want)} tensors absent, "
            f"first: {missing[:3]}"
        )
    if extra:
        raise SystemExit(f"UNEXPECTED KEYS: {extra[:5]}")

    # The artifact must load into a plain backbone, on one card, with no pipeline.
    whole.load_state_dict({k: v for k, v in state.items() if k in want}, strict=True)
    print("[ckpt] OK: whole model, global block indices, loads into LMBackbone")


def launch() -> None:
    """Run ``main`` on ``GPUS`` spawned ranks, or in place when ranks exist."""
    if os.environ.get("RANK") is not None:
        main()
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=GPUS or torch.cuda.device_count(),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="pp_checkpoint_smoke",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)()


if __name__ == "__main__":
    launch()
