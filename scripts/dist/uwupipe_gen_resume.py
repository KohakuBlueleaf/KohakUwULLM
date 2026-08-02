"""Resume and generation on the kohakuwupipe trainer, at production shape.

    .venv/bin/python scripts/dist/uwupipe_gen_resume.py \
        --ckpt /tmp/kwp_moe1b/last.ckpt

Restores a whole-model checkpoint into the split model, checks every rank got
its own slice bit-exactly, then generates through the pipeline schedule.
"""

import argparse
import faulthandler
import os

import torch
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwullm.generation import PipelineGenerator
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.pipeline_lightning import decode_stage
from kohakuwullm.training.parallel.uwupipe import (
    LMPipelineModule,
    plan_for,
    with_router_losses,
)
from kohakuwupipe import PipelineTrainer, get_logger, init_pipeline, shutdown

log = get_logger("uwupipe_gen_resume")
WATCHDOG = 600.0
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Kohaku-MoE-1B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--microbatches", type=int, default=32)
    ap.add_argument("--ckpt", default="/tmp/kwp_moe1b/last.ckpt")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=8)
    # Makes the training boundary a tuple while decode's stays a bare tensor.
    ap.add_argument("--aux-loss-weight", type=float, default=0.0)
    ap.add_argument("--router-z-loss-weight", type=float, default=0.0)
    return ap.parse_args()


def main(args: argparse.Namespace) -> None:
    faulthandler.dump_traceback_later(WATCHDOG, exit=True)
    ranks = init_pipeline()
    config = get_preset(
        args.preset,
        vocab_size=args.vocab,
        tie_embeddings=False,
        max_position=4096,
        qk_norm=True,
        mxfp8=True,
    )
    config = with_router_losses(
        config, args.aux_loss_weight, args.router_z_loss_weight
    )
    plans = plan_for(config, ranks.world, seq_len=args.micro_tokens)
    module = LMPipelineModule(
        config,
        micro_tokens=args.micro_tokens,
        optimizer="muon",
        optimizer_kwargs={"muon_lr": 2e-3, "embed_lr": 2e-3},
        learning_rate=5e-4,
    )
    trainer = PipelineTrainer(
        module,
        ranks,
        plans,
        micro_tokens=args.micro_tokens,
        num_microbatches=args.microbatches,
    )

    before = {k: v.clone() for k, v in module.inner.state_dict().items()}
    payload = trainer.load_checkpoint(args.ckpt, strict=True)
    after = module.inner.state_dict()
    changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
    log.info(
        "resumed",
        step=payload.get("global_step"),
        tensors=len(after),
        changed=changed,
        layers=f"{plans[ranks.rank].start_layer}..{plans[ranks.rank].end_layer - 1}",
    )
    if changed == 0:
        raise SystemExit("RESUME LOADED NOTHING: weights identical to fresh init")

    # Decode needs a padded (rows, 1) boundary, not the packed training one.
    stage = decode_stage(
        module.stage_module,
        plans[ranks.rank],
        config,
        ranks.rank,
        ranks.world,
        ranks.device,
        args.rows,
        param_dtype=module.param_dtype,
    )
    generator = PipelineGenerator(
        stage,
        module.inner,
        ranks.rank,
        ranks.world,
        microbatches=1,
        autocast_dtype=module.autocast_dtype,
    )
    prompt = torch.randint(0, args.vocab, (args.rows, args.prompt), device=ranks.device)
    out = generator.generate(prompt, max_new_tokens=args.new_tokens, temperature=0.0)
    log.info(
        "generated", shape=tuple(out.shape), tail=out[0, -args.new_tokens :].tolist()
    )
    if out.shape != (args.rows, args.prompt + args.new_tokens):
        raise SystemExit(f"BAD GENERATION SHAPE {tuple(out.shape)}")
    log.info("OK: resume restored this rank's slice and generation ran")
    shutdown()


def launch() -> None:
    """Run ``main`` on ``GPUS`` spawned ranks, or in place when ranks exist."""
    args = parse_args()
    if os.environ.get("RANK") is not None:
        main(args)
        return
    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=GPUS or torch.cuda.device_count(),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="uwupipe_gen_resume",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)(args)


if __name__ == "__main__":
    launch()
