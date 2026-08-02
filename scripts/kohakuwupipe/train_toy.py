"""Train a model you wrote yourself across pipeline stages, with kohakuwupipe.

    .venv/bin/python scripts/kohakuwupipe/train_toy.py

Nothing here imports ``kohakuwullm``: this is the template for using
``kohakuwupipe`` from another repository. Subclass :class:`PipelineModule` with
four things -- build the stage, declare its boundary, give an optimizer, define
the loss -- and :class:`PipelineTrainer` runs it.
"""

import argparse
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from kohakuwupipe import (
    Checkpoint,
    LossLog,
    PipelineModule,
    PipelineTrainer,
    Throughput,
    describe,
    get_logger,
    init_pipeline,
    plan_stages,
    shutdown,
)

log = get_logger("train_toy")
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


@dataclass
class Step:
    """One optimizer step: what the loop reads off a batch."""

    inputs: torch.Tensor
    target: torch.Tensor
    layout: None
    trained: int


class ToyStage(nn.Module):
    """A slice of a toy decoder: optional embedding, blocks, optional head."""

    def __init__(self, plan, vocab: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim) if plan.has_embed else None
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            for _ in range(plan.num_layers)
        )
        self.head = nn.Linear(dim, vocab, bias=False) if plan.has_head else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            x = self.embed(x)
        for block in self.blocks:
            x = x + block(x)
        return x

    def loss(self, hidden: torch.Tensor, target: torch.Tensor):
        """Cross-entropy on the last stage; a sum, which the loop normalizes."""
        logits = self.head(hidden).float()
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum"
        )
        return loss, {}

    def set_seq_info(self, layout) -> None:
        """A toy model carries no per-microbatch layout."""


class ToyModule(PipelineModule):
    """The four things kohakuwupipe needs to train any pipelined model."""

    def __init__(self, vocab: int, dim: int, tokens: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.tokens = tokens
        self.lr = lr

    def configure_model(self, plan, rank, world, device) -> nn.Module:
        return ToyStage(plan, self.vocab, self.dim).to(
            device=device, dtype=torch.bfloat16
        )

    def boundary_example(self, plan, device) -> torch.Tensor:
        if plan.has_embed:
            return torch.zeros(self.tokens, dtype=torch.long, device=device)
        return torch.zeros(self.tokens, self.dim, dtype=torch.bfloat16, device=device)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.stage_module.parameters(), lr=self.lr)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--ckpt", default="")
    return ap.parse_args()


def main(args: argparse.Namespace) -> None:
    ranks = init_pipeline()
    plans = plan_stages(
        depth=args.depth,
        num_stages=ranks.world,
        layer_cost=8.0 * args.dim * args.dim,
        head_cost=2.0 * args.dim * args.vocab,
        layer_params=8.0 * args.dim * args.dim,
        head_params=float(args.dim * args.vocab),
        embed_params=float(args.dim * args.vocab),
    )
    if ranks.rank == 0:
        log.info("stage split\n%s", describe(plans))

    callbacks = [Throughput(every_n_steps=10, warmup_steps=5), LossLog(every_n_steps=5)]
    if args.ckpt:
        callbacks.append(
            Checkpoint(args.ckpt, plans[ranks.rank].start_layer, args.steps)
        )

    trainer = PipelineTrainer(
        ToyModule(args.vocab, args.dim, args.tokens),
        ranks,
        plans,
        micro_tokens=args.tokens,
        num_microbatches=args.microbatches,
        callbacks=callbacks,
    )

    trained = args.tokens * args.microbatches
    generator = torch.Generator().manual_seed(0)

    def stream():
        """The same stream on every rank; only the endpoints read it."""
        while True:
            ids = torch.randint(0, args.vocab, (trained,), generator=generator)
            ids = ids.to(ranks.device)
            yield Step(inputs=ids, target=ids.roll(-1), layout=None, trained=trained)

    trainer.fit(stream(), max_steps=args.steps)
    log.info("done", steps=trainer.loop.global_step)
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
        run_id="train_toy",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)(args)


if __name__ == "__main__":
    launch()
