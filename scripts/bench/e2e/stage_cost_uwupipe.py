"""Per-stage cost of a pipeline split, each stage timed on its own card.

    torchrun --standalone --nproc_per_node=4 scripts/bench/e2e/stage_cost_uwupipe.py

Each rank builds only its own stage and times a forward+backward over its own
boundary shape, with no send/recv in the window. The spread across stages is
the imbalance the schedule then waits on. See docs/internals/pipeline.md.
"""

import argparse
import time

import torch
import torch.distributed as dist

from kohakuwullm.bench import make_info
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.uwupipe import LMPipelineModule, plan_for
from kohakuwupipe import get_logger, init_pipeline, shutdown

log = get_logger("stage_cost")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Kohaku-MoE-1B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--mxfp8", action="store_true")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--layers", default="")
    args = ap.parse_args()

    ranks = init_pipeline()
    config = get_preset(
        args.preset,
        vocab_size=args.vocab,
        tie_embeddings=False,
        max_position=4096,
        qk_norm=True,
    )
    layers = [int(n) for n in args.layers.split(",")] if args.layers else None
    plans = plan_for(config, ranks.world, seq_len=args.micro_tokens, layers=layers)
    plan = plans[ranks.rank]
    module = LMPipelineModule(config, micro_tokens=args.micro_tokens, mxfp8=args.mxfp8)
    stage = module.setup(plan, ranks.rank, ranks.world, ranks.device)

    info = make_info(args.micro_tokens, 50, 600, ranks.device, seed=0)
    stage.set_seq_info([info])
    if plan.has_embed:
        x = torch.zeros(args.micro_tokens, dtype=torch.long, device=ranks.device)
    else:
        x = torch.randn(
            args.micro_tokens, config.dim, device=ranks.device, dtype=module.param_dtype
        ).requires_grad_()
    labels = torch.randint(0, args.vocab, (args.micro_tokens,), device=ranks.device)

    def step():
        stage.set_seq_info([info])
        out = stage(x)
        if plan.has_head:
            loss, _ = stage.loss(out, labels)
        else:
            loss = out.float().sum()
        loss.backward()
        module.stage_module.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    ms = samples[len(samples) // 2] * 1e3

    stats = torch.tensor([ms], device=ranks.device)
    gathered = [torch.zeros_like(stats) for _ in range(ranks.world)]
    dist.all_gather(gathered, stats)
    if ranks.rank == 0:
        times = [float(g[0]) for g in gathered]
        slowest = max(times)
        total = sum(times)
        log.info("per-stage forward+backward, no seam in the window")
        for i, (p, t) in enumerate(zip(plans, times)):
            role = "embed" if p.has_embed else ("head" if p.has_head else "-")
            log.info(
                f"  stage {i}",
                layers=f"{p.start_layer}..{p.end_layer - 1}",
                n=p.num_layers,
                role=role,
                ms=t,
                share=100 * t / total,
            )
        log.info(
            "imbalance",
            slowest_ms=slowest,
            ideal_ms=total / ranks.world,
            wasted_pct=100 * (slowest - total / ranks.world) / slowest,
        )
    shutdown()


if __name__ == "__main__":
    main()
