"""Does pipelining actually overlap decode work, and what is the real ceiling?

    torchrun --standalone --nproc_per_node=4 scripts/dist/pp_generate_bench.py \
        --preset Kohaku-MoE-3B --requests 1 2 4 8 16 32

Every microbatch is ONE request, so raising the count adds independent work
rather than shrinking the existing work. Two arms over the same total requests:

* **sequential** -- one request at a time, no overlap available;
* **pipelined**  -- all requests in flight as microbatches.

Their ratio is the overlap pipelining actually delivers. Scored against a
roofline that charges DRAM *and* the seam. See docs/internals/pipeline.md.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from kohakuwullm.bench.model.ladder import census
from kohakuwullm.generation import PipelineGenerator
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.pipeline_lightning import build_stage

PEAK_GBPS = 1791.0
# Measured by scripts/dist/pp_link_bench.py on this box; no peer access.
HOP_LATENCY_S = 22.2e-6
BROADCAST_S = 47.3e-6
SEED = 1234


def stage_bytes(rung, stages: int, batch: int, context: int, elem: int) -> float:
    """Bytes the busiest stage reads per token step: its weights plus its KV."""
    weights = rung.compute_active * elem / stages
    kv = 2 * batch * context * rung.kv_heads * rung.head_dim * rung.depth * elem
    return weights + kv / stages


def roofline(rung, stages: int, requests: int, context: int, elem: int) -> dict:
    """Saturated tokens/s, and the one-request latency, both charging the seam.

    Saturated throughput is set by the slowest stage OR the seam, whichever is
    larger; latency adds every stage and every hop in series.
    """
    per_stage = stage_bytes(rung, stages, requests, context, elem) / (PEAK_GBPS * 1e9)
    seam = HOP_LATENCY_S
    throughput = requests / max(per_stage, seam)
    latency = per_stage * stages + (stages - 1) * seam + BROADCAST_S
    return {
        "stage_seconds": per_stage,
        "seam_seconds": seam,
        "bound": "dram" if per_stage >= seam else "seam",
        "saturated_tokens_per_s": throughput,
        "one_request_latency_s": latency,
    }


def timed(fn, iters: int = 3):
    """Median seconds over ``iters`` calls, after one warmup."""
    fn()
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Kohaku-MoE-3B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--prompt", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--out", default="out/bench/gen/pipeline")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    overrides = {
        "vocab_size": args.vocab,
        "tie_embeddings": False,
        "max_position": 4096,
    }
    config = get_preset(args.preset, **overrides)
    rung = census(args.preset, **overrides)
    elem, context = 2, args.prompt + args.new_tokens // 2

    # One request per microbatch, so the boundary is (1, 1) and never changes.
    torch.manual_seed(SEED)
    module, stage, _plan, _ = build_stage(
        config,
        rank,
        world,
        device,
        1,
        seq_len=args.prompt + args.new_tokens,
        param_dtype=torch.bfloat16,
        calibrate=False,
        decode=True,
    )
    inner = getattr(module, "module", module)

    if rank == 0:
        print(
            f"{args.preset}  active {rung.compute_active / 1e6:.0f}M  depth "
            f"{rung.depth}  stages {world}  hop {HOP_LATENCY_S * 1e6:.1f}us\n"
            f"{'reqs':>5} {'seq ms':>9} {'pipe ms':>9} {'overlap':>8} "
            f"{'pipe tok/s':>11} {'roof tok/s':>11} {'bound':>6} {'% roof':>7}",
            flush=True,
        )

    rows = []
    for requests in args.requests:
        gen = PipelineGenerator(
            stage, inner, rank, world, microbatches=1, autocast_dtype=torch.bfloat16
        )
        torch.manual_seed(SEED)
        one = torch.randint(0, args.vocab, (1, args.prompt), device=device)

        def sequential(engine=gen, n=requests, prompt=one):
            for _ in range(n):
                engine.generate(prompt, max_new_tokens=args.new_tokens, temperature=0.0)

        seq_s = timed(sequential)

        # All requests in flight at once: `requests` microbatches of one row.
        pipe_gen = PipelineGenerator(
            stage,
            inner,
            rank,
            world,
            microbatches=requests,
            autocast_dtype=torch.bfloat16,
        )
        batch = one.repeat(requests, 1)

        def pipelined(engine=pipe_gen, prompt=batch):
            engine.generate(prompt, max_new_tokens=args.new_tokens, temperature=0.0)

        pipe_s = timed(pipelined)

        roof = roofline(rung, world, requests, context, elem)
        tokens = requests * args.new_tokens
        rate = tokens / pipe_s
        rows.append(
            dict(
                requests=requests,
                sequential_s=seq_s,
                pipelined_s=pipe_s,
                overlap=seq_s / pipe_s,
                tokens_per_s=rate,
                **roof,
            )
        )
        if rank == 0:
            print(
                f"{requests:5d} {seq_s * 1e3:9.1f} {pipe_s * 1e3:9.1f} "
                f"{seq_s / pipe_s:7.2f}x {rate:11.0f} "
                f"{roof['saturated_tokens_per_s']:11.0f} {roof['bound']:>6} "
                f"{100 * rate / roof['saturated_tokens_per_s']:6.2f}%",
                flush=True,
            )
        gen = pipe_gen = None
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.preset}-overlap-pp{world}.json")
        with open(path, "w") as handle:
            json.dump(
                dict(
                    preset=args.preset,
                    stages=world,
                    new_tokens=args.new_tokens,
                    active_params=rung.compute_active,
                    hop_latency_s=HOP_LATENCY_S,
                    broadcast_s=BROADCAST_S,
                    peak_gbps=PEAK_GBPS,
                    device=torch.cuda.get_device_name(),
                    rows=rows,
                ),
                handle,
                indent=2,
            )
        print(f"wrote {path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
