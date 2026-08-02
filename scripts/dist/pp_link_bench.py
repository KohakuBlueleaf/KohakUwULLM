"""What one pipeline seam costs: per-hop latency and bandwidth around the chain.

    .venv/bin/python scripts/dist/pp_link_bench.py

Decode ships a few kB per token per seam, so the seam's *latency* sets the step,
not its bandwidth. A token is circulated 0->1->..->N-1->0 and the time divided by
the hop count. See docs/internals/pipeline.md.
"""

import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

SIZES = [4 << 10, 24 << 10, 96 << 10, 1 << 20, 16 << 20, 128 << 20]
ITERS = 100
WARMUP = 20
# Ranks to spawn when the caller started none. 0 uses every GPU.
GPUS = 0


def ring_hop(buf: torch.Tensor, rank: int, world: int, iters: int) -> float:
    """Median seconds for ONE hop, from a full circulation of the chain."""
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        if rank == 0:
            dist.send(buf, nxt)
            dist.recv(buf, prv)
        else:
            dist.recv(buf, prv)
            dist.send(buf, nxt)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters / world


def collective_cost(buf: torch.Tensor, iters: int, op: str) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        if op == "broadcast":
            dist.broadcast(buf, src=0)
        else:
            dist.all_reduce(buf)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)

    peer = torch.cuda.can_device_access_peer(0, 1) if world > 1 else False
    rows = []
    if rank == 0:
        print(f"world {world}  peer-access(0,1) {peer}")
        print(
            f"{'bytes':>10} {'hop us':>9} {'GB/s':>8} {'bcast us':>10} {'allred us':>10}"
        )

    for size in SIZES:
        buf = torch.empty(size, dtype=torch.uint8, device="cuda")
        ring_hop(buf, rank, world, WARMUP)
        collective_cost(buf, WARMUP, "broadcast")
        hop = ring_hop(buf, rank, world, ITERS)
        bcast = collective_cost(buf, ITERS, "broadcast")
        # all_reduce needs a float payload of the same footprint.
        red = torch.empty(size // 4, dtype=torch.float32, device="cuda")
        collective_cost(red, WARMUP, "all_reduce")
        allred = collective_cost(red, ITERS, "all_reduce")
        if rank == 0:
            rows.append(
                dict(
                    bytes=size,
                    hop_us=hop * 1e6,
                    gbps=size / hop / 1e9,
                    broadcast_us=bcast * 1e6,
                    all_reduce_us=allred * 1e6,
                )
            )
            print(
                f"{size:10d} {hop * 1e6:9.1f} {size / hop / 1e9:8.2f} "
                f"{bcast * 1e6:10.1f} {allred * 1e6:10.1f}",
                flush=True,
            )
        del buf, red
        torch.cuda.empty_cache()

    if rank == 0:
        out = "out/bench/gen/pipeline"
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"link-pp{world}.json")
        with open(path, "w") as handle:
            json.dump(
                dict(
                    world=world,
                    peer_access=peer,
                    device=torch.cuda.get_device_name(),
                    rows=rows,
                ),
                handle,
                indent=2,
            )
        print(f"wrote {path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


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
        run_id="pp_link_bench",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, main)()


if __name__ == "__main__":
    launch()
