"""Pipelined generation across stages, independent of the training loop.

torchrun --standalone --nproc_per_node=2 scripts/dist/pp_generate_smoke.py
"""

import os

import torch
import torch.distributed as dist

from kohakuwullm.generation import PipelineGenerator
from kohakuwullm.models import get_preset
from kohakuwullm.training.parallel.pipeline_lightning import build_stage

VOCAB = 2048
ROWS = 4
PROMPT = 8
NEW = 6


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    config = get_preset(
        "Nano-25M", vocab_size=VOCAB, tie_embeddings=False, max_position=256
    )
    # `microbatch_tokens` is the per-row prompt+generated length: one row per
    # microbatch, and the boundary shape is frozen at construction.
    module, stage, plan, _ = build_stage(
        config, rank, world, device, 1, seq_len=PROMPT + NEW, decode=True
    )
    inner = getattr(module, "module", module)

    generator = PipelineGenerator(stage, inner, rank, world)
    prompt = torch.randint(0, VOCAB, (ROWS, PROMPT), device=device)
    out = generator.generate(prompt, max_new_tokens=NEW, temperature=0.0)

    dist.barrier()
    print(
        f"[rank{rank}] GENERATE OK shape={tuple(out.shape)} tail={out[0, -NEW:].tolist()}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
