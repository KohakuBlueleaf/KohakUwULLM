"""Microbatch-count sweep: bubble rate against fixed per-boundary latency.

torchrun --standalone --nproc_per_node=4 scripts/bench/e2e/pp_micro_sweep.py
"""

import os
import time

import torch
import torch.distributed as dist

from kohakuwullm.models import get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.parallel.pipeline_lightning import (
    build_schedule,
    build_stage,
    run_step,
    stage_local_optimizer,
)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    preset = os.environ.get("PP_PRESET", "Kohaku-1B")
    vocab = 65536
    total_tokens = int(os.environ.get("PP_TOKENS", "16384"))
    cfg = get_preset(preset, vocab_size=vocab, tie_embeddings=False)

    if rank == 0:
        print(f"{preset}, {total_tokens} tokens/step, {world} stages", flush=True)
        print(
            f"{'n_micro':>8} {'tok/micro':>10} {'s/iter':>9} {'tok/s':>12}", flush=True
        )

    # 1F1B requires n_micro >= n_stages; below that the pipeline cannot fill.
    for n_micro in (4, 8, 16, 32, 64):
        if n_micro < world:
            continue
        per = total_tokens // n_micro
        if per < 128:
            continue
        module, stage, _, _ = build_stage(cfg, rank, world, device, per, seq_len=per)
        n_seq = max(1, per // 128)
        lengths = torch.full((n_seq,), per // n_seq, dtype=torch.int32)
        infos = [SeqInfo.from_lengths(lengths, device) for _ in range(n_micro)]
        tokens = torch.randint(0, vocab, (per * n_micro,), device=device)
        targets = tokens.roll(-1)

        # Bound at definition: the loop rebinds these each pass and frees them
        # at the end, so a late-binding closure would read the wrong stage.
        def loss_fn(hidden, target, module=module, denom=per * n_micro):
            loss, _ = module.loss(hidden, target)
            return loss / denom

        schedule = build_schedule(stage, n_micro, loss_fn=loss_fn, kind="1f1b")
        opt = stage_local_optimizer(module, lr=1e-4)

        def step(opt=opt, module=module, schedule=schedule, infos=infos):
            opt.zero_grad(set_to_none=True)
            module.set_seq_info(infos)
            losses = [] if rank == world - 1 else None
            run_step(
                schedule, rank, world, tokens=tokens, targets=targets, losses=losses
            )
            opt.step()

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = (time.perf_counter() - start) / iters
        if rank == 0:
            print(
                f"{n_micro:>8} {per:>10} {elapsed:>8.3f}s "
                f"{per * n_micro / elapsed:>11,.0f}",
                flush=True,
            )
        del module, stage, schedule, opt
        torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
