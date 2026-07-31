"""Does a pipelined model generate the same tokens as the whole model?

    torchrun --standalone --nproc_per_node=4 scripts/dist/pp_generate_equivalence.py

Greedy decode, so the two must agree token for token. The fixture is
deliberately chaotic: at the default init the model repeats one token and every
cache bug passes. See docs/guides/generation.md.
"""

import faulthandler
import os

import torch
import torch.distributed as dist

from kohakuwullm.generation import LocalGenerator, PipelineGenerator
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training.parallel.pipeline_lightning import build_stage

VOCAB = 2048
ROWS = 2
PROMPT = 12
NEW = 24
SEED = 7
WATCHDOG_SECONDS = 240.0


def main() -> None:
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    # A wide init makes the logits chaotic, so a wrong position or a stale KV
    # slot changes the argmax instead of being masked by a repeated token.
    config = get_preset(
        "Nano-25M",
        vocab_size=VOCAB,
        tie_embeddings=False,
        max_position=256,
        init_std=0.35,
    )
    torch.manual_seed(SEED)
    prompt = torch.randint(0, VOCAB, (ROWS, PROMPT), device=device)

    torch.manual_seed(SEED)
    whole = LMBackbone(config).to(device=device, dtype=torch.bfloat16).eval()
    reference = LocalGenerator(whole).generate(
        prompt, max_new_tokens=NEW, temperature=0.0
    )
    tail = reference[:, PROMPT:]
    distinct = int(tail.unique().numel())
    del whole
    torch.cuda.empty_cache()

    torch.manual_seed(SEED)
    module, stage, _plan, _ = build_stage(
        config,
        rank,
        world,
        device,
        ROWS,
        seq_len=PROMPT + NEW,
        param_dtype=torch.bfloat16,
        autocast_dtype=None,
        calibrate=False,
        decode=True,
    )
    inner = getattr(module, "module", module)
    got = PipelineGenerator(stage, inner, rank, world, microbatches=1).generate(
        prompt, max_new_tokens=NEW, temperature=0.0
    )

    if rank == 0:
        # A degenerate reference cannot detect anything; fail loudly instead.
        assert distinct >= 4, f"fixture is not chaotic: {distinct} distinct tokens"
        same = torch.equal(got, reference)
        print(f"reference tail: {tail[0].tolist()}", flush=True)
        print(f"pipelined tail: {got[0, PROMPT:].tolist()}", flush=True)
        if not same:
            differ = (got != reference).nonzero()
            raise SystemExit(
                f"PIPELINED GENERATION DIVERGES from the whole model at "
                f"{differ.shape[0]} of {got.numel()} positions, first {differ[0].tolist()}"
            )
        print(
            f"OK: {world} stages reproduce the whole model token for token "
            f"({distinct} distinct tokens in the fixture)",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
