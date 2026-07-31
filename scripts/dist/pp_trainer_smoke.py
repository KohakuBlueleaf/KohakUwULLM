"""Run PipelinedLMTrainer through a real Lightning Trainer, generating mid-run.

    torchrun --standalone --nproc_per_node=2 scripts/dist/pp_trainer_smoke.py

Checks the token counters and the per-stage FLOP shares against their closed
forms, and previews from inside the loop, which is where generation must work.
"""

import faulthandler
import os

import lightning.pytorch as pl
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import Callback

from kohakuwullm.data.loader.microbatch import MicroBatchedStep
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.loop.pipeline_trainer import PipelinedLMTrainer
from kohakuwullm.training.parallel.strategy import PipelineOnlyStrategy

VOCAB = 2048
MICRO = 256
N_MICRO = 4
STEPS = 6
PREVIEW_EVERY = 3
WATCHDOG_SECONDS = 180.0


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


class MidRunPreview(Callback):
    """Generate through the pipeline schedule from inside the loop, on every rank.

    Also checks the accounting at ``on_train_end``, which is the last hook before
    teardown moves the module to CPU and destroys the process group.
    """

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        step = trainer.global_step
        if step == 0 or step % PREVIEW_EVERY:
            return
        device = trainer.strategy.root_device
        prompt = torch.zeros(2, 8, dtype=torch.long, device=device)
        out = pl_module.generate(prompt, max_new_tokens=4, temperature=0.0)
        print(
            f"[rank{trainer.global_rank}] preview @ step {step} -> {tuple(out.shape)}",
            flush=True,
        )

    def on_train_end(self, trainer, pl_module) -> None:
        rank = trainer.global_rank
        # Counted on the ingesting stage only, so the reduced total is the truth once.
        snapshot = pl_module.token_snapshot()
        want = MICRO * N_MICRO * STEPS
        assert snapshot.seen == want, f"seen {snapshot.seen} != {want}"
        assert snapshot.trained == want, f"trained {snapshot.trained} != {want}"

        # Every stage's block share sums to one whole model.
        shares = torch.tensor(
            [pl_module._flop_share], device=trainer.strategy.root_device
        )
        torch.distributed.all_reduce(shares)
        assert abs(float(shares[0]) - 1.0) < 1e-9, f"shares sum to {float(shares[0])}"

        print(
            f"[rank{rank}] TRAIN OK | seen {snapshot.seen} trained {snapshot.trained}"
            f" | mflops {snapshot.model_flops:.3e} | share {pl_module._flop_share:.4f}",
            flush=True,
        )


def main() -> None:
    # A pipeline deadlock is silent; dump every rank's stack rather than hang.
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)
    world = int(os.environ.get("WORLD_SIZE", "2"))
    module = PipelinedLMTrainer(
        preset="Nano-25M",
        vocab_size=VOCAB,
        arch_overrides={"tie_embeddings": False, "max_position": 1024},
        micro_tokens=MICRO,
        num_microbatches=N_MICRO,
        optimizer="adamw",
        learning_rate=1e-4,
        scheduler_config={"lr": {"mode": "constant", "end": -1}},
        name="pp-smoke",
        log_interval=1,
    )
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=world,
        strategy=PipelineOnlyStrategy(process_group_backend="nccl"),
        precision="32-true",
        max_steps=STEPS,
        use_distributed_sampler=False,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[MidRunPreview()],
    )
    loader = data.DataLoader(Steps(), batch_size=None, shuffle=False)
    trainer.fit(module, loader)
    print(f"[rank{trainer.global_rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
