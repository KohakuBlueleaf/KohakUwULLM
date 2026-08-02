"""Pipeline-parallel variant of :class:`LMTrainer`.

Overrides model construction, batch placement, the step and the FLOP share;
everything else -- token accounting, logging, resume -- is inherited unchanged.
See docs/internals/pipeline.md.
"""

import torch

from kohakuwullm.generation import build_generator
from kohakuwullm.models.flops import document_lengths
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights
from kohakuwullm.training.loop.trainer import LMTrainer
from kohakuwullm.training.parallel.pipeline_lightning import (
    build_schedule,
    build_stage,
    decode_stage,
)
from kohakuwupipe.parallel.streams import reduce_accumulator

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class PipelinedLMTrainer(LMTrainer):
    """One pipeline stage per rank, driven by a ``torch.distributed`` schedule.

    Args:
        micro_tokens: tokens per microbatch, identical on every rank; it fixes the
            boundary shape ``PipelineStage`` freezes at construction.
        num_microbatches: microbatches per optimizer step; this is the gradient
            accumulation, so ``accumulate_grad_batches`` must stay 1.
        schedule: ``1f1b`` or ``gpipe``.
        param_dtype / autocast_dtype: stage parameter storage and forward autocast.
    """

    # `generate` runs the pipeline schedule, so every rank must enter it together.
    generate_is_collective = True

    def __init__(
        self,
        *,
        micro_tokens: int = 8192,
        num_microbatches: int = 32,
        schedule: str = "1f1b",
        param_dtype: str = "bf16",
        autocast_dtype: str | None = "bf16",
        seq_len: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.micro_tokens = micro_tokens
        self.num_microbatches = num_microbatches
        self.schedule_kind = schedule
        self.param_dtype = _DTYPES[param_dtype]
        self.autocast_dtype = (
            None if autocast_dtype is None else _DTYPES[autocast_dtype]
        )
        self.seq_len = seq_len
        self.stage_module = None
        self.pp_stage = None
        self.plan = None
        self._flop_share = 1.0
        self._generator = None
        self._generator_rows = None
        self._schedule = None
        self._denom = 1

    def configure_model(self) -> None:
        """Replace the full backbone with this rank's stage. Called more than once."""
        if self.pp_stage is not None:
            return
        rank = self.trainer.global_rank
        world = self.trainer.world_size
        device = self.trainer.strategy.root_device
        self.stage_module, self.pp_stage, self.plan, _plans = build_stage(
            self.arch,
            rank,
            world,
            device,
            self.micro_tokens,
            autocast_dtype=self.autocast_dtype,
            param_dtype=self.param_dtype,
            seq_len=self.seq_len,
            head_kwargs=self.head_kwargs,
        )
        inner = getattr(self.stage_module, "module", self.stage_module)
        # Everything inherited -- loss, param_summary, the fp8 refresh, the grad
        # norm -- reads `self.backbone`, so point it at this rank's slice.
        self.backbone = inner
        self._flop_share = self.plan.num_layers / max(self.arch.depth, 1)

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        """Stage 0 takes the tokens, the last stage the labels; neither takes both."""
        last = self.trainer.world_size - 1
        batch.seq_infos = [info.to(device) for info in batch.seq_infos]
        if self.trainer.global_rank == 0:
            batch.tokens = batch.tokens.to(device, non_blocking=True)
        if self.trainer.global_rank == last:
            batch.labels = batch.labels.to(device, non_blocking=True)
        return batch

    def training_step(self, batch, batch_idx):
        """One optimizer step, run as ``num_microbatches`` pipeline microbatches.

        Collective: every rank executes exactly one ``schedule.step``; no rank may
        return early. See docs/internals/pipeline.md.
        """
        rank = self.trainer.global_rank
        last = self.trainer.world_size - 1
        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad(set_to_none=True)

        self.backbone.set_seq_info(batch.seq_infos)
        # Every stage holds the same batch, so the count needs no collective and
        # no host sync. See docs/internals/pipeline.md.
        local_trained = int(sum(batch.trained))
        denom = max(local_trained, 1)

        self._denom = denom
        schedule = self._pipeline_schedule()
        losses = [] if rank == last else None
        # No `manual_backward`: the schedule already ran backward on every stage.
        if rank == 0:
            schedule.step(batch.tokens)
        elif rank == last:
            schedule.step(target=batch.labels, losses=losses)
        else:
            schedule.step()

        grad_norm = self._grad_norm()
        self.clip_gradients(
            opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm"
        )
        empty_step = local_trained == 0
        if not empty_step:
            opt.step()
            refresh_mxfp8_weights(self.backbone)
        opt.zero_grad(set_to_none=True)
        sched.step()
        moe_logs = self.backbone.update_router_bias()

        seen = self.micro_tokens * self.num_microbatches
        # `token_snapshot` sums over ranks and every stage holds the same batch, so
        # only the ingesting stage counts it. See docs/internals/pipeline.md.
        if self.plan.has_embed:
            self.token_counts += torch.tensor(
                [seen, local_trained],
                dtype=torch.int64,
                device=self.token_counts.device,
            )
        # Each stage charges its own layers, plus the head on whichever holds it.
        lengths = torch.cat([document_lengths(info) for info in batch.seq_infos])
        self._flop_counts += self.flop_model.stage_flops(
            seen, lengths, self._flop_share, self.plan.has_head
        )

        totals = {
            "ce": self._mean_loss(losses, last) * local_trained,
            "n": local_trained,
        }
        self._log_step(totals, {}, moe_logs, grad_norm, empty_step)

    def gathered_state_dict(self) -> dict:
        """Every stage's slice under whole-backbone names, plus this module's buffers.

        Collective: every rank must call. The result loads into a plain
        :class:`LMBackbone`. See docs/internals/pipeline.md.
        """
        local = self.backbone.global_state_dict()
        merged = {}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(parts, local)
            for part in parts:
                merged.update(part)
        else:
            merged.update(local)
        for name, buf in self.named_buffers(recurse=False):
            merged[name] = buf.detach().cpu()
        return merged

    def load_gathered_state_dict(self, state, strict: bool = True) -> None:
        """Load this rank's slice out of a whole-backbone checkpoint."""
        self.backbone.load_global_state_dict(state, strict=strict)
        for name, buf in self.named_buffers(recurse=False):
            if name in state:
                buf.copy_(state[name].to(buf.device))

    def _pipeline_schedule(self):
        """The schedule, built once; rebuilding it per step deadlocks the ranks."""
        if self._schedule is None:

            def loss_fn(output, target):
                # `stage_module`, not `backbone`: the schedule calls this outside
                # the stage forward, so only the wrapper carries the autocast.
                hidden, *streams = output if isinstance(output, tuple) else (output,)
                loss, _ = self.stage_module.loss(hidden, target)
                loss = loss / max(self._denom, 1)
                # An accumulator holds a per-microbatch mean; average, do not sum.
                aux_scale = 1.0 / max(self.num_microbatches, 1)
                for stream in streams:
                    loss = loss + aux_scale * reduce_accumulator(stream)
                return loss

            self._schedule = build_schedule(
                self.pp_stage,
                self.num_microbatches,
                loss_fn=loss_fn,
                kind=self.schedule_kind,
            )
        return self._schedule

    def _mean_loss(self, losses, last_rank: int) -> torch.Tensor:
        """Per-token mean loss, broadcast from the only rank that computes it.

        Stays a device tensor; microbatch losses **sum** to the mean.
        See docs/internals/pipeline.md.
        """
        device = self.token_counts.device
        buf = torch.zeros(1, device=device)
        if losses:
            buf[0] = torch.stack([x.detach().float() for x in losses]).sum()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(buf, src=last_rank)
        return buf[0]

    def generate(self, prompt_ids, **kwargs):
        """Sample across the stages. See docs/guides/generation.md.

        Collective: every rank must call this the same number of times with the
        same shapes, and every rank gets the tokens back.
        """
        rows = prompt_ids.shape[0]
        if self._generator is None or self._generator_rows != rows:
            # The training stage's boundary is packed `(micro_tokens,)`; decode
            # needs a padded `(rows, 1)` one over the same module.
            stage = decode_stage(
                self.stage_module,
                self.plan,
                self.arch,
                self.trainer.global_rank,
                self.trainer.world_size,
                self.trainer.strategy.root_device,
                rows,
                param_dtype=self.param_dtype,
            )
            self._generator = build_generator(
                stage=stage,
                head_module=self.backbone,
                rank=self.trainer.global_rank,
                world=self.trainer.world_size,
                microbatches=1,
                autocast_dtype=self.autocast_dtype,
            )
            self._generator_rows = rows
        return self._generator.generate(prompt_ids, **kwargs)
