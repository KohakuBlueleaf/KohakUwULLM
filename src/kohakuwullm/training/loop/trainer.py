"""The Lightning LM trainer: manual optimization over packed varlen batches.

See docs/guides/training.md.
"""

import contextlib
import math
import time

import lightning.pytorch as pl
import torch
from anyschedule import AnySchedule

from kohakuwullm.compile_utils import apply_compile
from kohakuwullm.data.loader.padded import split_padded
from kohakuwullm.data.packing import split_packed
from kohakuwullm.models import LMArchConfig, LMBackbone, get_preset
from kohakuwullm.models.flops import FlopCounter, document_lengths
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights
from kohakuwullm.training.loop.resume import (
    load_loader_state,
    load_rng_state,
    loader_state,
    resumable,
    rng_state,
)
from kohakuwullm.training.loop.sampling import PreviewSampler
from kohakuwullm.training.loop.tokens import TokenSnapshot, all_reduce_, all_reduce_int
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.utils import human_count

# Checkpoint key holding everything this trainer adds to a Lightning checkpoint.
RESUME_KEY = "kohakuwullm"

# Reference context the startup log reports FLOPs/token at.
REPORT_CONTEXT = 2048


class LMTrainer(pl.LightningModule):
    """Causal-LM training on packed variable-length batches."""

    def __init__(
        self,
        *,
        # architecture
        preset: str | None = "Nano-500M",
        arch_overrides: dict | None = None,
        vocab_size: int = 65536,
        head_kwargs: dict | None = None,
        # optimization
        optimizer: str = "adamw",
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        eps: float = 1e-8,
        use_mup: bool = False,
        base_dim: int = 256,
        scheduler_config: dict | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        optimizer_kwargs: dict | None = None,
        # runtime
        compile_config: dict | None = None,
        grad_checkpointing: bool = False,
        resume_state: bool = True,
        # logging
        name: str = "lm",
        log_interval: int = 10,
        **_unused,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if preset is None:
            arch = LMArchConfig(**(arch_overrides or {}))
        else:
            arch = get_preset(preset, **(arch_overrides or {}))
        arch.vocab_size = vocab_size
        arch.grad_ckpt = grad_checkpointing
        self.arch = arch
        self.head_kwargs = head_kwargs
        self.backbone = LMBackbone(arch, head_kwargs=head_kwargs)
        # Built from the uncompiled module, which still exposes the shape constants.
        self.flop_model = FlopCounter(self.backbone)
        self.backbone = apply_compile(self.backbone, compile_config)

        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.learning_rate = learning_rate
        self.betas = tuple(betas)
        self.weight_decay = weight_decay
        self.eps = eps
        self.use_mup = use_mup
        self.base_dim = base_dim
        self.scheduler_config = scheduler_config or {
            "lr": {"mode": "cosine", "min_value": 0.1, "end": -1}
        }
        self.grad_accum = gradient_accumulation_steps
        self.grad_clip = gradient_clip
        self.resume_state = resume_state

        self.name = name
        self.log_interval = log_interval
        # Rank-local running totals, reduced only when a snapshot is taken.
        self.register_buffer(
            "token_counts", torch.zeros(2, dtype=torch.int64), persistent=True
        )
        # [model, hardware] FLOPs. Not a buffer: `bf16-true` converts float buffers.
        self._flop_counts = torch.zeros(2, dtype=torch.float64)
        # Global totals carried in from a checkpoint, kept apart from the local counters.
        self._count_offset = (0, 0)
        self._flop_offset = (0.0, 0.0)
        self._elapsed_offset = 0.0
        self._clock_start: float | None = None
        self._last_snapshot: TokenSnapshot | None = None
        self._resume: dict | None = None
        self.sampler = PreviewSampler()
        self.ema_loss = -1.0

    # -- setup ----------------------------------------------------------- #

    def on_train_start(self) -> None:
        # First hook that runs after the checkpoint is restored under every strategy.
        self._apply_resume()
        self._flop_counts = self._flop_counts.to(self.device)
        self._clock_start = time.perf_counter()
        self._last_snapshot = self.token_snapshot()

        summary = self.backbone.param_summary()
        if self.trainer.is_global_zero:
            flops = human_count(int(self.flop_model.per_token(REPORT_CONTEXT)))
            print(
                f"[{self.name}] {self.arch.depth}L d{self.arch.dim} "
                f"heads {self.arch.heads}/{self.arch.kv_heads}\n"
                f"  parameters   total {human_count(summary['total'])} | "
                f"active {human_count(summary['active'])} | "
                f"embedding {human_count(summary['embedding'])}\n"
                f"  layer types  {self._layer_summary()}\n"
                f"  model FLOPs  {flops}/token fwd+bwd at ctx {REPORT_CONTEXT}"
            )
            if self._count_offset[0]:
                print(
                    f"  resuming at  {human_count(self._count_offset[0])} tokens seen"
                    f" | {human_count(self._count_offset[1])} trained"
                )
            for problem in self.resume_warnings():
                print(f"  [resume] {problem}")

    def _layer_summary(self) -> str:
        types = self.arch.layer_types
        moe = self.arch.moe_layers
        return (
            f"{sum(1 for t in types if t == 'full')} full / "
            f"{sum(1 for t in types if t == 'sliding')} sliding, "
            f"{sum(moe)} MoE / {len(moe) - sum(moe)} dense"
        )

    # -- training -------------------------------------------------------- #

    def _micro_batches(self, batch):
        """Split a batch into ``grad_accum`` micro-batches, by its own layout.

        Packed splits on document boundaries, padded on the batch axis.
        """
        if self.grad_accum == 1:
            return [batch]
        if batch.seq_info.packed:
            return split_packed(batch, self.grad_accum)
        return split_padded(batch, self.grad_accum)

    def training_step(self, batch, batch_idx):
        """One optimizer step over the batch's micro-batches, scaled by tokens.

        Collective: nothing here returns early. See docs/guides/training.md.
        """
        opt = self.optimizers()
        sched = self.lr_schedulers()
        opt.zero_grad(set_to_none=True)

        micro = self._micro_batches(batch)
        # Global, not rank-local: DDP averages gradients across unequal token counts.
        local_tokens = sum(m.num_trained for m in micro)
        step_tokens = all_reduce_int(local_tokens, self.token_counts.device)
        empty_step = step_tokens == 0
        denom = step_tokens / max(self.trainer.world_size, 1) if step_tokens else 1.0

        totals = {"ce": 0.0, "n": 0}
        extra_logs: dict[str, float] = {}
        # See docs/internals/moe-router-loss.md.
        router_scale = denom / len(micro)
        for chunk in micro:
            loss, logs = self.backbone.loss(
                chunk.tokens,
                chunk.labels,
                chunk.seq_info,
                reduction="sum",
                router_scale=router_scale,
            )
            self.manual_backward(loss / denom)
            totals["ce"] += logs["ce"].item()
            totals["n"] += int(logs["n_tokens"].item())
            for key in ("z_loss", "router_loss"):
                if key in logs:
                    extra_logs[key] = extra_logs.get(key, 0.0) + logs[key].item()
        # The z-loss arrives as a token sum, the router terms as per-token means.
        if "z_loss" in extra_logs:
            extra_logs["z_loss"] /= max(totals["n"], 1)
        if "router_loss" in extra_logs:
            extra_logs["router_loss"] /= len(micro)

        # Unscale before measuring, so the logged norm tracks the model not the scaler.
        unscale = getattr(self.trainer.precision_plugin, "unscale_gradients", None)
        if callable(unscale):
            unscale(opt)
        grad_norm = self._grad_norm()
        self.clip_gradients(
            opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm"
        )
        # Skipped on an all-masked step, which would apply weight decay and nothing else.
        if not empty_step:
            opt.step()
            # The fp8 weight cache is never invalidated; refresh it once per step.
            refresh_mxfp8_weights(self.backbone)
        opt.zero_grad(set_to_none=True)
        sched.step()

        # A per-optimizer-step rule: the load counter has to cover the whole step.
        moe_logs = self.backbone.update_router_bias()

        seen = sum(m.num_tokens for m in micro)
        # The local trained count, not the reduced one; a snapshot sums across ranks.
        self.token_counts += torch.tensor(
            [seen, local_tokens], dtype=torch.int64, device=self.token_counts.device
        )
        # Charged from the unsplit batch, whose document lengths the split preserves.
        self._flop_counts += self.flop_model.batch_flops(
            seen, document_lengths(batch.seq_info)
        )
        self._log_step(totals, extra_logs, moe_logs, grad_norm, empty_step)

    def _grad_norm(self) -> float:
        """Global L2 gradient norm, pre-clip."""
        norms = [
            torch.linalg.vector_norm(p.grad.detach())
            for p in self.backbone.parameters()
            if p.grad is not None
        ]
        if not norms:
            return 0.0
        return torch.linalg.vector_norm(torch.stack(norms)).item()

    def _log_step(
        self, totals, extra_logs, moe_logs, grad_norm, empty_step: bool = False
    ) -> None:
        """Log the step's loss and auxiliaries, plus progress on the interval.

        ``extra_logs`` arrives already normalized and is logged as given.
        Collective: gated on the *reduced* ``empty_step`` flag.
        """
        if not empty_step:
            loss_per_token = totals["ce"] / max(totals["n"], 1)
            decay = min(0.99, self.global_step / (self.global_step + 10))
            if self.ema_loss < 0:
                self.ema_loss = loss_per_token
            else:
                self.ema_loss = self.ema_loss * decay + loss_per_token * (1 - decay)

            self.log("train/loss", loss_per_token, prog_bar=True, sync_dist=True)
            self.log("train/loss_ema", self.ema_loss, prog_bar=True, sync_dist=False)
            self.log("train/ppl", math.exp(min(loss_per_token, 20)), sync_dist=True)
            if grad_norm is not None:
                self.log("train/grad_norm", grad_norm, sync_dist=True)
            for key, value in extra_logs.items():
                self.log(f"train/{key}", value, sync_dist=True)
            for key, value in (moe_logs or {}).items():
                self.log(f"train/{key}", value, sync_dist=True)

        if self.global_step % self.log_interval == 0:
            self._log_progress()

    def _log_progress(self) -> None:
        """Token totals and both rates, from one globally-reduced snapshot.

        Logged without ``sync_dist``: the values are already global.
        """
        total = self.token_snapshot()
        interval = total - (self._last_snapshot or total)
        self._last_snapshot = total

        self.log("train/tokens_seen", float(total.seen))
        self.log("train/tokens_trained", float(total.trained))
        # Billions as well as raw counts, which `self.log` stores as float32.
        self.log("train/b_tokens_seen", total.seen / 1e9)
        self.log("train/b_tokens_trained", total.trained / 1e9)
        self.log("train/trained_frac", total.trained_frac)

        self.log("train/tokens_per_sec", interval.tokens_per_sec)
        self.log("train/trained_tokens_per_sec", interval.trained_tokens_per_sec)
        self.log("train/tokens_per_sec_avg", total.tokens_per_sec)
        self.log("train/trained_tokens_per_sec_avg", total.trained_tokens_per_sec)
        self.log("train/b_tokens_per_day", interval.b_tokens_per_day)
        self.log("train/b_trained_tokens_per_day", interval.b_trained_tokens_per_day)

    def token_snapshot(self) -> TokenSnapshot:
        """Globally-reduced cumulative progress. Collective: every rank must call."""
        device = self.token_counts.device
        counts = all_reduce_(self.token_counts.detach().clone())
        # Matched to the counters' device, which may not be `self.device` yet.
        flops = all_reduce_(self._flop_counts.detach().clone().to(device))
        seen, trained = counts.tolist()
        model_flops, hardware_flops = flops.tolist()
        return TokenSnapshot(
            seen=self._count_offset[0] + int(seen),
            trained=self._count_offset[1] + int(trained),
            model_flops=self._flop_offset[0] + model_flops,
            hardware_flops=self._flop_offset[1] + hardware_flops,
            elapsed=self._elapsed(),
        )

    def _elapsed(self) -> float:
        """Training wall-clock, excluding the gap between a crash and a restart."""
        if self._clock_start is None:
            return self._elapsed_offset
        return self._elapsed_offset + (time.perf_counter() - self._clock_start)

    # -- resume ---------------------------------------------------------- #

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add what Lightning does not carry. Collective on every rank."""
        total = self.token_snapshot()
        state = {
            "tokens": [total.seen, total.trained],
            "flops": [total.model_flops, total.hardware_flops],
            "elapsed": total.elapsed,
            "ema_loss": self.ema_loss,
            "rng": rng_state(),
        }
        position = loader_state(self._train_loader())
        if position is not None:
            state["loader"] = position
        checkpoint[RESUME_KEY] = state

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Stash the totals; restore the RNG and the dataloader position.

        No trainer attached means ``load_from_checkpoint``, which restores the
        totals only. See docs/guides/training.md.
        """
        self._resume = checkpoint.get(RESUME_KEY)
        if self._resume is None or not self.resume_state:
            return
        if getattr(self, "_trainer", None) is None:
            return
        load_rng_state(self._resume.get("rng", {}))
        load_loader_state(self._train_loader(), self._resume.get("loader"))

    def _apply_resume(self) -> None:
        """Move the stashed checkpoint totals into the offsets."""
        state, self._resume = self._resume, None
        if state is None:
            return
        # The restored buffer holds rank 0's local count; the run's totals are the offsets.
        self.token_counts.zero_()
        self._flop_counts.zero_()
        tokens = state.get("tokens") or (0, 0)
        self._count_offset = (int(tokens[0]), int(tokens[1]))
        flops = state.get("flops") or (0.0, 0.0)
        self._flop_offset = (float(flops[0]), float(flops[1]))
        self._elapsed_offset = float(state.get("elapsed", 0.0))
        self.ema_loss = float(state.get("ema_loss", -1.0))

    def _train_loader(self):
        """The train dataloader, before or after the fit loop has wrapped it.

        ``None`` for a ``LightningDataModule``, which Lightning saves separately.
        """
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return None
        loader = getattr(trainer, "train_dataloader", None)
        if loader is None:
            source = getattr(getattr(trainer, "fit_loop", None), "_data_source", None)
            loader = getattr(source, "instance", None)
        if isinstance(loader, (pl.LightningDataModule, pl.LightningModule)):
            return None
        return loader

    def resume_warnings(self) -> list[str]:
        """What a crash would *not* restore about this run, for the startup log."""
        problems = []
        if not self.resume_state:
            problems.append(
                "resume_state=False: RNG and data position will not be restored"
            )
        elif not resumable(self._train_loader()):
            problems.append(
                "the train dataloader exposes no state_dict()/load_state_dict(), "
                "so a resumed run restarts its data stream from the beginning"
            )
        return problems

    # -- validation ------------------------------------------------------ #

    def validation_step(self, batch, batch_idx):
        if batch.num_trained == 0:
            return
        with torch.no_grad():
            _, logs = self.backbone.loss(
                batch.tokens, batch.labels, batch.seq_info, reduction="sum"
            )
        loss = logs["ce"].item() / max(int(logs["n_tokens"].item()), 1)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/ppl", math.exp(min(loss, 20)), sync_dist=True)

    # -- optimizer ------------------------------------------------------- #

    def _resolved_scheduler_config(self) -> dict:
        resolved = {}
        total = None
        for key, sub in self.scheduler_config.items():
            if isinstance(sub, dict) and sub.get("end", -1) in (None, -1):
                if total is None:
                    total = self._estimated_steps()
                if total > 0:
                    sub = {**sub, "end": total}
            resolved[key] = sub
        return resolved

    def _estimated_steps(self) -> int:
        """Lightning's step estimate -- asked for only if a schedule needs it.

        Reading it costs the saved dataloader position. See docs/guides/training.md.
        """
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, ValueError, OverflowError):
            return 0

    def configure_optimizers(self):
        opt = build_optimizer(
            self.backbone,
            name=self.optimizer_name,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
            eps=self.eps,
            use_mup=self.use_mup,
            base_dim=self.base_dim,
            **self.optimizer_kwargs,
        )
        sched = AnySchedule(opt, config=self._resolved_scheduler_config())
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    # -- generation ------------------------------------------------------ #

    def generate(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample a continuation of ``prompt_ids``, for the preview callback.

        ``kwargs`` reach :meth:`PreviewSampler.generate` unchanged. Runs under the
        precision plugin's forward context. See docs/guides/training.md.
        """
        context = (
            self._trainer.precision_plugin.forward_context()
            if self._trainer is not None
            else contextlib.nullcontext()
        )
        with context:
            return self.sampler.generate(self.backbone, prompt_ids, **kwargs)
