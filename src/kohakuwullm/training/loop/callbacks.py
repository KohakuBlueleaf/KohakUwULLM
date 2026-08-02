"""Training callbacks: progress, sample previews and throughput accounting.

All three are interval-gated and rank-0-only where they produce output; none is
required for training to run. See docs/guides/training.md.
"""

import warnings

import torch
from lightning.pytorch.callbacks import Callback, TQDMProgressBar


def _stage_plan(module):
    """A pipeline :class:`StagePlan` on this module or a wrapper deep, if any."""
    for start in (module, getattr(module, "backbone", None)):
        candidate = start
        # Bounded rather than `while`: a wrapper that returns itself would spin.
        for _ in range(4):
            if candidate is None:
                break
            plan = getattr(candidate, "plan", None)
            if getattr(plan, "num_stages", 1) > 1:
                return plan
            candidate = getattr(candidate, "module", None) or getattr(
                candidate, "_orig_mod", None
            )
    return None


class StepProgressBar(TQDMProgressBar):
    """Progress against the run's total step target, driven by ``global_step``."""

    def __init__(self, total_steps: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total_steps = total_steps

    def init_train_tqdm(self):
        """The bar, retotalled from the epoch's batches to the run's steps."""
        bar = super().init_train_tqdm()
        bar.total = self.total_steps
        bar.unit = "step"
        return bar

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # Deliberately not `super()`, which retotals to the epoch's batch count.
        bar = self.train_progress_bar
        bar.total = self.total_steps
        bar.initial = trainer.global_step
        bar.set_description("train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        bar = self.train_progress_bar
        # Assigned, not incremented: a step does not land on every batch.
        if trainer.global_step != bar.n:
            bar.n = min(trainer.global_step, self.total_steps)
            bar.refresh()
        self._update_metrics(trainer, pl_module)

    def _update_metrics(self, trainer, pl_module) -> None:
        """Refresh the postfix, minus the keys not worth the width."""
        metrics = self.get_metrics(trainer, pl_module)
        # `v_num` is the logger's run id: constant for the whole run.
        metrics.pop("v_num", None)
        self.train_progress_bar.set_postfix(metrics)


class SampleLogCallback(Callback):
    """Generate a few completions every ``every_n_steps`` and log them.

    Only rank 0 generates, unless the module sets ``generate_is_collective``, in
    which case every rank runs the loop and rank 0 reports. Disables itself under
    a pipeline split no single rank can drive. See docs/guides/training.md.

    Args:
        tokenizer: used to decode (and to encode the prompts).
        prompts: raw prompt strings; ``None`` uses a small TIPO-shaped default.
        every_n_steps: cadence; ``<= 0`` disables.
        max_new_tokens / temperature / top_p / top_k / min_p: sampling
            settings. ``max_new_tokens=None`` fills the model's context.
    """

    DEFAULT_PROMPTS = [
        "target: <|long|> <|tag_to_long|>\ntag: 1girl, solo, looking at viewer",
        "quality: masterpiece\nrating: general\ntarget: <|short|>\ntag: ",
        "target: <|very_long|> <|short_to_long|>\nshort: A cat sitting on a windowsill.\nlong:",
    ]

    def __init__(
        self,
        tokenizer,
        prompts: list[str] | None = None,
        every_n_steps: int = 1000,
        max_new_tokens: int | None = 96,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self._last_step = -1
        self._disabled = False
        self._collective = False

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = trainer.global_step
        # Every gate below this point reads rank-invariant state.
        if self.every_n_steps <= 0 or step % self.every_n_steps != 0:
            return
        if step == self._last_step:
            return
        self._last_step = step
        if self._disabled or self._disable_if_unavailable(pl_module):
            return
        if not (self._collective or trainer.is_global_zero):
            return
        self._log_samples(trainer, pl_module, step)

    def _disable_if_unavailable(self, pl_module) -> bool:
        """Turn previews off, once and loudly, when this rank cannot generate."""
        if not callable(getattr(pl_module, "generate", None)):
            self._disabled = True
            warnings.warn(
                f"sample previews disabled: {type(pl_module).__name__} has no "
                "generate()",
                stacklevel=2,
            )
            return True
        self._collective = bool(getattr(pl_module, "generate_is_collective", False))
        plan = _stage_plan(pl_module)
        if plan is not None and not self._collective:
            self._disabled = True
            warnings.warn(
                f"sample previews disabled: the model is split across "
                f"{plan.num_stages} pipeline stages and this rank holds layers "
                f"{plan.start_layer}..{plan.end_layer - 1} "
                f"(embedding={plan.has_embed}, head={plan.has_head}), so no rank "
                "can run a full forward on its own. Route generation through the "
                "pipeline schedule, or preview from a checkpoint instead.",
                stacklevel=2,
            )
            return True
        return False

    @torch.no_grad()
    def _log_samples(self, trainer, pl_module, step: int) -> None:
        """Complete every prompt, print them, and mirror to wandb if it is there.

        Under a collective ``generate`` every rank runs the loop; only rank 0
        decodes and reports.
        """
        texts = []
        for prompt in self.prompts:
            ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            ids = ids.to(pl_module.device)
            out = pl_module.generate(
                ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            completion = self.tokenizer.decode(
                out[0, ids.shape[1] :], skip_special_tokens=False
            )
            texts.append((prompt, completion))

        if not trainer.is_global_zero:
            return

        banner = "\n".join(
            f"--- prompt ---\n{p}\n--- completion ---\n{c}" for p, c in texts
        )
        print(f"\n[samples @ step {step}]\n{banner}\n")

        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb

            table = wandb.Table(columns=["step", "prompt", "completion"])
            for prompt, completion in texts:
                table.add_data(step, prompt, completion)
            experiment.log({"samples": table}, commit=False)
        except ImportError:
            pass


class ThroughputCallback(Callback):
    """Log token rates, MFU and HFU, both per interval and cumulative.

    Counters come from the trainer's ``token_snapshot()``; ``peak_flops``
    defaults to the detected device's :data:`PEAK_TFLOPS` entry.
    """

    # fp32-accumulate ceilings, measured rather than taken from the whitepaper.
    PEAK_TFLOPS = {
        "NVIDIA GeForce RTX 5090": 270.0,
        "NVIDIA GeForce RTX 4090": 165.2,
        "NVIDIA H100": 989.0,
        "NVIDIA A100": 312.0,
    }

    def __init__(
        self, every_n_steps: int = 50, peak_flops: float | None = None
    ) -> None:
        self.every_n_steps = every_n_steps
        self.peak_flops = peak_flops
        self._last = None
        self._disabled = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        """Resolve the device's peak rate, unless the caller supplied one."""
        if self.peak_flops is None and torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name
            for key, value in self.PEAK_TFLOPS.items():
                if key in name:
                    self.peak_flops = value * 1e12
                    break

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log this interval's rates against the previous snapshot."""
        step = trainer.global_step
        if self._disabled or self.every_n_steps <= 0 or step % self.every_n_steps:
            return
        snapshot = getattr(pl_module, "token_snapshot", None)
        if not callable(snapshot):
            self._disabled = True
            warnings.warn(
                f"throughput logging disabled: {type(pl_module).__name__} has no "
                "token_snapshot()",
                stacklevel=2,
            )
            return

        total = snapshot()
        # The first call only establishes the baseline.
        if self._last is None:
            self._last = total
            return
        interval = total - self._last
        self._last = total

        pl_module.log("perf/tokens_per_sec", interval.tokens_per_sec, sync_dist=False)
        pl_module.log(
            "perf/trained_tokens_per_sec",
            interval.trained_tokens_per_sec,
            sync_dist=False,
        )
        pl_module.log("perf/tokens_per_sec_avg", total.tokens_per_sec, sync_dist=False)
        pl_module.log("perf/trained_frac", interval.trained_frac, sync_dist=False)
        pl_module.log(
            "perf/b_tokens_per_day", interval.b_tokens_per_day, sync_dist=False
        )
        if self.peak_flops:
            pl_module.log("perf/mfu", interval.mfu(self.peak_flops), sync_dist=False)
            pl_module.log("perf/mfu_avg", total.mfu(self.peak_flops), sync_dist=False)
            pl_module.log("perf/hfu", interval.hfu(self.peak_flops), sync_dist=False)
