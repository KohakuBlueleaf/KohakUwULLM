"""LM-specific callbacks for the ``kohakuwupipe`` loop.

``kohakuwupipe`` stays architecture-free, so anything that knows about tokens,
a tokenizer or a sampler lives here. See docs/internals/pipeline.md.
"""

import time

import torch
import torch.distributed as dist
from anyschedule.utils import get_scheduler
from tqdm.auto import tqdm

from kohakuwullm.generation import LocalGenerator, build_generator
from kohakuwullm.models import LMBackbone
from kohakuwullm.training.parallel.pipeline_lightning import decode_stage
from kohakuwupipe import Callback


class RouterBiasSchedule(Callback):
    """Scale every router's bias update rate by an AnySchedule factor each step.

    The factor multiplies the rate the routers were built with, so a factor of
    zero freezes the bias while the load counter and the imbalance metric keep
    reporting. Resuming re-applies the factor for the restored step.
    See docs/internals/moe-router-loss.md.

    Args:
        module: the :class:`LMPipelineModule` being trained.
        config: one AnySchedule scheduler config, shaped as ``SCHEDULER_CONFIG``'s
            entries are. Read standalone, without an optimizer.
    """

    def __init__(self, module, config: dict) -> None:
        self.module = module
        self.schedule = get_scheduler(dict(config))
        self.base: float | None = None

    def apply(self, step: int) -> None:
        if self.base is not None:
            self.module.inner.set_bias_update_rate(self.base * self.schedule(step))

    def on_train_start(self, loop) -> None:
        self.base = self.module.inner.bias_update_rate()
        self.apply(loop.global_step)

    def on_train_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        self.apply(out.index)


class SamplePreview(Callback):
    """Generate completions through the schedule every ``every_n_steps``.

    Collective: every rank enters the schedule, and only rank 0 decodes. The
    decode boundary is padded ``(rows, 1)`` and separate from the training one,
    so the module must be in eval for the duration. See docs/guides/generation.md.

    Args:
        module: the :class:`LMPipelineModule` being trained.
        tokenizer: anything with ``__call__`` and ``decode``.
        ranks: from :func:`kohakuwupipe.init_pipeline`.
        prompts: ``(name, text)`` pairs; each is sampled ``samples`` times.
        samples: rows generated per prompt, and the decode batch width.
        every_n_steps: cadence; 0 disables.
        at_start: also preview on the first step, before any training.
        report: ``(step, rows) -> None`` on rank 0, where a row is
            ``(name, prompt, index, text)``. The text is never logged.
        local: gather the whole model and decode on one card. Off falls back to
            pipelined decode, which costs one pipeline traversal per token.
        forward_only: drive pipelined decode with a plain send/recv forward
            rather than a training schedule.
        max_new_tokens / temperature / top_p / top_k / min_p: sampling
            controls. ``max_new_tokens=None`` fills the model's context.
    """

    def __init__(
        self,
        module,
        tokenizer,
        ranks,
        prompts=None,
        every_n_steps: int = 1000,
        at_start: bool = True,
        samples: int = 16,
        report=None,
        max_new_tokens: int | None = 128,
        temperature: float = 0.35,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        local: bool = True,
        forward_only: bool = True,
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.ranks = ranks
        self.prompts = (
            list(prompts) if prompts else [("default", "target: <|long|>\ntag: 1girl")]
        )
        self.every_n_steps = every_n_steps
        self.at_start = at_start
        self.samples = samples
        self.report = report
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.local = local
        self.forward_only = forward_only
        self._generator = None
        self._local_model = None

    def on_train_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        first = self.at_start and batch_idx == 0
        if not first and (self.every_n_steps <= 0 or out.index % self.every_n_steps):
            return
        self.preview(loop, out.index)

    def preview(self, loop, step: int) -> None:
        """One collective round per prompt. Every rank must reach every round."""
        began = time.perf_counter()
        was_training = loop.stage_module.training
        loop.stage_module.eval()
        rows = []
        gathered = self._gathered(loop) if self.local else None
        bar = tqdm(
            self.prompts,
            desc=f"preview@{step}",
            unit="prompt",
            leave=False,
            disable=None if self.ranks.rank == 0 else True,
        )
        try:
            for name, text in bar:
                if gathered is not None and self.ranks.rank:
                    break
                prompt_ids = self._encode(text)
                generator = gathered or self._build()
                tokens = generator.generate(
                    prompt_ids.to(self.ranks.device),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                if self.ranks.rank:
                    continue
                rows += [
                    (
                        name,
                        text,
                        index,
                        self.tokenizer.decode(row.tolist(), skip_special_tokens=True),
                    )
                    for index, row in enumerate(tokens)
                ]
        finally:
            bar.close()
            loop.stage_module.train(was_training)
        if self.ranks.rank == 0:
            tokens_out = sum(len(r[3]) > 0 for r in rows)
            print(
                f"[preview@{step}] {time.perf_counter() - began:.2f}s "
                f"local={self.local} prompts={len(self.prompts)} rows={tokens_out}",
                flush=True,
            )
        if self.ranks.rank == 0 and self.report is not None:
            self.report(step, rows)

    def _gathered(self, loop):
        """A whole-model `LocalGenerator` on rank 0, or ``None`` off rank 0.

        Collective: every rank must reach the gather.
        See docs/guides/generation.md.
        """
        local = self.module.inner.global_state_dict()
        merged = {}
        if dist.is_available() and dist.is_initialized():
            parts = [None] * self.ranks.world
            dist.all_gather_object(parts, local)
            for part in parts:
                merged.update(part)
        else:
            merged.update(local)
        if self.ranks.rank:
            return None
        if self._local_model is None:
            self._local_model = LMBackbone(
                self.module.config, head_kwargs=self.module.head_kwargs
            ).to(self.ranks.device, dtype=self.module.param_dtype)
        missing, _ = self._local_model.load_state_dict(merged, strict=False)
        if missing:
            raise RuntimeError(f"preview gather missed {len(missing)}: {missing[:3]}")
        self._local_model.eval()
        return LocalGenerator(self._local_model)

    def _encode(self, text: str) -> torch.Tensor:
        """One prompt repeated ``samples`` times, the decode batch's shape."""
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        return ids.unsqueeze(0).expand(self.samples, -1).contiguous()

    def _build(self):
        """The decode-shaped generator, built once."""
        rows = self.samples
        if self._generator is None:
            stage = decode_stage(
                self.module.stage_module,
                self.module.plan,
                self.module.config,
                self.ranks.rank,
                self.ranks.world,
                self.ranks.device,
                rows,
                param_dtype=self.module.param_dtype,
            )
            self._generator = build_generator(
                stage=stage,
                head_module=self.module.inner,
                rank=self.ranks.rank,
                world=self.ranks.world,
                microbatches=1,
                autocast_dtype=self.module.autocast_dtype,
                forward_only_decode=self.forward_only,
            )
        return self._generator
