"""LM-specific callbacks for the ``kohakuwupipe`` loop.

``kohakuwupipe`` stays architecture-free, so anything that knows about tokens,
a tokenizer or a sampler lives here. See docs/internals/pipeline.md.
"""

import torch
from tqdm.auto import tqdm

from kohakuwullm.generation import build_generator
from kohakuwullm.training.parallel.pipeline_lightning import decode_stage
from kohakuwupipe import Callback


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
        report: ``(step, rows) -> None`` on rank 0, where a row is
            ``(name, prompt, index, text)``. The text is never logged.
        max_new_tokens / temperature / top_p: sampling controls.
    """

    def __init__(
        self,
        module,
        tokenizer,
        ranks,
        prompts=None,
        every_n_steps: int = 1000,
        samples: int = 16,
        report=None,
        max_new_tokens: int = 128,
        temperature: float = 0.35,
        top_p: float = 0.95,
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.ranks = ranks
        self.prompts = (
            list(prompts) if prompts else [("default", "target: <|long|>\ntag: 1girl")]
        )
        self.every_n_steps = every_n_steps
        self.samples = samples
        self.report = report
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._generator = None

    def on_train_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        self.preview(loop, out.index)

    def preview(self, loop, step: int) -> None:
        """One collective round per prompt. Every rank must reach every round."""
        was_training = loop.stage_module.training
        loop.stage_module.eval()
        rows = []
        bar = tqdm(
            self.prompts,
            desc=f"preview@{step}",
            unit="prompt",
            leave=False,
            disable=None if self.ranks.rank == 0 else True,
        )
        try:
            for name, text in bar:
                prompt_ids = self._encode(text)
                tokens = self._build().generate(
                    prompt_ids.to(self.ranks.device),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
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
        if self.ranks.rank == 0 and self.report is not None:
            self.report(step, rows)

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
            )
        return self._generator
