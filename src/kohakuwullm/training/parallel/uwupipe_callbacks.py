"""LM-specific callbacks for the ``kohakuwupipe`` loop.

``kohakuwupipe`` stays architecture-free, so anything that knows about tokens,
a tokenizer or a sampler lives here. See docs/internals/pipeline.md.
"""

import torch

from kohakuwullm.generation import build_generator
from kohakuwullm.training.parallel.pipeline_lightning import decode_stage
from kohakuwupipe import Callback, get_logger

log = get_logger(__name__)


class SamplePreview(Callback):
    """Generate completions through the schedule every ``every_n_steps``.

    Collective: every rank enters the schedule, and only rank 0 decodes. The
    decode boundary is padded ``(rows, 1)`` and separate from the training one,
    so the module must be in eval for the duration. See docs/guides/generation.md.

    Args:
        module: the :class:`LMPipelineModule` being trained.
        tokenizer: anything with ``__call__`` and ``decode``.
        ranks: from :func:`kohakuwupipe.init_pipeline`.
        prompts: strings to continue; one row each.
        every_n_steps: cadence; 0 disables.
        max_new_tokens / temperature / top_p: sampling controls.
    """

    def __init__(
        self,
        module,
        tokenizer,
        ranks,
        prompts=None,
        every_n_steps: int = 1000,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.module = module
        self.tokenizer = tokenizer
        self.ranks = ranks
        self.prompts = list(prompts) if prompts else ["1girl"]
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._generator = None

    def on_train_batch_end(self, loop, out, batch=None, batch_idx=0) -> None:
        if self.every_n_steps <= 0 or out.index % self.every_n_steps:
            return
        self.preview(loop, out.index)

    def preview(self, loop, step: int) -> None:
        """One collective round of sampling. Every rank must reach this."""
        prompt_ids = self._encode()
        was_training = loop.stage_module.training
        loop.stage_module.eval()
        try:
            tokens = self._build(prompt_ids).generate(
                prompt_ids.to(self.ranks.device),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        finally:
            loop.stage_module.train(was_training)
        if self.ranks.rank:
            return
        for prompt, row in zip(self.prompts, tokens):
            log.info(
                "preview",
                step=step,
                prompt=prompt,
                text=self.tokenizer.decode(row.tolist(), skip_special_tokens=True),
            )

    def _encode(self) -> torch.Tensor:
        """Left-padded prompt ids, one row per prompt, all the same length."""
        rows = [
            self.tokenizer(text, return_tensors="pt")["input_ids"][0]
            for text in self.prompts
        ]
        width = max(row.shape[0] for row in rows)
        pad = self.tokenizer.pad_token_id or 0
        return torch.stack(
            [
                torch.cat(
                    [torch.full((width - row.shape[0],), pad, dtype=row.dtype), row]
                )
                for row in rows
            ]
        )

    def _build(self, prompt_ids: torch.Tensor):
        """The decode-shaped generator, built once per row count."""
        rows = prompt_ids.shape[0]
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
