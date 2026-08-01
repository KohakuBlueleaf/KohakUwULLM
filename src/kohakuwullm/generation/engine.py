"""Generation engines: one for a whole model, one for a pipeline of stages.

Both take ``(B, S)`` padded prompts and return ``(B, S + n)``. See docs/guides/generation.md.
"""

from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist

from kohakuwullm.models.cache import KVCache
from kohakuwullm.training.parallel.pipeline_lightning import build_schedule

SAMPLE_SEED = 20090220
WARMUP_STEPS = 4


def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all but the ``top_k`` highest logits per row."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = logits.topk(top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero all but the smallest head of ``probs`` carrying mass ``top_p``."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs * keep)


def filter_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Zero tokens below ``min_p`` times the row's peak probability."""
    if min_p <= 0.0:
        return probs
    floor = probs.max(dim=-1, keepdim=True).values * min_p
    return probs * (probs >= floor)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One token per row. ``temperature <= 0`` is greedy and ignores every filter.

    Filters compose in the order top-k, top-p, min-p: top-k bounds the candidate
    count, top-p bounds their mass, min-p bounds their ratio to the peak.
    """
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    # fp32: top-p sums the whole vocabulary, and 65536 fp16 terms lose percent-
    # level accuracy. See docs/guides/generation.md.
    logits = logits.float()
    probs = torch.softmax(filter_top_k(logits, top_k) / temperature, dim=-1)
    probs = filter_min_p(filter_top_p(probs, top_p), min_p)
    probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
    # multinomial asserts on the device; a finite check here names the cause.
    if not torch.isfinite(probs).all():
        raise RuntimeError(
            "sampling probabilities are not finite; logits carried inf/nan "
            f"(temperature={temperature}, top_p={top_p}, top_k={top_k})"
        )
    return torch.multinomial(probs, 1, generator=generator)


def token_budget(max_new_tokens: int | None, prompt_len: int, max_position: int) -> int:
    """Tokens to generate. ``None`` fills the model's context; anything past it is
    clamped, because RoPE and the KV cache are only defined inside it."""
    room = max(max_position - prompt_len, 0)
    return room if max_new_tokens is None else min(max_new_tokens, room)


def advance(tokens, nxt, finished, eos_token_id):
    """Append one sampled token per row, holding already-finished rows at EOS.

    Returns ``(tokens, finished)``. Without the hold, a row that emitted EOS keeps
    sampling, so no batch ever satisfies an all-rows stop.
    See docs/guides/generation.md.
    """
    if eos_token_id is not None:
        nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
        finished = finished | (nxt == eos_token_id)
    return torch.cat([tokens, nxt], dim=1), finished


class Generator:
    """Common sampling state: a private RNG stream and the token-choice rule."""

    def __init__(self, seed: int = SAMPLE_SEED) -> None:
        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

    def generator(self, device: torch.device) -> torch.Generator:
        """This device's generator, created on first use."""
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._generators[device] = gen
        return gen

    def _choose(self, logits, generator, **opts):
        return sample(logits, generator=generator, **opts)

    def generate(self, prompt_ids, **kwargs):
        raise NotImplementedError


class LocalGenerator(Generator):
    """Every rank holds the whole model; generation is rank-local and collective-free.

    Args:
        backbone: an :class:`LMBackbone` with an ``LMHead``.
    """

    def __init__(self, backbone, seed: int = SAMPLE_SEED) -> None:
        super().__init__(seed)
        self.backbone = backbone

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int | None = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sample a continuation of ``prompt_ids`` ``(B, S)``.

        ``max_new_tokens=None`` runs until every row has emitted EOS or the
        model's context is full. ``use_cache=False`` re-runs the whole prefix each
        step; the two must agree token for token at ``temperature=0``.
        """
        was_training = self.backbone.training
        generator = generator or self.generator(prompt_ids.device)
        rows, prompt_len = prompt_ids.shape
        budget = token_budget(
            max_new_tokens, prompt_len, self.backbone.config.max_position
        )
        self.backbone.eval()
        try:
            tokens = prompt_ids.clone()
            finished = torch.zeros(rows, 1, dtype=torch.bool, device=prompt_ids.device)
            cache = (
                KVCache.from_config(
                    self.backbone.config,
                    batch_size=rows,
                    max_length=prompt_len + budget,
                    device=prompt_ids.device,
                )
                if use_cache
                else None
            )
            step = tokens
            for _ in range(budget):
                # New iteration for cudagraph trees; see docs/guides/generation.md.
                torch.compiler.cudagraph_mark_step_begin()
                hidden = self.backbone(step, cache=cache)
                logits = self.backbone.head.logits(hidden[:, -1]).float()
                nxt = self._choose(
                    logits,
                    generator,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
                tokens, finished = advance(tokens, nxt, finished, eos_token_id)
                step = tokens[:, -1:].contiguous() if cache is not None else tokens
                if bool(finished.all()):
                    break
        finally:
            self.backbone.train(was_training)
        return tokens


class PipelineGenerator(Generator):
    """The model is split over ranks; one prompt row per microbatch.

    Every rank must call :meth:`generate` the same number of times with the same
    shapes; only stage 0 needs the prompt and only the last stage produces logits.

    Args:
        stage: this rank's ``PipelineStage``.
        head_module: the stage module, for ``head.logits`` on the last rank.
        rank / world: pipeline position and size.
        autocast_dtype: the stage's own autocast dtype; the head runs under it
            because it sits outside the stage forward the schedule wraps.
    """

    def __init__(
        self,
        stage,
        head_module,
        rank: int,
        world: int,
        seed: int = SAMPLE_SEED,
        microbatches: int | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(seed)
        self.microbatches = microbatches
        self.stage = stage
        self.head_module = head_module
        self.rank = rank
        self.world = world
        self.autocast_dtype = autocast_dtype
        self.head_context = (
            nullcontext
            if autocast_dtype is None
            else partial(torch.autocast, "cuda", dtype=autocast_dtype)
        )
        self.schedule = None
        self._chunks = 0

    def cache_dtype(self) -> torch.dtype:
        """Dtype the key/value buffers are stored in."""
        return self.head_module.boundary_dtype or self.autocast_dtype or torch.bfloat16

    def _caches(self, chunks: int, rows: int, length: int, device):
        """One :class:`KVCache` per microbatch, sized for ``length`` positions."""
        return [
            KVCache.from_config(
                self.head_module.config,
                batch_size=rows,
                max_length=length,
                device=device,
                dtype=self.cache_dtype(),
            )
            for _ in range(chunks)
        ]

    @torch.no_grad()
    def prepare(self, chunks: int, rows: int, device) -> None:
        """Build the schedule for ``chunks`` microbatches and take its first step.

        The step runs against throwaway caches, so the extra forward a schedule
        makes when it first runs never reaches the caches :meth:`generate`
        decodes from. Collective, and a no-op once built.
        See docs/guides/generation.md.
        """
        if self.schedule is not None and self._chunks == chunks:
            return
        self.schedule = build_schedule(self.stage, chunks, loss_fn=None, kind="gpipe")
        self._chunks = chunks
        self.head_module.set_cache(self._caches(chunks, rows, WARMUP_STEPS, device))
        try:
            if self.rank == 0:
                self.schedule.step(
                    torch.zeros(rows * chunks, 1, dtype=torch.long, device=device)
                )
            else:
                self.schedule.step()
        finally:
            self.head_module.set_cache(None)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int | None = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a continuation of ``prompt_ids`` ``(B, S)`` across the stages.

        Cached decode: one token per step, so the pipeline boundary shape is
        constant, which is what ``PipelineStage`` freezes. Every rank must call
        this the same number of times with the same shapes.
        ``max_new_tokens=None`` runs until every row has emitted EOS or the
        model's context is full.
        """
        device = prompt_ids.device
        last = self.world - 1
        rows, prompt_len = prompt_ids.shape
        total = prompt_len + token_budget(
            max_new_tokens, prompt_len, self.head_module.config.max_position
        )
        was_training = self.head_module.training
        self.head_module.eval()
        generator = generator or self.generator(device)

        chunks = self.microbatches or rows
        if rows % chunks:
            raise ValueError(f"{rows} rows do not split into {chunks} microbatches")
        per_chunk = rows // chunks
        try:
            self.prepare(chunks, per_chunk, device)
            schedule = self.schedule
            self.head_module.set_cache(self._caches(chunks, per_chunk, total, device))
            tokens = prompt_ids.clone()
            finished = torch.zeros(rows, 1, dtype=torch.bool, device=device)
            for pos in range(total - 1):
                # `clone`, not `contiguous`: a one-element slice reports itself
                # contiguous while keeping the source row stride.
                step = tokens[:, pos : pos + 1].clone(
                    memory_format=torch.contiguous_format
                )
                # New iteration for cudagraph trees; see docs/guides/generation.md.
                torch.compiler.cudagraph_mark_step_begin()
                hidden = schedule.step(step) if self.rank == 0 else schedule.step()
                if pos < prompt_len - 1:
                    continue
                nxt = torch.zeros(rows, 1, dtype=torch.long, device=device)
                if self.rank == last and hidden is not None:
                    # The head sits outside the stage forward the schedule wraps.
                    with self.head_context():
                        logits = self.head_module.head.logits(hidden[:, -1]).float()
                    nxt = self._choose(
                        logits,
                        generator,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                    )
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(nxt, src=last)
                # Every rank sees the same `nxt`, so every rank stops on the same
                # step without another collective.
                tokens, finished = advance(tokens, nxt, finished, eos_token_id)
                if bool(finished.all()):
                    break
        finally:
            self.head_module.set_cache(None)
            self.head_module.train(was_training)
        return tokens


def build_generator(
    backbone=None, stage=None, head_module=None, rank: int = 0, world: int = 1, **kwargs
) -> Generator:
    """Select the engine once, at build time, from whether a stage was given."""
    if stage is None:
        if backbone is None:
            raise ValueError("a whole-model generator needs `backbone`")
        return LocalGenerator(backbone, **kwargs)
    if head_module is None:
        raise ValueError("a pipeline generator needs `head_module`")
    return PipelineGenerator(stage, head_module, rank, world, **kwargs)
