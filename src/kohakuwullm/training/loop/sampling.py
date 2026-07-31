"""Sampling for the in-training preview callback.

KV-cached by default, and drawing from its own RNG stream rather than the
default one. See docs/guides/training.md.
"""

import torch

from kohakuwullm.models.cache import KVCache

# Fixed rather than derived from the run seed, so two runs draw the same noise.
SAMPLE_SEED = 20090220


class PreviewSampler:
    """Top-p sampling over a backbone, with a private per-device RNG stream."""

    def __init__(self, seed: int = SAMPLE_SEED) -> None:
        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

    def generator(self, device: torch.device) -> torch.Generator:
        """This device's generator, created on first use."""
        generator = self._generators.get(device)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[device] = generator
        return generator

    @torch.no_grad()
    def generate(
        self,
        backbone,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend ``prompt_ids`` in the padded layout.

        ``use_cache=False`` re-runs the whole prefix each step; the two must
        agree token for token at ``temperature=0``. See docs/concepts/architecture.md.
        """
        was_training = backbone.training
        generator = generator or self.generator(prompt_ids.device)
        backbone.eval()
        try:
            tokens = prompt_ids.clone()
            cache = (
                KVCache.from_config(
                    backbone.config,
                    batch_size=tokens.shape[0],
                    max_length=tokens.shape[1] + max_new_tokens,
                    device=tokens.device,
                )
                if use_cache
                else None
            )
            step = tokens
            for _ in range(max_new_tokens):
                hidden = backbone(step, cache=cache)
                logits = backbone.head.logits(hidden[:, -1]).float()
                if temperature <= 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    nxt = top_p_sample(probs, top_p, generator)
                tokens = torch.cat([tokens, nxt], dim=1)
                step = nxt if cache is not None else tokens
                if eos_token_id is not None and (nxt == eos_token_id).all():
                    break
        finally:
            # Restored even if sampling raises; gradient checkpointing keys off it.
            backbone.train(was_training)
        return tokens


def top_p_sample(
    probs: torch.Tensor, top_p: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample one token per row from the smallest mass-``top_p`` head of ``probs``."""
    if top_p >= 1.0:
        return torch.multinomial(probs, 1, generator=generator)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    # Keep the first token that crosses the threshold, so the set is never empty.
    keep = cumulative - sorted_probs < top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-9)
    choice = torch.multinomial(sorted_probs, 1, generator=generator)
    return sorted_idx.gather(-1, choice)
