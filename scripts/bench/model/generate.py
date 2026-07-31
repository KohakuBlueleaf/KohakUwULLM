"""KV-cached generation against re-running the prefix every step.

The cache removes work that grows with the square of the sample length, but a
one-token decode step launches the same ~500 kernels either way, so at preview
scale the saving is competing with dispatch overhead rather than with FLOPs.
Every row therefore reports the **host** share of its wall time beside the
speedup: a row that is 95% host-bound is measuring Python, not attention.

Every row also re-checks the equivalence the cache is only useful under, because a
generation path that is fast and wrong is not a result. The check is run in fp32
against a **noise floor measured the same way**: ``lrel`` is the logit difference
between the cached and cache-free paths, ``self`` is the difference between two
identical cache-free runs, and ``ok`` requires the first to sit inside the second.
A bare equality column would report a false alarm on every MoE row -- routed
expert combination is atomic and its forward does not repeat itself. ``agree``
counts leading tokens that survive the benchmark's own bf16, where an untrained
model's top-2 logit gap is at the noise floor and the count means nothing on its
own. See docs/concepts/architecture.md §15.

Usage:
    .venv/bin/python scripts/bench/model/generate.py
    .venv/bin/python scripts/bench/model/generate.py --out out/bench/model/generate.json
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from kohakuwullm.bench import device_name, rel_error
from kohakuwullm.models import KVCache, LMBackbone, get_preset
from kohakuwullm.training.loop.sampling import PreviewSampler

VOCAB = 32768


@dataclass
class Row:
    model: str
    params_m: float
    batch: int
    prompt: int
    new_tokens: int
    cached_ms: float
    plain_ms: float
    cached_host_frac: float
    speedup: float
    cached_tok_s: float
    peak_gib: float
    within_noise: bool
    fp32_cache_rel: float
    fp32_self_rel: float
    agree_tokens: int


def _time(fn, repeats: int) -> tuple[float, float]:
    """Median wall and host milliseconds over ``repeats`` calls, after one warmup."""
    wall, host = [], []
    fn()
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, cpu_start = time.perf_counter(), time.process_time()
        fn()
        cpu = time.process_time() - cpu_start
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - start) * 1e3)
        host.append(cpu * 1e3)
    return statistics.median(wall), statistics.median(host)


def measure(
    name: str, config, batch: int, prompt_len: int, new_tokens: int, repeats: int
) -> Row:
    # The previous row's model is unreferenced by now but still holds its blocks.
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    backbone = LMBackbone(config).cuda().to(torch.bfloat16).eval()
    prompt = torch.randint(0, VOCAB, (batch, prompt_len), device="cuda")
    sampler = PreviewSampler()

    def cached():
        return sampler.generate(
            backbone, prompt, max_new_tokens=new_tokens, temperature=0.0
        )

    def plain():
        return sampler.generate(
            backbone,
            prompt,
            max_new_tokens=new_tokens,
            temperature=0.0,
            use_cache=False,
        )

    agree = _bf16_agreement(backbone, prompt, new_tokens)
    cache_rel, self_rel = _fp32_check(config, prompt)

    torch.cuda.reset_peak_memory_stats()
    cached_ms, cached_host = _time(cached, repeats)
    peak = torch.cuda.max_memory_allocated() / 2**30
    plain_ms, _ = _time(plain, repeats)

    row = Row(
        model=name,
        params_m=sum(p.numel() for p in backbone.parameters()) / 1e6,
        batch=batch,
        prompt=prompt_len,
        new_tokens=new_tokens,
        cached_ms=cached_ms,
        plain_ms=plain_ms,
        cached_host_frac=cached_host / cached_ms,
        speedup=plain_ms / cached_ms,
        cached_tok_s=batch * new_tokens / (cached_ms / 1e3),
        peak_gib=peak,
        within_noise=cache_rel <= max(2 * self_rel, 1e-5),
        fp32_cache_rel=cache_rel,
        fp32_self_rel=self_rel,
        agree_tokens=agree,
    )
    return row


def _greedy_steps(backbone, prompt, steps: int, use_cache: bool):
    """Greedy decode, returning the tokens and every step's logits."""
    tokens = prompt.clone()
    cache = (
        KVCache.from_config(
            backbone.config,
            batch_size=tokens.shape[0],
            max_length=tokens.shape[1] + steps,
            device=tokens.device,
        )
        if use_cache
        else None
    )
    step, logits_per_step = tokens, []
    for _ in range(steps):
        hidden = backbone(step, cache=cache)
        logits = backbone.head.logits(hidden[:, -1]).float()
        logits_per_step.append(logits)
        nxt = logits.argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
        step = nxt if cache is not None else tokens
    return tokens[:, prompt.shape[1] :], logits_per_step


@torch.no_grad()
def _bf16_agreement(backbone, prompt, steps: int) -> int:
    """Leading generated tokens the two paths agree on, in the benchmark dtype."""
    cached, _ = _greedy_steps(backbone, prompt, steps, True)
    plain, _ = _greedy_steps(backbone, prompt, steps, False)
    return int((cached == plain).all(dim=0).cumprod(0).sum())


def _forced_logits(backbone, tokens, prompt_len: int, use_cache: bool):
    """Per-step logits with the context pinned to ``tokens`` rather than resampled.

    Free-running greedy would compare two trajectories; once they part, the logits
    are answers to different questions.
    """
    out = []
    if use_cache:
        cache = KVCache.from_config(
            backbone.config,
            batch_size=tokens.shape[0],
            max_length=tokens.shape[1],
            device=tokens.device,
        )
        chunks = [tokens[:, :prompt_len]]
        chunks += [tokens[:, i : i + 1] for i in range(prompt_len, tokens.shape[1])]
        for chunk in chunks:
            hidden = backbone(chunk, cache=cache)
            out.append(backbone.head.logits(hidden[:, -1]).float())
    else:
        for end in range(prompt_len, tokens.shape[1] + 1):
            hidden = backbone(tokens[:, :end])
            out.append(backbone.head.logits(hidden[:, -1]).float())
    return out


@torch.no_grad()
def _fp32_check(config, prompt, steps: int = 32) -> tuple[float, float]:
    """Cached-vs-plain and plain-vs-plain logit error in fp32, teacher-forced."""
    torch.manual_seed(0)
    backbone = LMBackbone(config).cuda().eval()
    prompt_len = prompt.shape[1]
    tail = torch.randint(
        0, config.vocab_size, (prompt.shape[0], steps), device=prompt.device
    )
    tokens = torch.cat([prompt, tail], dim=1)

    cached = _forced_logits(backbone, tokens, prompt_len, True)
    plain = _forced_logits(backbone, tokens, prompt_len, False)
    again = _forced_logits(backbone, tokens, prompt_len, False)
    cache_rel = statistics.median(rel_error(c, p) for c, p in zip(cached, plain))
    self_rel = statistics.median(rel_error(a, p) for a, p in zip(again, plain))
    return cache_rel, self_rel


def sweep(repeats: int) -> list[Row]:
    rows = []
    for batch in (1, 4):
        for new_tokens in (128, 512):
            rows.append(
                measure(
                    "Kohaku-200M",
                    get_preset("Kohaku-200M", vocab_size=VOCAB),
                    batch,
                    32,
                    new_tokens,
                    repeats,
                )
            )
    rows.append(
        measure(
            "Kohaku-200M w=128",
            get_preset("Kohaku-200M", vocab_size=VOCAB, sliding_window=128),
            1,
            32,
            512,
            repeats,
        )
    )
    rows.append(
        measure(
            "Kohaku-MoE-1B",
            get_preset("Kohaku-MoE-1B", vocab_size=VOCAB),
            1,
            32,
            128,
            repeats,
        )
    )
    rows.append(
        measure(
            "Kohaku-200M long prompt",
            get_preset("Kohaku-200M", vocab_size=VOCAB),
            1,
            1024,
            128,
            repeats,
        )
    )
    return rows


def report(rows: list[Row]) -> None:
    print(f"device: {device_name()}  dtype: bfloat16")
    header = (
        f"{'model':24s} {'B':>2s} {'prompt':>6s} {'new':>5s} "
        f"{'cached ms':>10s} {'plain ms':>10s} {'speedup':>8s} "
        f"{'host%':>6s} {'tok/s':>8s} {'GiB':>6s} {'ok':>4s} "
        f"{'lrel':>8s} {'self':>8s} {'agree':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.model:24s} {r.batch:2d} {r.prompt:6d} {r.new_tokens:5d} "
            f"{r.cached_ms:10.1f} {r.plain_ms:10.1f} {r.speedup:7.2f}x "
            f"{r.cached_host_frac * 100:5.0f}% {r.cached_tok_s:8.1f} "
            f"{r.peak_gib:6.2f} {'yes' if r.within_noise else 'NO':>4s} "
            f"{r.fp32_cache_rel:8.1e} {r.fp32_self_rel:8.1e} "
            f"{r.agree_tokens:3d}/{r.new_tokens:<4d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rows = sweep(args.repeats)
    report(rows)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {"device": device_name(), "rows": [asdict(r) for r in rows]},
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
