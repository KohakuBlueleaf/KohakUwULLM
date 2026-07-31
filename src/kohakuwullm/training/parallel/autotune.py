"""Measured per-layer costs for the pipeline split.

Times one block of each distinct layer type at the real micro-batch shape, plus
the head and the embedding, so the partitioner divides milliseconds instead of
predicted FLOPs. See docs/internals/pipeline.md.
"""

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from kohakuwullm.models.backbone import LMBackbone
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.models.head import LMHead
from kohakuwullm.models.mxfp8_swap import swap_mxfp8
from kohakuwullm.registry import POSENC, build

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class LayerCosts:
    """Milliseconds and parameters per layer, plus the two endpoints.

    ``layers`` and ``params`` are one entry per layer, in model order.
    See docs/internals/pipeline.md.
    """

    layers: list[float]
    head: float
    embed: float
    params: list[float]
    head_params: float
    embed_params: float

    def summary(self) -> str:
        """One line per distinct cost, for the startup log."""
        rows = [f"head {self.head:.2f} ms", f"embed {self.embed:.2f} ms"]
        seen: dict[float, int] = {}
        for cost in self.layers:
            seen[cost] = seen.get(cost, 0) + 1
        rows += [f"{n}x layer {cost:.2f} ms" for cost, n in seen.items()]
        return "measured stage costs: " + ", ".join(rows)


def layer_key(config, index: int) -> tuple:
    """What makes two layers cost the same: sparsity, backend, window."""
    return (
        config.moe_layers[index],
        repr(config.attn_for(index)),
        config.window_for(index),
    )


def probe_layout(micro_tokens: int, doc_len: int, device) -> SeqInfo:
    """A packed layout of ``micro_tokens`` split into ``doc_len``-token documents.

    Attention is quadratic in the *document* length, not in the buffer length, so
    a probe over one giant document mis-times every attention layer.
    """
    doc_len = max(1, min(doc_len, micro_tokens))
    whole, remainder = divmod(micro_tokens, doc_len)
    lengths = [doc_len] * whole + ([remainder] if remainder else [])
    return SeqInfo.from_lengths(torch.tensor(lengths, dtype=torch.int32), device)


def measure_costs(
    config,
    device,
    micro_tokens: int,
    doc_len: int = 325,
    param_dtype: str = "bf16",
    autocast_dtype: str | None = "bf16",
    mxfp8: bool = False,
    warmup: int = 3,
    iters: int = 8,
) -> LayerCosts:
    """Time every distinct layer type, the head and the embedding, on this device.

    ``doc_len`` is the corpus's typical document length. Costs are wall-clock
    milliseconds for one forward plus backward at ``micro_tokens``.
    """
    dtype = _DTYPES[param_dtype]
    autocast = None if autocast_dtype is None else _DTYPES[autocast_dtype]
    info = probe_layout(micro_tokens, doc_len, device)
    pos_enc = build(
        config.posenc,
        POSENC,
        head_dim=config.head_dim,
        theta=config.rope_theta,
        scaling=config.rope_scaling,
        factor=config.rope_factor,
        partial_rotary_factor=config.rope_partial,
        original_context=config.max_position,
    ).to(device)

    keys = [layer_key(config, i) for i in range(config.depth)]
    measured: dict[tuple, tuple[float, float]] = {}
    for index, key in enumerate(keys):
        if key not in measured:
            measured[key] = _time_block(
                config,
                index,
                device,
                micro_tokens,
                info,
                pos_enc,
                dtype,
                autocast,
                mxfp8,
                warmup,
                iters,
            )
    head_ms, head_params = _time_head(
        config, device, micro_tokens, dtype, autocast, warmup, iters
    )
    embed_ms, embed_params = _time_embed(
        config, device, micro_tokens, dtype, warmup, iters
    )
    return LayerCosts(
        layers=[measured[key][0] for key in keys],
        head=head_ms,
        embed=embed_ms,
        params=[measured[key][1] for key in keys],
        head_params=head_params,
        embed_params=embed_params,
    )


def broadcast_costs(costs: LayerCosts, device, src: int = 0) -> LayerCosts:
    """Replace every rank's costs with rank ``src``'s. Collective.

    Identical cards still time slightly differently, and ranks that disagreed
    about the split would build mismatched stages.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return costs
    flat = torch.tensor(
        [*costs.layers, costs.head, costs.embed], dtype=torch.float64, device=device
    )
    dist.broadcast(flat, src=src)
    depth = len(costs.layers)
    values = flat.tolist()
    return LayerCosts(
        layers=values[:depth],
        head=values[depth],
        embed=values[depth + 1],
        params=costs.params,
        head_params=costs.head_params,
        embed_params=costs.embed_params,
    )


def _time(fn, warmup: int, iters: int) -> float:
    """Median wall-clock ms per call, synchronized either side of each sample."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def _params(module) -> float:
    return float(sum(p.numel() for p in module.parameters()))


def _time_block(
    config,
    index,
    device,
    micro_tokens,
    info,
    pos_enc,
    dtype,
    autocast,
    mxfp8,
    warmup,
    iters,
) -> tuple[float, float]:
    """``(ms, parameters)`` for one block with layer ``index``'s properties."""
    block = LMBackbone.build_block(config, index).to(device=device, dtype=dtype)
    if mxfp8:
        swap_mxfp8(block)
    x = torch.randn(micro_tokens, config.dim, device=device, dtype=dtype)
    x = x.requires_grad_()
    posenc = pos_enc.prepare(info.position_ids, device, dtype)

    def step():
        block.zero_grad(set_to_none=True)
        x.grad = None
        with _maybe_autocast(autocast):
            out = block(x, info, posenc, None)
        out.float().sum().backward()

    ms = _time(step, warmup, iters)
    params = _params(block)
    torch.cuda.empty_cache()
    return ms, params


def _time_head(
    config, device, micro_tokens, dtype, autocast, warmup, iters
) -> tuple[float, float]:
    """``(ms, parameters)`` for the final norm plus the head's loss."""
    head = LMHead(
        config.dim,
        config.vocab_size,
        tie_embeddings=False,
        z_loss_weight=config.z_loss_weight,
        soft_cap=config.logit_soft_cap,
    ).to(device=device, dtype=dtype)
    hidden = torch.randn(micro_tokens, config.dim, device=device, dtype=dtype)
    hidden = hidden.requires_grad_()
    labels = torch.randint(0, config.vocab_size, (micro_tokens,), device=device)

    def step():
        head.zero_grad(set_to_none=True)
        hidden.grad = None
        with _maybe_autocast(autocast):
            loss, _ = head.loss(hidden, labels, reduction="sum")
        loss.backward()

    ms = _time(step, warmup, iters)
    params = _params(head)
    torch.cuda.empty_cache()
    return ms, params


def _time_embed(config, device, micro_tokens, dtype, warmup, iters):
    """``(ms, parameters)`` for the token embedding gather."""
    embed = torch.nn.Embedding(config.vocab_size, config.dim).to(
        device=device, dtype=dtype
    )
    tokens = torch.randint(0, config.vocab_size, (micro_tokens,), device=device)

    def step():
        embed.zero_grad(set_to_none=True)
        embed(tokens).float().sum().backward()

    ms = _time(step, warmup, iters)
    params = _params(embed)
    torch.cuda.empty_cache()
    return ms, params


def _maybe_autocast(dtype):
    """Autocast when one is configured, otherwise a no-op context."""
    if dtype is None:
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=dtype)
