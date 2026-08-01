"""Pipeline parallelism: cost-balanced stage splitting.

Splits parameters across ranks so each holds 1/N, shipping only boundary
activations. Sequence metadata is not pipelined -- every stage derives its own
SeqInfo from lengths set via :meth:`LMStage.set_seq_info`. See docs/internals/pipeline.md.
"""

import time
import warnings
from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from kohakuwullm.models.backbone import LMBackbone
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.utils import count_active_parameters, count_parameters
from kohakuwupipe.parallel.plan import (
    StagePlan,
    plan_from_layers,
    plan_stages,
)
from kohakuwupipe.parallel.streams import accumulate, accumulator


@dataclass(frozen=True)
class LMStagePlan(StagePlan):
    """A stage plan that names an LM's ends: the embedding table and the head."""

    @property
    def has_embed(self) -> bool:
        return self.is_first

    @property
    def has_head(self) -> bool:
        return self.is_last


def _ffn_hidden(config, index: int | None = None) -> int:
    """Feed-forward width, read off a block the model builds on the meta device.

    Not recomputed: ``resolve_hidden``'s GLU correction and its ``multiple_of``
    ceiling both have to match what was constructed, and a second derivation
    drifts. See docs/internals/pipeline.md.
    """
    with torch.device("meta"):
        block = LMBackbone.build_block(
            config, config.depth - 1 if index is None else index
        )
    return int(block.mlp.hidden)


def _block_cost(config, seq_len: int) -> float:
    """Relative forward cost of one decoder block, per token."""
    dim = config.dim
    kv_dim = config.kv_heads * config.head_dim
    q_dim = config.heads * config.head_dim
    attn_proj = dim * (q_dim + 2 * kv_dim) + q_dim * dim
    # Attention itself is quadratic in sequence length; per token that is linear.
    attn_score = 2 * q_dim * seq_len
    hidden = _ffn_hidden(config)
    if config.moe_every > 0:
        ffn = 3 * dim * hidden * (config.moe_top_k + config.moe_num_shared)
    else:
        ffn = 3 * dim * hidden
    return float(attn_proj + attn_score + ffn)


HEAD_INEFFICIENCY = 1.55


def measure_head_scale(
    config,
    device,
    microbatch_tokens: int,
    warmup: int = 2,
    iters: int = 5,
    probe_tokens: int = 2048,
) -> float:
    """Time one block and the head, and return the head's cost correction.

    The ratio of measured to predicted head cost, for
    ``plan_stages(head_scale=...)``. Timed on a one-layer copy of ``config`` at
    ``min(microbatch_tokens, probe_tokens)`` tokens.
    """
    tokens_used = min(microbatch_tokens, probe_tokens)
    probe = replace(config, depth=1)
    model = LMBackbone(probe).to(device=device, dtype=torch.float32)
    lengths = torch.full((1,), tokens_used, dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, device)
    tokens = torch.randint(0, probe.vocab_size, (tokens_used,), device=device)
    labels = tokens.roll(-1)
    hidden = torch.randn(
        tokens_used, probe.dim, device=device, dtype=torch.float32
    ).requires_grad_(True)

    def block_step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(tokens, info)
        out.float().sum().backward()

    def head_step():
        model.zero_grad(set_to_none=True)
        if hidden.grad is not None:
            hidden.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.head.loss(hidden, labels, reduction="sum")
        loss.backward()

    block_ms = _time(block_step, warmup, iters)
    head_ms = _time(head_step, warmup, iters)

    predicted = float(config.dim * config.vocab_size) / _block_cost(config, tokens_used)
    return (head_ms / max(block_ms, 1e-6)) / max(predicted, 1e-9)


def _time(fn, warmup: int, iters: int) -> float:
    """Mean wall-clock ms per call, synchronized either side of the timed loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def _block_params(config) -> float:
    """Parameters stored per decoder block, counting every expert."""
    dim = config.dim
    kv_dim = config.kv_heads * config.head_dim
    q_dim = config.heads * config.head_dim
    attn = dim * (q_dim + 2 * kv_dim) + q_dim * dim
    hidden = _ffn_hidden(config)
    if config.moe_every > 0:
        experts = config.moe_num_experts + config.moe_num_shared
        ffn = 3 * dim * hidden * experts + dim * config.moe_num_experts
    else:
        ffn = 3 * dim * hidden
    return float(attn + ffn + 2 * dim)


def plan_for(
    config,
    num_stages: int,
    seq_len: int = 2048,
    head_scale: float | None = None,
    layers=None,
    costs=None,
):
    """The stage split for ``config``.

    ``costs`` is a measured :class:`~kohakuwullm.training.parallel.autotune.LayerCosts`
    and replaces the analytic model outright; ``layers`` pins an explicit
    per-stage count and skips both. See docs/internals/pipeline.md.
    """
    scale = HEAD_INEFFICIENCY if head_scale is None else head_scale
    vocab_params = float(config.dim * config.vocab_size)
    layer_cost = costs.layers if costs is not None else _block_cost(config, seq_len)
    head_cost = costs.head if costs is not None else vocab_params * scale
    if layers is not None:
        if isinstance(layers, str) or not hasattr(layers, "__iter__"):
            raise TypeError(f"layers must be a sequence of ints, got {layers!r}")
        counts = [int(n) for n in layers]
        if sum(counts) != config.depth:
            raise ValueError(
                f"layers {counts} sum to {sum(counts)}, not depth {config.depth}"
            )
        if len(counts) != num_stages:
            raise ValueError(f"layers {counts} is not {num_stages} stages")
        return plan_from_layers(
            counts, layer_cost=layer_cost, head_cost=head_cost, plan_cls=LMStagePlan
        )
    return plan_stages(
        depth=config.depth,
        num_stages=num_stages,
        layer_cost=layer_cost,
        head_cost=head_cost,
        layer_params=(costs.params if costs is not None else _block_params(config)),
        head_params=(
            costs.head_params
            if costs is not None
            else (0.0 if config.tie_embeddings else vocab_params)
        ),
        embed_params=costs.embed_params if costs is not None else vocab_params,
        plan_cls=LMStagePlan,
    )


def router_stream_enabled(blocks) -> bool:
    """Whether any block emits a router auxiliary term.

    Answered from the whole backbone, not one stage's slice, so every rank
    agrees on the boundary shape. See docs/internals/moe-router-loss.md.
    """
    return any(getattr(block.mlp, "emits_router_loss", False) for block in blocks)


class LMStage(nn.Module):
    """One contiguous slice of :class:`~kohakuwullm.models.backbone.LMBackbone`.

    ``forward`` is ``tokens (T,) -> hidden (T, D)`` on the first stage and
    ``hidden -> hidden`` elsewhere; the loss is applied outside, through
    :meth:`loss` on the last stage. Call :meth:`set_seq_info` with one entry per
    microbatch before the schedule runs, and match ``boundary_dtype`` to the
    dtype declared to ``PipelineStage(input_args=...)``. See docs/internals/pipeline.md.
    """

    def __init__(
        self,
        backbone,
        plan: StagePlan,
        boundary_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.config = backbone.config
        self.grad_ckpt = backbone.grad_ckpt

        self.embed = backbone.embed if plan.has_embed else None
        self.embed_scale = backbone.embed_scale
        # Parameterless (buffers only), so replicating per stage is free.
        self.pos_enc = backbone.pos_enc
        self.blocks = nn.ModuleList(backbone.blocks[plan.start_layer : plan.end_layer])
        self.final_norm = backbone.final_norm if plan.has_head else None
        self.head = backbone.head if plan.has_head else None

        # A tied weight cannot span two ranks; untie into a stage-local copy.
        if plan.has_head and not plan.has_embed and self.head is not None:
            if self.head.tie_embeddings:
                self._untie_head(backbone)

        # Stages hold different numbers of MoE layers, so a router that reduced
        # over the pipeline group would hang. See docs/internals/pipeline.md.
        if plan.num_stages > 1:
            for block in self.blocks:
                router = getattr(block.mlp, "router", None)
                if router is not None:
                    router.reduce_load = False

        # This stage's sparse layers, and whether the boundary carries their
        # auxiliary terms. See docs/internals/moe-router-loss.md.
        self.moe_blocks = [b for b in self.blocks if hasattr(b.mlp, "router_losses")]
        self.router_stream = plan.num_stages > 1 and router_stream_enabled(
            backbone.blocks
        )

        self.boundary_dtype = None if plan.has_head else boundary_dtype

        self._seq_infos: list[SeqInfo] = []
        self._cache = None
        self._cache_cursor = 0
        self._cursor = 0

    def _untie_head(self, backbone) -> None:
        """Give this stage its own copy of the projection weight."""
        weight = backbone.embed.weight.detach().clone()
        self.head.tie_embeddings = False
        self.head._tied = ()
        self.head.weight = nn.Parameter(weight)
        warnings.warn(
            "pipeline split separates the embedding (stage 0) from the LM head "
            "(last stage); the tied weight was untied into a stage-local copy. "
            "Set tie_embeddings=False in the arch config to make this explicit.",
            stacklevel=3,
        )

    def update_router_bias(self) -> dict[str, torch.Tensor]:
        """Advance this stage's MoE routers; call once per optimizer step.

        Values stay on the device: reading them costs one stall per router.
        See docs/internals/pipeline.md.
        """
        ratios = [
            r
            for block in self.blocks
            if (router := getattr(block.mlp, "router", None)) is not None
            and (r := router.update_bias()) is not None
        ]
        if not ratios:
            return {}
        stacked = torch.stack(ratios)
        return {
            "moe/load_imbalance_max": stacked.max(),
            "moe/load_imbalance_mean": stacked.mean(),
        }

    def set_bias_update_rate(self, rate: float) -> None:
        """Set every router's balancing step size. Zero freezes the bias.

        The load counter and the imbalance metric keep running, so a frozen run
        still reports what the routing is doing.
        See docs/internals/moe-router-loss.md.
        """
        for block in self.blocks:
            router = getattr(block.mlp, "router", None)
            if router is not None:
                router.bias_update_rate = rate

    def global_names(self) -> dict[str, str]:
        """This stage's state-dict keys mapped to their whole-backbone names.

        Only ``blocks.*`` moves: a stage numbers its blocks from zero, the
        backbone from ``plan.start_layer``. See docs/internals/pipeline.md.
        """
        offset = self.plan.start_layer
        names = {}
        for key in self.state_dict():
            group, _, rest = key.partition(".")
            if group == "blocks":
                index, _, tail = rest.partition(".")
                names[key] = f"blocks.{int(index) + offset}.{tail}"
            else:
                names[key] = key
        return names

    def global_state_dict(self) -> dict[str, torch.Tensor]:
        """This stage's tensors under whole-backbone names, detached on the CPU."""
        local = self.state_dict()
        return {
            name: local[key].detach().cpu() for key, name in self.global_names().items()
        }

    def load_global_state_dict(self, state, strict: bool = True) -> None:
        """Take this stage's slice out of a whole-backbone state dict."""
        names = self.global_names()
        missing = [name for name in names.values() if name not in state]
        if missing and strict:
            raise KeyError(
                f"stage {self.plan.index} needs {len(missing)} tensors the "
                f"checkpoint does not carry, first: {missing[:3]}"
            )
        local = {key: state[name] for key, name in names.items() if name in state}
        self.load_state_dict(local, strict=strict)

    def param_summary(self) -> dict[str, int]:
        """Parameter counts for this stage's slice. See docs/internals/pipeline.md."""
        total = count_parameters(self, trainable_only=False)
        embed = 0
        if self.embed is not None:
            embed += self.embed.weight.numel()
        if self.head is not None and not self.head.tie_embeddings:
            embed += self.head.weight.numel()
        return {
            "total": total,
            "active": count_active_parameters(self),
            "embedding": embed,
            "non_embedding": total - embed,
        }

    def set_seq_info(self, seq_infos: list[SeqInfo]) -> None:
        """Register per-microbatch layouts for the step about to run."""
        self._seq_infos = seq_infos
        self._cursor = 0

    def _next_seq_info(self, hidden: torch.Tensor) -> SeqInfo:
        """The layout for the microbatch about to run; one whole doc if unset."""
        if not self._seq_infos:
            return SeqInfo.from_lengths(
                torch.tensor([hidden.shape[0]], dtype=torch.int32), hidden.device
            )
        info = self._seq_infos[self._cursor % len(self._seq_infos)]
        self._cursor += 1
        return info

    def set_cache(self, cache) -> None:
        """Attach one :class:`KVCache` for every microbatch, or ``None`` to detach.

        A list is consumed round-robin, one entry per ``forward``, matching the
        way ``set_seq_info`` feeds per-microbatch layouts.
        """
        self._cache = cache
        self._cache_cursor = 0

    def _next_cache(self):
        if self._cache is None:
            return None
        if not isinstance(self._cache, (list, tuple)):
            return self._cache
        entry = self._cache[self._cache_cursor % len(self._cache)]
        self._cache_cursor += 1
        return entry

    def router_losses(self) -> torch.Tensor | None:
        """Summed MoE auxiliary terms from this stage's last forward, or ``None``."""
        return accumulate(None, (b.mlp.router_losses() for b in self.moe_blocks))

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None):
        """Run this stage's slice: tokens or hidden in, hidden out.

        Returns ``(hidden, aux)`` while training with ``router_stream`` on, the
        second stream carrying every stage's summed MoE auxiliary terms.
        See docs/internals/moe-router-loss.md.
        """
        cache = self._next_cache()
        if cache is None:
            seq_info = self._next_seq_info(x)
        else:
            # Later stages carry hidden, not ids; the cache only needs the layout.
            seq_info = cache.seq_info(x if x.dim() == 2 else x[..., 0])
        if self.embed is not None:
            x = self.embed(x)
            if self.embed_scale != 1.0:
                x = x * self.embed_scale
        steps = x.shape[1] if x.dim() > 1 else x.shape[0]
        posenc = self.pos_enc.prepare(seq_info.position_ids, x.device, x.dtype)
        for offset, block in enumerate(self.blocks):
            # Global index: the cache is sized for the whole model, this stage owns
            # a contiguous slice of its layers.
            layer = (
                None if cache is None else cache.layer(self.plan.start_layer + offset)
            )
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, seq_info, posenc, layer, use_reentrant=False
                )
            else:
                x = block(x, seq_info, posenc, layer)
        # Every stage advances its own copy, so the next step's positions continue.
        if cache is not None:
            cache.advance(steps)
        if self.final_norm is not None:
            x = self.final_norm(x)
        if self.boundary_dtype is not None:
            x = x.to(self.boundary_dtype)
        if not (self.router_stream and self.training):
            return x
        carried = accumulator(x.device) if aux is None else aux
        return x, accumulate(carried, (self.router_losses(),))

    def loss(self, hidden: torch.Tensor, labels: torch.Tensor):
        """Head loss, for the last stage only."""
        if self.head is None:
            raise RuntimeError("loss() called on a stage that does not own the head")
        return self.head.loss(hidden, labels, reduction="sum")


class AutocastStage(nn.Module):
    """Runs a stage forward and loss under autocast while parameters stay fp32."""

    def __init__(self, module: nn.Module, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype

    @property
    def router_stream(self) -> bool:
        return self.module.router_stream

    def router_losses(self):
        return self.module.router_losses()

    def set_seq_info(self, seq_infos) -> None:
        self.module.set_seq_info(seq_infos)

    def loss(self, hidden, labels):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module.loss(hidden, labels)

    def forward(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=self.dtype):
            return self.module(*args, **kwargs)
