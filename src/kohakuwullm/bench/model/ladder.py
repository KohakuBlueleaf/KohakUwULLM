"""Parameter census of a preset ladder, measured rather than solved.

Counts come from a real :class:`~kohakuwullm.models.LMBackbone` built on the meta
device. ``active`` / ``routed_active`` / ``compute_active`` differ; see
docs/performance/benchmarking.md for which you want.
"""

import math
from dataclasses import dataclass

import torch

from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.presets import KOHAKU_LADDER


@dataclass(frozen=True)
class RungCensus:
    """Measured shape and parameter counts for one preset."""

    name: str
    dim: int
    depth: int
    heads: int
    kv_heads: int
    head_dim: int
    mlp_hidden: int
    num_experts: int
    top_k: int
    expert_hidden: int
    num_shared: int
    total: int
    active: int
    embedding: int
    router: int

    @property
    def sparse(self) -> bool:
        return self.num_experts > 0

    @property
    def compute_active(self) -> int:
        """Active parameters that do arithmetic: no embedding *table*, head kept."""
        if get_preset(self.name).tie_embeddings:
            return self.active
        # `embedding` counts table and head together when untied; the head stays.
        return self.active - self.embedding // 2

    @property
    def routed_active(self) -> int:
        """Body parameters one token touches: no embedding, no head, no router."""
        return self.active - self.embedding - self.router

    @property
    def kappa(self) -> float:
        """Sparsity ``top_k / num_experts``; 1.0 for a dense rung."""
        return self.top_k / self.num_experts if self.sparse else 1.0

    @property
    def gqa_ratio(self) -> float:
        return self.heads / self.kv_heads

    @property
    def kv_out(self) -> int:
        """Width of ``k_proj`` / ``v_proj`` -- an MXFP8 alignment axis."""
        return self.kv_heads * self.head_dim

    @property
    def capacity(self) -> float:
        """``sqrt(active * total)``, ``active`` being :attr:`routed_active` when sparse."""
        return math.sqrt(
            (self.total if not self.sparse else self.routed_active) * self.total
        )

    @property
    def capacity_full(self) -> float:
        """``sqrt(active * total)`` with :attr:`active` throughout."""
        return math.sqrt(self.active * self.total)


def census(name: str, **overrides) -> RungCensus:
    """Build ``name`` on the meta device and count it."""
    config = get_preset(name, **overrides)
    # A shape query on meta, so it allocates nothing and needs no device.
    with torch.device("meta"):
        model = LMBackbone(config)
    summary = model.param_summary()
    experts = [m for m in model.modules() if isinstance(m, MoEMLP)]
    router = sum(m.router.weight.numel() for m in experts)
    # The first *dense* block, not `blocks[0]`, which may be an expert layer.
    dense = [b for b, is_moe in zip(model.blocks, config.moe_layers) if not is_moe]
    return RungCensus(
        name=name,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
        # From the built module, not the config, so the width is the resolved one.
        mlp_hidden=dense[0].mlp.hidden if dense else 0,
        num_experts=experts[0].num_experts if experts else 0,
        top_k=experts[0].top_k if experts else 0,
        expert_hidden=experts[0].hidden if experts else 0,
        num_shared=experts[0].num_shared if experts else 0,
        total=summary["total"],
        active=summary["active"],
        embedding=summary["embedding"],
        router=router,
    )


def ladder_census(names: tuple[str, ...] = KOHAKU_LADDER) -> list[RungCensus]:
    """Census every rung, in ladder order."""
    return [census(name) for name in names]
