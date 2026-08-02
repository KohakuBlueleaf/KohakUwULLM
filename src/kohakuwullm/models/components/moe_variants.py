"""Router and expert variants beyond the DeepSeekMoE default.

None of these is the default; each exists so an ablation is a config change.
See docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kohakuwullm.models.components.mlp import GLUMLP
from kohakuwullm.models.components.moe import MoEMLP, expert_counts
from kohakuwullm.registry import MLP, ROUTER


@ROUTER.register("sinkhorn")
class SinkhornRouter(nn.Module):
    """Balanced assignment by Sinkhorn normalization instead of a bias nudge.

    Alternates row and column normalization of the score matrix toward
    doubly-stochastic. See docs/concepts/architecture.md.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 8,
        n_iters: int = 3,
        temperature: float = 0.05,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        **_unused,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_iters = n_iters
        self.temperature = temperature
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)
        self.expert_bias = None
        self.load_accum = None
        self.aux_loss = None
        self.z_loss = None
        self.emits_loss = False

    @torch.no_grad()
    def _sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        out = torch.exp(cost / self.temperature)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-9)
        for _ in range(self.n_iters):
            out = out / out.sum(dim=0, keepdim=True).clamp_min(1e-9)
            out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return out

    def forward(self, x: torch.Tensor):
        logits = F.linear(x.float(), self.weight.float())
        # Assignment from the balanced matrix, weight from the raw scores.
        with torch.no_grad():
            assignment = self._sinkhorn(logits)
        topk_idx = assignment.topk(self.top_k, dim=-1).indices
        topk_weight = logits.softmax(-1).gather(1, topk_idx)
        if self.norm_topk_prob and self.top_k > 1:
            topk_weight = topk_weight / topk_weight.sum(-1, keepdim=True).clamp_min(
                1e-9
            )
        return topk_idx, (topk_weight * self.routed_scaling_factor).to(x.dtype)

    def route(self, x: torch.Tensor):
        """``(topk_idx, topk_weight, counts)`` -- the interface ``MoEMLP`` calls."""
        idx, weight = self(x)
        return idx, weight, expert_counts(idx.reshape(-1), self.num_experts)

    @torch.no_grad()
    def update_bias(self):
        return None


@ROUTER.register("expert_choice")
class ExpertChoiceRouter(nn.Module):
    """Experts pick tokens, instead of tokens picking experts.

    **Not wired to :class:`~kohakuwullm.models.components.moe.MoEMLP`** --
    :meth:`route` raises. See docs/concepts/architecture.md.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 8,
        capacity_factor: float = 1.0,
        **_unused,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)
        self.expert_bias = None
        self.load_accum = None
        self.aux_loss = None
        self.z_loss = None
        self.emits_loss = False

    def forward(self, x: torch.Tensor):
        tokens = x.shape[0]
        scores = F.linear(x.float(), self.weight.float()).softmax(-1)
        capacity = max(
            1, int(tokens * self.top_k / self.num_experts * self.capacity_factor)
        )
        capacity = min(capacity, tokens)
        # (num_experts, capacity): which tokens each expert claims.
        chosen = scores.T.topk(capacity, dim=-1).indices
        weights = scores.T.gather(1, chosen)
        # Re-expressed as a per-token list so the shared expert path is unchanged.
        flat_tokens = chosen.reshape(-1)
        flat_experts = torch.arange(
            self.num_experts, device=x.device
        ).repeat_interleave(capacity)
        return (flat_tokens, flat_experts, weights.reshape(-1).to(x.dtype)), None

    def route(self, x: torch.Tensor):
        raise NotImplementedError(
            "expert_choice emits a (token, expert) pair list; MoEMLP's dispatch "
            "needs a (T, slots) index matrix and derives a pair's token as "
            "pair_index // slots, which expert choice deliberately breaks. "
            "expert_sort must take an explicit per-pair token index first."
        )

    @torch.no_grad()
    def update_bias(self):
        return None


class PerLayerEmbedding(nn.Module):
    """Gemma-3n-style per-layer embeddings (PLE). See docs/concepts/architecture.md.

    Args:
        vocab_size: token vocabulary.
        depth: number of blocks that will consume a slice.
        ple_dim: width of the per-layer vector before projection.
        dim: model width to project into.
    """

    def __init__(self, vocab_size: int, depth: int, ple_dim: int, dim: int) -> None:
        super().__init__()
        self.depth = depth
        self.ple_dim = ple_dim
        # One table over all layers' slices, so the lookup is a single gather.
        self.table = nn.Embedding(vocab_size, depth * ple_dim)
        nn.init.normal_(self.table.weight, std=ple_dim**-0.5)
        self.proj = nn.ModuleList(
            nn.Linear(ple_dim, dim, bias=False) for _ in range(depth)
        )
        for layer in self.proj:
            nn.init.zeros_(layer.weight)  # start as a no-op on the residual stream
        self.norm = nn.ModuleList(
            nn.LayerNorm(ple_dim, elementwise_affine=False) for _ in range(depth)
        )

    def lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(T,) -> (T, depth, ple_dim)``; done once per forward, not per layer."""
        flat = self.table(tokens)
        return flat.reshape(*tokens.shape, self.depth, self.ple_dim)

    def for_layer(self, cache: torch.Tensor, layer: int) -> torch.Tensor:
        """The additive contribution for one block, ``(T, dim)``."""
        return self.proj[layer](self.norm[layer](cache[..., layer, :]))


@ROUTER.register("relu")
class ReLURouter(nn.Module):
    """ReMoE: ReLU gating with no top-k, sparsity held by an adaptive L1 penalty.

    An expert is active iff its score is positive, and the L1 multiplier follows
    ``lambda *= alpha ** sign(target - observed)``. Load balancing folds into the
    same penalty. See docs/concepts/architecture.md, including its cost against ``topk``.

    Args:
        dim: model width.
        num_experts: expert count.
        top_k: *target* average experts per token, not enforced per token.
        l1_alpha: multiplicative step on the L1 coefficient.
        l1_init: initial L1 coefficient.
        max_slots: hard cap on active experts per token, bounding the dispatch
            buffer through the early phase where nearly every expert is active.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 8,
        l1_alpha: float = 1.2,
        l1_init: float = 1e-8,
        max_slots: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.l1_alpha = l1_alpha
        self.max_slots = max_slots or min(num_experts, 4 * top_k)
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)
        self.register_buffer("l1_coeff", torch.tensor(float(l1_init)))
        self.aux_loss: torch.Tensor | None = None
        self.z_loss: torch.Tensor | None = None
        # Every training forward emits the L1 penalty; it is not optional.
        self.emits_loss = True

    def forward(self, x: torch.Tensor):
        """``x`` is ``(T, D)`` flat. Returns ``(topk_idx, topk_weight)``.

        A fixed ``max_slots`` width, so the dispatch keeps a static shape.
        """
        scores = F.relu(F.linear(x.float(), self.weight.float()))
        active = (scores > 0).float()
        target = 1.0 - self.top_k / self.num_experts
        observed = 1.0 - active.mean()

        if self.training:
            # Per-expert weighting folds balancing into the same term.
            share = active.mean(0).detach()
            penalty = (scores * share.unsqueeze(0)).sum(-1).mean()
            # clone(), not the live buffer: `mul_` below bumps its version.
            self.aux_loss = self.l1_coeff.clone() * penalty
            # Device-side; `.item()` here would stall the host every micro-batch.
            self.l1_coeff.mul_(self.l1_alpha ** torch.sign(target - observed))
        else:
            self.aux_loss = None
        self.z_loss = None

        top = scores.topk(self.max_slots, dim=-1)
        weight = top.values
        # Inactive slots go to the sentinel bucket, which has no weight matrix,
        # so their rows are never computed.
        idx = torch.where(weight > 0, top.indices, self.num_experts)
        return idx, weight.to(x.dtype)

    @property
    def num_buckets(self) -> int:
        """Dispatch buckets, one more than the experts: the last is the sentinel."""
        return self.num_experts + 1

    def route(self, x: torch.Tensor):
        idx, weight = self(x)
        return idx, weight, expert_counts(idx.reshape(-1), self.num_buckets)

    @torch.no_grad()
    def update_bias(self) -> torch.Tensor | None:
        """No bias to update: sparsity and balance both live in the L1 term."""
        return None


@MLP.register("latent_moe")
class LatentMoEMLP(nn.Module):
    """Kimi-K3 style: routed experts live in a compressed latent space.

    Only the expert *input* dimension moves. See docs/concepts/architecture.md.

    Args:
        dim: model width.
        latent: expert input width. ``None`` uses ``dim // alpha``.
        alpha: compression factor when ``latent`` is not given.
        hidden / multiple_of: geometry of one routed expert.
        num_experts / top_k / num_shared / router: as :class:`MoEMLP`.
    """

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        ratio: float = 4.0,
        multiple_of: int = 64,
        latent: int | None = None,
        alpha: int = 2,
        num_experts: int = 64,
        top_k: int = 6,
        num_shared: int = 1,
        router="topk",
        dense_fallback: bool = False,
        **router_kwargs,
    ) -> None:
        super().__init__()
        self.latent = latent or max(multiple_of, dim // alpha)
        self.down = nn.Linear(dim, self.latent, bias=False)
        self.up = nn.Linear(self.latent, dim, bias=False)
        # The inner block owns routing and the expert bank at latent width; its
        # own shared expert is off because this class provides them at full width.
        self.inner = MoEMLP(
            self.latent,
            hidden=hidden,
            ratio=ratio,
            multiple_of=multiple_of,
            num_experts=num_experts,
            top_k=top_k,
            num_shared=0,
            router=router,
            router_dim=dim,
            dense_fallback=dense_fallback,
            **router_kwargs,
        )
        self.shared = (
            GLUMLP(dim, hidden=hidden, ratio=ratio, multiple_of=multiple_of)
            if num_shared > 0
            else None
        )

    @property
    def router(self):
        """The inner router, so ``update_bias`` and loss collection still reach it."""
        return self.inner.router

    @property
    def w_in(self):
        """Delegated so depth-scaled init reaches the expert stacks it owns."""
        return self.inner.w_in

    @property
    def w_out(self):
        return self.inner.w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Experts see the compressed activation; routing reads the full residual.
        routed = self.up(self.inner(self.down(x), route_on=x))
        if self.shared is not None:
            routed = routed + self.shared(x)
        return routed

    def router_losses(self) -> torch.Tensor | None:
        return self.inner.router_losses()

    @property
    def emits_router_loss(self) -> bool:
        return self.inner.emits_router_loss
