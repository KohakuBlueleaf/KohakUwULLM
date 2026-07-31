"""Optimizer construction: parameter grouping, muP, and the optimizer registry.

Weight decay applies to matrices but not to vectors and not to the input
embedding; Muon adds a second axis to the split. See docs/internals/optimizers.md.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from kohakuwullm.registry import OPTIMIZER, resolve
from kohakuwullm.training.optim.fused_adamw import FusedAdamW
from kohakuwullm.training.optim.muon import MuonW
from kohakuwullm.training.optim.torchao_optim import QUANTIZED_ADAMW

OPTIMIZER.register("adamw")(optim.AdamW)
OPTIMIZER.register("adam")(optim.Adam)
OPTIMIZER.register("sgd")(optim.SGD)
OPTIMIZER.register("fused_adamw")(FusedAdamW)


def _is_decay_param(name: str, param: nn.Parameter) -> bool:
    """Matrices decay; vectors do not."""
    if param.ndim <= 1:
        return False
    return not name.endswith("expert_bias")


def is_embedding(name: str) -> bool:
    """Whether ``name`` is the input embedding table, matched on path component."""
    return "embed" in name.split(".")


def is_hidden_matrix(name: str, param: nn.Parameter) -> bool:
    """Whether Muon may orthogonalize this parameter.

    True for a linear map between two *feature* spaces; ``embed`` / ``head`` /
    ``router`` are excluded, matched on path components. See docs/internals/optimizers.md.
    """
    if param.ndim < 2:
        return False
    return not {"embed", "head", "router"} & set(name.split("."))


def group_parameters(
    model: nn.Module,
    weight_decay: float,
    use_mup: bool = False,
    base_dim: int = 256,
    lr: float = 3e-4,
    muon_lr: float | None = None,
    muon_filter=is_hidden_matrix,
    muon_mup_exponent: float = 0.0,
    decay_embeddings: bool = False,
    embed_lr: float | None = None,
) -> list[dict]:
    """Split parameters into decay / no-decay groups, optionally muP-scaled.

    Under muP a hidden matrix's lr scales as ``base_dim / fan_in`` and its decay
    as the inverse. ``muon_lr`` (spectral-norm units) pulls the hidden matrices
    into their own ``use_muon`` group before the muP split, and
    ``muon_mup_exponent`` scales that lr by ``(base_dim / fan_in) ** e``.
    ``decay_embeddings`` puts the input embedding back in the decay group.
    ``embed_lr`` gives the input embedding its own group at that lr. See
    docs/internals/optimizers.md.
    """
    decay, no_decay, embed, mup_groups = [], [], [], {}
    # Keyed by fan_in only when an exponent asks for it.
    muon_by_fan_in: dict[int | None, list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if muon_lr is not None and muon_filter(name, param):
            key = param.shape[-1] if muon_mup_exponent else None
            muon_by_fan_in.setdefault(key, []).append(param)
            continue
        if embed_lr is not None and is_embedding(name):
            embed.append(param)
            continue
        if not _is_decay_param(name, param) or (
            not decay_embeddings and is_embedding(name)
        ):
            no_decay.append(param)
            continue
        # Same predicate as the decay exclusion above, not a second spelling of it.
        if use_mup and not is_embedding(name):
            fan_in = param.shape[-1]
            scale = base_dim / fan_in
            mup_groups.setdefault(round(scale, 6), []).append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr": lr})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
    if embed:
        wd = weight_decay if decay_embeddings else 0.0
        groups.append({"params": embed, "weight_decay": wd, "lr": embed_lr})
    for scale, params in sorted(mup_groups.items()):
        groups.append(
            {
                "params": params,
                # Decay scales with the lr, keeping the decay-to-update ratio fixed.
                "weight_decay": weight_decay / scale,
                "lr": lr * scale,
            }
        )
    for fan_in, params in sorted(muon_by_fan_in.items(), key=lambda kv: kv[0] or 0):
        group_lr = muon_lr * (
            1.0 if fan_in is None else (base_dim / fan_in) ** muon_mup_exponent
        )
        groups.append(
            {
                "params": params,
                "lr": group_lr,
                # Rescaled to the Muon lr, so the per-step shrink matches AdamW's.
                "weight_decay": weight_decay * lr / group_lr,
                "use_muon": True,
            }
        )
    if muon_lr is not None:
        for group in groups:
            group.setdefault("use_muon", False)
    return groups


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    eps: float = 1e-8,
    use_mup: bool = False,
    base_dim: int = 256,
    muon_lr: float = 0.02,
    muon_mup_exponent: float = 0.0,
    decay_embeddings: bool = False,
    embed_lr: float | None = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer over correctly-grouped parameters."""
    # `resolve`, not `build`: the groups are assembled below and passed at the end.
    cls = resolve(name, OPTIMIZER)
    groups = group_parameters(
        model,
        weight_decay,
        use_mup=use_mup,
        base_dim=base_dim,
        lr=lr,
        muon_lr=muon_lr if getattr(cls, "muon_groups", False) else None,
        muon_mup_exponent=muon_mup_exponent,
        decay_embeddings=decay_embeddings,
        embed_lr=embed_lr,
    )
    # The quantized variants are matched by name: they are deferred factories.
    if cls in (optim.AdamW, optim.Adam, FusedAdamW, MuonW) or name in QUANTIZED_ADAMW:
        kwargs.setdefault("betas", betas)
        kwargs.setdefault("eps", eps)
    if cls in (optim.AdamW, optim.Adam):
        # Not `fused=True`: it would disable Lightning's AMP gradient clipping.
        kwargs.setdefault("foreach", torch.cuda.is_available())
    return cls(groups, lr=lr, **kwargs)
