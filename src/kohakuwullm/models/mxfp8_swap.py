"""Replace a built backbone's projections with MXFP8 linears, in place, and account.

``nn.Linear`` children are selected by parent module type; matmul held as a bare
parameter is declared through :mod:`kohakuwullm.models.mxfp8_protocol`. Every
non-fp8 outcome is charged to a bucket and blocks the run.

See docs/concepts/architecture.md and docs/internals/mxfp8.md.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear
from kohakuwullm.models.components.attention import BaseAttention
from kohakuwullm.models.components.mlp import GELUMLP, GLUMLP
from kohakuwullm.models.components.moe_variants import LatentMoEMLP
from kohakuwullm.models.head import LMHead
from kohakuwullm.models.mxfp8_protocol import CONVERT, DECLARE, Matmul

ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
# Fused and unfused GLU layouts plus the plain MLP's pair; only the attributes a
# given module actually has are visited.
MLP_PROJECTIONS = ("w_in", "gate_proj", "up_proj", "w_out", "fc1", "fc2")
# `latent_moe`'s compression pair, whose parent is neither attention nor an MLP.
LATENT_PROJECTIONS = ("down", "up")

# Parent type -> the `nn.Linear` attributes it may own. Pairs, not a dict keyed by
# type, because `isinstance` has to see a group as one alternative.
LINEAR_PARENTS = (
    (BaseAttention, ATTN_PROJECTIONS),
    ((GLUMLP, GELUMLP), MLP_PROJECTIONS),
    (LatentMoEMLP, LATENT_PROJECTIONS),
)

# Where a tensor's per-token arithmetic is charged. `never` -- head, embedding,
# routers -- is not a failure; the other three all block.
BUCKETS = ("fp8", "skipped", "unreached", "never")


@dataclass
class SwapReport:
    """What the surgery did, what it refused, and what it could not reach.

    ``swapped`` names *tensors*; ``modules`` names the modules holding an fp8
    cache, which is the count :func:`refresh_mxfp8_weights` returns.
    """

    swapped: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    unreached: list[tuple[str, str]] = field(default_factory=list)
    by_design: list[str] = field(default_factory=list)
    unaccounted: list[str] = field(default_factory=list)
    # Left bf16 by a `scope` argument: reported, and not blocking.
    out_of_scope: list[str] = field(default_factory=list)
    mac: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))

    @property
    def blocking(self) -> bool:
        """Whether the model would train as a bf16/fp8 mixture if this were accepted."""
        return bool(self.skipped or self.unreached or self.unaccounted)

    @property
    def total_mac(self) -> int:
        return sum(self.mac.values())

    def share(self, bucket: str) -> float:
        """``bucket``'s fraction of per-token matmul; 0 when there is no matmul."""
        total = self.total_mac
        return self.mac[bucket] / total if total else 0.0

    def summary(self) -> str:
        total = self.total_mac
        lines = [
            f"MXFP8: {len(self.swapped)} tensor(s) in fp8 over {len(self.modules)} "
            f"module(s), {self.share('fp8'):.1%} of {total / 1e6:.1f}M per-token MAC"
        ]
        for label, entries in (
            ("SKIPPED", self.skipped),
            ("UNREACHED", self.unreached),
        ):
            for name, reason in entries:
                lines.append(f"  {label} {name}: {reason}")
        for bucket in ("skipped", "unreached"):
            if self.mac[bucket]:
                lines.append(
                    f"  {bucket} matmul: {self.mac[bucket] / 1e6:.1f}M per token, "
                    f"{self.share(bucket):.1%} of the model's"
                )
        if self.unaccounted:
            # Outside the MAC census: an unclaimed tensor has no declared cost.
            lines.append(
                f"  UNACCOUNTED (not in the census above): {len(self.unaccounted)} "
                f"matmul-shaped parameter(s): {', '.join(self.unaccounted)}"
            )
        if self.by_design:
            lines.append(
                f"  bf16 by design: {len(self.by_design)} tensor(s), "
                f"{self.share('never'):.1%} of per-token MAC"
            )
        return "\n".join(lines)


def _reject(module: nn.Linear) -> str | None:
    """Why ``MXFP8Linear`` cannot stand in for this layer, or ``None``."""
    if module.bias is not None:
        return "has a bias; MXFP8Linear is bias-free"
    # Only `in_features` blocks; `out_features` is zero-padded exactly.
    if module.in_features % 128:
        return f"in_features={module.in_features} is not a multiple of 128"
    return None


def _qualify(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _resolve(module: nn.Module, path: str) -> torch.Tensor:
    """Follow a dotted attribute path, so a declaration can name a child's tensor."""
    obj: object = module
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _by_design(module: nn.Module) -> dict[str, Matmul]:
    """Matmul refused by type rather than by declaration: the head and embeddings.

    Routers are declared by the MoE layer that owns them, not here.
    """
    if isinstance(module, LMHead):
        vocab, dim = module.projection.shape
        return {
            "projection": Matmul(
                vocab * dim, "the head contracts over the vocabulary", never=True
            )
        }
    if isinstance(module, nn.Embedding):
        return {"weight": Matmul(0, "an embedding is a gather, not a GEMM", never=True)}
    return {}


def _in_scope(qualified: str, scope: tuple[str, ...] | None) -> bool:
    return scope is None or any(part in qualified for part in scope)


def _visit_linear_children(
    parent: nn.Module,
    prefix: str,
    linear_cls: type,
    report: SwapReport,
    accounted: set[int],
    scope: tuple[str, ...] | None = None,
) -> None:
    for types, names in LINEAR_PARENTS:
        if not isinstance(parent, types):
            continue
        # First match wins and returns: a module matching two entries would
        # otherwise have shared attribute names visited and charged twice.
        return _swap_children(
            parent, prefix, names, linear_cls, report, accounted, scope
        )


def _swap_children(
    parent: nn.Module,
    prefix: str,
    names: tuple[str, ...],
    linear_cls: type,
    report: SwapReport,
    accounted: set[int],
    scope: tuple[str, ...] | None = None,
) -> None:
    for name in names:
        child = getattr(parent, name, None)
        qualified = _qualify(prefix, name)
        if isinstance(child, nn.Linear) and not _in_scope(qualified, scope):
            # Accounted, or the closing sweep would call the weight unclaimed.
            accounted.add(id(child.weight))
            report.out_of_scope.append(qualified)
            report.mac["never"] += child.in_features * child.out_features
            continue
        if isinstance(child, linear_cls):
            # Already fp8: a second swap over the same model.
            accounted.add(id(child.weight))
            report.swapped.append(qualified)
            report.modules.append(qualified)
            report.mac["fp8"] += child.in_features * child.out_features
            continue
        if not isinstance(child, nn.Linear):
            continue
        accounted.add(id(child.weight))
        mac = child.in_features * child.out_features
        reason = _reject(child)
        if reason is not None:
            report.skipped.append((qualified, reason))
            report.mac["skipped"] += mac
            continue
        replacement = linear_cls(
            child.in_features, child.out_features, dtype=child.weight.dtype
        )
        with torch.no_grad():
            replacement.weight.copy_(child.weight)
        replacement.to(child.weight.device)
        setattr(parent, name, replacement)
        accounted.add(id(replacement.weight))
        report.swapped.append(qualified)
        report.modules.append(qualified)
        report.mac["fp8"] += mac


def _visit_declared(
    parent: nn.Module,
    prefix: str,
    report: SwapReport,
    accounted: set[int],
    convert_declared: bool,
    scope: tuple[str, ...] | None = None,
) -> None:
    """Account, and where possible convert, the matmul this module declares."""
    declared: dict[str, Matmul] = _by_design(parent)
    declare = getattr(parent, DECLARE, None)
    if declare is not None:
        declared = {**declared, **declare()}
    if not declared:
        return
    for name in declared:
        accounted.add(id(_resolve(parent, name)))

    for name, entry in declared.items():
        if entry.never:
            report.by_design.append(_qualify(prefix, name))
            report.mac["never"] += entry.mac_per_token

    # Before the refusal logic: an out-of-scope module is not being asked to
    # convert, so a refusal it would have raised is not a finding about this run.
    if not _in_scope(prefix, scope):
        for name, entry in declared.items():
            if entry.never:
                continue
            report.out_of_scope.append(_qualify(prefix, name))
            report.mac["never"] += entry.mac_per_token
        return

    blocked = {
        n: e for n, e in declared.items() if e.refusal is not None and not e.never
    }
    eligible = {n: e for n, e in declared.items() if e.refusal is None}
    if blocked:
        # One refused tensor refuses the whole module; a sibling's own reason is
        # replaced by a pointer to it.
        for name, entry in blocked.items():
            report.skipped.append((_qualify(prefix, name), entry.refusal))
            report.mac["skipped"] += entry.mac_per_token
        for name, entry in eligible.items():
            report.skipped.append(
                (
                    _qualify(prefix, name),
                    f"a sibling tensor in {prefix or '<root>'} is refused",
                )
            )
            report.mac["skipped"] += entry.mac_per_token
        return
    if not eligible:
        return

    convert = getattr(parent, CONVERT, None) if convert_declared else None
    if convert is None:
        why = (
            f"{type(parent).__name__} declares it fp8-eligible but has no "
            f"{CONVERT}(), so nothing converted it"
            if convert_declared
            else "the caller passed convert_declared=False"
        )
        for name, entry in eligible.items():
            report.unreached.append((_qualify(prefix, name), why))
            report.mac["unreached"] += entry.mac_per_token
        return
    convert()
    report.modules.append(prefix)
    for name, entry in eligible.items():
        report.swapped.append(_qualify(prefix, name))
        report.mac["fp8"] += entry.mac_per_token


def _sweep_unaccounted(
    model: nn.Module, report: SwapReport, accounted: set[int]
) -> None:
    """Every matmul-shaped parameter no type rule and no declaration claimed.

    Rank is the filter: a 1-D parameter is a norm gain or an attention sink,
    never a matmul. See docs/concepts/architecture.md.
    """
    for name, param in model.named_parameters():
        if param.dim() >= 2 and id(param) not in accounted:
            report.unaccounted.append(f"{name}{tuple(param.shape)}")


def swap_mxfp8(
    model: nn.Module,
    linear_cls: type = MXFP8Linear,
    convert_declared: bool = True,
    scope: tuple[str, ...] | None = None,
) -> SwapReport:
    """Convert every reachable matmul to ``linear_cls`` / fp8, in place, and account.

    Weights are copied, so parameter values and dtype are unchanged. See
    docs/concepts/architecture.md.

    Args:
        model: the built backbone, surgery applied in place.
        linear_cls: stand-in for ``nn.Linear``; a variant quantizer is how the
            round-to-nearest arm is built.
        scope: qualified-name substrings to restrict conversion to, e.g.
            ``("attn.",)``. Everything outside is charged to ``out_of_scope``
            and left bf16, which does not block. ``None`` converts everything.
        convert_declared: convert the bare-parameter matmul modules declare.
            ``False`` charges it to ``unreached``, so the report still refuses.
    """
    report = SwapReport()
    accounted: set[int] = set()
    for prefix, parent in model.named_modules():
        _visit_linear_children(parent, prefix, linear_cls, report, accounted, scope)
        _visit_declared(parent, prefix, report, accounted, convert_declared, scope)
    _sweep_unaccounted(model, report, accounted)
    return report


def refresh_mxfp8_weights(model: nn.Module) -> int:
    """Requantize every MXFP8 weight cache. Call after **every** optimizer step.

    Returns the number of modules refreshed; assert it against
    ``len(report.modules)``, which counts modules and not tensors.
    """
    count = 0
    for module in model.modules():
        refresh = getattr(module, "refresh_quantized_weight", None)
        if callable(refresh):
            refresh()
            count += 1
    return count
