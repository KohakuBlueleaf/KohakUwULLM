"""Muon: momentum orthogonalized by Newton-Schulz, with an internal AdamW.

Matrix parameters take the orthogonalized step; everything else takes AdamW,
selected per parameter group by a ``use_muon`` flag. See docs/internals/optimizers.md.
"""

import math
from collections.abc import Callable
from functools import partial

import torch
import torch._dynamo

from kohakuwullm.compile_utils import raise_recompile_limit
from kohakuwullm.kernels.optim.adamw16 import adamw16_step
from kohakuwullm.kernels.optim.stochastic_round import stochastic_round_update_
from kohakuwullm.registry import OPTIMIZER

# Jordan quintic coefficients, for five iterations.
NS_COEFFS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

# Per-iteration cubic coefficients; the schedule fixes the step count at five.
NS_CUBIC5: tuple[tuple[float, float], ...] = (
    (3.3656576, -3.3420992),
    (2.5744352, -1.4957376),
    (2.5368962, -1.4312570),
    (2.4418906, -1.2764040),
    (2.2230472, -0.9630650),
)


def newton_schulz(
    grad: torch.Tensor,
    steps: int = 5,
    dtype: torch.dtype = torch.bfloat16,
    coeffs: tuple[float, float, float] = NS_COEFFS,
) -> torch.Tensor:
    """Approximate the orthogonal polar factor ``U V^T`` of ``grad``.

    Batched over every leading dimension, so an ``(experts, out, in)`` stack is
    orthogonalized one expert at a time. Returns a tensor in ``dtype``.
    """
    a, b, c = coeffs
    # Normalize inside the iteration's convergence radius, reducing in fp32.
    norm = (
        torch.linalg.vector_norm(grad, dim=(-2, -1), keepdim=True, dtype=torch.float32)
        .clamp_min(1e-7)
        .mul_(1.01)
    )
    # Copy before the in-place divide: without Nesterov `grad` is the momentum buffer.
    x = grad.to(dtype=dtype, copy=True).div_(norm.to(dtype))
    # Iterate on the short side; the polar factor of the transpose is the transpose.
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


# Step grouping for a matrix wide enough that gram-space accumulation pays.
NS_PHASES_GRAM: tuple[int, ...] = (2, 3)
# One step per group -- the unfactored iteration.
NS_PHASES_DIRECT: tuple[int, ...] = (1, 1, 1, 1, 1)


def newton_schulz_cubic(
    grad: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    schedule: tuple[tuple[float, float], ...] = NS_CUBIC5,
    phases: tuple[int, ...] = NS_PHASES_DIRECT,
) -> torch.Tensor:
    """Polar factor via the cubic schedule -- two matmuls per step, not three.

    ``phases`` groups consecutive steps and must sum to ``len(schedule)``, which
    fixes the iteration count. See docs/internals/optimizers.md.
    """
    if sum(phases) != len(schedule):
        raise ValueError(
            f"phases {phases} sum to {sum(phases)}, not the schedule's "
            f"{len(schedule)} steps"
        )
    # Same normalization as `newton_schulz`.
    norm = (
        torch.linalg.vector_norm(grad, dim=(-2, -1), keepdim=True, dtype=torch.float32)
        .clamp_min(1e-7)
        .mul_(1.01)
    )
    x = grad.to(dtype=dtype, copy=True).div_(norm.to(dtype))
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    start = 0
    for length in phases:
        gram = x @ x.mT
        product = None
        for offset, (a, b) in enumerate(schedule[start : start + length]):
            step = gram * b
            step.diagonal(dim1=-2, dim2=-1).add_(a)
            product = step if product is None else step @ product
            # The last step in a group never reads its own gram back.
            if offset + 1 < length:
                gram = step @ gram @ step
        x = product @ x
        start += length
    return x.mT if transposed else x


def _nesterov_direction(
    buffer: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    beta: float,
    one_minus: float,
) -> None:
    """Advance the momentum EMA and write the look-ahead into ``out``."""
    buffer.lerp_(grad, one_minus)
    out.copy_(grad.lerp(buffer, beta))


def _heavyball_direction(
    buffer: torch.Tensor,
    grad: torch.Tensor,
    out: torch.Tensor,
    beta: float,
    one_minus: float,
) -> None:
    """Advance the momentum EMA and copy it into ``out``."""
    buffer.lerp_(grad, one_minus)
    out.copy_(buffer)


def _decay_and_step(
    param: torch.Tensor, update: torch.Tensor, keep: torch.Tensor, alpha: torch.Tensor
) -> None:
    """Decoupled decay and the orthogonalized step, in one pass over ``param``.

    ``keep`` and ``alpha`` are 0-d fp32 tensors, not floats.
    """
    param.mul_(keep).addcmul_(update, alpha)


def _decay_and_step_sr(
    param: torch.Tensor,
    update: torch.Tensor,
    decay: float,
    alpha: float,
    seed: int,
    rng_offset: int,
) -> None:
    """The same writeback, rounded stochastically into a low-precision parameter.

    ``decay`` and ``alpha`` are plain floats here, not 0-d tensors.
    """
    # cubic5's output is strided; the kernel needs it contiguous.
    if not update.is_contiguous():
        update = update.contiguous()
    stochastic_round_update_(
        param, update, seed, decay=decay, alpha=alpha, rng_offset=rng_offset
    )


def orthogonal_update_scale(shape: torch.Size, mode: str, rms_target: float) -> float:
    """Multiplier on the orthogonalized update, in units of ``lr``."""
    fan_out, fan_in = shape[-2], shape[-1]
    match mode:
        case "spectral":
            # The gradient dualized under the RMS->RMS operator norm.
            return max(1.0, fan_out / fan_in) ** 0.5
        case "rms":
            # Moonlight's rule: force the update's RMS to `rms_target`.
            return rms_target * max(fan_out, fan_in) ** 0.5
        case _:
            raise ValueError(f"unknown update_scale {mode!r}; use spectral or rms")


@OPTIMIZER.register("muon")
class MuonW(torch.optim.Optimizer):
    """Muon for the hidden matrices, decoupled-decay AdamW for everything else.

    Every parameter group must carry ``use_muon``, which ``optim.group_parameters``
    produces; a missing flag raises. See docs/internals/optimizers.md.

    Args:
        lr: spectral-norm step for the Muon groups; the group dicts carry their own.
        momentum: Muon's SGD momentum.
        nesterov: apply the momentum look-ahead before orthogonalizing.
        ns_steps: Newton-Schulz iterations. Read by the quintic variant only.
        ns_variant: ``"cubic5"`` or ``"quintic"``.
        ns_dtype: precision of the Newton-Schulz matmuls.
        ns_batch_elems: element budget for one batched Newton-Schulz call.
        compile_ns: compile the iteration and its elementwise passes; ``None``
            enables it only when the parameters are on CUDA.
        gram_aspect: aspect ratio at or above which a shape uses the grouped
            phases (:data:`NS_PHASES_GRAM`); ``inf`` disables grouping.
        update_scale: ``"spectral"`` or ``"rms"``.
        rms_target: target update RMS for ``update_scale="rms"``.
        rounding: ``"nearest"`` or ``"stochastic"`` writeback into the parameter.
        seed: base seed for the stochastic-rounding stream.
        weight_decay, betas, eps: AdamW hyperparameters for the non-Muon groups.
    """

    # Read by build_optimizer to decide whether to produce a Muon split.
    muon_groups = True

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_variant: str = "cubic5",
        ns_dtype: torch.dtype = torch.bfloat16,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        update_scale: str = "spectral",
        rms_target: float = 0.2,
        compile_ns: bool | None = None,
        ns_batch_elems: int = 64 << 20,
        gram_aspect: float = 1.5,
        rounding: str = "nearest",
        seed: int = 0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_dtype=ns_dtype,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            update_scale=update_scale,
            rms_target=rms_target,
            use_muon=None,
        )
        if ns_variant not in ("cubic5", "quintic"):
            raise ValueError(f"unknown ns_variant {ns_variant!r}")
        self.ns_variant = ns_variant
        self.ns_steps = ns_steps
        self.gram_aspect = gram_aspect
        # Set before `super().__init__`, which calls `add_param_group`.
        self._phases: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._orthogonalize: dict[tuple[int, ...], Callable] | None = None
        self._scalars: tuple[torch.Tensor, torch.Tensor] | None = None
        super().__init__(params, defaults)
        self.ns_batch_elems = ns_batch_elems
        # Resolved from the parameters: compiling emits Triton, which has no CPU backend.
        if compile_ns is None:
            compile_ns = any(
                param.is_cuda
                for group in self.param_groups
                for param in group["params"]
            )
        if compile_ns:
            # Two graphs per shape: a full batch and the remainder chunk.
            raise_recompile_limit(2 * len(self._phases) + 8)
        wrap = (
            (lambda fn: torch.compile(fn, dynamic=False))
            if compile_ns
            else (lambda fn: fn)
        )
        base = newton_schulz_cubic if ns_variant == "cubic5" else newton_schulz
        self._newton_schulz = wrap(base)
        self._momentum = {
            True: wrap(_nesterov_direction),
            False: wrap(_heavyball_direction),
        }
        self._decay_and_step = wrap(_decay_and_step)
        # Resolved once; the step never asks which rule is active.
        if rounding not in ("nearest", "stochastic"):
            raise ValueError(f"unknown rounding rule {rounding!r}")
        self._sr = rounding == "stochastic"
        self._sr_seed = seed
        self._sr_offset = 0
        self._orthogonalize = {
            shape: self._bind(phases) for shape, phases in self._phases.items()
        }

    def _bind(self, phases: tuple[int, ...]) -> Callable:
        """The Newton-Schulz call for one shape, callable as ``fn(x, dtype=...)``."""
        if self.ns_variant == "cubic5":
            return partial(self._newton_schulz, phases=phases)
        return partial(self._newton_schulz, steps=self.ns_steps)

    def add_param_group(self, param_group: dict) -> None:
        """Resolve each parameter's shape-dependent update scale, once."""
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if group["use_muon"] is None:
            raise ValueError(
                "every MuonW param group needs a `use_muon` flag; "
                "build groups with training.optim.group_parameters"
            )
        if not group["use_muon"]:
            return
        for param in group["params"]:
            if param.ndim < 2:
                raise ValueError(
                    f"a {param.ndim}-D parameter reached a Muon group; "
                    "vectors and scalars have no singular values to equalize"
                )
            self.state[param]["scale"] = orthogonal_update_scale(
                param.shape, group["update_scale"], group["rms_target"]
            )
            self._record_shape(tuple(param.shape))

    def _record_shape(self, shape: tuple[int, ...]) -> None:
        """Choose this shape's Newton-Schulz phases, once."""
        if shape in self._phases:
            return
        short, long = sorted(shape[-2:])
        self._phases[shape] = (
            NS_PHASES_GRAM if long >= self.gram_aspect * short else NS_PHASES_DIRECT
        )
        # None until `__init__` has resolved compilation; a later group binds here.
        if self._orthogonalize is not None:
            self._orthogonalize[shape] = self._bind(self._phases[shape])

    @torch.no_grad()
    def step(self, closure=None):
        """One step, each group taking the algorithm its ``use_muon`` flag selects."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group(group)
            else:
                self._adam_group(group)
        return loss

    def _muon_group(self, group: dict) -> None:
        """Orthogonalized step, batched over the matrices that share a shape."""
        lr, decay = group["lr"], group["weight_decay"]
        pending: dict[tuple[int, ...], list] = {}
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
            pending.setdefault(tuple(param.shape), []).append(param)

        direction = self._momentum[bool(group["nesterov"])]
        for shape, params in pending.items():
            self._apply_shaped(params, shape, group, direction, lr, decay)

    def _apply_shaped(
        self,
        params: list,
        shape: tuple[int, ...],
        group: dict,
        direction,
        lr: float,
        decay: float,
    ) -> None:
        """Orthogonalize one shape's matrices as one batch, then apply them."""
        orthogonalize = self._orthogonalize[shape]
        per_chunk = self._ns_batch(shape)
        for start in range(0, len(params), per_chunk):
            chunk = params[start : start + per_chunk]
            batch = torch.empty(
                (len(chunk), *shape), dtype=chunk[0].dtype, device=chunk[0].device
            )
            beta = group["momentum"]
            for slot, param in zip(batch, chunk):
                # Out-of-place: the trainer reads .grad after the step.
                direction(
                    self.state[param]["momentum_buffer"],
                    param.grad,
                    slot,
                    beta,
                    1 - beta,
                )
            updates = orthogonalize(batch, dtype=group["ns_dtype"])
            scale = self.state[chunk[0]]["scale"]
            if self._sr:
                # Walk the RNG offset by each parameter's numel.
                for param, update in zip(chunk, updates):
                    _decay_and_step_sr(
                        param,
                        update,
                        lr * decay,
                        -lr * scale,
                        self._sr_seed,
                        self._sr_offset,
                    )
                    self._sr_offset += param.numel()
            else:
                keep, alpha = self._step_scalars(chunk[0], 1 - lr * decay, -lr * scale)
                for param, update in zip(chunk, updates):
                    self._decay_and_step(param, update, keep, alpha)

    def _step_scalars(
        self, sample: torch.Tensor, keep: float, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's two scalars as device tensors, refilled in place."""
        if self._scalars is None:
            self._scalars = (
                torch.zeros((), dtype=torch.float32, device=sample.device),
                torch.zeros((), dtype=torch.float32, device=sample.device),
            )
        self._scalars[0].fill_(keep)
        self._scalars[1].fill_(alpha)
        return self._scalars

    def _ns_batch(self, shape: tuple[int, ...]) -> int:
        """How many same-shape matrices to stack at once, bounded by elements."""
        return max(1, self.ns_batch_elems // max(math.prod(shape), 1))

    def _adam_group(self, group: dict) -> None:
        """Decoupled-decay AdamW, for everything Muon does not apply to.

        State is held in the parameter dtype; the arithmetic is fp32.
        """
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_rms"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            adamw16_step(
                param,
                param.grad.to(param.dtype),
                state["exp_avg"],
                state["exp_avg_rms"],
                state["step"],
                group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                stochastic=self._sr and param.dtype is torch.bfloat16,
                seed=self._sr_seed,
            )
