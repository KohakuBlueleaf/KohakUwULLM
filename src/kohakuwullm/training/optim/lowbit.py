"""Native low-precision parameters: the cast policy and a rounding-aware AdamW.

See docs/internals/optimizers.md.
"""

import torch
import torch.nn as nn

# Names containing any of these stay fp32 under `cast_parameters_`.
KEEP_FP32_DEFAULT: tuple[str, ...] = (
    "norm",
    "router",
    "inv_freq",
    "freq_dirs",
    "sink",
    "bias",
)


def cast_parameters_(
    module: nn.Module,
    dtype: torch.dtype,
    keep_fp32: tuple[str, ...] = KEEP_FP32_DEFAULT,
    cast_buffers: bool = False,
) -> dict[str, int]:
    """Cast ``module``'s floating-point parameters to ``dtype`` in place.

    Names containing a ``keep_fp32`` entry are left alone, buffers included;
    integer buffers are never touched. Returns the counts moved and kept.
    """
    moved = kept = 0
    for name, param in module.named_parameters():
        if not param.is_floating_point():
            continue
        if any(token in name for token in keep_fp32):
            kept += param.numel()
            continue
        param.data = param.data.to(dtype)
        if param.grad is not None:
            param.grad = param.grad.to(dtype)
        moved += param.numel()
    if cast_buffers:
        for name, buffer in module.named_buffers():
            if not buffer.is_floating_point():
                continue
            if any(token in name for token in keep_fp32):
                continue
            buffer.data = buffer.data.to(dtype)
    return {"cast": moved, "kept_fp32": kept}


def stochastic_round_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Write ``src`` (fp32) into ``dst``, rounding stochastically. bf16 only."""
    if dst.dtype is not torch.bfloat16:
        raise ValueError(f"stochastic rounding is bf16-only, got {dst.dtype}")
    bits = src.view(torch.int32)
    # A uniform 16-bit addend carries into the bf16 mantissa with the right probability.
    noise = torch.randint(0, 1 << 16, src.shape, dtype=torch.int32, device=src.device)
    truncated = torch.bitwise_and(bits + noise, -(1 << 16))
    return dst.copy_(truncated.view(torch.float32))


def nearest_round_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Plain round-to-nearest writeback."""
    return dst.copy_(src)


class StochasticAdamW(torch.optim.Optimizer):
    """AdamW that updates low-precision parameters without losing small updates.

    The moments and the update stay fp32; only the writeback is rounded by the
    selected rule, and fp32 parameters take the same path with a no-op rule. See
    docs/internals/optimizers.md.

    Args:
        rounding: ``"stochastic"``, ``"kahan"`` or ``"nearest"``.
        state_dtype: dtype of ``exp_avg`` / ``exp_avg_sq``. ``None`` means fp32.
        clip_grad_norm: when set, the optimizer owns clipping and the trainer
            must not clip again; the *unclipped* gradient is left behind.
    """

    _ROUNDING = {"stochastic": stochastic_round_, "nearest": nearest_round_}

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        rounding: str = "stochastic",
        state_dtype: torch.dtype | str | None = None,
        clip_grad_norm: float | None = None,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if clip_grad_norm is not None and clip_grad_norm <= 0.0:
            raise ValueError(f"clip_grad_norm must be positive, got {clip_grad_norm}")
        if rounding not in ("stochastic", "kahan", "nearest"):
            raise ValueError(f"unknown rounding rule {rounding!r}")

        if isinstance(state_dtype, str):
            state_dtype = getattr(torch, state_dtype)
        if state_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError(
                f"state_dtype must be None, float32 or bfloat16, got {state_dtype}"
            )

        super().__init__(
            params,
            dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay),
        )
        self.rounding = rounding
        self.state_dtype = state_dtype or torch.float32
        self.clip_grad_norm = clip_grad_norm
        # Resolved once; the step never asks which rule is active.
        self._round = self._ROUNDING.get(rounding, nearest_round_)
        self._kahan = rounding == "kahan"
        self._last_grad_norm: torch.Tensor | None = None

    # -- state ----------------------------------------------------------- #

    def _state_for(self, param: torch.Tensor) -> dict:
        """This parameter's moments, allocated on first use."""
        state = self.state[param]
        if state:
            return state
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(param, dtype=self.state_dtype)
        state["exp_avg_sq"] = torch.zeros_like(param, dtype=self.state_dtype)
        if self._kahan and param.dtype is not torch.float32:
            state["compensation"] = torch.zeros_like(param)
        return state

    # -- clipping -------------------------------------------------------- #

    def _clip_scale(self) -> torch.Tensor | None:
        """Global clip coefficient as a device tensor, reduced in fp32."""
        self._last_grad_norm = None
        if self.clip_grad_norm is None:
            return None
        grads = [
            p.grad
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grads:
            return None
        # Cast each per-tensor norm before the outer reduction.
        norms = [n.float() for n in torch._foreach_norm(grads, 2.0)]
        total = torch.linalg.vector_norm(torch.stack(norms))
        self._last_grad_norm = total
        return (self.clip_grad_norm / total.clamp_min(1e-6)).clamp(max=1.0)

    def grad_norm(self) -> torch.Tensor | None:
        """Pre-clip global gradient norm from the last :meth:`step`."""
        return self._last_grad_norm

    # -- step ------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        """One AdamW step over every group, clipping first if this owns clipping."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        clip = self._clip_scale()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("StochasticAdamW does not support sparse grads")
                self._update(param, group, beta1, beta2, clip)
        return loss

    def _update(self, param, group, beta1, beta2, clip) -> None:
        """Assemble one parameter's update in fp32 and write it back rounded."""
        state = self._state_for(param)
        state["step"] += 1
        step = state["step"]
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

        # `copy=True` throughout: `.float()` on an fp32 tensor returns itself.
        grad = param.grad.to(torch.float32, copy=True)
        if clip is not None:
            grad.mul_(clip)
        exp_avg.lerp_(grad.to(exp_avg.dtype), 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Bias corrections in fp32 even under bf16 state; see docs/internals/optimizers.md.
        denom = exp_avg_sq.to(torch.float32, copy=True)
        denom.div_(1.0 - beta2**step).sqrt_().add_(group["eps"])
        update = exp_avg.to(torch.float32, copy=True)
        update.div_(denom).mul_(-group["lr"] / (1.0 - beta1**step))

        if param.dtype is torch.float32:
            param.mul_(1.0 - group["lr"] * group["weight_decay"]).add_(update)
            return

        working = param.to(torch.float32, copy=True)
        working.mul_(1.0 - group["lr"] * group["weight_decay"])
        if self._kahan:
            compensation = state["compensation"]
            update.add_(compensation)
            working.add_(update)
            nearest_round_(param, working)
            # Carry what the writeback discarded, decay included, to the next step.
            compensation.copy_(working.sub_(param.float()))
            return
        self._round(param, working.add_(update))
