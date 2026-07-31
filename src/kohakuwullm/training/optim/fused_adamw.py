"""AdamW on the fused ATen kernel, with clipping folded in. See docs/internals/optimizers.md."""

import torch
import torch.distributed as dist


class FusedAdamW(torch.optim.Optimizer):
    """AdamW on ``torch._fused_adamw_``, optionally clipping inside the step.

    Requires fp32 parameters. With ``clip_grad_norm`` set the optimizer owns
    clipping and the trainer must not clip again. ``state_dtype="bfloat16"``
    halves optimizer state and is CUDA-only. See docs/internals/optimizers.md.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        clip_grad_norm: float | None = None,
        sharded_norm: bool = False,
        state_dtype: torch.dtype | str | None = None,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if clip_grad_norm is not None and clip_grad_norm <= 0.0:
            raise ValueError(f"clip_grad_norm must be positive, got {clip_grad_norm}")

        if isinstance(state_dtype, str):
            state_dtype = getattr(torch, state_dtype)
        if state_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError(
                f"state_dtype must be None, float32 or bfloat16, got {state_dtype}"
            )

        super().__init__(
            params,
            # `fused` tells `load_state_dict` to host `step` on the parameter's device.
            dict(
                lr=lr,
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay,
                fused=True,
            ),
        )
        self.clip_grad_norm = clip_grad_norm
        self.sharded_norm = sharded_norm
        self.state_dtype = state_dtype
        self._last_grad_norm: torch.Tensor | None = None

    # -- state ----------------------------------------------------------- #

    def _state_for(self, param: torch.Tensor) -> dict:
        """This parameter's moments and step counter, allocated on first use."""
        state = self.state[param]
        if state:
            return state
        if param.dtype in (torch.bfloat16, torch.float16):
            # The ATen kernel rounds to nearest; see docs/internals/optimizers.md.
            raise ValueError(
                f"FusedAdamW needs float32 parameters, got {param.dtype}. Use "
                "bf16-mixed, or a stochastically-rounded optimizer for bf16-true."
            )
        dtype = self.state_dtype or param.dtype
        if dtype is torch.bfloat16 and param.dtype is not torch.float32:
            raise ValueError(
                "bfloat16 optimizer state is only supported for float32 parameters, "
                f"got {param.dtype}"
            )
        if dtype is not param.dtype and param.device.type != "cuda":
            raise ValueError(
                "bfloat16 optimizer state requires the CUDA kernel, "
                f"got a parameter on {param.device}"
            )
        # The fused kernel reads the step count as an fp32 scalar on device.
        state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
        state["exp_avg"] = torch.zeros_like(param, dtype=dtype)
        state["exp_avg_sq"] = torch.zeros_like(param, dtype=dtype)
        return state

    def _collect(self) -> list[tuple[dict, list, list, list, list, list]]:
        """Group each param group's tensors by (device, dtype), by hand."""
        collected: list[tuple[dict, list, list, list, list, list]] = []
        for group in self.param_groups:
            buckets: dict[tuple, tuple[list, list, list, list, list]] = {}
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("FusedAdamW does not support sparse gradients")
                state = self._state_for(param)
                key = (param.device, param.dtype)
                bucket = buckets.setdefault(key, ([], [], [], [], []))
                bucket[0].append(param)
                bucket[1].append(param.grad)
                bucket[2].append(state["exp_avg"])
                bucket[3].append(state["exp_avg_sq"])
                bucket[4].append(state["step"])
            for bucket in buckets.values():
                collected.append((group, *bucket))
        return collected

    # -- clipping -------------------------------------------------------- #

    def _clip_scale(self, collected: list) -> torch.Tensor | None:
        """Global L2 norm as a divisor for the fused kernel's ``grad_scale``."""
        self._last_grad_norm = None
        if self.clip_grad_norm is None:
            return None
        grads = [g for entry in collected for g in entry[2]]
        if not grads:
            return None

        # Cast each per-tensor norm before the outer reduction.
        norms = [n.float() for n in torch._foreach_norm(grads, 2.0)]
        total = torch.linalg.vector_norm(torch.stack(norms))
        if self.sharded_norm and dist.is_available() and dist.is_initialized():
            squared = total.square()
            dist.all_reduce(squared)
            total = squared.sqrt()
        self._last_grad_norm = total

        # `grad_scale` divides, so this is the reciprocal of the clip coefficient.
        return (total / self.clip_grad_norm).clamp(min=1.0)

    def grad_norm(self) -> torch.Tensor | None:
        """Pre-clip global gradient norm as a device tensor, or ``None`` if unclipped."""
        return self._last_grad_norm

    # -- step ------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        """One fused AdamW step per (device, dtype) bucket."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        collected = self._collect()
        grad_scale = self._clip_scale(collected)

        for group, params, grads, exp_avgs, exp_avg_sqs, steps in collected:
            beta1, beta2 = group["betas"]
            torch._foreach_add_(steps, 1)
            scale = grad_scale
            if scale is not None and scale.device != params[0].device:
                scale = scale.to(params[0].device, non_blocking=True)
            torch._fused_adamw_(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                [],
                steps,
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                amsgrad=False,
                maximize=False,
                grad_scale=scale,
                found_inf=None,
            )
        return loss
