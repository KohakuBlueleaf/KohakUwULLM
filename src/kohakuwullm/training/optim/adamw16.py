"""AdamW for 16-bit parameters: state 16-bit in memory, arithmetic in fp32.

The counterpart to :class:`~kohakuwullm.training.optim.fused_adamw.FusedAdamW`,
which requires fp32 parameters and refuses these. See docs/internals/optimizers.md.
"""

import torch

from kohakuwullm.kernels.optim.adamw16 import adamw16_step

_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)


class AdamW16(torch.optim.Optimizer):
    """AdamW over fp16 / bf16 parameters, one fused Triton launch per tensor.

    Both moments are stored in the parameter's own dtype and read into fp32
    registers, so the update never rounds through 16 bits. ``exp_avg_sq`` is
    held as ``sqrt(v)``.

    Args:
        params: parameters or param groups.
        lr / betas / eps / weight_decay: as AdamW; decay is decoupled.
        stochastic: round the parameter writeback stochastically. bf16 only --
            the discarded-bit width is exponent-dependent for fp16.

    Clipping is the caller's; this optimizer does not fold it in.
    See docs/internals/optimizers.md.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        stochastic: bool = False,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                stochastic=stochastic,
            ),
        )
        self.seed = seed

    @torch.no_grad()
    def step(self, closure=None):
        """One AdamW step over every parameter that has a gradient."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("AdamW16 does not support sparse gradients")
                if param.dtype not in _SUPPORTED:
                    raise ValueError(
                        f"AdamW16 takes fp16/bf16/fp32 parameters, got {param.dtype}"
                    )
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_rms"] = torch.zeros_like(param)
                state["step"] += 1
                stochastic = group["stochastic"] and param.dtype is torch.bfloat16
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
                    stochastic=stochastic,
                    seed=self.seed,
                )
        return loss
