"""Fused AdamW whose state is 16-bit in memory and fp32 in registers.

The second moment is stored as ``sqrt(v)``, not ``v``, so the squared exponent
never reaches memory. Rounding on the writeback is a caller argument.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _adamw16_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    r_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    eps,
    wd,
    m_scale,
    r_scale,
    seed,
    STOCHASTIC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One AdamW step. ``r`` holds ``sqrt(v)``; ``m_scale``/``r_scale`` debias.

    The debias reciprocals arrive already computed on the host.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    m = beta1 * m + (1.0 - beta1) * g
    # The stored quantity is the root, squared back here.
    v = beta2 * (r * r) + (1.0 - beta2) * (g * g)
    r_new = tl.sqrt(v)

    update = (m * m_scale) / (r_new * r_scale + eps)
    # Decoupled decay: applied to the parameter, not folded into the gradient.
    p = p - lr * (update + wd * p)

    if STOCHASTIC:
        # Uniform noise across the bits the narrowing cast discards. bf16 only.
        rnd = tl.randint(seed, offs)
        p_bits = p.to(tl.int32, bitcast=True) + (rnd.to(tl.int32) >> 16)
        p_out = (p_bits & 0xFFFF0000).to(tl.float32, bitcast=True)
        m_bits = m.to(tl.int32, bitcast=True) + (
            tl.randint(seed + 1, offs).to(tl.int32) >> 16
        )
        m_out = (m_bits & 0xFFFF0000).to(tl.float32, bitcast=True)
    else:
        p_out = p
        m_out = m

    tl.store(p_ptr + offs, p_out.to(p_ptr.dtype.element_ty), mask=mask)
    tl.store(m_ptr + offs, m_out.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(r_ptr + offs, r_new.to(r_ptr.dtype.element_ty), mask=mask)


def adamw16_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_rms: torch.Tensor,
    step: int,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    stochastic: bool = False,
    seed: int = 0,
) -> None:
    """In-place AdamW step over 16-bit state, computing in fp32.

    ``exp_avg_rms`` stores ``sqrt(v)``, not ``v``: callers converting from a torch
    optimizer's ``exp_avg_sq`` must take the square root. ``stochastic`` is
    bf16-only, since the discarded-bit width is exponent-dependent for fp16.
    """
    if stochastic and param.dtype is not torch.bfloat16:
        raise ValueError("stochastic rounding here is bf16-only; see the docstring")
    for name, t in (("grad", grad), ("exp_avg", exp_avg), ("exp_avg_rms", exp_avg_rms)):
        if t.shape != param.shape:
            raise ValueError(f"{name} shape {t.shape} != param {param.shape}")

    beta1, beta2 = betas
    n = param.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _adamw16_kernel[grid](
        param,
        grad,
        exp_avg,
        exp_avg_rms,
        n,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        1.0 / (1.0 - beta1**step),
        1.0 / (1.0 - beta2**step) ** 0.5,
        seed + step,
        STOCHASTIC=stochastic,
        BLOCK=1024,
        num_warps=4,
    )
