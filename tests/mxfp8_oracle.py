"""fp64 reference for the MX format, and the routing fixture the expert tests share.

Not a test module. ``test_kernels_mxfp8_experts*.py`` all measure the same fused
and unfused path against the same oracle, and three copies of an 80-line
reference is how two of them end up disagreeing about what the kernel owes.

The quantizer here is deliberately **not** a call into the kernel's own. An
oracle that shares the kernel's rounding proves only that the kernel is
self-consistent; this one re-derives the block exponent in fp64, so a change to
the rounding rule shows up as a failure rather than as a matched pair of changes.
"""

import torch

from kohakuwullm.kernels.mxfp8 import BLOCK_SCALE, E4M3_MAX


def _mx_quantize_fp64(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch mirror of ``mxfp8._quantize_block``: the exponent rounds **up**.

    A second implementation of the quantizer, not a call into the kernel's, so a
    change to the rounding rule shows up here as a failure rather than being
    tracked silently by the reference.
    """
    rows, cols = x.shape
    blocks = x.double().reshape(rows, cols // BLOCK_SCALE, BLOCK_SCALE)
    amax = blocks.abs().amax(dim=2)
    exponent = torch.ceil(torch.log2(amax.clamp_min(1e-30) / E4M3_MAX)).clamp_min(
        -127.0
    )
    q = (blocks / torch.exp2(exponent)[:, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(rows, cols), exponent


def _mx_dequantize_fp64(q: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    blocks = q.double().reshape(rows, cols // BLOCK_SCALE, BLOCK_SCALE)
    return (blocks * torch.exp2(exponent)[:, :, None]).reshape(rows, cols)


def mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    return _mx_dequantize_fp64(*_mx_quantize_fp64(x))


def routing(tokens, top_k, experts, dtype, seed=0, skew=False, sentinel=0):
    """Sorted-order routing indices, plus a sentinel bucket when asked for.

    ``skew`` puts half the pairs on one expert and empties another, which is what
    exercises the tile-owner search rather than the balanced case where every
    expert has the same row count and an off-by-one would cancel.
    """
    torch.manual_seed(seed)
    buckets = experts + (1 if sentinel else 0)
    pairs = tokens * top_k
    if skew:
        expert_of = torch.randint(0, experts, (pairs,), device="cuda")
        expert_of[: pairs // 2] = min(3, experts - 1)
        expert_of[expert_of == experts - 1] = 0
    else:
        expert_of = torch.arange(pairs, device="cuda") % experts
    if sentinel:
        expert_of[:sentinel] = experts
    counts = torch.bincount(expert_of, minlength=buckets)
    offsets = torch.zeros(buckets + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    order = expert_of.argsort(stable=True).to(torch.int32)
    token_of = torch.div(order, top_k, rounding_mode="floor").to(torch.int32)
    gate = (torch.rand(pairs, device="cuda") + 0.5).to(dtype)
    return offsets, order, token_of, gate


def experts_oracle(
    x, w_in, w_out, gate, token_of, order, offsets, hidden, dout, wgrad_fp8=False
):
    """fp64 re-derivation of the fused path, algorithm for algorithm.

    **Not autograd over an unquantized model.** That measures how lossy MXFP8 is --
    6% on this shape -- and would pass with a kernel that had the SwiGLU halves
    swapped. This mirrors every quantization and rounding point the kernels
    perform, including the two *different* fp8 copies of each weight that FPROP and
    DGRAD read, so what is left over is only fp32 accumulation order and the 16-bit
    output atomics. It therefore holds to a few ULP, and a real bug moves it.

    ``wgrad_fp8`` selects which operand the two weight gradients contract, and it is a
    parameter rather than a constant because the answer is asymmetric and that asymmetry
    is the thing worth pinning. FPROP, DGRAD and ``dgate`` all read the fp8 copies, so
    those terms are unaffected; WGRAD multiplies the **16-bit** activation and the
    16-bit ``h``, matching ``MXFP8Linear``'s WGRAD and for the same reason -- a
    weight-gradient error integrates into optimizer state as systematic bias. The
    ``True`` branch reproduces the fp8-operand version that measured 0.21 nats of loss,
    so a test can assert the kernel is nearer the right oracle than the wrong one.
    """
    experts, two_h, dim = w_in.shape
    dtype = x.dtype
    off = offsets.tolist()
    tok, prs = token_of.long(), order.long()
    xd = mx_roundtrip(x)
    xg = xd.index_select(0, tok)
    # The same gather without the fp8 round trip. GEMM1 contracts `xg`; WGRAD contracts
    # this one, and holding both is what encodes the asymmetry rather than assuming it.
    xg16 = x.double().index_select(0, tok)
    dd = mx_roundtrip(dout)

    out = torch.zeros(x.shape[0], dim, dtype=torch.float64, device=x.device)
    dx = torch.zeros_like(out)
    dgate = torch.zeros(gate.numel(), dtype=torch.float64, device=x.device)
    dw_in = torch.zeros(experts, two_h, dim, dtype=torch.float64, device=x.device)
    dw_out = torch.zeros(experts, dim, hidden, dtype=torch.float64, device=x.device)

    for e in range(experts):
        lo, hi = off[e], off[e + 1]
        if hi <= lo:
            continue
        w_f = mx_roundtrip(w_in[e])
        w_d = mx_roundtrip(w_in[e].t().contiguous())
        o_f = mx_roundtrip(w_out[e])
        o_d = mx_roundtrip(w_out[e].t().contiguous())

        pre = (xg[lo:hi] @ w_f.T).to(dtype).double()
        gate_h, value = pre[:, :hidden], pre[:, hidden:]
        sig = torch.sigmoid(gate_h)
        hd = mx_roundtrip(gate_h * sig * value)
        # `h` rebuilt from the stored pre-activation the way the backward does it: silu
        # in the storage dtype, then the product in the storage dtype. Two roundings,
        # not one of an fp64 product, because that is what the elementwise op performs.
        h16 = (
            (torch.nn.functional.silu(gate_h.to(dtype)).double() * value)
            .to(dtype)
            .double()
        )

        rows_t, pairs = tok[lo:hi], prs[lo:hi]
        weight = gate.double()[pairs]
        out.index_add_(0, rows_t, (hd @ o_f.T) * weight[:, None])

        unscaled = dd[rows_t] @ o_d.T
        dgate.index_add_(0, pairs, (unscaled * hd).sum(1))
        dh = unscaled * weight[:, None]
        dpre = torch.cat(
            [dh * value * sig * (1.0 + gate_h * (1.0 - sig)), dh * gate_h * sig], dim=1
        )
        dx.index_add_(0, rows_t, mx_roundtrip(dpre) @ w_d.T)
        b_out, b_in = (hd, xg[lo:hi]) if wgrad_fp8 else (h16, xg16[lo:hi])
        # The product stays in fp64. It used to mirror the kernel's
        # `.to(tl.float16)`, and that mirror is *why* the fp16 downcast survived
        # every test in this file: an oracle that reproduces the narrowing agrees
        # with the kernel by construction, which is the "reference derived the same
        # way" trap this module's own grouped test warns about. The operands are
        # narrowed -- they are what the kernel reads -- and nothing after them is.
        dw_out[e] = (dout.double()[rows_t] * weight[:, None]).T @ b_out
        dw_in[e] = dpre.to(dtype).double().T @ b_in
    return out, dx, dw_in, dw_out, dgate
