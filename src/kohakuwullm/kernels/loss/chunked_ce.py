"""Linear + cross-entropy over a bounded scratch buffer: cuBLAS GEMMs, Triton epilogue.

Logits are produced by cuBLAS into a reused ``chunk x vocab_block`` tile that a
Triton epilogue consumes in place, so peak scratch is set by the tile rather than
by ``T``. ``dX`` and ``dW`` accumulate in fp32.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100


@triton.jit
def _lse_accum_kernel(
    logits_ptr,
    tgt_ptr,
    max_ptr,
    sum_ptr,
    tgtlogit_ptr,
    v_start,
    VB,
    stride_row,
    BLOCK_V: tl.constexpr,
):
    """Fold one ``(rows, VB)`` logit tile into running max / sumexp / target logit.

    Online softmax, so the tile is read once.
    """
    row = tl.program_id(0)
    base = logits_ptr + row * stride_row
    tgt = tl.load(tgt_ptr + row)
    run_max = tl.load(max_ptr + row)
    run_sum = tl.load(sum_ptr + row)
    tgt_logit = tl.load(tgtlogit_ptr + row)

    for v0 in range(0, VB, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < VB
        vals = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(vals))
        # exp(-inf - -inf) is nan; before the first tile there is no mass to rescale.
        scale = tl.where(run_max == float("-inf"), 0.0, tl.exp(run_max - new_max))
        run_sum = run_sum * scale + tl.sum(
            tl.where(mask, tl.exp(vals - new_max), 0.0), axis=0
        )
        run_max = new_max
        # `mask` is load-bearing: an out-of-range lane's global index can still
        # collide with a target in a later tile.
        hit = mask & (v_start + offs == tgt)
        tgt_logit += tl.sum(tl.where(hit, vals, 0.0), axis=0)

    tl.store(max_ptr + row, run_max)
    tl.store(sum_ptr + row, run_sum)
    tl.store(tgtlogit_ptr + row, tgt_logit)


@triton.jit
def _dlogit_kernel(
    logits_ptr,
    tgt_ptr,
    lse_ptr,
    grad_ptr,
    v_start,
    VB,
    stride_row,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    """Rewrite a logit tile as ``(softmax - onehot) * dloss`` in place.

    Two-dimensional grid, unlike the forward: this step is elementwise, so rows
    and vocabulary blocks are independent.
    """
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = offs < VB

    tgt = tl.load(tgt_ptr + row)
    lse = tl.load(lse_ptr + row)
    grad = tl.load(grad_ptr + row)
    grad = tl.where(tgt == ignore_index, 0.0, grad)

    ptr = logits_ptr + row * stride_row + offs
    vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    prob = tl.exp(vals - lse)
    dlogit = (prob - tl.where(v_start + offs == tgt, 1.0, 0.0)) * grad
    tl.store(ptr, dlogit, mask=mask)


class _TilePlan:
    """Tile geometry and the logit cache budget, resolved once per call."""

    def __init__(self, T, V, chunk, vocab_block, retain):
        self.chunk = max(1, T if chunk is None else min(chunk, T))
        self.vocab_block = max(1, V if vocab_block is None else min(vocab_block, V))
        self.stride = self.chunk * self.vocab_block
        self.block_v = min(4096, triton.next_power_of_2(self.vocab_block))
        # Vocabulary-major order, so the fp32 dW accumulator spans `vocab_block`
        # rows rather than the whole of (V, D).
        self.t_tiles = triton.cdiv(T, self.chunk)
        # One token chunk per vocabulary block leaves nothing to accumulate across,
        # so the dW GEMM writes the staging buffer instead of adding to it.
        self.dw_direct = self.t_tiles == 1
        self.tiles = [
            (t0, min(self.chunk, T - t0), v0, min(self.vocab_block, V - v0))
            for v0 in range(0, V, self.vocab_block)
            for t0 in range(0, T, self.chunk)
        ]
        self.cached = min(len(self.tiles), round(max(0.0, retain) * len(self.tiles)))

    def buffers(self, device, dtype):
        cache = (
            torch.empty(self.cached * self.stride, device=device, dtype=dtype)
            if self.cached
            else None
        )
        scratch = (
            torch.empty(self.stride, device=device, dtype=dtype)
            if self.cached < len(self.tiles)
            else None
        )
        return cache, scratch

    def tile(self, index, rows, cols, cache, scratch):
        """The ``(rows, cols)`` view backing tile ``index``, and whether it is cached."""
        hit = index < self.cached
        buf = cache if hit else scratch
        start = index * self.stride if hit else 0
        return buf[start : start + rows * cols].view(rows, cols), hit


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target, ignore_index, chunk, vocab_block, retain):
        x = x.contiguous()
        weight = weight.contiguous()
        target = target.contiguous()
        T, D = x.shape
        V = weight.shape[0]
        plan = _TilePlan(T, V, chunk, vocab_block, retain)
        cache, scratch = plan.buffers(x.device, x.dtype)

        run_max = torch.full((T,), float("-inf"), device=x.device, dtype=torch.float32)
        run_sum = torch.zeros(T, device=x.device, dtype=torch.float32)
        tgt_logit = torch.zeros(T, device=x.device, dtype=torch.float32)

        for index, (t0, rows, v0, cols) in enumerate(plan.tiles):
            tile, _ = plan.tile(index, rows, cols, cache, scratch)
            torch.mm(x[t0 : t0 + rows], weight[v0 : v0 + cols].t(), out=tile)
            _lse_accum_kernel[(rows,)](
                tile,
                target[t0 : t0 + rows],
                run_max[t0 : t0 + rows],
                run_sum[t0 : t0 + rows],
                tgt_logit[t0 : t0 + rows],
                v0,
                cols,
                tile.stride(0),
                BLOCK_V=plan.block_v,
                num_warps=8,
            )

        lse = run_max + torch.log(run_sum)
        loss = torch.where(
            target == ignore_index, torch.zeros_like(lse), lse - tgt_logit
        )

        ctx.save_for_backward(x, weight, target, lse, cache)
        ctx.ignore_index = ignore_index
        ctx.plan = plan
        ctx.consumed = False
        ctx.grads = None
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, weight, target, lse, cache = ctx.saved_tensors
        plan = ctx.plan
        # A split backward asks for the input grad and the weight grad in two
        # calls, and the epilogue has already overwritten the cached tiles by the
        # end of the first. Both were computed then, so replay serves the cache.
        if ctx.grads is not None:
            dx_held, dw_held = ctx.grads
            return (
                dx_held if ctx.needs_input_grad[0] else None,
                dw_held if ctx.needs_input_grad[1] else None,
                None,
                None,
                None,
                None,
                None,
            )
        ctx.consumed = True
        T, D = x.shape
        V = weight.shape[0]
        dloss = dloss.contiguous().float()

        need_dx, need_dw = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dx = (
            torch.zeros(T, D, device=x.device, dtype=torch.float32) if need_dx else None
        )
        dw = torch.empty(V, D, device=x.device, dtype=weight.dtype) if need_dw else None
        dw_acc = (
            torch.empty(plan.vocab_block, D, device=x.device, dtype=torch.float32)
            if need_dw
            else None
        )
        scratch = (
            torch.empty(plan.stride, device=x.device, dtype=x.dtype)
            if plan.cached < len(plan.tiles)
            else None
        )

        acc = None
        for index, (t0, rows, v0, cols) in enumerate(plan.tiles):
            last_chunk = index % plan.t_tiles == plan.t_tiles - 1
            if dw_acc is not None and index % plan.t_tiles == 0:
                acc = dw_acc[:cols]
                if not plan.dw_direct:
                    acc.zero_()
            xc = x[t0 : t0 + rows]
            wv = weight[v0 : v0 + cols]
            tile, hit = plan.tile(index, rows, cols, cache, scratch)
            if not hit:
                torch.mm(xc, wv.t(), out=tile)
            _dlogit_kernel[(rows, triton.cdiv(cols, plan.block_v))](
                tile,
                target[t0 : t0 + rows],
                lse[t0 : t0 + rows],
                dloss[t0 : t0 + rows],
                v0,
                cols,
                tile.stride(0),
                ctx.ignore_index,
                BLOCK_V=plan.block_v,
                num_warps=4,
            )
            if need_dx:
                view = dx[t0 : t0 + rows]
                torch.addmm(view, tile, wv, out=view, out_dtype=torch.float32)
            if need_dw:
                if plan.dw_direct:
                    torch.mm(tile.t(), xc, out=acc, out_dtype=torch.float32)
                else:
                    torch.addmm(acc, tile.t(), xc, out=acc, out_dtype=torch.float32)
                if last_chunk:
                    dw[v0 : v0 + cols].copy_(acc)

        dx_out = dx.to(x.dtype) if need_dx else None
        ctx.grads = (dx_out, dw)
        return (dx_out, dw, None, None, None, None, None)


def chunked_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    chunk: int | None = 8192,
    vocab_block: int | None = 8192,
    retain: float = 0.0,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-token cross-entropy of ``x @ weight.T`` against ``target``.

    Returns an unreduced ``(T,)`` fp32 tensor; reduction is left to the caller so
    a token-weighted mean can be taken in fp32. Ignored positions return exactly
    zero and contribute nothing to either gradient.

    Args:
        chunk: rows per logit tile. ``None`` means the whole batch.
        vocab_block: columns per logit tile. ``None`` means the whole vocabulary.
        compute_dtype: dtype the GEMMs run in; accumulation stays fp32. Defaults
            to the activation dtype, and the weight follows.
        retain: fraction of logit tiles kept from the forward for backward to
            reuse. ``0.0`` recomputes every tile (one extra GEMM, ``O(1)``
            scratch); ``1.0`` keeps all ``T * V``. Above ``0.0``, backward may
            only be called once.

    ``dloss`` is expected to be ``O(1)``: reduce with ``.sum()`` and divide by the
    token count afterwards.
    """
    if compute_dtype is not None and compute_dtype is not x.dtype:
        x = x.to(compute_dtype)
    if weight.dtype is not x.dtype:
        weight = weight.to(x.dtype)
    return _ChunkedLinearCE.apply(
        x, weight, target, ignore_index, chunk, vocab_block, retain
    )
