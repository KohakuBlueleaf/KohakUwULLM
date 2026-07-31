"""Expert-sort indices for MoE dispatch, without a sort, and the gated combine.

Grouping ``(token, slot)`` pairs by expert is a counting sort over the histogram
the fused router already returns. Order within an expert is not deterministic.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _expert_offsets(
    counts_ptr,
    offsets_ptr,
    cursor_ptr,
    max_rows_ptr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Exclusive/inclusive scan of the per-expert counts, in one program.

    ``E <= 128``, so the whole histogram is a single register-resident vector.
    """
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    counts = tl.load(counts_ptr + offs, mask=mask, other=0).to(tl.int32)
    inclusive = tl.cumsum(counts, axis=0)

    tl.store(offsets_ptr, 0)
    tl.store(offsets_ptr + 1 + offs, inclusive, mask=mask)
    tl.store(cursor_ptr + offs, inclusive - counts, mask=mask)
    # Load-imbalance diagnostic. Nothing on the hot path reads it.
    tl.store(max_rows_ptr, tl.max(tl.where(mask, counts, 0)))


@triton.jit
def _place_pairs(
    expert_ptr,
    cursor_ptr,
    order_ptr,
    token_ptr,
    N,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scatter each ``(token, slot)`` pair into its expert's row range."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    expert = tl.load(expert_ptr + offs, mask=mask, other=0).to(tl.int32)
    # Per-expert running cursor: a distinct old value per lane even on collision.
    pos = tl.atomic_add(cursor_ptr + expert, 1, mask=mask)
    tl.store(order_ptr + pos, offs, mask=mask)
    tl.store(token_ptr + pos, offs // TOP_K, mask=mask)


@triton.jit
def _combine_fwd(
    out_ptr,
    rows_ptr,
    weight_ptr,
    order_ptr,
    token_ptr,
    valid_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    """``out[token_of[p]] += rows[p] * weight[order[p]]`` for every sorted row.

    Rows at or past ``valid_ptr[0]`` are skipped outright rather than scaled by
    their zero gate weight. The bound is read from device memory, never passed as
    an int.
    """
    pid = tl.program_id(0)
    if pid >= tl.load(valid_ptr):
        return
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    scale = tl.load(weight_ptr + tl.load(order_ptr + pid))
    token = tl.load(token_ptr + pid)
    rows = tl.load(rows_ptr + pid * D + offs_d, mask=mask, other=0.0)
    scaled = (rows.to(tl.float32) * scale).to(rows.dtype)
    # Atomic in the output's own dtype; the sum is depth top_k.
    tl.atomic_add(out_ptr + token * D + offs_d, scaled, mask=mask)


@triton.jit
def _combine_bwd(
    grad_out_ptr,
    rows_ptr,
    weight_ptr,
    order_ptr,
    token_ptr,
    grad_rows_ptr,
    grad_weight_ptr,
    valid_ptr,
    N,
    D,
    BLOCK_D: tl.constexpr,
):
    """Both gradients of the combine in one pass, since both read the same two rows.

    ``grad_weight`` is a full-width dot product per routed row split across
    ``D / BLOCK_D`` programs, so it accumulates through an fp32 atomic. Skipped rows
    get an explicit zero rather than an early return.
    """
    pid = tl.program_id(0)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D
    if pid >= tl.load(valid_ptr):
        tl.store(grad_rows_ptr + pid * D + offs_d, 0.0, mask=mask)
        return

    pair = tl.load(order_ptr + pid)
    token = tl.load(token_ptr + pid)
    scale = tl.load(weight_ptr + pair)
    grad = tl.load(grad_out_ptr + token * D + offs_d, mask=mask, other=0.0)
    rows = tl.load(rows_ptr + pid * D + offs_d, mask=mask, other=0.0)

    tl.store(
        grad_rows_ptr + pid * D + offs_d,
        (grad.to(tl.float32) * scale).to(rows.dtype),
        mask=mask,
    )
    partial = tl.sum(tl.where(mask, grad.to(tl.float32) * rows.to(tl.float32), 0.0))
    tl.atomic_add(grad_weight_ptr + pair, partial)


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, weight, order, token_of, num_tokens, valid_rows):
        n, dim = rows.shape
        block_d = min(1024, max(16, triton.next_power_of_2(dim)))
        out = torch.zeros(num_tokens, dim, dtype=rows.dtype, device=rows.device)
        _combine_fwd[(n, triton.cdiv(dim, block_d))](
            out, rows, weight, order, token_of, valid_rows, n, dim, BLOCK_D=block_d
        )
        ctx.save_for_backward(rows, weight, order, token_of, valid_rows)
        ctx.block_d = block_d
        return out

    @staticmethod
    def backward(ctx, grad_out):
        rows, weight, order, token_of, valid_rows = ctx.saved_tensors
        n, dim = rows.shape
        grad_rows = torch.empty_like(rows)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        _combine_bwd[(n, triton.cdiv(dim, ctx.block_d))](
            grad_out.contiguous(),
            rows,
            weight,
            order,
            token_of,
            grad_rows,
            grad_weight,
            valid_rows,
            n,
            dim,
            BLOCK_D=ctx.block_d,
        )
        return grad_rows, grad_weight.to(weight.dtype), None, None, None, None


def combine_routed(
    rows: torch.Tensor,
    weight: torch.Tensor,
    order: torch.Tensor,
    token_of: torch.Tensor,
    num_tokens: int,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale each routed row by its gate weight and sum it back onto its token.

    Args:
        rows: ``(T * top_k, D)`` expert outputs in sorted order.
        weight: ``(T * top_k,)`` gate weights, indexed by pair index.
        order: sorted position -> pair index.
        token_of: sorted position -> token index.
        num_tokens: T, the row count of the returned tensor.
        valid_rows: one-element device tensor; sorted positions at or past it are
            skipped as uncomputed. ``None`` means every row is valid. Kept on the
            device, never read on the host.
    """
    if rows.shape[0] != order.numel():
        raise ValueError(f"rows {rows.shape[0]} != order {order.numel()}")
    if not rows.is_cuda:
        return combine_routed_reference(
            rows, weight, order, token_of, num_tokens, valid_rows
        )
    if valid_rows is None:
        valid_rows = torch.full(
            (1,), rows.shape[0], dtype=torch.int32, device=rows.device
        )
    return _Combine.apply(
        rows.contiguous(), weight, order, token_of, num_tokens, valid_rows
    )


def combine_routed_reference(
    rows: torch.Tensor,
    weight: torch.Tensor,
    order: torch.Tensor,
    token_of: torch.Tensor,
    num_tokens: int,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-composed equivalent of :func:`combine_routed`, and the CPU path."""
    scaled = rows * weight.index_select(0, order.long()).reshape(-1, 1)
    if valid_rows is not None:
        # Mask skipped rows to zero before the sum: an uncomputed row may hold nan.
        keep = torch.arange(rows.shape[0], device=rows.device) < valid_rows.to(
            rows.device
        )
        scaled = torch.where(keep.unsqueeze(1), scaled, torch.zeros_like(scaled))
    out = torch.zeros(num_tokens, rows.shape[1], dtype=rows.dtype, device=rows.device)
    return out.index_add(0, token_of.long(), scaled)


def expert_sort(
    flat_expert: torch.Tensor, counts: torch.Tensor, top_k: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group ``(token, slot)`` pairs by expert using their known histogram.

    Args:
        flat_expert: ``(T * top_k,)`` expert id per pair.
        counts: ``(E,)`` rows per expert, as returned by the router.
        top_k: experts per token, used to recover the token from a pair index.
        num_experts: E.

    Returns:
        ``(offsets, order, token_of, max_rows)``. ``offsets`` is int32 ``(E + 1,)``
        for the grouped GEMM; ``order`` maps sorted position -> pair index;
        ``token_of`` maps sorted position -> token index; ``max_rows`` is a
        one-element device tensor holding the largest per-expert count, a
        diagnostic and not an input to anything.
    """
    if not flat_expert.is_cuda:
        return expert_sort_reference(flat_expert, counts, top_k, num_experts)

    n = flat_expert.numel()
    device = flat_expert.device
    block_e = max(16, triton.next_power_of_2(num_experts))

    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    max_rows = torch.empty(1, dtype=torch.int32, device=device)
    _expert_offsets[(1,)](
        counts, offsets, cursor, max_rows, E=num_experts, BLOCK_E=block_e
    )

    order = torch.empty(n, dtype=torch.int32, device=device)
    token_of = torch.empty(n, dtype=torch.int32, device=device)
    block = 1024
    _place_pairs[(triton.cdiv(n, block),)](
        flat_expert, cursor, order, token_of, n, TOP_K=top_k, BLOCK=block
    )
    return offsets, order, token_of, max_rows


def expert_sort_reference(
    flat_expert: torch.Tensor, counts: torch.Tensor, top_k: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort-based equivalent of :func:`expert_sort`, and the CPU path.

    Uses a stable sort, so this side is reproducible where the CUDA path's atomic
    cursor is not.
    """
    device = flat_expert.device
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    order = flat_expert.argsort(stable=True).to(torch.int32)
    token_of = torch.div(order, top_k, rounding_mode="floor").to(torch.int32)
    max_rows = (
        counts.max().reshape(1).to(torch.int32)
        if counts.numel()
        else torch.zeros(1, dtype=torch.int32, device=device)
    )
    return offsets, order, token_of, max_rows
