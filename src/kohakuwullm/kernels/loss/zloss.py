"""Chunked ``logsumexp(x @ W.T) ** 2`` -- the z-loss, without materializing logits.

The forward walks the batch in chunks under ``no_grad`` keeping only the per-row
scalar; the backward recomputes each chunk's softmax to build
``d/dx [lse(xW)^2] = 2 * lse(xW) * softmax(xW) @ W``. Peak memory is one chunk of
logits, so the whole thing is O(1) in ``N``.

See docs/internals/kernels.md.
"""

import torch


def _chunk_rows(n: int, vocab: int, dim: int) -> int:
    """Rows per chunk, sized so one chunk of logits matches the input's footprint."""
    ratio = max(1, vocab // max(dim, 1))
    rows = max(256, n // max(ratio, 1))
    return min(n, 1 << (rows - 1).bit_length())


class _LogSumExpSquare(torch.autograd.Function):
    # Casts the float arguments to the autocast dtype. See docs/internals/kernels.md.
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, labels, ignore_index):
        n, dim = hidden.shape
        vocab = weight.shape[0]
        rows = _chunk_rows(n, vocab, dim)
        valid = labels != ignore_index

        out = torch.empty(n, device=hidden.device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, n, rows):
                stop = min(start + rows, n)
                logits = torch.mm(hidden[start:stop], weight.T).float()
                out[start:stop] = torch.logsumexp(logits, dim=-1)
        out = torch.where(valid, out, torch.zeros_like(out))

        ctx.save_for_backward(hidden, weight, out, valid)
        ctx.rows = rows
        return out.pow(2)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        hidden, weight, lse, valid = ctx.saved_tensors
        n = hidden.shape[0]
        rows = ctx.rows
        # d(lse^2)/d(lse) = 2 * lse, folded with the incoming grad.
        coeff = (2.0 * lse * grad_out.float() * valid).to(hidden.dtype)

        d_hidden = torch.empty_like(hidden)
        d_weight = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            h = hidden[start:stop]
            probs = torch.softmax(torch.mm(h, weight.T).float(), dim=-1)
            probs = probs.mul_(coeff[start:stop, None].float()).to(hidden.dtype)
            d_hidden[start:stop] = torch.mm(probs, weight)
            d_weight += torch.mm(probs.T, h).float()
        return d_hidden, d_weight.to(weight.dtype), None, None


def logsumexp_square(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """``(N,)`` of ``logsumexp(hidden @ weight.T) ** 2``, zero where ignored."""
    return _LogSumExpSquare.apply(hidden, weight, labels, ignore_index)
