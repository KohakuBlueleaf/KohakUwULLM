"""The output head: a vocabulary projection whose loss never materializes logits.

* ``loss(hidden, labels)`` -- training. Fused, no logits ever exist.
* ``logits(hidden)`` -- generation / analysis. Materializes.

Two loss kernels, selected once by ``kernel``. See docs/concepts/architecture.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions

from kohakuwullm.kernels.loss.chunked_ce import chunked_linear_cross_entropy
from kohakuwullm.kernels.loss.zloss import logsumexp_square


class LMHead(nn.Module):
    """Vocabulary projection + loss. See docs/concepts/architecture.md.

    Args:
        dim: model width.
        vocab_size: output vocabulary.
        tie_embeddings: share the weight with the input embedding. Every preset
            leaves this **off**.
        embedding: the ``nn.Embedding`` to tie to (required when tying).
        z_loss_weight: penalize ``logsumexp(logits)^2``, via
            :mod:`kohakuwullm.kernels.loss.zloss`.
        soft_cap: ``cap * tanh(logits / cap)`` before the loss; forces the
            materializing path.
        label_smoothing: passed through to cross-entropy.
        ignore_index: label value excluded from the loss.
        chunked: ATen path only: pass an explicit ``LinearCrossEntropyOptions()``
            to engage its chunked path.
        kernel: ``"chunked_ce"`` (ours) or ``"torch"`` (ATen), selected once.
        chunk: rows per logit tile for ``chunked_ce``.
        vocab_block: columns per logit tile; also bounds the fp32 ``dW``
            accumulator.
        retain: fraction of forward logit tiles cached for backward. Any value
            above 0 makes a second backward on the same graph raise.
        compute_dtype: dtype the ``chunked_ce`` GEMMs run in.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        tie_embeddings: bool = True,
        embedding: nn.Embedding | None = None,
        z_loss_weight: float = 0.0,
        soft_cap: float | None = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        chunked: bool = False,
        kernel: str = "chunked_ce",
        chunk: int = 8192,
        vocab_block: int = 4096,
        retain: float = 1.0,
        compute_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.z_loss_weight = z_loss_weight
        self.soft_cap = soft_cap
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.options = LinearCrossEntropyOptions() if chunked else None
        self.kernel = kernel
        self.chunk = chunk
        self.vocab_block = vocab_block
        self.retain = retain
        self.compute_dtype = compute_dtype
        if kernel not in ("torch", "chunked_ce"):
            raise ValueError(f"unknown head kernel {kernel!r}")
        if kernel == "chunked_ce" and label_smoothing:
            raise ValueError("chunked_ce does not implement label smoothing")

        if tie_embeddings:
            if embedding is None:
                raise ValueError("tie_embeddings=True requires the `embedding` module")
            self._tied = (embedding,)  # a tuple, so it is not a submodule
            self.weight = None
        else:
            self._tied = ()
            self.weight = nn.Parameter(torch.empty(vocab_size, dim))
            nn.init.normal_(self.weight, std=dim**-0.5)

    @property
    def projection(self) -> torch.Tensor:
        return self._tied[0].weight if self.tie_embeddings else self.weight

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Materialize logits. Generation / analysis only -- never in training."""
        out = F.linear(hidden, self.projection)
        if self.soft_cap is not None:
            out = self.soft_cap * torch.tanh(out / self.soft_cap)
        return out

    def token_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-token cross-entropy, ``(N,)`` in fp32, zero at ignored positions.

        Always ``reduction="none"``, so the reduction happens here in fp32.
        """
        hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = labels.reshape(-1)
        if self.soft_cap is not None:
            logits = self.logits(hidden)
            per_token = F.cross_entropy(
                logits.float(),
                labels,
                ignore_index=self.ignore_index,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
        elif self.kernel == "chunked_ce":
            per_token = chunked_linear_cross_entropy(
                hidden,
                self.projection,
                labels,
                ignore_index=self.ignore_index,
                chunk=self.chunk,
                vocab_block=self.vocab_block,
                retain=self.retain,
                compute_dtype=self.compute_dtype,
            ).float()
        else:
            per_token = F.linear_cross_entropy(
                hidden,
                self.projection,
                labels,
                ignore_index=self.ignore_index,
                reduction="none",
                label_smoothing=self.label_smoothing,
                options=self.options,
            ).float()
        return per_token

    def loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "sum",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cross-entropy (+ z-loss) over ``(N, D)`` hidden states and ``(N,)`` labels.

        With ``reduction="sum"`` the caller divides by the *global* trained-token
        count itself. Returns ``(loss, logs)``, ``logs`` carrying the
        cross-entropy separately from the z-loss.
        """
        per_token = self.token_loss(hidden, labels)
        n_valid = (labels.reshape(-1) != self.ignore_index).sum()
        ce = (
            per_token.sum()
            if reduction == "sum"
            else per_token.sum() / n_valid.clamp_min(1)
        )
        logs = {"ce": ce.detach(), "n_tokens": n_valid.detach()}

        if self.z_loss_weight <= 0:
            return ce, logs
        z_sq = logsumexp_square(
            hidden.reshape(-1, hidden.shape[-1]),
            self.projection,
            labels.reshape(-1),
            self.ignore_index,
        )
        z = z_sq.sum() if reduction == "sum" else z_sq.sum() / n_valid.clamp_min(1)
        logs["z_loss"] = z.detach()
        return ce + self.z_loss_weight * z, logs
