"""``SeqInfo`` -- packed vs padded batch layout. Only attention reads it.

See docs/concepts/architecture.md.
"""

from dataclasses import dataclass

import torch


@dataclass
class SeqInfo:
    """Layout descriptor for one batch.

    Attributes:
        cu_seqlens: ``(N+1,)`` int32 cumulative document lengths, or ``None``
            for the padded layout.
        max_seqlen: longest document in the batch (the varlen kernel needs it
            to size its tiles).
        position_ids: ``(T,)`` packed or ``(B, S)`` padded; restarts per doc.
        num_seqs: number of documents.
    """

    cu_seqlens: torch.Tensor | None
    max_seqlen: int
    position_ids: torch.Tensor
    num_seqs: int

    @property
    def packed(self) -> bool:
        return self.cu_seqlens is not None

    @property
    def seqlens(self) -> torch.Tensor:
        """``(N,)`` per-document lengths."""
        if self.cu_seqlens is None:
            raise ValueError("seqlens is only defined for the packed layout")
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]

    def to(self, device: torch.device) -> "SeqInfo":
        return SeqInfo(
            cu_seqlens=None if self.cu_seqlens is None else self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            position_ids=self.position_ids.to(device),
            num_seqs=self.num_seqs,
        )

    @staticmethod
    def from_lengths(
        lengths: torch.Tensor, device: torch.device | None = None
    ) -> "SeqInfo":
        """Build a packed :class:`SeqInfo` from ``(N,)`` document lengths."""
        lengths = lengths.to(device=device, dtype=torch.int32)
        cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
        torch.cumsum(lengths, 0, out=cu[1:])
        position_ids = (
            torch.cat(
                [torch.arange(int(n), device=lengths.device) for n in lengths.tolist()]
            )
            if lengths.numel()
            else torch.zeros(0, dtype=torch.long, device=lengths.device)
        )
        return SeqInfo(
            cu_seqlens=cu,
            max_seqlen=int(lengths.max().item()) if lengths.numel() else 0,
            position_ids=position_ids.long(),
            num_seqs=lengths.numel(),
        )

    @staticmethod
    def padded(
        batch_size: int, seq_len: int, device: torch.device | None = None
    ) -> "SeqInfo":
        """Build a padded :class:`SeqInfo` for a dense ``(B, S)`` batch."""
        pos = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        return SeqInfo(
            cu_seqlens=None,
            max_seqlen=seq_len,
            position_ids=pos,
            num_seqs=batch_size,
        )
