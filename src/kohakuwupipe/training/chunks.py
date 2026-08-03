"""Several model chunks on one rank, behind a single module.

Interleaved and V-shaped schedules give a rank non-adjacent slices of the model,
while the loop, the optimizer and the grad scaler each want one module.
See docs/kohakuwupipe/schedules.md.
"""

import torch.nn as nn
from torch.distributed.pipelining.schedules import generate_stage_to_rank_mapping


class ChunkedStage(nn.Module):
    """This rank's chunks, ordered by stage index.

    ``set_seq_info`` reaches every chunk; ``loss`` reaches the chunk holding the
    head, which under a V placement is on the same rank as the embedding.
    """

    def __init__(self, chunks, plans) -> None:
        super().__init__()
        self.chunks = nn.ModuleList(chunks)
        self.plans = list(plans)

    def set_seq_info(self, layout) -> None:
        for chunk in self.chunks:
            setter = getattr(chunk, "set_seq_info", None)
            if setter is not None:
                setter(layout)

    def loss(self, hidden, target):
        for chunk, plan in zip(self.chunks, self.plans):
            if plan.is_last:
                return chunk.loss(hidden, target)
        raise RuntimeError("loss called on a rank that holds no final chunk")


def local_stages(num_stages: int, rank: int, world: int, style: str) -> list[int]:
    """Stage indices this rank owns, ascending.

    ``style`` is ``"v"`` for the V-shaped schedules and ``"loop"`` for the
    interleaved ones; the mapping itself is torch's.
    """
    mapping = generate_stage_to_rank_mapping(world, num_stages, style=style)
    return [index for index in sorted(mapping) if mapping[index] == rank]
